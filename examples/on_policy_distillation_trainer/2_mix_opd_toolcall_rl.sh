#!/usr/bin/env bash
# OPD on the general pool + tool-call RL on the tool pool, in one run.
#
# The objective is split per row, not per config flag.
# `distillation.distillation_loss.use_task_rewards` is global, so it is turned
# ON here and the routing happens inside opd_reward.py: the four general
# data_sources are pinned to a constant 0.0, which centres their GRPO group to
# an advantage of exactly zero and removes them from the policy-gradient term,
# while the teacher-KL term -- which builds its own advantage from the negated
# per-token KL -- still applies to every row. Net effect:
#
#   general 15k (math/if/code) : teacher KL only          <- OPD
#   tool     4k (ToolAlpaca)   : teacher KL + exact match <- OPD + RL
#
# Both pools must live in ONE parquet. verl ends RLHFDataset with
# datasets.concatenate_datasets, which rejects the two source files outright
# (large_string vs string, and extra_info structs sharing 4 of 17 fields), so
# build_mixed_rl.py rewrites them against a single schema:
#
#   python3 Open-MOPD/experiments/data/build_mixed_rl.py
#
# Validation runs on the held-out 97 tool prompts, which share neither a query
# nor an API with training. With val_before_train the step-0 number should
# reproduce the offline 50.52 exact for the SFT student. aime25 is deliberately
# not added alongside it -- val_files is concatenated by the same intolerant
# code path, and its schema is further from the tool set than the two training
# pools were from each other. The general side stays on the offline harness
# under Open-MOPD/evals, which is what produced the leaderboard numbers.

set -xeuo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# ---- user-adjustable ----
STUDENT_MODEL=${STUDENT_MODEL:-/mnt/afs/huangxinyue/model_hf/qwen2.5-7b-toolcall-sft}
TEACHER_MODEL=${TEACHER_MODEL:-/mnt/afs/huangxinyue/model_hf/Qwen2.5-7B-Instruct}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-6}
TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE:-2}

distillation_loss_mode=${DISTILLATION_LOSS_MODE:-k1}
use_policy_gradient=${USE_POLICY_GRADIENT:-True}
distillation_topk=${DISTILLATION_TOPK:-64}

# Scales the teacher-KL term against the task-reward term. There is no matching
# task_loss_coef, so this is the only dial on the balance between the two -- and
# it is global, so lowering it to amplify the tool reward also weakens OPD on
# the general pool.
distillation_loss_coef=${DISTILLATION_LOSS_COEF:-1.0}

# A group of one is special cased to mean=0/std=1, so n=1 still learns from the
# raw reward, just without a baseline. n>1 buys the centred GRPO signal, at the
# cost of sampling the general rows -- which only feed the KL term -- n times too.
rollout_n=${ROLLOUT_N:-8}
rollout_temperature=${ROLLOUT_TEMPERATURE:-1.0}

# Length weighting decides how much of the batch the tool reward actually gets.
# Under verl's default token-mean, pg_loss is normalised by every response token
# in the batch while only the tool tokens carry a non-zero advantage; tool gold
# averages ~58 tokens against a 1024-token budget for math/code CoT, so a 21%
# row share collapses to roughly 2-3% of the gradient. seq-mean-token-mean
# weights each sequence equally and restores the row share -- at the price of
# also reweighting the distillation term, which no longer matches 0_origin_opd.sh.
loss_agg_mode=${LOSS_AGG_MODE:-seq-mean-token-mean}

# 18992 prompts, dp=6 -> 96 is divisible by 6; 197 steps per epoch.
train_batch_size=${TRAIN_BATCH_SIZE:-96}
ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-96}
max_prompt_length=${MAX_PROMPT_LENGTH:-2048}
max_response_length=${MAX_RESPONSE_LENGTH:-1024}
ppo_max_token_len_per_gpu=${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}

actor_lr=${ACTOR_LR:-1e-6}

rollout_tp=${ROLLOUT_TP:-1}
rollout_gpu_mem_util=${ROLLOUT_GPU_MEM_UTIL:-0.4}
teacher_tp=${TEACHER_TP:-1}
teacher_gpu_mem_util=${TEACHER_GPU_MEM_UTIL:-0.8}

total_epochs=${TOTAL_EPOCHS:-3}
save_freq=${SAVE_FREQ:-20}
test_freq=${TEST_FREQ:-20}

project_name=${PROJECT_NAME:-verl_toolcall_opd}
experiment_name=${EXPERIMENT_NAME:-qwen2_5_7b_mix_opd_toolcall_rl}
# ---- end user-adjustable ----

DATA_ROOT=${DATA_ROOT:-/mnt/afs/huangxinyue/projects/Open-MOPD/data}
mixed_train=${MIXED_TRAIN:-${DATA_ROOT}/mixed_opd_toolcall.parquet}
toolcall_test=${TOOLCALL_TEST:-${DATA_ROOT}/toolcall_rl_test97.parquet}

train_files="['$mixed_train']"
val_files="['$toolcall_test']"

max_num_tokens=$(( max_prompt_length + max_response_length + 1 ))

########################### local-disk caches ###########################
# /mnt/afs is a FUSE mount (quarkfs) where fcntl.flock() on an already-unlinked
# inode returns ENOENT, which breaks filelock inside datasets.load_dataset().
LOCAL_CACHE_ROOT=${LOCAL_CACHE_ROOT:-/tmp/${USER:-$(id -un)}/verl_cache}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$LOCAL_CACHE_ROOT/hf_datasets}
export HF_MODULES_CACHE=${HF_MODULES_CACHE:-$LOCAL_CACHE_ROOT/hf_modules}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$LOCAL_CACHE_ROOT/triton}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-$LOCAL_CACHE_ROOT/inductor}
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-$LOCAL_CACHE_ROOT/vllm}
mkdir -p "$HF_DATASETS_CACHE" "$HF_MODULES_CACHE" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$VLLM_CACHE_ROOT"

########################### sanity checks ###########################
for f in "$mixed_train" "$toolcall_test"; do
    [[ -f "$f" ]] || { echo "missing parquet: $f (run build_mixed_rl.py)" >&2; exit 1; }
done
total_gpus=$(( NGPUS_PER_NODE * NNODES + TEACHER_WORLD_SIZE * NNODES ))
if (( NGPUS_PER_NODE % rollout_tp != 0 )); then
    echo "rollout_tp=${rollout_tp} must divide NGPUS_PER_NODE=${NGPUS_PER_NODE}" >&2
    exit 1
fi
if (( TEACHER_WORLD_SIZE % teacher_tp != 0 )); then
    echo "teacher_tp=${teacher_tp} must divide TEACHER_WORLD_SIZE=${TEACHER_WORLD_SIZE}" >&2
    exit 1
fi
dp_size=$(( NGPUS_PER_NODE / rollout_tp ))
if (( train_batch_size % dp_size != 0 )); then
    echo "train_batch_size=${train_batch_size} must be divisible by dp=${dp_size}" >&2
    exit 1
fi
if (( rollout_n < 2 )); then
    echo "WARNING: rollout_n=${rollout_n} gives GRPO no group to centre against, so the" >&2
    echo "         task reward acts as a baseline-free REINFORCE signal" >&2
fi
echo "requires ${total_gpus} GPUs total: ${NGPUS_PER_NODE} trainer + ${TEACHER_WORLD_SIZE} teacher"

########################### parameter arrays ###########################

DATA=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files="$train_files"
    data.val_files="$val_files"
    data.train_batch_size=${train_batch_size}
    data.max_prompt_length=${max_prompt_length}
    data.max_response_length=${max_response_length}
    data.filter_overlong_prompts=True
    data.truncation='error'
    # Shuffling is what puts general and tool rows in the same batch, which the
    # per-row routing relies on to keep both terms alive at every step.
    data.shuffle=True
)

MODEL=(
    actor_rollout_ref.model.path="$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=True
    actor_rollout_ref.model.enable_gradient_checkpointing=True
)

ACTOR=(
    actor_rollout_ref.actor.use_torch_compile=True
    actor_rollout_ref.actor.optim.lr=${actor_lr}
    actor_rollout_ref.actor.ppo_mini_batch_size=${ppo_mini_batch_size}
    actor_rollout_ref.actor.use_dynamic_bsz=True
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}
    # The student is already constrained by the teacher; a second pull toward a
    # reference policy would double-regularize it. It also keeps ppo_loss purely
    # advantage-driven, which is what makes the zero-advantage routing exact.
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
)

ROLLOUT=(
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=${rollout_tp}
    actor_rollout_ref.rollout.gpu_memory_utilization=${rollout_gpu_mem_util}
    actor_rollout_ref.rollout.n=${rollout_n}
    actor_rollout_ref.rollout.temperature=${rollout_temperature}
    actor_rollout_ref.rollout.max_model_len=${max_num_tokens}
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_max_token_len_per_gpu}
    # Greedy validation so the online tool number is comparable to the offline
    # eval_vllm.py run, which uses temperature=0.
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0
    actor_rollout_ref.rollout.val_kwargs.n=1
)

TRAINER=(
    trainer.balance_batch=True
    trainer.logger='["wandb"]'
    trainer.project_name=${project_name}
    trainer.experiment_name=${experiment_name}
    trainer.n_gpus_per_node=${NGPUS_PER_NODE}
    trainer.nnodes=${NNODES}
    # Step-0 baseline: the tool split should land on the offline 50.52 exact.
    trainer.val_before_train=True
    trainer.save_freq=${save_freq}
    trainer.test_freq=${test_freq}
    trainer.total_epochs=${total_epochs}
)

REWARD=(
    # Holds the OPD/RL routing table, and keeps an unregistered data_source from
    # raising inside the agent loop and discarding the whole trajectory.
    reward.custom_reward_function.path="$SCRIPT_DIR/opd_reward.py"
)

EXTRA=(
    distillation.enabled=True
    distillation.n_gpus_per_node=${TEACHER_WORLD_SIZE}
    distillation.nnodes=${NNODES}
    distillation.teacher_models.teacher_model.model_path="$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=${teacher_tp}
    distillation.teacher_models.teacher_model.inference.name=vllm
    distillation.teacher_models.teacher_model.inference.gpu_memory_utilization=${teacher_gpu_mem_util}
    distillation.teacher_models.teacher_model.inference.max_model_len=${max_num_tokens}
    distillation.distillation_loss.loss_mode=${distillation_loss_mode}
    distillation.distillation_loss.topk=${distillation_topk}
    # Global on: the per-row routing in opd_reward.py decides who actually feels it.
    distillation.distillation_loss.use_task_rewards=True
    distillation.distillation_loss.distillation_loss_coef=${distillation_loss_coef}
    distillation.distillation_loss.use_policy_gradient=${use_policy_gradient}
    distillation.distillation_loss.clip_ratio_low=0.2
    distillation.distillation_loss.clip_ratio_high=0.28
    distillation.distillation_loss.loss_max_clamp=10.0
    distillation.distillation_loss.log_prob_min_clamp=-10.0
)

########################### interpreter ###########################
# Importing transformers out of the conda env on /mnt/afs takes ~25 minutes.
PYTHON=${PYTHON:-/tmp/envs/verl/bin/python3}
if [[ ! -x $PYTHON ]]; then
    echo "WARNING: $PYTHON missing, falling back to the /mnt/afs env (very slow startup)" >&2
    PYTHON=/mnt/afs/huangxinyue/miniconda3/envs/verl/bin/python3
fi
"$PYTHON" -c 'import sys; print("interpreter:", sys.executable)'

########################### launch ###########################
LOG_FILE=${LOG_FILE:-$SCRIPT_DIR/logs/${experiment_name}_$(date +%Y%m%d_%H%M%S).log}
mkdir -p "$(dirname "$LOG_FILE")"
echo "logging to ${LOG_FILE}"

"$PYTHON" -m verl.trainer.main_ppo \
    "${DATA[@]}" \
    "${MODEL[@]}" \
    "${ACTOR[@]}" \
    "${ROLLOUT[@]}" \
    "${TRAINER[@]}" \
    "${REWARD[@]}" \
    "${EXTRA[@]}" \
    "$@" 2>&1 | tee "$LOG_FILE"
