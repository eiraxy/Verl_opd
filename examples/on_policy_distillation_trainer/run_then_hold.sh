#!/usr/bin/env bash
# Train, then park a GPU load on every card so the container is not reclaimed.
#
# The OPD run finishes in a few hours and an idle container gets killed, so the
# cards must never be left empty. The holder starts whether training succeeded or
# crashed -- a 3am crash is exactly when we need the cards held.
#
#   ./run_then_hold.sh                    # train, then hold
#   HOLD_MEM_UTIL=0.6 ./run_then_hold.sh  # lower the water mark
#   TRAIN=0 ./run_then_hold.sh            # hold only, skip training
#
# Extra args are forwarded to 0_origin_opd.sh as Hydra overrides.
#
# Stop the holder with 0_inference/stop_stress.sh (or touch 0_inference/STOP).
# To abort the whole chain, kill this wrapper's process group -- killing only the
# trainer is read as "training ended" and moves us straight to the holder.

set -uo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
INFER_DIR=${INFER_DIR:-/mnt/afs/huangxinyue/projects/0_inference}
# Invoke ray as a module, not via bin/ray: the copied env's console scripts kept
# a shebang pointing back at the conda env on quarkfs, where `ray stop` spends
# minutes importing instead of seconds.
PYTHON=${PYTHON:-/tmp/envs/verl/bin/python3}

# The requirement is a 60% floor per card. vLLM settles a few points above its
# gpu_memory_utilization target, so 0.7 keeps us clear of the floor even if the
# scheduler samples a trough.
HOLD_MEM_UTIL=${HOLD_MEM_UTIL:-0.7}

WRAP_LOG=$SCRIPT_DIR/logs/run_then_hold_$(date +%Y%m%d_%H%M%S).log
mkdir -p "$(dirname "$WRAP_LOG")"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$WRAP_LOG"; }

log "wrapper log: $WRAP_LOG"

if [[ ${TRAIN:-1} == 1 ]]; then
    log "starting training"
    bash "$SCRIPT_DIR/0_origin_opd.sh" "$@"
    log "training exited rc=$?"
else
    log "TRAIN=0, skipping training"
fi

# vLLM engine subprocesses and Ray workers outlive the driver, and they keep
# holding device memory -- which makes the holder's own
# gpu_memory_utilization check fail.
log "tearing down trainer leftovers"
timeout 120 "$PYTHON" -m ray.scripts.scripts stop --force >/dev/null 2>&1
pkill -f 'verl.trainer.main_ppo' 2>/dev/null
pkill -f 'VLLM::' 2>/dev/null
pkill -f 'ray/core/src/ray/(raylet|gcs)' 2>/dev/null

used=""
for _ in $(seq 1 90); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -rn | head -1)
    [[ $used -lt 2000 ]] && break
    sleep 2
done
if [[ ${used:-99999} -lt 2000 ]]; then
    log "gpu memory drained (max ${used} MiB)"
else
    log "WARNING: gpu memory still at ${used} MiB after 180s, starting the holder anyway"
fi

log "starting gpu holder at gpu_memory_utilization=$HOLD_MEM_UTIL"
exec bash "$INFER_DIR/run_stress_vllm.sh" "$HOLD_MEM_UTIL"
