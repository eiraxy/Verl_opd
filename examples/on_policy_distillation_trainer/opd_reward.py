"""Reward shim for on-policy distillation, and the per-dataset OPD/RL router.

Two jobs.

**Never lose a trajectory.**  `distillation.distillation_loss.use_task_rewards=False`
drops the task reward from the loss, but it does not stop verl from computing
one: RewardLoopManager hands out live workers whenever no reward model is
configured, so AgentLoopWorker scores every trajectory inline.  An exception
raised there is caught by the agent loop and the whole trajectory is discarded
as a failure, which means an unregistered data_source silently deletes most of
each batch and the replay buffer backfills with all-zero padding rows.  Hence
the blanket except below: a bad score must never cost us the rollout.

**Route the task reward per dataset.**  `use_task_rewards` is one global flag,
so "OPD on the general pool, OPD + RL on the tool pool" cannot be expressed in
config.  It can be expressed here, because of how the two loss terms get their
advantages (verl/trainer/distillation/losses.py):

  - the distillation term builds its own advantage from the negated per-token
    teacher KL, so it applies to every row no matter what we return;
  - the task term is `ppo_loss`, which is linear in `advantages` -- and with
    `entropy_coeff=0` and `use_kl_loss=False` that is all it is.

A constant 0.0 reward therefore switches the task term off for a row, under
both branches of `compute_grpo_outcome_advantage`: a singleton group takes
mean=0/std=1 and yields the raw score, and a group of equal scores is centred
to exactly zero.  So the general pool gets teacher KL only, the tool pool gets
teacher KL plus a real exact-match gradient, in one run with the flag on.

Pinning the general sources explicitly matters -- three of the four happen to
be unregistered and would fall through to 0.0 anyway, but `taco` is registered
and would otherwise contribute a live sandbox-scored gradient. Validation
sources are deliberately not listed, so their metrics stay real.
"""

import logging
import os

from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)

# Training data_sources of the general OPD pool (train_5k_per_domain.parquet).
# Override with OPD_ONLY_DATA_SOURCES="a,b" (empty string disables the routing).
_DEFAULT_OPD_ONLY = "math_dapo_boxed,nemotron_if_rl,primeintellect,taco"
OPD_ONLY_DATA_SOURCES = {
    name.strip()
    for name in os.environ.get("OPD_ONLY_DATA_SOURCES", _DEFAULT_OPD_ONLY).split(",")
    if name.strip()
}

_warned: set[str] = set()


def _warn_once(data_source: str, reason: str) -> None:
    if data_source not in _warned:
        _warned.add(data_source)
        logger.warning("opd_reward: scoring data_source=%s as 0.0 (%s)", data_source, reason)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    if data_source in OPD_ONLY_DATA_SOURCES:
        # Distillation-only row: a constant reward flattens its GRPO group, so
        # the policy-gradient term sees an advantage of exactly zero.
        return 0.0
    try:
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    except NotImplementedError:
        _warn_once(str(data_source), "no reward function registered")
    except Exception as e:  # noqa: BLE001 - a bad score must never cost us the trajectory
        _warn_once(str(data_source), f"{type(e).__name__}: {e}")
    return 0.0
