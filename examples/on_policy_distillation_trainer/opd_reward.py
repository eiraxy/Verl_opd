"""Reward shim for on-policy distillation.

`distillation.distillation_loss.use_task_rewards=False` drops the task reward from
the loss, but it does not stop verl from computing one: RewardLoopManager hands out
live workers whenever no reward model is configured, so AgentLoopWorker scores every
trajectory inline. An exception raised there is caught by the agent loop and the
whole trajectory is discarded as a failure, which means an unregistered data_source
silently deletes most of each batch and the replay buffer backfills with all-zero
padding rows.

Scores that are actually read (validation metrics) still come from
`default_compute_score`; everything else degrades to a harmless 0.0 instead of a
dropped rollout.
"""

import logging

from verl.utils.reward_score import default_compute_score

logger = logging.getLogger(__name__)

_warned: set[str] = set()


def _warn_once(data_source: str, reason: str) -> None:
    if data_source not in _warned:
        _warned.add(data_source)
        logger.warning("opd_reward: scoring data_source=%s as 0.0 (%s)", data_source, reason)


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    try:
        return default_compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs)
    except NotImplementedError:
        _warn_once(str(data_source), "no reward function registered")
    except Exception as e:  # noqa: BLE001 - a bad score must never cost us the trajectory
        _warn_once(str(data_source), f"{type(e).__name__}: {e}")
    return 0.0
