from __future__ import annotations

from dataclasses import dataclass

try:
    from .tasks import TaskConfig
except ImportError:
    from tasks import TaskConfig


@dataclass
class EpisodeMetrics:
    total_reward: float
    average_reward: float
    bottlenecks: int
    optimal_steps: int
    average_balance_gap: float
    throughput_served: float
    steps_completed: int
    retail_delivered: float = 0.0
    sla_success_rate: float = 0.0
    disruption_recovery_score: float = 0.0
    invalid_actions: int = 0


def grade_episode(task: TaskConfig, metrics: EpisodeMetrics) -> float:
    """Deterministic multi-component grader normalized to [0.0, 1.0]."""

    bottleneck_score = max(
        0.0,
        1.0 - max(0, metrics.bottlenecks - task.target_bottlenecks) / max(1, task.max_steps / 8),
    )
    balance_score = max(
        0.0,
        1.0 - max(0.0, metrics.average_balance_gap - task.target_balance_gap) / 0.65,
    )
    reward_score = min(1.0, metrics.average_reward / max(task.minimum_avg_reward, 0.01))
    delivery_score = min(1.0, metrics.retail_delivered / max(task.target_retail_delivery, 1.0))
    sla_score = min(1.0, metrics.sla_success_rate / max(task.target_sla, 0.01))
    recovery_score = max(0.0, min(1.0, metrics.disruption_recovery_score))
    validity_score = max(0.0, 1.0 - metrics.invalid_actions / max(1, metrics.steps_completed))

    final_score = (
        0.12 * bottleneck_score
        + 0.10 * balance_score
        + 0.10 * reward_score
        + 0.32 * delivery_score
        + 0.20 * sla_score
        + 0.10 * recovery_score
        + 0.06 * validity_score
    )
    return round(max(0.0, min(1.0, final_score)), 3)
