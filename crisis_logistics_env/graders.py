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


def grade_episode(task: TaskConfig, metrics: EpisodeMetrics) -> float:
    bottleneck_score = max(
        0.0,
        1.0 - max(0, metrics.bottlenecks - task.target_bottlenecks) / max(1, task.max_steps / 4),
    )
    balance_score = max(
        0.0,
        1.0 - max(0.0, metrics.average_balance_gap - task.target_balance_gap) / 40.0,
    )
    efficiency_score = min(1.0, metrics.average_reward / max(task.minimum_avg_reward, 0.01))
    stability_score = metrics.optimal_steps / task.max_steps

    final_score = (
        0.35 * bottleneck_score
        + 0.25 * balance_score
        + 0.20 * efficiency_score
        + 0.20 * stability_score
    )
    return round(max(0.0, min(1.0, final_score)), 3)
