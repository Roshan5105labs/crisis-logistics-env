from __future__ import annotations

from dataclasses import dataclass

from graders import EpisodeMetrics, grade_episode
from models import CrisisLogisticsAction
from server.crisis_logistics_env_environment import (
    CrisisLogisticsEnvironment,
    choose_network_action,
)
from tasks import list_tasks


@dataclass
class EpisodeSummary:
    task_id: str
    total_reward: float
    score: float
    bottlenecks: int


def run_policy(task_id: str, policy: str) -> EpisodeSummary:
    env = CrisisLogisticsEnvironment()
    observation = env.reset(task_id=task_id)
    round_robin_step = 0

    while not observation.done:
        if policy == "round_robin":
            action = round_robin_step % 3
            round_robin_step += 1
            observation = env.step(CrisisLogisticsAction(target_hub=action))
        else:
            observation = env.step(choose_network_action(observation))

    metrics = EpisodeMetrics(
        total_reward=env.total_reward,
        average_reward=env.total_reward / max(env.step_count, 1),
        bottlenecks=env.bottlenecks,
        optimal_steps=env.optimal_steps,
        average_balance_gap=sum(env.balance_gap_history) / max(len(env.balance_gap_history), 1),
        throughput_served=env.throughput_served,
        steps_completed=env.step_count,
        retail_delivered=env.retail_delivered,
        sla_success_rate=env._sla_success_rate(),
        disruption_recovery_score=sum(env.recovery_history) / max(len(env.recovery_history), 1),
        invalid_actions=env.invalid_actions,
    )
    score = grade_episode(env.task, metrics)
    return EpisodeSummary(
        task_id=task_id,
        total_reward=round(env.total_reward, 2),
        score=score,
        bottlenecks=env.bottlenecks,
    )


def main() -> None:
    print("LogiFlow-RL Benchmarks")
    print("----------------------")
    for policy in ("round_robin", "heuristic"):
        print(f"\nPolicy: {policy}")
        scores = []
        for task in list_tasks():
            summary = run_policy(task.task_id, policy)
            scores.append(summary.score)
            print(
                f"{summary.task_id:6} | reward={summary.total_reward:6.2f} | "
                f"score={summary.score:0.3f} | bottlenecks={summary.bottlenecks}"
            )
        avg_score = sum(scores) / len(scores)
        print(f"average | score={avg_score:0.3f}")


if __name__ == "__main__":
    main()
