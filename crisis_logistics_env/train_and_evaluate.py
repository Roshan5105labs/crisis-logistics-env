from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import List, Optional

try:
    from crisis_logistics_env.graders import EpisodeMetrics, grade_episode
    from crisis_logistics_env.models import CrisisLogisticsAction
    from crisis_logistics_env.server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_network_action,
        choose_resilient_action,
    )
    from crisis_logistics_env.tasks import list_tasks
except ImportError:
    from graders import EpisodeMetrics, grade_episode
    from models import CrisisLogisticsAction
    from server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_network_action,
        choose_resilient_action,
    )
    from tasks import list_tasks


@dataclass
class EpisodeSummary:
    task_id: str
    policy: str
    total_reward: float
    average_reward: float
    score: float
    bottlenecks: int
    retail_delivered: float
    sla_success_rate: float
    priority_service_rate: float
    average_pressure: float
    invalid_actions: int
    reward_curve: List[float]


def run_policy(task_id: str, policy: str) -> EpisodeSummary:
    env = CrisisLogisticsEnvironment()
    observation = env.reset(task_id=task_id)
    round_robin_step = 0
    reward_curve: List[float] = []
    pressure_curve: List[float] = []

    while not observation.done:
        if policy == "round_robin":
            action = CrisisLogisticsAction(target_hub=round_robin_step % 3)
            round_robin_step += 1
        elif policy == "heuristic":
            action = choose_network_action(observation)
        elif policy == "resilient":
            action = choose_resilient_action(observation)
        else:
            raise ValueError(f"Unknown policy: {policy}")

        observation = env.step(action)
        reward_curve.append(float(observation.reward or 0.0))
        pressure_curve.append(float(observation.dynamic_pressure))

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
        policy=policy,
        total_reward=round(env.total_reward, 3),
        average_reward=round(metrics.average_reward, 3),
        score=score,
        bottlenecks=env.bottlenecks,
        retail_delivered=round(env.retail_delivered, 2),
        sla_success_rate=env._sla_success_rate(),
        priority_service_rate=env._priority_service_rate(),
        average_pressure=round(mean(pressure_curve) if pressure_curve else 0.0, 3),
        invalid_actions=env.invalid_actions,
        reward_curve=[round(v, 3) for v in reward_curve],
    )


def export_artifacts(summaries: List[EpisodeSummary]) -> tuple[Path, Path, Optional[Path]]:
    artifacts_dir = Path(__file__).resolve().parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifacts_dir / "benchmark_summary.json"
    curves_path = artifacts_dir / "reward_curves.csv"

    per_policy: dict[str, dict[str, float]] = {}
    for policy in sorted({summary.policy for summary in summaries}):
        rows = [summary for summary in summaries if summary.policy == policy]
        per_policy[policy] = {
            "avg_score": round(mean([row.score for row in rows]), 3),
            "avg_reward": round(mean([row.average_reward for row in rows]), 3),
            "avg_sla_success_rate": round(mean([row.sla_success_rate for row in rows]), 3),
            "avg_priority_service_rate": round(mean([row.priority_service_rate for row in rows]), 3),
            "avg_invalid_actions": round(mean([row.invalid_actions for row in rows]), 3),
        }

    payload = {
        "policies": per_policy,
        "runs": [{k: v for k, v in asdict(summary).items() if k != "reward_curve"} for summary in summaries],
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    with curves_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["task_id", "policy", "step", "reward"])
        for summary in summaries:
            for step, reward in enumerate(summary.reward_curve, start=1):
                writer.writerow([summary.task_id, summary.policy, step, reward])

    plot_path: Optional[Path] = None
    try:
        import matplotlib.pyplot as plt

        plot_path = artifacts_dir / "reward_curves.png"
        plt.figure(figsize=(10, 5))
        for policy in sorted({summary.policy for summary in summaries}):
            curves = [summary.reward_curve for summary in summaries if summary.policy == policy]
            max_len = max((len(curve) for curve in curves), default=0)
            if max_len == 0:
                continue
            mean_curve = [
                mean([curve[step] for curve in curves if step < len(curve)])
                for step in range(max_len)
            ]
            plt.plot(range(1, max_len + 1), mean_curve, linewidth=2, label=policy)
        plt.xlabel("Step")
        plt.ylabel("Reward")
        plt.title("LogiFlow-RL Baseline Reward Curves")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.savefig(plot_path, dpi=160, bbox_inches="tight")
        plt.close()
    except Exception as exc:
        print(f"Warning: matplotlib plot unavailable ({exc}); trying PIL fallback.")
        try:
            from PIL import Image, ImageDraw

            plot_path = artifacts_dir / "reward_curves.png"
            width, height = 1100, 560
            margin_left, margin_right = 70, 30
            margin_top, margin_bottom = 40, 60
            plot_w = width - margin_left - margin_right
            plot_h = height - margin_top - margin_bottom

            image = Image.new("RGB", (width, height), "white")
            draw = ImageDraw.Draw(image)

            per_policy_curves: dict[str, List[float]] = {}
            for policy in sorted({summary.policy for summary in summaries}):
                curves = [summary.reward_curve for summary in summaries if summary.policy == policy]
                max_len = max((len(curve) for curve in curves), default=0)
                if max_len == 0:
                    continue
                per_policy_curves[policy] = [
                    mean([curve[step] for curve in curves if step < len(curve)])
                    for step in range(max_len)
                ]

            if per_policy_curves:
                all_values = [value for curve in per_policy_curves.values() for value in curve]
                y_min = min(all_values)
                y_max = max(all_values)
                if abs(y_max - y_min) < 1e-6:
                    y_max = y_min + 1.0

                def map_x(step: int, max_len: int) -> int:
                    if max_len <= 1:
                        return margin_left
                    return int(margin_left + (step / (max_len - 1)) * plot_w)

                def map_y(value: float) -> int:
                    return int(margin_top + (1 - (value - y_min) / (y_max - y_min)) * plot_h)

                draw.rectangle(
                    [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
                    outline="#333333",
                    width=1,
                )

                grid_steps = 5
                for i in range(grid_steps + 1):
                    y = margin_top + int(i * plot_h / grid_steps)
                    draw.line([(margin_left, y), (margin_left + plot_w, y)], fill="#E6E6E6", width=1)

                colors = {
                    "round_robin": "#A84B4B",
                    "heuristic": "#2E7D32",
                    "resilient": "#1565C0",
                }
                legend_y = margin_top + 8
                legend_x = margin_left + 8

                for policy, curve in per_policy_curves.items():
                    color = colors.get(policy, "#444444")
                    points = [
                        (map_x(step, len(curve)), map_y(value))
                        for step, value in enumerate(curve)
                    ]
                    if len(points) >= 2:
                        draw.line(points, fill=color, width=3)
                    elif len(points) == 1:
                        x, y = points[0]
                        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
                    draw.rectangle((legend_x, legend_y, legend_x + 14, legend_y + 10), fill=color)
                    draw.text((legend_x + 20, legend_y - 1), policy, fill="#111111")
                    legend_y += 18

                draw.text((margin_left, height - 45), "Step", fill="#111111")
                draw.text((10, margin_top), "Reward", fill="#111111")
                draw.text((margin_left, 10), "LogiFlow-RL Baseline Reward Curves", fill="#111111")

                image.save(plot_path)
            else:
                plot_path = None
        except Exception as fallback_exc:
            print(f"Warning: could not generate reward_curves.png with PIL fallback ({fallback_exc})")
            plot_path = None

    return summary_path, curves_path, plot_path


def _print_table(summaries: List[EpisodeSummary]) -> None:
    print("task    | policy     | score | avg_reward | sla  | priority | bottlenecks | invalid")
    print("--------+------------+-------+------------+------+----------+-------------+--------")
    for summary in summaries:
        print(
            f"{summary.task_id:7} | {summary.policy:10} | {summary.score:0.3f} | "
            f"{summary.average_reward:0.3f}     | {summary.sla_success_rate:0.3f} | "
            f"{summary.priority_service_rate:0.3f}    | {summary.bottlenecks:11} | {summary.invalid_actions:7}"
        )


def main() -> None:
    print("LogiFlow-RL Benchmarks (Hackathon Evidence)")
    print("-------------------------------------------")
    print("Note: this script benchmarks non-LLM baselines only.")
    print("Run train_grpo.py for GRPO LLM training and before/after model evaluation.\n")
    summaries: List[EpisodeSummary] = []
    policies = ("round_robin", "heuristic", "resilient")

    for policy in policies:
        for task in list_tasks():
            summary = run_policy(task.task_id, policy)
            summaries.append(summary)

    _print_table(summaries)

    summary_path, curves_path, plot_path = export_artifacts(summaries)
    print("\nArtifacts")
    print(f"- Summary JSON: {summary_path}")
    print(f"- Reward curves: {curves_path}")
    if plot_path:
        print(f"- Reward curves plot: {plot_path}")

    baseline_scores = [row.score for row in summaries if row.policy == "round_robin"]
    resilient_scores = [row.score for row in summaries if row.policy == "resilient"]
    if baseline_scores and resilient_scores:
        baseline_avg = mean(baseline_scores)
        resilient_avg = mean(resilient_scores)
        delta = resilient_avg - baseline_avg
        pct = (delta / baseline_avg * 100.0) if baseline_avg > 0 else 0.0
        print(
            f"\nResilient policy improvement vs round_robin: "
            f"{delta:+0.3f} score points ({pct:+0.1f}%)."
        )


if __name__ == "__main__":
    main()
