from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    title: str
    objective: str
    max_steps: int
    initial_loads: List[float]
    drain_rates: List[float]
    incoming_schedule: List[float]
    event_schedule: List[str]
    target_bottlenecks: int
    target_balance_gap: float
    minimum_avg_reward: float


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        title="Steady-State Rebalancing",
        objective="Keep all hubs in the 30-70 utilization band during predictable daytime demand.",
        max_steps=12,
        initial_loads=[28.0, 42.0, 35.0],
        drain_rates=[8.0, 7.0, 6.0],
        incoming_schedule=[9.0, 8.0, 11.0, 10.0, 9.0, 12.0, 8.0, 10.0, 9.0, 11.0, 8.0, 10.0],
        event_schedule=[
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
            "normal",
        ],
        target_bottlenecks=0,
        target_balance_gap=25.0,
        minimum_avg_reward=0.55,
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        title="Flash Sale Containment",
        objective="Absorb an afternoon flash-sale surge without letting any single hub overload.",
        max_steps=14,
        initial_loads=[34.0, 48.0, 31.0],
        drain_rates=[8.0, 7.0, 6.0],
        incoming_schedule=[10.0, 12.0, 18.0, 22.0, 20.0, 16.0, 11.0, 13.0, 17.0, 14.0, 10.0, 9.0, 8.0, 10.0],
        event_schedule=[
            "normal",
            "normal",
            "flash_sale",
            "flash_sale",
            "flash_sale",
            "weather_disruption",
            "normal",
            "normal",
            "flash_sale",
            "weather_disruption",
            "normal",
            "normal",
            "normal",
            "normal",
        ],
        target_bottlenecks=0,
        target_balance_gap=28.0,
        minimum_avg_reward=0.48,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        title="Cascading Disruption Recovery",
        objective="Stabilize the network through repeated surge waves and weather disruptions while preserving throughput.",
        max_steps=16,
        initial_loads=[45.0, 57.0, 40.0],
        drain_rates=[8.0, 7.0, 6.0],
        incoming_schedule=[16.0, 22.0, 19.0, 24.0, 18.0, 20.0, 23.0, 14.0, 25.0, 17.0, 15.0, 21.0, 13.0, 18.0, 14.0, 12.0],
        event_schedule=[
            "weather_disruption",
            "flash_sale",
            "weather_disruption",
            "flash_sale",
            "normal",
            "weather_disruption",
            "flash_sale",
            "normal",
            "flash_sale",
            "weather_disruption",
            "normal",
            "flash_sale",
            "normal",
            "weather_disruption",
            "normal",
            "normal",
        ],
        target_bottlenecks=1,
        target_balance_gap=35.0,
        minimum_avg_reward=0.40,
    ),
}


def list_tasks() -> List[TaskConfig]:
    return [TASKS["easy"], TASKS["medium"], TASKS["hard"]]


def get_task(task_id: str) -> TaskConfig:
    return TASKS[task_id]
