from __future__ import annotations

import uuid
from typing import List

from openenv.core.env_server import Environment

try:
    from ..graders import EpisodeMetrics, grade_episode
    from ..models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from ..tasks import get_task, list_tasks
except ImportError:
    from graders import EpisodeMetrics, grade_episode
    from models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from tasks import get_task, list_tasks


class CrisisLogisticsEnvironment(
    Environment[CrisisLogisticsAction, CrisisLogisticsObservation, CrisisLogisticsState]
):
    """
    LogiFlow-RL: a deterministic supply-chain balancing benchmark.

    The environment exposes three benchmark tasks with increasing difficulty.
    Each episode provides a fixed shipment schedule so scores are reproducible.
    """

    def __init__(self):
        super().__init__()
        self.optimal_zone = (30.0, 70.0)
        self.available_tasks = list_tasks()
        self.reset(task_id="easy")

    def reset(
        self, seed=None, episode_id=None, task_id: str = "easy", **kwargs
    ) -> CrisisLogisticsObservation:
        self.task = get_task(task_id)
        self.episode_id = episode_id or str(uuid.uuid4())
        self.hub_loads = self.task.initial_loads[:]
        self.drain_rates = self.task.drain_rates[:]
        self.step_count = 0
        self.schedule_index = 0
        self.done = False
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.optimal_steps = 0
        self.bottlenecks = 0
        self.balance_gap_history: List[float] = []
        self.throughput_served = 0.0
        self.event_label = self.task.event_schedule[0]
        self.incoming_load = self.task.incoming_schedule[0]
        self.score = 0.0
        return self._get_observation(
            f"Task '{self.task.title}' initialized. Route the scheduled shipment stream."
        )

    def step(
        self, action: CrisisLogisticsAction, timeout_s=None, **kwargs
    ) -> CrisisLogisticsObservation:
        if self.done:
            observation = self._get_observation("Episode already finished.")
            observation.reward = 0.0
            return observation

        selected = action.target_hub
        if selected not in (0, 1, 2):
            observation = self._get_observation("Invalid hub selected.")
            observation.reward = 0.0
            return observation

        self.step_count += 1

        for hub_index in range(3):
            if hub_index != selected:
                self.hub_loads[hub_index] = max(
                    0.0, self.hub_loads[hub_index] - self.drain_rates[hub_index]
                )

        self.hub_loads[selected] += self.incoming_load
        self.throughput_served += self.incoming_load

        overloaded = any(load > 100.0 for load in self.hub_loads)
        if overloaded:
            self.bottlenecks += 1

        reward = self._calculate_step_reward(selected)
        self.total_reward += reward
        self.last_reward = reward

        if self._is_optimal_state():
            self.optimal_steps += 1

        self.balance_gap_history.append(max(self.hub_loads) - min(self.hub_loads))

        if self.step_count >= self.task.max_steps:
            self.done = True
        else:
            self.schedule_index = self.step_count
            self.event_label = self.task.event_schedule[self.schedule_index]
            self.incoming_load = self.task.incoming_schedule[self.schedule_index]

        self.score = self._compute_score()
        observation = self._get_observation(
            f"Shipment routed to Hub {selected}. Loads: {[round(load, 1) for load in self.hub_loads]}"
        )
        observation.reward = reward
        observation.done = self.done
        return observation

    def _calculate_step_reward(self, selected: int) -> float:
        target_load = self.hub_loads[selected]
        center = 50.0
        zone_bonus = 1.0 if 30.0 <= target_load <= 70.0 else 0.0
        load_penalty = min(abs(target_load - center) / 50.0, 1.0)
        balance_gap = max(self.hub_loads) - min(self.hub_loads)
        balance_penalty = min(balance_gap / 100.0, 1.0)
        overload_penalty = 1.0 if target_load > 100.0 else 0.0

        reward = 0.55 + 0.35 * zone_bonus - 0.25 * load_penalty - 0.20 * balance_penalty - 0.45 * overload_penalty
        return round(max(0.0, min(1.0, reward)), 2)

    def _is_optimal_state(self) -> bool:
        low, high = self.optimal_zone
        return all(low <= load <= high for load in self.hub_loads)

    def _compute_score(self) -> float:
        metrics = EpisodeMetrics(
            total_reward=self.total_reward,
            average_reward=self.total_reward / max(self.step_count, 1),
            bottlenecks=self.bottlenecks,
            optimal_steps=self.optimal_steps,
            average_balance_gap=sum(self.balance_gap_history) / max(len(self.balance_gap_history), 1),
            throughput_served=self.throughput_served,
            steps_completed=self.step_count,
        )
        return grade_episode(self.task, metrics)

    def _get_observation(self, message: str) -> CrisisLogisticsObservation:
        overloaded_hubs = sum(1 for load in self.hub_loads if load > 100.0)
        next_incoming = 0.0 if self.done else self.incoming_load
        next_event = "completed" if self.done else self.event_label
        return CrisisLogisticsObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            hub_loads=[round(load, 2) for load in self.hub_loads],
            drain_rates=self.drain_rates[:],
            incoming_load=next_incoming,
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            overloaded_hubs=overloaded_hubs,
            cumulative_score=self.score,
            last_reward=self.last_reward,
            event_label=next_event,
            message=message,
            reward=self.last_reward,
            done=self.done,
            metadata={
                "title": self.task.title,
                "available_tasks": [task.task_id for task in self.available_tasks],
                "bottlenecks": self.bottlenecks,
            },
        )

    @property
    def state(self) -> CrisisLogisticsState:
        return CrisisLogisticsState(
            episode_id=self.episode_id,
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            step_count=self.step_count,
            hub_loads=[round(load, 2) for load in self.hub_loads],
            incoming_index=self.schedule_index,
            bottlenecks=self.bottlenecks,
            score=self.score,
        )


def choose_balancing_action(observation: CrisisLogisticsObservation) -> int:
    """Deterministic heuristic baseline used for smoke tests and offline fallback."""

    best_idx = 0
    best_score = float("inf")
    for index in range(3):
        projected = observation.hub_loads[:]
        for drain_idx in range(3):
            if drain_idx != index:
                projected[drain_idx] = max(
                    0.0, projected[drain_idx] - observation.drain_rates[drain_idx]
                )
        projected[index] += observation.incoming_load

        balance_gap = max(projected) - min(projected)
        overload_penalty = 40.0 if projected[index] > 100.0 else 0.0
        zone_penalty = sum(abs(load - 50.0) for load in projected) / 3.0
        projected_score = overload_penalty + balance_gap + zone_penalty
        if projected_score < best_score:
            best_score = projected_score
            best_idx = index
    return best_idx
