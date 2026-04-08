from __future__ import annotations

import random
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class LogiFlowGymEnv(gym.Env):
    """Gymnasium wrapper for the 3-hub logistics balancing problem."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, max_steps: int = 100):
        super().__init__()
        self.max_steps = max_steps
        self.base_drain_rates = np.array([8.0, 7.0, 6.0], dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([150, 150, 150, 20, 20, 20, 40, 2], dtype=np.float32),
            dtype=np.float32,
        )
        self.reset()

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            random.seed(seed)

        self.hub_loads = np.array([24.0, 36.0, 30.0], dtype=np.float32)
        self.drain_rates = self.base_drain_rates.copy()
        self.step_count = 0
        self.event_label = "normal"
        self.incoming_load = self._sample_incoming_load()
        return self._get_obs(), self._get_info(0.0)

    def step(self, action: int):
        self.step_count += 1

        for idx in range(3):
            if idx != action:
                self.hub_loads[idx] = max(0.0, self.hub_loads[idx] - self.drain_rates[idx])

        self.hub_loads[action] += self.incoming_load
        reward = self._calculate_reward(action)

        self.event_label = self._sample_event_label()
        self.incoming_load = self._sample_incoming_load(self.event_label)

        terminated = False
        truncated = self.step_count >= self.max_steps
        return self._get_obs(), reward, terminated, truncated, self._get_info(reward)

    def render(self):
        print(
            f"step={self.step_count} loads={self.hub_loads.round(1).tolist()} "
            f"incoming={self.incoming_load:.1f} event={self.event_label}"
        )

    def _get_obs(self) -> np.ndarray:
        event_id = {"normal": 0.0, "weather_disruption": 1.0, "flash_sale": 2.0}[self.event_label]
        return np.array(
            [
                self.hub_loads[0],
                self.hub_loads[1],
                self.hub_loads[2],
                self.drain_rates[0],
                self.drain_rates[1],
                self.drain_rates[2],
                self.incoming_load,
                event_id,
            ],
            dtype=np.float32,
        )

    def _get_info(self, reward: float) -> dict[str, Any]:
        return {
            "reward": reward,
            "overloaded_hubs": int(np.sum(self.hub_loads > 100.0)),
            "event_label": self.event_label,
        }

    def _sample_event_label(self) -> str:
        roll = random.random()
        if roll < 0.15:
            return "flash_sale"
        if roll < 0.25:
            return "weather_disruption"
        return "normal"

    def _sample_incoming_load(self, event_label: str | None = None) -> float:
        label = event_label or self.event_label
        if label == "flash_sale":
            return random.uniform(16.0, 24.0)
        if label == "weather_disruption":
            return random.uniform(11.0, 18.0)
        return random.uniform(6.0, 12.0)

    def _calculate_reward(self, action: int) -> float:
        reward = 0.5
        target_load = self.hub_loads[action]

        if 30.0 <= target_load <= 70.0:
            reward += 5.0
        else:
            reward -= min(abs(target_load - 50.0) / 15.0, 4.0)

        if target_load > 100.0:
            reward -= 20.0

        reward -= float(np.max(self.hub_loads) - np.min(self.hub_loads)) / 50.0
        reward -= float(np.sum(self.hub_loads > 100.0)) * 3.0
        return round(reward, 2)
