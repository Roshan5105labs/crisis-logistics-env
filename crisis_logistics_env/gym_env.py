from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .models import CrisisLogisticsAction
    from .server.crisis_logistics_env_environment import CrisisLogisticsEnvironment
except ImportError:
    from models import CrisisLogisticsAction
    from server.crisis_logistics_env_environment import CrisisLogisticsEnvironment


class LogiFlowGymEnv(gym.Env):
    """Gymnasium wrapper for the 12-node delayed logistics benchmark."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(self, task_id: str = "easy"):
        super().__init__()
        self.task_id = task_id
        self.env = CrisisLogisticsEnvironment()
        self.action_space = spaces.Dict(
            {
                "source_node": spaces.Discrete(12),
                "dest_node": spaces.Discrete(12),
                "shipment_volume": spaces.Box(low=1.0, high=60.0, shape=(), dtype=np.float32),
            }
        )
        self.observation_space = spaces.Box(low=0.0, high=1.5, shape=(20,), dtype=np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        task_id = (options or {}).get("task_id", self.task_id)
        self.observation = self.env.reset(seed=seed, task_id=task_id)
        return self._flatten(self.observation), self._info()

    def step(self, action: dict[str, Any]):
        env_action = CrisisLogisticsAction(
            source_node=int(action["source_node"]),
            dest_node=int(action["dest_node"]),
            shipment_volume=float(action["shipment_volume"]),
        )
        self.observation = self.env.step(env_action)
        terminated = bool(self.observation.done)
        truncated = False
        return self._flatten(self.observation), float(self.observation.reward or 0.0), terminated, truncated, self._info()

    def render(self):
        print(
            f"step={self.env.step_count} score={self.env.score:.3f} "
            f"retail={self.env.retail_delivered:.1f} transit={len(self.env.in_transit)}"
        )

    def _flatten(self, observation) -> np.ndarray:
        util = list(observation.node_utilization[:12])
        while len(util) < 12:
            util.append(0.0)
        extras = [
            observation.incoming_load / 60.0,
            len(observation.in_transit_shipments) / 25.0,
            len(observation.active_disruptions) / 12.0,
            observation.cumulative_score,
            observation.step_count / max(observation.max_steps, 1),
            observation.dynamic_pressure,
            observation.priority_service_rate,
            min(1.0, observation.adaptive_disruption_rate),
        ]
        return np.array(util + extras, dtype=np.float32)

    def _info(self) -> dict[str, Any]:
        return {
            "score": self.env.score,
            "bottlenecks": self.env.bottlenecks,
            "retail_delivered": self.env.retail_delivered,
            "sla_success_rate": self.env._sla_success_rate(),
            "dynamic_pressure": self.env.dynamic_pressure,
            "priority_service_rate": self.env._priority_service_rate(),
        }
