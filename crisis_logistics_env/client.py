# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Client for the LogiFlow-RL environment server."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import CrisisLogisticsAction, CrisisLogisticsObservation, CrisisLogisticsState


class CrisisLogisticsEnv(
    EnvClient[CrisisLogisticsAction, CrisisLogisticsObservation, CrisisLogisticsState]
):
    """Thin client that talks to the HTTP or WebSocket server."""

    def _step_payload(self, action: CrisisLogisticsAction) -> Dict:
        return {"target_hub": action.target_hub}

    def _parse_result(self, payload: Dict) -> StepResult[CrisisLogisticsObservation]:
        obs_data = payload.get("observation", {})
        observation = CrisisLogisticsObservation(
            task_id=obs_data.get("task_id", "easy"),
            difficulty=obs_data.get("difficulty", "easy"),
            objective=obs_data.get("objective", ""),
            hub_loads=obs_data.get("hub_loads", [0.0, 0.0, 0.0]),
            drain_rates=obs_data.get("drain_rates", [6.0, 5.0, 4.0]),
            incoming_load=obs_data.get("incoming_load", 0.0),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 100),
            overloaded_hubs=obs_data.get("overloaded_hubs", 0),
            cumulative_score=obs_data.get("cumulative_score", 0.0),
            last_reward=obs_data.get("last_reward", 0.0),
            event_label=obs_data.get("event_label", "normal"),
            message=obs_data.get("message", ""),
            reward=payload.get("reward"),
            done=payload.get("done", False),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> CrisisLogisticsState:
        return CrisisLogisticsState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", "easy"),
            difficulty=payload.get("difficulty", "easy"),
            hub_loads=payload.get("hub_loads", [0.0, 0.0, 0.0]),
            incoming_index=payload.get("incoming_index", 0),
            bottlenecks=payload.get("bottlenecks", 0),
            score=payload.get("score", 0.0),
        )
