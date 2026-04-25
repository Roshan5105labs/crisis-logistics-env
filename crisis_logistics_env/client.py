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
        return action.model_dump(exclude_none=True)

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
            dynamic_pressure=obs_data.get("dynamic_pressure", 0.0),
            adaptive_disruption_rate=obs_data.get("adaptive_disruption_rate", 0.0),
            priority_target_node=obs_data.get("priority_target_node", 10),
            priority_target_name=obs_data.get("priority_target_name", "Retail North"),
            priority_queue_depth=obs_data.get("priority_queue_depth", 0),
            priority_service_rate=obs_data.get("priority_service_rate", 0.0),
            message=obs_data.get("message", ""),
            node_names=obs_data.get("node_names", []),
            node_types=obs_data.get("node_types", []),
            node_loads=obs_data.get("node_loads", []),
            node_capacities=obs_data.get("node_capacities", []),
            node_utilization=obs_data.get("node_utilization", []),
            node_drain_rates=obs_data.get("node_drain_rates", []),
            node_risk_scores=obs_data.get("node_risk_scores", []),
            connectivity=obs_data.get("connectivity", {}),
            visible_node_ids=obs_data.get("visible_node_ids", []),
            observed_node_loads=obs_data.get("observed_node_loads", []),
            visible_connectivity=obs_data.get("visible_connectivity", {}),
            in_transit_shipments=obs_data.get("in_transit_shipments", []),
            active_disruptions=obs_data.get("active_disruptions", []),
            reward_breakdown=obs_data.get("reward_breakdown", {}),
            last_action=obs_data.get("last_action", {}),
            pending_source_node=obs_data.get("pending_source_node", 0),
            retail_delivered=obs_data.get("retail_delivered", 0.0),
            sla_success_rate=obs_data.get("sla_success_rate", 0.0),
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
            node_loads=payload.get("node_loads", []),
            node_utilization=payload.get("node_utilization", []),
            in_transit_count=payload.get("in_transit_count", 0),
            active_disruptions=payload.get("active_disruptions", []),
            retail_delivered=payload.get("retail_delivered", 0.0),
            sla_success_rate=payload.get("sla_success_rate", 0.0),
            dynamic_pressure=payload.get("dynamic_pressure", 0.0),
            adaptive_disruption_rate=payload.get("adaptive_disruption_rate", 0.0),
            priority_target_node=payload.get("priority_target_node", 10),
            priority_service_rate=payload.get("priority_service_rate", 0.0),
        )
