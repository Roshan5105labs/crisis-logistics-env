# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the LogiFlow-RL environment."""

from typing import Any, Dict, List, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CrisisLogisticsAction(Action):
    """Move freight through the supply-chain network.

    The Phase 2 action uses source/destination/volume. The older target_hub
    field remains supported so existing validators and demos keep working.
    """

    target_hub: Optional[int] = Field(
        default=None,
        ge=0,
        le=11,
        description=(
            "Backward-compatible route selector. If source_node/dest_node are omitted, "
            "this chooses one outgoing edge from the current shipment source."
        ),
    )
    source_node: Optional[int] = Field(
        default=None,
        ge=0,
        le=11,
        description="Node id that dispatches freight.",
    )
    dest_node: Optional[int] = Field(
        default=None,
        ge=0,
        le=11,
        description="Connected downstream node id that receives freight after transit delay.",
    )
    shipment_volume: Optional[float] = Field(
        default=None,
        gt=0.0,
        le=60.0,
        description="Freight volume to move. Defaults to the active shipment volume.",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Optional natural-language rationale from the agent before the structured action.",
    )


class CrisisLogisticsObservation(Observation):
    """Partially observable state of the supply-chain network."""

    task_id: str = Field(default="easy", description="Active benchmark task identifier.")
    difficulty: str = Field(default="easy", description="Difficulty label for the active task.")
    objective: str = Field(default="", description="Task objective visible to the agent.")

    # Backward-compatible 3-value tier summary used by the existing visualizer.
    hub_loads: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Aggregate utilization for supplier, warehouse, and downstream tiers.",
    )
    drain_rates: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Aggregate drain rates for supplier, warehouse, and downstream tiers.",
    )

    incoming_load: float = Field(
        default=0.0,
        description="Volume of the active shipment introduced at the current source node.",
    )
    step_count: int = Field(default=0, description="Current step in the episode.")
    max_steps: int = Field(default=80, description="Maximum episode length.")
    overloaded_hubs: int = Field(
        default=0,
        description="Number of nodes currently above capacity.",
    )
    cumulative_score: float = Field(
        default=0.0,
        description="Normalized score in the range [0.0, 1.0] for progress so far.",
    )
    last_reward: float = Field(default=0.0, description="Reward from the previous action.")
    event_label: str = Field(
        default="normal",
        description="Current network condition, e.g. normal, port_closure, or supplier_failure.",
    )
    dynamic_pressure: float = Field(
        default=0.0,
        description="Adaptive stress estimate in [0, 1] driven by congestion, SLA risk, and delivery shortfall.",
    )
    adaptive_disruption_rate: float = Field(
        default=0.0,
        description="Effective disruption probability after adaptive pressure scaling.",
    )
    priority_target_node: int = Field(
        default=10,
        description="Retail node that should receive urgent demand in the current step window.",
    )
    priority_target_name: str = Field(
        default="Retail North",
        description="Human-readable name for the active priority retail target.",
    )
    priority_queue_depth: int = Field(
        default=0,
        description="Number of in-transit urgent shipments still pending completion.",
    )
    priority_service_rate: float = Field(
        default=0.0,
        description="Fraction of urgent shipments delivered on target and within SLA.",
    )
    message: str = Field(default="", description="Human-readable state summary.")

    node_names: List[str] = Field(default_factory=list, description="Names for all network nodes.")
    node_types: List[str] = Field(default_factory=list, description="Node type for each network node.")
    node_loads: List[float] = Field(default_factory=list, description="Current load at each node.")
    node_capacities: List[float] = Field(default_factory=list, description="Capacity for each node.")
    node_utilization: List[float] = Field(default_factory=list, description="Load/capacity for each node.")
    node_drain_rates: List[float] = Field(default_factory=list, description="Processing rate for each node.")
    node_risk_scores: List[float] = Field(default_factory=list, description="Disruption risk for each node.")
    connectivity: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Directed connectivity graph. Keys are node ids encoded as strings.",
    )

    visible_node_ids: List[int] = Field(
        default_factory=list,
        description="Nodes visible to the agent under the two-hop partial-observation rule.",
    )
    observed_node_loads: List[Optional[float]] = Field(
        default_factory=list,
        description="Node loads with hidden nodes represented as null.",
    )
    visible_connectivity: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Connectivity restricted to currently visible nodes.",
    )
    in_transit_shipments: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Shipments currently moving between nodes with delayed arrival.",
    )
    active_disruptions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active stochastic disruptions and their remaining duration.",
    )
    reward_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Inspectible per-step reward components.",
    )
    last_action: Dict[str, Any] = Field(default_factory=dict, description="Last resolved structured action.")
    pending_source_node: int = Field(default=0, description="Source node for the active incoming shipment.")
    retail_delivered: float = Field(default=0.0, description="Total volume that has arrived at retail sinks.")
    sla_success_rate: float = Field(default=0.0, description="Fraction of retail deliveries within SLA.")


class CrisisLogisticsState(State):
    """Internal environment state exposed for validation and debugging."""

    task_id: str = Field(default="easy", description="Active task identifier.")
    difficulty: str = Field(default="easy", description="Task difficulty label.")
    hub_loads: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Backward-compatible aggregate tier utilization.",
    )
    incoming_index: int = Field(default=0, description="Index of the next scheduled shipment.")
    bottlenecks: int = Field(default=0, description="Total overload events encountered this episode.")
    score: float = Field(default=0.0, description="Current normalized task score.")

    node_loads: List[float] = Field(default_factory=list, description="Current load at each node.")
    node_utilization: List[float] = Field(default_factory=list, description="Load/capacity for each node.")
    in_transit_count: int = Field(default=0, description="Number of delayed shipments in transit.")
    active_disruptions: List[Dict[str, Any]] = Field(default_factory=list, description="Active disruptions.")
    retail_delivered: float = Field(default=0.0, description="Total retail-arrived volume.")
    sla_success_rate: float = Field(default=0.0, description="Fraction of retail deliveries within SLA.")
    dynamic_pressure: float = Field(default=0.0, description="Current adaptive network pressure in [0, 1].")
    adaptive_disruption_rate: float = Field(default=0.0, description="Effective disruption probability this step.")
    priority_target_node: int = Field(default=10, description="Retail node targeted for urgent demand.")
    priority_service_rate: float = Field(default=0.0, description="Urgent shipment success rate.")
