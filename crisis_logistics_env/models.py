# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the LogiFlow-RL environment."""

from typing import List

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class CrisisLogisticsAction(Action):
    """Route the next shipment to one of the available hubs."""

    target_hub: int = Field(
        ...,
        ge=0,
        le=2,
        description="Index of the hub that should receive the incoming shipment: 0, 1, or 2.",
    )


class CrisisLogisticsObservation(Observation):
    """Current state of the regional hub network."""

    task_id: str = Field(default="easy", description="Active benchmark task identifier.")
    difficulty: str = Field(default="easy", description="Difficulty label for the active task.")
    objective: str = Field(default="", description="Task objective visible to the agent.")
    hub_loads: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Current utilization percentage for each hub.",
    )
    drain_rates: List[float] = Field(
        default_factory=lambda: [6.0, 5.0, 4.0],
        description="How much load each hub clears per timestep when not selected.",
    )
    incoming_load: float = Field(
        default=0.0,
        description="Load percentage of the shipment that must be assigned this step.",
    )
    step_count: int = Field(default=0, description="Current step in the episode.")
    max_steps: int = Field(default=100, description="Maximum episode length.")
    overloaded_hubs: int = Field(
        default=0,
        description="Number of hubs currently above 100 percent utilization.",
    )
    cumulative_score: float = Field(
        default=0.0,
        description="Normalized score in the range [0.0, 1.0] for progress so far in the episode.",
    )
    last_reward: float = Field(default=0.0, description="Reward from the previous action.")
    event_label: str = Field(
        default="normal",
        description="Traffic condition for the current shipment, e.g. normal or flash_sale.",
    )
    message: str = Field(default="", description="Human-readable state summary.")


class CrisisLogisticsState(State):
    """Internal environment state exposed for validation and debugging."""

    task_id: str = Field(default="easy", description="Active task identifier.")
    difficulty: str = Field(default="easy", description="Task difficulty label.")
    hub_loads: List[float] = Field(
        default_factory=lambda: [0.0, 0.0, 0.0],
        description="Current utilization values for each hub.",
    )
    incoming_index: int = Field(default=0, description="Index of the next scheduled shipment.")
    bottlenecks: int = Field(default=0, description="Total bottlenecks encountered this episode.")
    score: float = Field(default=0.0, description="Current normalized task score.")
