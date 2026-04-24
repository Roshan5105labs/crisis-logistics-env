from __future__ import annotations

import random
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from openenv.core.env_server import Environment

try:
    from ..graders import EpisodeMetrics, grade_episode
    from ..models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from ..tasks import ScheduledShipment, get_task, list_tasks
except ImportError:
    from graders import EpisodeMetrics, grade_episode
    from models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from tasks import ScheduledShipment, get_task, list_tasks


@dataclass
class TransitShipment:
    shipment_id: str
    source: int
    dest: int
    volume: float
    remaining_steps: int
    deadline_step: int
    started_step: int
    event_label: str


@dataclass
class ActiveDisruption:
    node_id: int
    kind: str
    remaining_steps: int
    severity: float


class CrisisLogisticsEnvironment(
    Environment[CrisisLogisticsAction, CrisisLogisticsObservation, CrisisLogisticsState]
):
    """Resilient supply-chain routing benchmark with delayed outcomes."""

    def __init__(self):
        super().__init__()
        self.available_tasks = list_tasks()
        self.reset(task_id="easy")

    def reset(
        self, seed: Optional[int] = None, episode_id: Optional[str] = None, task_id: str = "easy", **kwargs
    ) -> CrisisLogisticsObservation:
        self.task = get_task(task_id)
        self.rng = random.Random(seed if seed is not None else self.task.seed)
        self.episode_id = episode_id or str(uuid.uuid4())
        self.step_count = 0
        self.schedule_index = 0
        self.done = False
        self.last_reward = 0.0
        self.total_reward = 0.0
        self.score = 0.0
        self.last_action: Dict[str, Any] = {}
        self.last_reward_breakdown: Dict[str, float] = {}
        self.invalid_actions = 0

        self.nodes = self.task.nodes
        self.node_names = [node.name for node in self.nodes]
        self.node_types = [node.node_type for node in self.nodes]
        self.node_capacities = [node.capacity for node in self.nodes]
        self.node_loads = [node.initial_load for node in self.nodes]
        self.base_drain_rates = [node.drain_rate for node in self.nodes]
        self.node_risk_scores = [node.risk_score for node in self.nodes]
        self.connectivity = {node.node_id: node.connections[:] for node in self.nodes}

        self.in_transit: List[TransitShipment] = []
        self.active_disruptions: List[ActiveDisruption] = []
        self.bottlenecks = 0
        self.balance_gap_history: List[float] = []
        self.optimal_steps = 0
        self.throughput_served = 0.0
        self.retail_delivered = 0.0
        self.sla_deliveries = 0
        self.total_retail_deliveries = 0
        self.recovery_history: List[float] = []

        current = self._current_shipment()
        self.incoming_load = current.volume
        self.event_label = current.event_hint
        return self._get_observation(
            f"Task '{self.task.title}' initialized on a 12-node supply-chain network."
        )

    def step(
        self, action: CrisisLogisticsAction, timeout_s: Optional[float] = None, **kwargs
    ) -> CrisisLogisticsObservation:
        if self.done:
            observation = self._get_observation("Episode already finished.")
            observation.reward = 0.0
            return observation

        current = self._current_shipment()
        self.event_label = current.event_hint
        self.incoming_load = current.volume
        self.step_count += 1

        self._maybe_start_disruption(current)
        self.node_loads[current.source_node] += current.volume

        resolved_action, valid, invalid_reason = self._resolve_action(action, current)
        dispatched_volume = 0.0
        if valid:
            dispatched_volume = self._dispatch(resolved_action, current)
        else:
            self.invalid_actions += 1

        arrivals = self._advance_transit()
        retail_arrived, sla_hits = self._apply_arrivals(arrivals)
        self._drain_nodes()
        self._apply_cascades()
        self._tick_disruptions()

        overloaded_nodes = self._overloaded_nodes()
        if overloaded_nodes:
            self.bottlenecks += len(overloaded_nodes)

        reward, breakdown = self._calculate_step_reward(
            valid=valid,
            dispatched_volume=dispatched_volume,
            retail_arrived=retail_arrived,
            sla_hits=sla_hits,
            invalid_reason=invalid_reason,
        )
        self.last_reward = reward
        self.last_reward_breakdown = breakdown
        self.total_reward += reward

        if self._network_balanced():
            self.optimal_steps += 1
        self.balance_gap_history.append(self._balance_gap())
        self.throughput_served += dispatched_volume
        self.score = self._compute_score()

        self.schedule_index += 1
        if self.schedule_index >= self.task.max_steps:
            self.done = True
        else:
            next_shipment = self._current_shipment()
            self.incoming_load = next_shipment.volume
            self.event_label = next_shipment.event_hint

        message = self._build_message(resolved_action, valid, invalid_reason, retail_arrived)
        observation = self._get_observation(message)
        observation.reward = reward
        observation.done = self.done
        return observation

    def _current_shipment(self) -> ScheduledShipment:
        index = min(self.schedule_index, len(self.task.incoming_schedule) - 1)
        return self.task.incoming_schedule[index]

    def _resolve_action(
        self, action: CrisisLogisticsAction, current: ScheduledShipment
    ) -> tuple[Dict[str, Any], bool, str]:
        if action.source_node is not None and action.dest_node is not None:
            source = action.source_node
            dest = action.dest_node
        else:
            source = current.source_node
            options = self.connectivity.get(source, [])
            if not options:
                return {}, False, "source_has_no_outbound_edges"
            selector = action.target_hub if action.target_hub is not None else 0
            dest = options[selector % len(options)]

        volume = action.shipment_volume if action.shipment_volume is not None else current.volume
        resolved = {
            "source_node": source,
            "dest_node": dest,
            "shipment_volume": round(float(volume), 2),
            "reasoning": action.reasoning or "",
        }
        self.last_action = resolved

        if source < 0 or source >= len(self.nodes) or dest < 0 or dest >= len(self.nodes):
            return resolved, False, "node_out_of_range"
        if dest not in self.connectivity.get(source, []):
            return resolved, False, "dest_not_connected"
        if self._is_blocked(source):
            return resolved, False, "source_blocked_by_disruption"
        if volume <= 0:
            return resolved, False, "non_positive_volume"
        if self.node_loads[source] + 1e-6 < volume:
            return resolved, False, "insufficient_source_load"
        return resolved, True, ""

    def _dispatch(self, resolved: Dict[str, Any], current: ScheduledShipment) -> float:
        source = int(resolved["source_node"])
        dest = int(resolved["dest_node"])
        volume = min(float(resolved["shipment_volume"]), self.node_loads[source])
        self.node_loads[source] -= volume
        delay = self._transit_delay(source, dest)
        self.in_transit.append(
            TransitShipment(
                shipment_id=f"ship-{self.step_count}-{len(self.in_transit)}",
                source=source,
                dest=dest,
                volume=round(volume, 2),
                remaining_steps=delay,
                deadline_step=self.step_count + current.deadline_steps,
                started_step=self.step_count,
                event_label=current.event_hint,
            )
        )
        return volume

    def _advance_transit(self) -> List[TransitShipment]:
        arrivals: List[TransitShipment] = []
        still_in_transit: List[TransitShipment] = []
        for shipment in self.in_transit:
            shipment.remaining_steps -= 1
            if shipment.remaining_steps <= 0:
                arrivals.append(shipment)
            else:
                still_in_transit.append(shipment)
        self.in_transit = still_in_transit
        return arrivals

    def _apply_arrivals(self, arrivals: List[TransitShipment]) -> tuple[float, int]:
        retail_arrived = 0.0
        sla_hits = 0
        for shipment in arrivals:
            self.node_loads[shipment.dest] += shipment.volume
            if self.node_types[shipment.dest] == "retail":
                retail_arrived += shipment.volume
                self.retail_delivered += shipment.volume
                self.total_retail_deliveries += 1
                if self.step_count <= shipment.deadline_step:
                    self.sla_deliveries += 1
                    sla_hits += 1
        return retail_arrived, sla_hits

    def _drain_nodes(self) -> None:
        for index, load in enumerate(self.node_loads):
            drain = self._effective_drain(index)
            self.node_loads[index] = max(0.0, load - drain)

    def _maybe_start_disruption(self, current: ScheduledShipment) -> None:
        candidate = current.source_node
        if current.event_hint == "weather_disruption":
            candidate = self.rng.choice([7, 8, 9])
        elif current.event_hint == "supplier_failure":
            candidate = current.source_node
        elif current.event_hint == "flash_sale":
            candidate = self.rng.choice([10, 11])

        risk = self.node_risk_scores[candidate]
        event_boost = 0.18 if current.event_hint != "normal" else 0.0
        if self.rng.random() < self.task.disruption_rate + risk * 0.08 + event_boost:
            self._add_disruption(candidate, current.event_hint)

    def _apply_cascades(self) -> None:
        for node_id in self._overloaded_nodes():
            downstream = self.connectivity.get(node_id, [])
            if not downstream:
                continue
            if self.rng.random() < self.task.cascade_rate:
                self._add_disruption(self.rng.choice(downstream), "cascade_spillover")

    def _add_disruption(self, node_id: int, kind: str) -> None:
        if any(d.node_id == node_id and d.kind == kind for d in self.active_disruptions):
            return
        duration = self.rng.randint(3, 8)
        severity = round(self.rng.uniform(0.25, 0.65), 2)
        self.active_disruptions.append(
            ActiveDisruption(node_id=node_id, kind=kind, remaining_steps=duration, severity=severity)
        )

    def _tick_disruptions(self) -> None:
        remaining: List[ActiveDisruption] = []
        for disruption in self.active_disruptions:
            disruption.remaining_steps -= 1
            if disruption.remaining_steps > 0:
                remaining.append(disruption)
        self.active_disruptions = remaining

    def _transit_delay(self, source: int, dest: int) -> int:
        base_delay = 3
        if self.node_types[source] == "warehouse":
            base_delay = 4
        if self.node_types[source] == "distribution":
            base_delay = 3
        disruption_delay = 0
        for disruption in self.active_disruptions:
            if disruption.node_id in {source, dest}:
                disruption_delay += 1 + int(disruption.severity * 2)
        return max(2, min(8, base_delay + disruption_delay))

    def _effective_drain(self, node_id: int) -> float:
        drain = self.base_drain_rates[node_id]
        for disruption in self.active_disruptions:
            if disruption.node_id == node_id:
                drain *= max(0.2, 1.0 - disruption.severity)
        return drain

    def _is_blocked(self, node_id: int) -> bool:
        return any(d.node_id == node_id and d.severity >= 0.6 for d in self.active_disruptions)

    def _calculate_step_reward(
        self,
        valid: bool,
        dispatched_volume: float,
        retail_arrived: float,
        sla_hits: int,
        invalid_reason: str,
    ) -> tuple[float, Dict[str, float]]:
        expected = max(self.incoming_load, 1.0)
        throughput_score = min(1.0, retail_arrived / expected)
        sla_score = self._sla_success_rate() if self.total_retail_deliveries > 0 else 0.45
        balance_score = max(0.0, 1.0 - self._balance_gap() / 0.85)
        recovery_score = max(0.0, 1.0 - (len(self._overloaded_nodes()) + len(self.active_disruptions)) / 8.0)
        valid_score = 1.0 if valid else 0.0
        overload_penalty = min(1.0, len(self._overloaded_nodes()) / 4.0)
        invalid_penalty = 0.35 if invalid_reason else 0.0
        anti_gaming_penalty = 0.08 if dispatched_volume > 0 and retail_arrived == 0 else 0.0

        reward = (
            0.18 * valid_score
            + 0.22 * throughput_score
            + 0.18 * sla_score
            + 0.20 * balance_score
            + 0.17 * recovery_score
            - 0.22 * overload_penalty
            - invalid_penalty
            - anti_gaming_penalty
        )
        breakdown = {
            "valid_action": round(valid_score, 3),
            "throughput": round(throughput_score, 3),
            "sla": round(sla_score, 3),
            "network_balance": round(balance_score, 3),
            "disruption_recovery": round(recovery_score, 3),
            "overload_penalty": round(overload_penalty, 3),
            "anti_gaming_penalty": round(anti_gaming_penalty, 3),
            "invalid_penalty": round(invalid_penalty, 3),
        }
        self.recovery_history.append(recovery_score)
        return round(max(0.0, min(1.0, reward)), 2), breakdown

    def _compute_score(self) -> float:
        metrics = self._metrics()
        return grade_episode(self.task, metrics)

    def _metrics(self) -> EpisodeMetrics:
        return EpisodeMetrics(
            total_reward=self.total_reward,
            average_reward=self.total_reward / max(self.step_count, 1),
            bottlenecks=self.bottlenecks,
            optimal_steps=self.optimal_steps,
            average_balance_gap=sum(self.balance_gap_history) / max(len(self.balance_gap_history), 1),
            throughput_served=self.throughput_served,
            steps_completed=self.step_count,
            retail_delivered=self.retail_delivered,
            sla_success_rate=self._sla_success_rate(),
            disruption_recovery_score=sum(self.recovery_history) / max(len(self.recovery_history), 1),
            invalid_actions=self.invalid_actions,
        )

    def _overloaded_nodes(self) -> List[int]:
        return [
            index
            for index, load in enumerate(self.node_loads)
            if load > self.node_capacities[index]
        ]

    def _node_utilization(self) -> List[float]:
        return [
            round(self.node_loads[index] / self.node_capacities[index], 3)
            for index in range(len(self.node_loads))
        ]

    def _balance_gap(self) -> float:
        utilizations = self._node_utilization()
        visible = [u for u in utilizations if u >= 0]
        return max(visible) - min(visible) if visible else 0.0

    def _network_balanced(self) -> bool:
        return self._balance_gap() <= 0.35 and len(self._overloaded_nodes()) == 0

    def _sla_success_rate(self) -> float:
        if self.total_retail_deliveries == 0:
            return 0.0
        return round(self.sla_deliveries / self.total_retail_deliveries, 3)

    def _tier_summary(self) -> tuple[List[float], List[float]]:
        tiers = [
            [0, 1, 2, 3],
            [4, 5, 6],
            [7, 8, 9, 10, 11],
        ]
        utilizations = self._node_utilization()
        loads = [round(sum(utilizations[i] for i in tier) / len(tier) * 100, 2) for tier in tiers]
        drains = [round(sum(self.base_drain_rates[i] for i in tier) / len(tier), 2) for tier in tiers]
        return loads, drains

    def _visible_nodes(self) -> Set[int]:
        current = self._current_shipment()
        center = current.source_node
        visible = {center}
        frontier = {center}
        reverse: Dict[int, List[int]] = {index: [] for index in range(len(self.nodes))}
        for source, dests in self.connectivity.items():
            for dest in dests:
                reverse[dest].append(source)
        for _ in range(2):
            next_frontier: Set[int] = set()
            for node in frontier:
                next_frontier.update(self.connectivity.get(node, []))
                next_frontier.update(reverse.get(node, []))
            visible.update(next_frontier)
            frontier = next_frontier
        visible.update(disruption.node_id for disruption in self.active_disruptions)
        visible.update(shipment.source for shipment in self.in_transit[:4])
        visible.update(shipment.dest for shipment in self.in_transit[:4])
        return {node for node in visible if 0 <= node < len(self.nodes)}

    def _active_disruption_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                "node_id": disruption.node_id,
                "node_name": self.node_names[disruption.node_id],
                "kind": disruption.kind,
                "remaining_steps": disruption.remaining_steps,
                "severity": disruption.severity,
            }
            for disruption in self.active_disruptions
        ]

    def _transit_dicts(self) -> List[Dict[str, Any]]:
        return [
            {
                "shipment_id": shipment.shipment_id,
                "source": shipment.source,
                "dest": shipment.dest,
                "volume": shipment.volume,
                "remaining_steps": shipment.remaining_steps,
                "deadline_step": shipment.deadline_step,
                "event_label": shipment.event_label,
            }
            for shipment in self.in_transit[:12]
        ]

    def _build_message(
        self, resolved_action: Dict[str, Any], valid: bool, invalid_reason: str, retail_arrived: float
    ) -> str:
        if not valid:
            return f"Action rejected: {invalid_reason}. Active disruptions: {len(self.active_disruptions)}."
        source = resolved_action.get("source_node")
        dest = resolved_action.get("dest_node")
        volume = resolved_action.get("shipment_volume")
        return (
            f"Dispatched {volume} units from Node {source} to Node {dest}. "
            f"Retail arrivals this step: {round(retail_arrived, 2)}."
        )

    def _get_observation(self, message: str) -> CrisisLogisticsObservation:
        visible_nodes = sorted(self._visible_nodes())
        visible_set = set(visible_nodes)
        node_utilization = self._node_utilization()
        observed_loads: List[Optional[float]] = [
            round(self.node_loads[index], 2) if index in visible_set else None
            for index in range(len(self.node_loads))
        ]
        visible_connectivity = {
            str(source): [dest for dest in dests if dest in visible_set]
            for source, dests in self.connectivity.items()
            if source in visible_set
        }
        hub_loads, drain_rates = self._tier_summary()
        overloaded = len(self._overloaded_nodes())
        current = self._current_shipment()
        return CrisisLogisticsObservation(
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            objective=self.task.objective,
            hub_loads=hub_loads,
            drain_rates=drain_rates,
            incoming_load=0.0 if self.done else current.volume,
            step_count=self.step_count,
            max_steps=self.task.max_steps,
            overloaded_hubs=overloaded,
            cumulative_score=self.score,
            last_reward=self.last_reward,
            event_label="completed" if self.done else current.event_hint,
            message=message,
            node_names=self.node_names,
            node_types=self.node_types,
            node_loads=[round(load, 2) for load in self.node_loads],
            node_capacities=self.node_capacities[:],
            node_utilization=node_utilization,
            node_drain_rates=[round(self._effective_drain(i), 2) for i in range(len(self.nodes))],
            node_risk_scores=self.node_risk_scores[:],
            connectivity={str(source): dests[:] for source, dests in self.connectivity.items()},
            visible_node_ids=visible_nodes,
            observed_node_loads=observed_loads,
            visible_connectivity=visible_connectivity,
            in_transit_shipments=self._transit_dicts(),
            active_disruptions=self._active_disruption_dicts(),
            reward_breakdown=self.last_reward_breakdown,
            last_action=self.last_action,
            pending_source_node=current.source_node,
            retail_delivered=round(self.retail_delivered, 2),
            sla_success_rate=self._sla_success_rate(),
            reward=self.last_reward,
            done=self.done,
            metadata={
                "title": self.task.title,
                "available_tasks": [task.task_id for task in self.available_tasks],
                "node_count": len(self.nodes),
                "network_shape": "4 suppliers -> 3 warehouses -> 3 distribution centers -> 2 retail sinks",
                "partial_observability": "Agent sees a two-hop neighborhood plus active disruptions.",
                "bottlenecks": self.bottlenecks,
            },
        )

    @property
    def state(self) -> CrisisLogisticsState:
        hub_loads, _ = self._tier_summary()
        return CrisisLogisticsState(
            episode_id=self.episode_id,
            task_id=self.task.task_id,
            difficulty=self.task.difficulty,
            step_count=self.step_count,
            hub_loads=hub_loads,
            incoming_index=self.schedule_index,
            bottlenecks=self.bottlenecks,
            score=self.score,
            node_loads=[round(load, 2) for load in self.node_loads],
            node_utilization=self._node_utilization(),
            in_transit_count=len(self.in_transit),
            active_disruptions=self._active_disruption_dicts(),
            retail_delivered=round(self.retail_delivered, 2),
            sla_success_rate=self._sla_success_rate(),
        )


def choose_network_action(observation: CrisisLogisticsObservation) -> CrisisLogisticsAction:
    """Deterministic baseline that pushes freight toward retail while avoiding overload."""

    loads = observation.node_loads or []
    capacities = observation.node_capacities or []
    connectivity = {int(k): v for k, v in observation.connectivity.items()} if observation.connectivity else {}
    node_types = observation.node_types or []

    if not loads or not capacities or not connectivity:
        return CrisisLogisticsAction(target_hub=0)

    candidate_sources = [
        idx
        for idx, load in enumerate(loads)
        if load > 1.0 and connectivity.get(idx) and idx in set(observation.visible_node_ids or range(len(loads)))
    ]
    if not candidate_sources:
        candidate_sources = [observation.pending_source_node]

    def downstream_priority(node_id: int) -> float:
        node_type = node_types[node_id] if node_id < len(node_types) else ""
        if node_type == "retail":
            return -0.25
        if node_type == "distribution":
            return 0.0
        if node_type == "warehouse":
            return 0.15
        return 0.3

    best: Optional[Dict[str, Any]] = None
    best_score = float("inf")
    for source in candidate_sources:
        for dest in connectivity.get(source, []):
            free_capacity = max(0.0, capacities[dest] - loads[dest])
            if free_capacity <= 0:
                continue
            volume = min(loads[source], free_capacity * 0.65, max(observation.incoming_load, 8.0), 18.0)
            if volume <= 0:
                continue
            projected_dest_util = (loads[dest] + volume) / capacities[dest]
            source_relief = loads[source] / capacities[source]
            score = projected_dest_util + downstream_priority(dest) - source_relief * 0.2
            if score < best_score:
                best_score = score
                best = {
                    "source_node": source,
                    "dest_node": dest,
                    "shipment_volume": round(volume, 2),
                    "reasoning": "Heuristic routes visible backlog toward lower-utilized downstream capacity.",
                }

    if best is None:
        return CrisisLogisticsAction(target_hub=0)
    return CrisisLogisticsAction(**best)


def choose_balancing_action(observation: CrisisLogisticsObservation) -> int:
    """Backward-compatible 0/1/2 selector used by the visualizer and old smoke tests."""

    source = observation.pending_source_node
    options = observation.connectivity.get(str(source), [])
    if not options:
        return 0
    loads = observation.node_loads or []
    capacities = observation.node_capacities or []
    best_offset = 0
    best_util = float("inf")
    for offset, dest in enumerate(options):
        if dest < len(loads) and dest < len(capacities):
            projected = (loads[dest] + observation.incoming_load) / max(capacities[dest], 1.0)
        else:
            projected = 1.0
        if projected < best_util:
            best_util = projected
            best_offset = offset
    return best_offset
