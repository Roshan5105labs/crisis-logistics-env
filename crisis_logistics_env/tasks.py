from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class NodeConfig:
    node_id: int
    name: str
    node_type: str
    capacity: float
    initial_load: float
    drain_rate: float
    risk_score: float
    connections: List[int]


@dataclass(frozen=True)
class ScheduledShipment:
    source_node: int
    volume: float
    deadline_steps: int
    event_hint: str


@dataclass(frozen=True)
class TaskConfig:
    task_id: str
    difficulty: str
    title: str
    objective: str
    max_steps: int
    nodes: List[NodeConfig]
    incoming_schedule: List[ScheduledShipment]
    disruption_rate: float
    cascade_rate: float
    target_bottlenecks: int
    target_balance_gap: float
    target_sla: float
    target_retail_delivery: float
    minimum_avg_reward: float
    seed: int

    @property
    def initial_loads(self) -> List[float]:
        return [node.initial_load for node in self.nodes]

    @property
    def drain_rates(self) -> List[float]:
        return [node.drain_rate for node in self.nodes]

    @property
    def event_schedule(self) -> List[str]:
        return [shipment.event_hint for shipment in self.incoming_schedule]


CONNECTIVITY: Dict[int, List[int]] = {
    0: [4, 5],
    1: [4, 6],
    2: [5, 6],
    3: [4, 5, 6],
    4: [7, 8],
    5: [8, 9],
    6: [7, 9],
    7: [10],
    8: [10, 11],
    9: [11],
    10: [],
    11: [],
}


NAMES = [
    "Supplier North",
    "Supplier West",
    "Supplier Port",
    "Supplier Inland",
    "Warehouse Alpha",
    "Warehouse Beta",
    "Warehouse Gamma",
    "DC Metro",
    "DC Central",
    "DC Coastal",
    "Retail North",
    "Retail South",
]

TYPES = [
    "supplier",
    "supplier",
    "supplier",
    "supplier",
    "warehouse",
    "warehouse",
    "warehouse",
    "distribution",
    "distribution",
    "distribution",
    "retail",
    "retail",
]


def _nodes(loads: List[float], risks: List[float], drain_scale: float = 1.0) -> List[NodeConfig]:
    capacities = [95.0, 92.0, 90.0, 88.0, 130.0, 125.0, 120.0, 105.0, 110.0, 105.0, 90.0, 90.0]
    drains = [6.0, 5.5, 5.0, 5.0, 9.0, 8.5, 8.0, 7.0, 7.0, 6.5, 11.0, 11.0]
    return [
        NodeConfig(
            node_id=index,
            name=NAMES[index],
            node_type=TYPES[index],
            capacity=capacities[index],
            initial_load=loads[index],
            drain_rate=round(drains[index] * drain_scale, 2),
            risk_score=risks[index],
            connections=CONNECTIVITY[index],
        )
        for index in range(12)
    ]


def _schedule(
    steps: int,
    base: float,
    sources: List[int],
    surge_steps: set[int],
    weather_steps: set[int],
    failure_steps: set[int],
) -> List[ScheduledShipment]:
    shipments: List[ScheduledShipment] = []
    for step in range(steps):
        source = sources[step % len(sources)]
        volume = base + (step % 5) * 1.5
        event = "normal"
        deadline = 12
        if step in surge_steps:
            volume += 11.0
            event = "flash_sale"
            deadline = 10
        if step in weather_steps:
            volume += 5.0
            event = "weather_disruption"
            deadline = 14
        if step in failure_steps:
            volume += 8.0
            source = 2
            event = "supplier_failure"
            deadline = 15
        shipments.append(
            ScheduledShipment(
                source_node=source,
                volume=round(volume, 2),
                deadline_steps=deadline,
                event_hint=event,
            )
        )
    return shipments


TASKS: dict[str, TaskConfig] = {
    "easy": TaskConfig(
        task_id="easy",
        difficulty="easy",
        title="Regional Network Balancing",
        objective=(
            "Operate a 12-node supplier-warehouse-DC-retail network for 50 steps. "
            "Keep utilization balanced while moving freight to retail within SLA."
        ),
        max_steps=50,
        nodes=_nodes(
            [24, 31, 27, 22, 46, 51, 43, 38, 44, 36, 20, 24],
            [0.10, 0.12, 0.16, 0.10, 0.15, 0.18, 0.14, 0.16, 0.15, 0.17, 0.12, 0.12],
        ),
        incoming_schedule=_schedule(
            50,
            9.0,
            [0, 1, 2, 3],
            surge_steps={12, 13, 14, 30},
            weather_steps={22, 23, 38},
            failure_steps=set(),
        ),
        disruption_rate=0.05,
        cascade_rate=0.10,
        target_bottlenecks=2,
        target_balance_gap=0.42,
        target_sla=0.65,
        target_retail_delivery=180.0,
        minimum_avg_reward=0.35,
        seed=101,
    ),
    "medium": TaskConfig(
        task_id="medium",
        difficulty="medium",
        title="Flash Sale With Port Risk",
        objective=(
            "Recover from burst demand and port slowdowns over 70 steps. "
            "Plan around delayed transit and prevent warehouse spillovers."
        ),
        max_steps=70,
        nodes=_nodes(
            [30, 34, 42, 25, 58, 62, 48, 46, 53, 44, 28, 26],
            [0.14, 0.15, 0.32, 0.12, 0.22, 0.24, 0.18, 0.20, 0.22, 0.28, 0.15, 0.16],
            drain_scale=0.95,
        ),
        incoming_schedule=_schedule(
            70,
            10.5,
            [2, 0, 1, 3],
            surge_steps={8, 9, 10, 11, 28, 29, 46, 47},
            weather_steps={18, 19, 20, 52, 53},
            failure_steps={35, 36},
        ),
        disruption_rate=0.09,
        cascade_rate=0.16,
        target_bottlenecks=4,
        target_balance_gap=0.48,
        target_sla=0.55,
        target_retail_delivery=250.0,
        minimum_avg_reward=0.30,
        seed=202,
    ),
    "hard": TaskConfig(
        task_id="hard",
        difficulty="hard",
        title="Cascading Disruption Recovery",
        objective=(
            "Stabilize a partially observable 12-node chain across 90 steps while weather, "
            "port closure, and supplier failures cascade through the network."
        ),
        max_steps=90,
        nodes=_nodes(
            [39, 36, 55, 30, 70, 66, 60, 55, 62, 58, 36, 34],
            [0.18, 0.22, 0.38, 0.16, 0.30, 0.28, 0.24, 0.26, 0.30, 0.34, 0.18, 0.18],
            drain_scale=0.85,
        ),
        incoming_schedule=_schedule(
            90,
            12.0,
            [2, 0, 1, 3],
            surge_steps={6, 7, 8, 21, 22, 23, 44, 45, 46, 70, 71},
            weather_steps={14, 15, 32, 33, 58, 59, 60},
            failure_steps={26, 27, 52, 76, 77},
        ),
        disruption_rate=0.12,
        cascade_rate=0.22,
        target_bottlenecks=7,
        target_balance_gap=0.56,
        target_sla=0.45,
        target_retail_delivery=300.0,
        minimum_avg_reward=0.25,
        seed=303,
    ),
}


def list_tasks() -> List[TaskConfig]:
    return [TASKS["easy"], TASKS["medium"], TASKS["hard"]]


def get_task(task_id: str) -> TaskConfig:
    return TASKS.get(task_id, TASKS["easy"])
