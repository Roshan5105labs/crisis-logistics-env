---
title: LogiFlow-RL
emoji: 🚚
colorFrom: blue
colorTo: green
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - logistics
  - reinforcement-learning
---

# LogiFlow-RL

LogiFlow-RL is an OpenEnv benchmark for resilient supply-chain routing. The agent acts as a live logistics crisis manager across a 12-node network:

`4 suppliers -> 3 warehouses -> 3 distribution centers -> 2 retail sinks`

The environment includes delayed in-transit shipments, stochastic disruption cascades, partial observability, and dense reward shaping. It is designed as a focused benchmark for dynamic supply-chain optimization rather than a full enterprise logistics platform.

## Action Space

The preferred Phase 2 action is:

- `source_node`: node dispatching freight
- `dest_node`: connected downstream node receiving freight
- `shipment_volume`: freight volume to move
- `reasoning`: optional natural-language rationale

For backward compatibility, `target_hub` is still accepted by `/step` and maps to an outgoing route from the active shipment source.

## Observation Space

The observation includes:

- 12-node loads, capacities, utilization, risk scores, and connectivity
- visible two-hop neighborhood for partial observability
- active disruptions and in-transit shipments
- incoming shipment source, volume, and event label
- reward breakdown, cumulative score, retail delivery, and SLA success
- backward-compatible tier summaries in `hub_loads`

## Tasks

- `easy`: 50-step regional network balancing
- `medium`: 70-step flash-sale and port-risk recovery
- `hard`: 90-step cascading disruption recovery

Each task has deterministic configuration and a grader score in `[0.0, 1.0]`.

## Run

```bash
python test_engine.py
python train_and_evaluate.py
python inference.py
```

Start the API and visualizer:

```bash
uv run server
```

Then open:

- `/web`
- `/visualizer`
- `/docs`
- `/health`
