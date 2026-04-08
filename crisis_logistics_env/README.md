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

## Overview

LogiFlow-RL is an OpenEnv environment for dynamic supply-chain workload balancing. The agent acts as a regional routing controller that must assign each incoming shipment to one of three hubs while keeping the network stable under predictable demand, flash-sale bursts, and cascading disruptions.

This is a real-world task rather than a toy game: the agent is learning a simplified version of freight routing and capacity balancing, which are core operations in modern logistics networks.

## Why this environment is useful

Static routing rules such as round robin often fail under bursty demand because they react too slowly to overload risk. LogiFlow-RL provides a reproducible environment for training and evaluating routing agents that must reason about load, capacity, drift, and operational resilience.

## Action Space

The action is a typed Pydantic model:

- `target_hub: int`
  - `0` route shipment to Hub A
  - `1` route shipment to Hub B
  - `2` route shipment to Hub C

## Observation Space

The observation is a typed Pydantic model containing:

- `task_id`: active benchmark task
- `difficulty`: easy, medium, or hard
- `objective`: natural-language task goal
- `hub_loads`: current utilization for the 3 hubs
- `drain_rates`: per-step clearing rates for non-selected hubs
- `incoming_load`: next scheduled shipment size
- `step_count` and `max_steps`
- `overloaded_hubs`
- `cumulative_score`: normalized task score in `[0.0, 1.0]`
- `last_reward`: shaped reward for the previous action in `[0.0, 1.0]`
- `event_label`: normal, flash_sale, weather_disruption, or completed

## Reward Design

The environment provides dense reward over the full trajectory:

- Positive reward when the selected hub stays in the optimal 30-70 utilization zone
- Penalties when the chosen hub drifts too far from the center of the target range
- Penalties for global imbalance across hubs
- Strong penalty when a routing decision causes overload above 100 utilization

Each step reward is normalized to `[0.0, 1.0]`.

## Tasks and Graders

The environment ships with three deterministic benchmark tasks and programmatic graders.

### Easy: `easy`

Title: Steady-State Rebalancing

Objective: Keep all hubs in the 30-70 utilization band during predictable daytime demand.

Expected challenge: The agent should quickly learn basic balancing behavior and avoid unnecessary skew.

### Medium: `medium`

Title: Flash Sale Containment

Objective: Absorb an afternoon flash-sale surge without letting any single hub overload.

Expected challenge: The agent must react to bursty schedules and preserve balance under transient spikes.

### Hard: `hard`

Title: Cascading Disruption Recovery

Objective: Stabilize the network through repeated surge waves and weather disruptions while preserving throughput.

Expected challenge: The agent must recover from repeated shocks and still maintain usable balance.

### Grader behavior

Each task has a deterministic grader that returns a final score in `[0.0, 1.0]` based on:

- Bottleneck avoidance
- Average inter-hub balance gap
- Average trajectory reward
- Fraction of steps spent in the optimal operating zone

The grader implementation is in [graders.py](C:\Users\rosha\crisis-logistics-env\crisis_logistics_env\graders.py).

## Baselines

The repo includes two reproducible local baselines:

- `round_robin`
- `heuristic`

Run:

```bash
python train_and_evaluate.py
```

Current local benchmark scores:

- `round_robin`
  - easy: `0.800`
  - medium: `0.886`
  - hard: `0.863`
- `heuristic`
  - easy: `0.800`
  - medium: `0.957`
  - hard: `0.900`

The submission baseline entrypoint is [inference.py](C:\Users\rosha\crisis-logistics-env\inference.py), which uses the OpenAI client and emits the required structured logs.

## Project Structure

```text
crisis_logistics_env/
├── __init__.py
├── client.py
├── graders.py
├── gym_env.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── README.md
├── tasks.py
├── test_engine.py
├── train_and_evaluate.py
└── server/
    ├── app.py
    ├── crisis_logistics_env_environment.py
    └── Dockerfile
```

## Setup

### Local run

```bash
python test_engine.py
python train_and_evaluate.py
```

### Baseline inference

Set the required environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN` or `OPENAI_API_KEY`

Then run from the repository root:

```bash
python inference.py
```

### Local server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### Docker

```bash
docker build -t logiflow-rl -f server/Dockerfile .
docker run -p 8000:8000 logiflow-rl
```

## Validation Notes

The environment includes:

- Typed action, observation, and state models
- Deterministic tasks with increasing difficulty
- Programmatic graders with scores in `[0.0, 1.0]`
- Dense partial-progress reward shaping
- Root-level `inference.py`
- Dockerfile and OpenEnv manifest

## Submission One-Liner

LogiFlow-RL is an OpenEnv benchmark for dynamic freight routing where an agent must balance shipments across regional hubs under flash-sale and disruption scenarios, with deterministic tasks and normalized graders for reproducible evaluation.
