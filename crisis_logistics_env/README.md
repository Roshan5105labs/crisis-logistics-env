---
title: LogiFlow-RL
emoji: ":truck:"
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

# LogiFlow-RL (OpenEnv Hackathon Phase-2)

LogiFlow-RL is an **OpenEnv-compliant** RL environment where an LLM routes freight in a dynamic 12-node crisis supply network:

`4 suppliers -> 3 warehouses -> 3 distribution centers -> 2 retail sinks`

This repo is structured to satisfy Phase-2 judging requirements: environment quality, clear story, measurable reward improvement, and reproducible training.

## Quick Links (Fill Before Final Submit)

- Hugging Face Space (environment URL): `<add-your-space-url>`
- Colab (minimal TRL GRPO): `<add-colab-link>`
- Colab (full notebook): `<add-colab-link>`
- Mini blog / `<2 min` video: `<add-link>`
- Optional W&B run: `<add-link>`

## Why This Environment Is Interesting

- **Dynamic pressure engine**: congestion, SLA risk, delivery shortfall, and disruptions continuously alter the task.
- **Delayed transit + cascading disruptions**: actions have lagged downstream effects; local optimization can fail globally.
- **Priority-demand windows**: the policy must serve urgent retail targets under time constraints.
- **Partial observability**: agent sees a two-hop neighborhood, not full perfect state.
- **Anti-gaming reward design**: independent reward components with penalties for invalid/looping/risky behavior.

## OpenEnv Compliance

- Uses latest OpenEnv core release line: `openenv-core[core]>=0.2.3`
- Implements standard Gym-style interfaces through OpenEnv server primitives:
  - `reset()`
  - `step(action)`
  - `state`
- Includes required environment manifest: `openenv.yaml`
- Includes FastAPI app for Space deployment: `server/app.py`
- MCP reserved names are not used as custom tools.

## Environment API Summary

### Action

Preferred structured action:

- `source_node`
- `dest_node`
- `shipment_volume`
- `reasoning` (optional)

Backward compatibility:

- `target_hub` is still supported for old tests/visualizer flows.

### Observation

Each step includes:

- node loads/capacities/utilization/risk
- visible subgraph (`visible_node_ids`, `observed_node_loads`, `visible_connectivity`)
- active disruptions and in-transit shipments
- adaptive fields (`dynamic_pressure`, `adaptive_disruption_rate`)
- priority fields (`priority_target_node`, `priority_service_rate`, queue depth)
- reward decomposition (`reward_breakdown`)

### Tasks

- `easy` (50 steps): regional balancing
- `medium` (70 steps): flash-sale + port risk
- `hard` (90 steps): cascading disruption recovery

## Reward Design

Per-step reward combines:

- valid action
- throughput
- SLA
- network balance
- disruption recovery
- priority service
- transit health

Penalties include:

- overload
- invalid action
- repetitive route loops
- route risk
- pressure stress
- anti-gaming repetitive low-impact routing

All components are visible in `observation.reward_breakdown`.

## Training Pipeline (HF TRL GRPO)

Main script: `train_grpo.py`

- Uses **HF TRL GRPO** (`GRPOTrainer`, `GRPOConfig`)
- Uses LoRA adapters via PEFT
- Includes reward sanity checks before training
- Runs before/after policy evaluation on all tasks
- Saves artifacts for README/blog/video evidence

Minimal Colab path:

- `notebooks/minimal_trl_grpo_colab.ipynb`

Full Colab notebook:

- `notebooks/logiflow_grpo_colab.ipynb`
- `notebooks/colab_space_grpo_unsloth.ipynb` (single-file Space-connected Unsloth+GRPO workflow with smoke test, training, plots, and before/after evaluation)

## Baseline + Evidence Artifacts

Baseline benchmarking script:

- `train_and_evaluate.py`

Exports:

- `artifacts/benchmark_summary.json`
- `artifacts/reward_curves.csv`
- `artifacts/reward_curves.png` (if matplotlib available)

After any environment or reward-function change, rerun the benchmark/training scripts so artifacts reflect the latest code.

Current baseline snapshot (`artifacts/benchmark_summary.json`):

- `round_robin` avg score: `0.469`
- `heuristic` avg score: `0.782`
- `resilient` avg score: `0.776`
- resilient vs round_robin delta: `+0.307` score points (`+65.5%`)

GRPO training script output (default):

- `outputs/logiflow-grpo-script/artifacts/reward_curve.png`
- `outputs/logiflow-grpo-script/artifacts/evaluation_before.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_after.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_summary.csv`
- `outputs/logiflow-grpo-script/artifacts/evaluation_summary.json`
- `outputs/logiflow-grpo-script/artifacts/improvement.json`

## Run Locally

```bash
pip install -e ".[train]"
python test_engine.py
python train_and_evaluate.py
python train_grpo.py --model-name Qwen/Qwen2.5-0.5B-Instruct --max-steps 140
python inference.py
```

Start the API server:

```bash
uv run server
```

Then open:

- `/web`
- `/visualizer`
- `/docs`
- `/health`

## Phase-2 Checklist

See full checklist:

- `PHASE2_SUBMISSION.md`

Critical final-step reminders:

- Add the **HF Space URL**
- Add the **Colab URL**
- Add the **mini-blog/video URL**
- Keep large media out of repo; link externally

Optional helper template for rapid storytelling:

- `MINI_BLOG_OR_VIDEO_SCRIPT.md`
