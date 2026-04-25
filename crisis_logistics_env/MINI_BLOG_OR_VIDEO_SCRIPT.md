# Mini Blog / <2 Minute Video Script (Template)

Use this directly for the required storytelling deliverable.

## Title

LogiFlow-RL: Training LLMs for Crisis Logistics Routing with OpenEnv + GRPO

## 1) Problem (15-20 sec)

Large language models are strong at text tasks, but they are weaker at sequential operational decisions under uncertainty.  
LogiFlow-RL targets this gap using crisis logistics: delayed transit, disruptions, partial observability, and urgent retail demand.

## 2) Environment (25-30 sec)

The environment is a 12-node network: suppliers, warehouses, distribution centers, and retail sinks.  
At each step, the agent chooses structured routing actions (`source_node`, `dest_node`, `shipment_volume`).  
The world changes dynamically with adaptive pressure, stochastic disruptions, and priority-demand windows.

## 3) Reward + Anti-Hacking (20-25 sec)

Reward is multi-component: valid action, throughput, SLA, balance, recovery, priority-service, and transit health.  
Penalties cover overload, invalid actions, route loops, risky routing, and pressure misuse.  
The reward breakdown is exposed each step for inspection and anti-gaming diagnostics.

## 4) Training + Results (20-25 sec)

We train with HF TRL `GRPOTrainer` using verifiable reward functions.  
We show baseline policies versus trained model behavior and include reward curves plus before/after evaluation artifacts in the repo.

## 5) Why It Matters (10-15 sec)

This benchmark simulates realistic operational stress and is useful for training agents that must make reliable, step-wise decisions under uncertainty, not just generate plausible text.

## Link Block (must include in blog/video description)

- Repo: `<your-repo-link>`
- HF Space: `<your-space-link>`
- Colab notebook: `<your-colab-link>`
- Evidence artifacts (reward curve + before/after): `<repo-folder-link>`
