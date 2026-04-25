import os
import json
from typing import List, Optional

from openai import OpenAI

try:
    from crisis_logistics_env import CrisisLogisticsAction
    from crisis_logistics_env.server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_resilient_action,
    )
    from crisis_logistics_env.tasks import list_tasks
except ImportError:
    from models import CrisisLogisticsAction
    from server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_resilient_action,
    )
    from tasks import list_tasks


API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("BENCHMARK") or "logiflow_rl"
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")

SYSTEM_PROMPT = (
    "You are a live logistics crisis manager controlling a 12-node supply-chain network. "
    "Reason about visible node loads, active disruptions, delayed in-transit shipments, and SLA pressure. "
    "Return exactly one JSON object with keys: reasoning, source_node, dest_node, shipment_volume. "
    "The destination must be connected to the source."
)


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(observation, task_title: str) -> str:
    return (
        f"Task: {task_title}\n"
        f"Objective: {observation.objective}\n"
        f"Step: {observation.step_count + 1}/{observation.max_steps}\n"
        f"Visible nodes: {observation.visible_node_ids}\n"
        f"Observed node loads: {observation.observed_node_loads}\n"
        f"Node capacities: {observation.node_capacities}\n"
        f"Visible connectivity: {observation.visible_connectivity}\n"
        f"Active disruptions: {observation.active_disruptions}\n"
        f"In-transit shipments: {observation.in_transit_shipments[:8]}\n"
        f"Incoming shipment: source={observation.pending_source_node}, volume={observation.incoming_load}\n"
        f"Traffic event: {observation.event_label}\n"
        f"Dynamic pressure: {observation.dynamic_pressure}\n"
        f"Priority target node: {observation.priority_target_node} ({observation.priority_target_name})\n"
        f"Adaptive disruption rate: {observation.adaptive_disruption_rate}\n"
        f"Priority service rate: {observation.priority_service_rate}\n"
        f"Current score: {observation.cumulative_score:.3f}\n"
        "Return one compact JSON object only."
    )


def choose_action_with_model(client: OpenAI, prompt: str) -> CrisisLogisticsAction:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=180,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    decoder = json.JSONDecoder()
    candidates = []
    for idx, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    if candidates:
        required = {"reasoning", "source_node", "dest_node", "shipment_volume"}
        for payload in reversed(candidates):
            if required.issubset(payload.keys()):
                return CrisisLogisticsAction(**payload)
        return CrisisLogisticsAction(**candidates[-1])
    raise ValueError(f"invalid_model_output:{text}")


def run_task(task_id: str, client: Optional[OpenAI]) -> float:
    env = CrisisLogisticsEnvironment()
    observation = env.reset(task_id=task_id)
    rewards: List[float] = []
    last_error: Optional[str] = None
    max_steps = min(
        env.task.max_steps,
        int(MAX_STEPS_OVERRIDE) if MAX_STEPS_OVERRIDE else env.task.max_steps,
    )

    log_start(task_id, BENCHMARK, MODEL_NAME)

    try:
        while not observation.done and observation.step_count < max_steps:
            action = choose_resilient_action(observation)
            if client is not None:
                prompt = build_user_prompt(observation, env.task.title)
                try:
                    action = choose_action_with_model(client, prompt)
                    last_error = None
                except Exception as exc:
                    last_error = str(exc)

            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            action_label = (
                f"route({action.source_node}->{action.dest_node},vol={action.shipment_volume})"
                if action.source_node is not None and action.dest_node is not None
                else f"route({action.target_hub})"
            )
            log_step(
                step=observation.step_count,
                action=action_label,
                reward=reward,
                done=observation.done,
                error=last_error,
            )

        final_score = observation.cumulative_score
        success = final_score >= 0.65
        return_score = final_score
        log_end(success, observation.step_count, return_score, rewards)
        return return_score
    except Exception:
        log_end(False, observation.step_count, 0.0, rewards)
        raise


def main() -> None:
    client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL) if API_KEY else None
    for task in list_tasks():
        run_task(task.task_id, client)


if __name__ == "__main__":
    main()
