import os
from typing import List, Optional

from openai import OpenAI

from crisis_logistics_env import CrisisLogisticsAction
from crisis_logistics_env.server.crisis_logistics_env_environment import (
    CrisisLogisticsEnvironment,
    choose_balancing_action,
)
from crisis_logistics_env.tasks import list_tasks


API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
BENCHMARK = os.getenv("BENCHMARK") or "logiflow_rl"
MAX_STEPS_OVERRIDE = os.getenv("MAX_STEPS")

SYSTEM_PROMPT = (
    "You are controlling a logistics routing environment with 3 hubs. "
    "Reply with exactly one digit: 0, 1, or 2. "
    "Choose the hub that best keeps the network balanced, avoids overload above 100, "
    "and keeps hubs near the 30-70 utilization band."
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


def build_user_prompt(task_title: str, objective: str, step: int, hub_loads: List[float], incoming_load: float, event_label: str, score: float) -> str:
    return (
        f"Task: {task_title}\n"
        f"Objective: {objective}\n"
        f"Step: {step}\n"
        f"Hub loads: {hub_loads}\n"
        f"Incoming shipment: {incoming_load}\n"
        f"Traffic event: {event_label}\n"
        f"Current score: {score:.3f}\n"
        "Return only one hub id: 0, 1, or 2."
    )


def choose_action_with_model(client: OpenAI, prompt: str) -> int:
    response = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=0.0,
        max_tokens=4,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    text = (response.choices[0].message.content or "").strip()
    if text and text[0] in {"0", "1", "2"}:
        return int(text[0])
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
            action_value = choose_balancing_action(observation)
            if client is not None:
                prompt = build_user_prompt(
                    env.task.title,
                    observation.objective,
                    observation.step_count + 1,
                    observation.hub_loads,
                    observation.incoming_load,
                    observation.event_label,
                    observation.cumulative_score,
                )
                try:
                    action_value = choose_action_with_model(client, prompt)
                    last_error = None
                except Exception as exc:
                    last_error = str(exc)

            action = CrisisLogisticsAction(target_hub=action_value)
            observation = env.step(action)
            reward = float(observation.reward or 0.0)
            rewards.append(reward)
            log_step(
                step=observation.step_count,
                action=f"route({action_value})",
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
