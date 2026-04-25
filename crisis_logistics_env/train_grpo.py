from __future__ import annotations

import argparse
import inspect
import json
import random
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List

try:
    import pandas as pd
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer
except ImportError as exc:
    raise RuntimeError(
        "Missing GRPO training dependencies. Install with:\n"
        "  pip install -e .[train]\n"
        "or install torch/trl/transformers/datasets/peft/accelerate manually."
    ) from exc

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None

try:
    from crisis_logistics_env.models import CrisisLogisticsAction
    from crisis_logistics_env.server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_network_action,
    )
    from crisis_logistics_env.tasks import list_tasks
except ImportError:
    from models import CrisisLogisticsAction
    from server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_network_action,
    )
    from tasks import list_tasks


SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)


@dataclass
class EvalResult:
    policy: str
    task_id: str
    score: float
    total_reward: float
    retail_delivered: float
    sla_success_rate: float
    priority_service_rate: float
    invalid_actions: int


def build_prompt(observation, task_title: str) -> str:
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
        f"Priority target: {observation.priority_target_name} (node {observation.priority_target_node})\n"
        "Return exactly one JSON object with keys: reasoning, source_node, dest_node, shipment_volume."
    )


def build_training_rows(samples_per_task: int = 42) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for task in list_tasks():
        env = CrisisLogisticsEnvironment()
        obs = env.reset(task_id=task.task_id)
        for _ in range(samples_per_task):
            source = int(obs.pending_source_node)
            rows.append(
                {
                    "prompt": build_prompt(obs, task.title),
                    "task_id": task.task_id,
                    "source_node": source,
                    "valid_dests": json.dumps(obs.connectivity.get(str(source), [])),
                    "incoming_load": float(obs.incoming_load),
                    "priority_target_node": int(obs.priority_target_node),
                }
            )
            obs = env.step(choose_network_action(obs))
            if obs.done:
                break
    return rows


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, dict):
        content = completion.get("content", completion)
        if isinstance(content, list):
            chunks = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    chunks.append(str(item["text"]))
                else:
                    chunks.append(str(item))
            return "\n".join(chunks)
        return str(content)
    if isinstance(completion, list):
        chunks = []
        for item in completion:
            if isinstance(item, dict):
                content = item.get("content", "")
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and "text" in part:
                            chunks.append(str(part["text"]))
                        else:
                            chunks.append(str(part))
                else:
                    chunks.append(str(content))
            else:
                chunks.append(str(item))
        return "\n".join(chunks)
    return str(completion)


def extract_json(text: str) -> Dict[str, Any]:
    if not isinstance(text, str):
        return {}
    decoder = json.JSONDecoder()
    candidates: List[Dict[str, Any]] = []
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            payload, _ = decoder.raw_decode(text[index:])
        except Exception:
            continue
        if isinstance(payload, dict):
            candidates.append(payload)
    if not candidates:
        return {}
    required = {"reasoning", "source_node", "dest_node", "shipment_volume"}
    for payload in reversed(candidates):
        if required.issubset(payload.keys()):
            return payload
    return candidates[-1]


def as_batch(values: Any, n: int) -> List[Any]:
    if values is None:
        return [None] * n
    if isinstance(values, (list, tuple)):
        if len(values) == n:
            return list(values)
        if len(values) == 1:
            return [values[0]] * n
    return [values] * n


def action_reward(
    completions,
    source_node=None,
    valid_dests=None,
    incoming_load=None,
    priority_target_node=None,
    **kwargs,
):
    n = len(completions)
    source_batch = as_batch(source_node, n)
    dest_batch = as_batch(valid_dests, n)
    incoming_batch = as_batch(incoming_load, n)
    priority_batch = as_batch(priority_target_node, n)

    rewards = []
    for index, completion in enumerate(completions):
        reward = 0.0
        parsed = extract_json(completion_to_text(completion))

        if parsed:
            reward += 0.20

        required = {"reasoning", "source_node", "dest_node", "shipment_volume"}
        if required.issubset(parsed.keys()):
            reward += 0.20

        try:
            src = int(parsed.get("source_node", -999))
            expected = int(source_batch[index]) if source_batch[index] is not None else None
            if expected is not None and src == expected:
                reward += 0.20
        except Exception:
            pass

        try:
            allowed = (
                json.loads(dest_batch[index])
                if isinstance(dest_batch[index], str)
                else (dest_batch[index] or [])
            )
            dest = int(parsed.get("dest_node", -999))
            if dest in allowed:
                reward += 0.20
        except Exception:
            pass

        try:
            volume = float(parsed.get("shipment_volume", -1))
            incoming = float(incoming_batch[index]) if incoming_batch[index] is not None else 10.0
            if 0 < volume <= 60:
                reward += 0.15
            closeness = max(0.0, 1.0 - abs(volume - incoming) / max(1.0, incoming))
            reward += 0.03 * closeness
        except Exception:
            pass

        try:
            target = int(priority_batch[index]) if priority_batch[index] is not None else None
            dest = int(parsed.get("dest_node", -999))
            if target is not None and dest == target:
                reward += 0.02
        except Exception:
            pass

        if not parsed:
            reward -= 0.05

        rewards.append(float(max(0.0, min(1.0, reward))))
    return rewards


def reward_sanity_check() -> None:
    good = ['{"reasoning":"route","source_node":2,"dest_node":4,"shipment_volume":11}']
    bad = ["not json at all"]
    good_score = action_reward(
        good,
        source_node=[2],
        valid_dests=["[4,5]"],
        incoming_load=[10],
        priority_target_node=[4],
    )[0]
    bad_score = action_reward(
        bad,
        source_node=[2],
        valid_dests=["[4,5]"],
        incoming_load=[10],
        priority_target_node=[4],
    )[0]
    print(f"reward_sanity good={good_score:.3f} bad={bad_score:.3f}")
    if good_score <= 0.60 or bad_score >= 0.20:
        raise RuntimeError("Reward sanity check failed; refusing to start GRPO training.")


def supported_kwargs(cls, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    supported = set(inspect.signature(cls).parameters)
    dropped = sorted(set(kwargs) - supported)
    if dropped:
        print(f"Skipping unsupported {cls.__name__} args: {dropped}")
    return {key: value for key, value in kwargs.items() if key in supported}


def generate_action(model, tokenizer, prompt: str) -> CrisisLogisticsAction:
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=110,
            do_sample=False,
            temperature=0.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    prompt_tokens = int(inputs["input_ids"].shape[1])
    generated_tokens = output[0][prompt_tokens:]
    text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
    if not text:
        text = tokenizer.decode(output[0], skip_special_tokens=True).strip()
    payload = extract_json(text)
    if not payload:
        return CrisisLogisticsAction(target_hub=0)
    try:
        return CrisisLogisticsAction(**payload)
    except Exception:
        return CrisisLogisticsAction(target_hub=0)


def run_policy(policy: str, task_id: str, model=None, tokenizer=None) -> EvalResult:
    env = CrisisLogisticsEnvironment()
    obs = env.reset(task_id=task_id)
    round_robin_step = 0

    while not obs.done:
        if policy == "round_robin":
            action = CrisisLogisticsAction(target_hub=round_robin_step % 3)
            round_robin_step += 1
        elif policy == "heuristic":
            action = choose_network_action(obs)
        elif policy == "model":
            if model is None or tokenizer is None:
                raise ValueError("model policy requires model and tokenizer")
            action = generate_action(model, tokenizer, build_prompt(obs, env.task.title))
        else:
            raise ValueError(f"Unknown policy: {policy}")
        obs = env.step(action)

    return EvalResult(
        policy=policy,
        task_id=task_id,
        score=float(env.score),
        total_reward=float(env.total_reward),
        retail_delivered=float(env.retail_delivered),
        sla_success_rate=float(env._sla_success_rate()),
        priority_service_rate=float(env._priority_service_rate()),
        invalid_actions=int(env.invalid_actions),
    )


def summarize_policy(results: Iterable[EvalResult], phase: str) -> pd.DataFrame:
    frame = pd.DataFrame([asdict(item) for item in results])
    summary = frame.groupby("policy", as_index=False).agg(
        avg_score=("score", "mean"),
        avg_reward=("total_reward", "mean"),
        avg_sla=("sla_success_rate", "mean"),
    )
    summary["phase"] = phase
    return summary


def extract_reward_history(log_history: List[Dict[str, Any]]) -> List[float]:
    keys = ["rewards/mean", "reward", "train/reward", "objective/reward"]
    history: List[float] = []
    for row in log_history:
        for key in keys:
            if key in row and isinstance(row[key], (int, float)):
                history.append(float(row[key]))
                break
    return history


def save_reward_curve_png(reward_history: List[float], output_path: Path) -> None:
    if plt is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(reward_history, linewidth=2)
        plt.xlabel("Logging Step")
        plt.ylabel("Mean Reward")
        plt.title("GRPO Reward Curve - LogiFlow-RL")
        plt.grid(alpha=0.2)
        plt.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close()
        return

    try:
        from PIL import Image, ImageDraw

        width, height = 1100, 540
        margin_left, margin_right = 70, 30
        margin_top, margin_bottom = 40, 60
        plot_w = width - margin_left - margin_right
        plot_h = height - margin_top - margin_bottom
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        y_min = min(reward_history)
        y_max = max(reward_history)
        if abs(y_max - y_min) < 1e-6:
            y_max = y_min + 1.0

        def map_x(step: int) -> int:
            if len(reward_history) <= 1:
                return margin_left
            return int(margin_left + (step / (len(reward_history) - 1)) * plot_w)

        def map_y(value: float) -> int:
            return int(margin_top + (1 - (value - y_min) / (y_max - y_min)) * plot_h)

        draw.rectangle(
            [margin_left, margin_top, margin_left + plot_w, margin_top + plot_h],
            outline="#333333",
            width=1,
        )
        points = [(map_x(step), map_y(value)) for step, value in enumerate(reward_history)]
        if len(points) >= 2:
            draw.line(points, fill="#1565C0", width=3)
        elif len(points) == 1:
            x, y = points[0]
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill="#1565C0")

        draw.text((margin_left, 10), "GRPO Reward Curve - LogiFlow-RL", fill="#111111")
        draw.text((margin_left, height - 45), "Logging Step", fill="#111111")
        draw.text((10, margin_top), "Mean Reward", fill="#111111")
        image.save(output_path)
    except Exception as exc:
        raise RuntimeError(f"Could not save reward curve png without matplotlib: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description="GRPO training script for LogiFlow-RL.")
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--samples-per-task", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=140)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--output-dir", default="outputs/logiflow-grpo-script")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    artifacts_dir = output_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    reward_sanity_check()

    train_rows = build_training_rows(samples_per_task=args.samples_per_task)
    train_ds = Dataset.from_list(train_rows)
    print(f"dataset rows={len(train_ds)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    training_args = GRPOConfig(
        **supported_kwargs(
            GRPOConfig,
            dict(
                output_dir=str(output_dir),
                learning_rate=1e-5,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=8,
                num_generations=args.num_generations,
                max_prompt_length=1024,
                max_completion_length=128,
                max_steps=args.max_steps,
                logging_steps=5,
                save_steps=max(20, args.max_steps // 2),
                report_to=[],
                remove_unused_columns=False,
                bf16=torch.cuda.is_available(),
            ),
        )
    )

    trainer = GRPOTrainer(
        **supported_kwargs(
            GRPOTrainer,
            dict(
                model=args.model_name,
                args=training_args,
                train_dataset=train_ds,
                reward_funcs=[action_reward],
                peft_config=peft_config,
                processing_class=tokenizer,
            ),
        )
    )

    before_results: List[EvalResult] = []
    for task in list_tasks():
        before_results.append(run_policy("round_robin", task.task_id, trainer.model, tokenizer))
        before_results.append(run_policy("heuristic", task.task_id, trainer.model, tokenizer))
        before_results.append(run_policy("model", task.task_id, trainer.model, tokenizer))

    trainer.train()

    reward_history = extract_reward_history(trainer.state.log_history)
    if not reward_history:
        raise RuntimeError("No reward points found in trainer logs.")

    reward_curve_path = artifacts_dir / "reward_curve.png"
    save_reward_curve_png(reward_history, reward_curve_path)

    after_results: List[EvalResult] = []
    for task in list_tasks():
        after_results.append(run_policy("round_robin", task.task_id, trainer.model, tokenizer))
        after_results.append(run_policy("heuristic", task.task_id, trainer.model, tokenizer))
        after_results.append(run_policy("model", task.task_id, trainer.model, tokenizer))

    before_df = pd.DataFrame([asdict(item) for item in before_results])
    after_df = pd.DataFrame([asdict(item) for item in after_results])

    summary_before = summarize_policy(before_results, "before")
    summary_after = summarize_policy(after_results, "after")
    summary = pd.concat([summary_before, summary_after], ignore_index=True)

    model_before = float(summary_before.loc[summary_before["policy"] == "model", "avg_score"].iloc[0])
    model_after = float(summary_after.loc[summary_after["policy"] == "model", "avg_score"].iloc[0])
    improvement = {
        "model_avg_score_before": model_before,
        "model_avg_score_after": model_after,
        "delta_score": model_after - model_before,
        "reward_history_points": len(reward_history),
        "reward_history_mean": round(mean(reward_history), 6),
    }

    before_df.to_csv(artifacts_dir / "evaluation_before.csv", index=False)
    after_df.to_csv(artifacts_dir / "evaluation_after.csv", index=False)
    summary.to_csv(artifacts_dir / "evaluation_summary.csv", index=False)

    payload = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_name": args.model_name,
        "seed": SEED,
        "before": before_df.to_dict(orient="records"),
        "after": after_df.to_dict(orient="records"),
        "summary": summary.to_dict(orient="records"),
        "improvement": improvement,
    }
    (artifacts_dir / "evaluation_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (artifacts_dir / "improvement.json").write_text(json.dumps(improvement, indent=2), encoding="utf-8")

    trainer.save_model(str(output_dir))

    print("Saved artifacts:")
    print(f"- {reward_curve_path}")
    print(f"- {artifacts_dir / 'evaluation_before.csv'}")
    print(f"- {artifacts_dir / 'evaluation_after.csv'}")
    print(f"- {artifacts_dir / 'evaluation_summary.csv'}")
    print(f"- {artifacts_dir / 'evaluation_summary.json'}")
    print(f"- {artifacts_dir / 'improvement.json'}")


if __name__ == "__main__":
    main()
