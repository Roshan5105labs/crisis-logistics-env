# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the LogiFlow-RL OpenEnv environment."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

try:
    from openenv.core.env_server.types import (
        EnvironmentMetadata,
        HealthResponse,
        HealthStatus,
        ResetRequest,
        ResetResponse,
        SchemaResponse,
        StepRequest,
        StepResponse,
    )
    from ..models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from .crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_resilient_action,
    )
except ImportError:
    from openenv.core.env_server.types import (
        EnvironmentMetadata,
        HealthResponse,
        HealthStatus,
        ResetRequest,
        ResetResponse,
        SchemaResponse,
        StepRequest,
        StepResponse,
    )
    from models import (
        CrisisLogisticsAction,
        CrisisLogisticsObservation,
        CrisisLogisticsState,
    )
    from server.crisis_logistics_env_environment import (
        CrisisLogisticsEnvironment,
        choose_resilient_action,
    )


app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="1.1.0",
    description=(
        "HTTP API for interacting with the LogiFlow-RL environment through "
        "a standardized OpenEnv-style interface."
    ),
)

env = CrisisLogisticsEnvironment()
VISUALIZER_PATH = Path(__file__).resolve().parent.parent / "visualisation" / "logiflow_visualizer.html"


class PolicyStepRequest(BaseModel):
    mode: Literal["heuristic", "llm"] = "heuristic"
    timeout_s: Optional[float] = None


class PolicyStepResponse(BaseModel):
    observation: Dict[str, Any]
    reward: float
    done: bool
    policy_mode: Literal["heuristic", "llm"]
    action_source: Literal["heuristic", "llm"]
    action: Dict[str, Any]
    llm_model: Optional[str] = None
    llm_raw_output: Optional[str] = None


def _build_policy_prompt(observation: CrisisLogisticsObservation, title: str) -> str:
    return (
        f"Task: {title}\n"
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


def _extract_json_payload(text: str) -> Dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates = []
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


def _resolve_llm_action(observation: CrisisLogisticsObservation) -> tuple[CrisisLogisticsAction, str, str]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="LLM mode needs HF_TOKEN or OPENAI_API_KEY set in Space secrets.",
        )
    base_url = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
    model_name = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

    try:
        from openai import OpenAI
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"openai client import failed: {exc}") from exc

    prompt = _build_policy_prompt(observation, env.task.title)
    system_prompt = (
        "You are a logistics routing policy for a crisis supply chain environment. "
        "Always return exactly one JSON object with keys: reasoning, source_node, dest_node, shipment_volume."
    )

    client = OpenAI(api_key=api_key, base_url=base_url)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0.0,
            max_tokens=180,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"LLM request failed: {exc}") from exc

    raw_text = (response.choices[0].message.content or "").strip()
    payload = _extract_json_payload(raw_text)
    if not payload:
        raise HTTPException(
            status_code=422,
            detail=f"LLM output did not contain valid action JSON. Raw output: {raw_text[:600]}",
        )
    try:
        action = CrisisLogisticsAction(**payload)
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"LLM output JSON could not be parsed as action: {exc}. Raw output: {raw_text[:600]}",
        ) from exc
    return action, model_name, raw_text


def _read_visualizer_html() -> str:
    """Load the standalone visualizer HTML bundled with the project."""
    if VISUALIZER_PATH.exists():
        return VISUALIZER_PATH.read_text(encoding="utf-8")
    return """
    <html>
      <head><title>LogiFlow-RL Visualizer Missing</title></head>
      <body style="font-family: Arial, sans-serif; margin: 40px;">
        <h1>Visualizer file not found</h1>
        <p>Expected to find the visualizer at <code>/visualisation/logiflow_visualizer.html</code>.</p>
      </body>
    </html>
    """


@app.get("/", response_class=HTMLResponse, tags=["Environment Info"])
async def root() -> HTMLResponse:
    return HTMLResponse(
        """
        <html>
          <head><title>LogiFlow-RL</title></head>
          <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>LogiFlow-RL</h1>
            <p>OpenEnv benchmark for dynamic logistics routing.</p>
            <ul>
              <li><a href="/docs">API docs</a></li>
              <li><a href="/health">Health</a></li>
              <li><a href="/schema">Schema</a></li>
              <li><a href="/web">Live visualizer</a></li>
            </ul>
          </body>
        </html>
        """
    )


@app.get("/web", response_class=HTMLResponse, tags=["Environment Info"])
async def web_landing() -> HTMLResponse:
    return HTMLResponse(_read_visualizer_html())


@app.get("/visualizer", response_class=HTMLResponse, tags=["Environment Info"])
async def visualizer() -> HTMLResponse:
    return HTMLResponse(_read_visualizer_html())


@app.post("/reset", response_model=ResetResponse, tags=["Environment Control"])
async def reset_environment(request: Optional[ResetRequest] = None) -> ResetResponse:
    request = request or ResetRequest()
    task_id = getattr(request, "task_id", None) or "easy"
    observation = env.reset(seed=request.seed, episode_id=request.episode_id, task_id=task_id)
    return ResetResponse(
        observation=observation.model_dump(),
        reward=float(observation.reward or 0.0),
        done=observation.done,
    )


@app.post("/step", response_model=StepResponse, tags=["Environment Control"])
async def step_environment(request: StepRequest) -> StepResponse:
    action = CrisisLogisticsAction(**request.action)
    observation = env.step(action, timeout_s=request.timeout_s)
    return StepResponse(
        observation=observation.model_dump(),
        reward=float(observation.reward or 0.0),
        done=observation.done,
    )


@app.post("/policy_step", response_model=PolicyStepResponse, tags=["Environment Control"])
async def policy_step(request: PolicyStepRequest) -> PolicyStepResponse:
    """Execute one environment step using either heuristic or strict LLM policy mode."""
    # Build current observation snapshot for policy selection.
    observation = env._get_observation("Policy evaluation snapshot.")

    if request.mode == "heuristic":
        action = choose_resilient_action(observation)
        policy_mode = "heuristic"
        action_source = "heuristic"
        llm_model = None
        llm_raw_output = None
    elif request.mode == "llm":
        action, llm_model, llm_raw_output = _resolve_llm_action(observation)
        policy_mode = "llm"
        action_source = "llm"
    else:
        raise HTTPException(status_code=400, detail=f"Unsupported policy mode: {request.mode}")

    next_observation = env.step(action, timeout_s=request.timeout_s)
    return PolicyStepResponse(
        observation=next_observation.model_dump(),
        reward=float(next_observation.reward or 0.0),
        done=next_observation.done,
        policy_mode=policy_mode,
        action_source=action_source,
        action=action.model_dump(exclude_none=True),
        llm_model=llm_model,
        llm_raw_output=llm_raw_output if request.mode == "llm" else None,
    )


@app.get("/state", response_model=CrisisLogisticsState, tags=["State Management"])
async def get_state() -> CrisisLogisticsState:
    return env.state


@app.get("/metadata", response_model=EnvironmentMetadata, tags=["Environment Info"])
async def get_metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="LogiFlow-RL",
        description=(
            "Adaptive logistics routing benchmark with dynamic pressure, priority demand, "
            "and multi-component verifiable rewards."
        ),
        version="1.1.0",
    )


@app.get("/schema", response_model=SchemaResponse, tags=["Schema"])
async def get_schema() -> SchemaResponse:
    return SchemaResponse(
        action=CrisisLogisticsAction.model_json_schema(),
        observation=CrisisLogisticsObservation.model_json_schema(),
        state=CrisisLogisticsState.model_json_schema(),
    )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health() -> HealthResponse:
    return HealthResponse(status=HealthStatus.HEALTHY)


def main() -> None:
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
