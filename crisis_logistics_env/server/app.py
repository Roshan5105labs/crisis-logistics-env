# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the LogiFlow-RL OpenEnv environment."""

from typing import Optional

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

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
    from .crisis_logistics_env_environment import CrisisLogisticsEnvironment
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
    from server.crisis_logistics_env_environment import CrisisLogisticsEnvironment


app = FastAPI(
    title="OpenEnv Environment HTTP API",
    version="1.0.0",
    description=(
        "HTTP API for interacting with the LogiFlow-RL environment through "
        "a standardized OpenEnv-style interface."
    ),
)

env = CrisisLogisticsEnvironment()


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
              <li><a href="/web">Web landing page</a></li>
            </ul>
          </body>
        </html>
        """
    )


@app.get("/web", response_class=HTMLResponse, tags=["Environment Info"])
async def web_landing() -> HTMLResponse:
    return HTMLResponse(
        """
        <html>
          <head><title>LogiFlow-RL Web</title></head>
          <body style="font-family: Arial, sans-serif; margin: 40px;">
            <h1>LogiFlow-RL</h1>
            <p>The environment is live. Use the links below to explore it.</p>
            <ul>
              <li><a href="/docs">Swagger docs</a></li>
              <li><a href="/health">Health endpoint</a></li>
              <li><a href="/schema">JSON schema</a></li>
            </ul>
          </body>
        </html>
        """
    )


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


@app.get("/state", response_model=CrisisLogisticsState, tags=["State Management"])
async def get_state() -> CrisisLogisticsState:
    return env.state


@app.get("/metadata", response_model=EnvironmentMetadata, tags=["Environment Info"])
async def get_metadata() -> EnvironmentMetadata:
    return EnvironmentMetadata(
        name="LogiFlow-RL",
        description="Deterministic logistics routing benchmark with easy, medium, and hard tasks.",
        version="1.0.0",
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
