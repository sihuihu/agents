# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import os
import uuid

import google.auth
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from google.adk.cli.fast_api import get_fast_api_app
from google.adk.runners import InMemoryRunner
from google.cloud import logging as google_cloud_logging
from google.genai import types as genai_types

from app.agent import app as adk_app
from app.app_utils.telemetry import setup_telemetry
from app.app_utils.typing import Feedback

setup_telemetry()
_, project_id = google.auth.default()
logging_client = google_cloud_logging.Client()
logger = logging_client.logger(__name__)
allow_origins = (
    os.getenv("ALLOW_ORIGINS", "").split(",") if os.getenv("ALLOW_ORIGINS") else None
)

# Artifact bucket for ADK (created by Terraform, passed via env var)
logs_bucket_name = os.environ.get("LOGS_BUCKET_NAME")

AGENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# In-memory session configuration - no persistent storage
session_service_uri = None

artifact_service_uri = f"gs://{logs_bucket_name}" if logs_bucket_name else None

app: FastAPI = get_fast_api_app(
    agents_dir=AGENT_DIR,
    web=True,
    artifact_service_uri=artifact_service_uri,
    allow_origins=allow_origins,
    session_service_uri=session_service_uri,
    otel_to_cloud=True,
)
app.title = "agent-news-digest"
app.description = "API for interacting with the Agent agent-news-digest"


@app.post("/trigger/daily-digest")
async def trigger_daily_digest() -> JSONResponse:
    """Trigger the daily AI agent news digest.

    Called by Cloud Scheduler (or manually via curl). Runs the full
    search → format → email pipeline and returns a summary.
    """
    runner = InMemoryRunner(app=adk_app)
    session = await runner.session_service.create_session(
        app_name=runner.app_name, user_id=f"scheduler-{uuid.uuid4().hex[:8]}"
    )
    result_parts: list[str] = []
    async for ev in runner.run_async(
        user_id=session.user_id,
        session_id=session.id,
        new_message=genai_types.Content(
            role="user",
            parts=[genai_types.Part(text="Run the daily agent news digest.")],
        ),
    ):
        if ev.content and ev.content.parts:
            for p in ev.content.parts:
                t = getattr(p, "text", None)
                if t:
                    result_parts.append(t)

    summary = result_parts[-1] if result_parts else "No output"
    logger.log_struct({"event": "daily_digest_triggered", "summary": summary}, severity="INFO")
    return JSONResponse({"status": "ok", "summary": summary})


@app.post("/feedback")
def collect_feedback(feedback: Feedback) -> dict[str, str]:
    """Collect and log feedback.

    Args:
        feedback: The feedback data to log

    Returns:
        Success message
    """
    logger.log_struct(feedback.model_dump(), severity="INFO")
    return {"status": "success"}


# Main execution
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
