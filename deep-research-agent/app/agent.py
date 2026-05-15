# ruff: noqa
# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import os

import subprocess

import google.auth
from google.adk.agents import Agent
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools import AgentTool, google_search
from google.adk.tools.load_web_page import load_web_page
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import SseConnectionParams
from google.genai import types

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# Cloud Run URL takes priority; fall back to local SSE for dev.
_CLOUD_RUN_URL = "https://gemini-advisor-mcp-299457666029.us-central1.run.app"
_MCP_SSE_PORT = int(os.environ.get("GEMINI_ADVISOR_PORT", "8765"))
_MCP_SSE_URL = os.environ.get("GEMINI_ADVISOR_URL", f"{_CLOUD_RUN_URL}/sse")

def _auth_headers() -> dict[str, str]:
    """Return an OIDC identity token header for Cloud Run; empty for local dev."""
    if _MCP_SSE_URL.startswith("http://"):
        return {}  # local dev — no auth needed
    token = subprocess.check_output(
        ["gcloud", "auth", "print-identity-token"], text=True
    ).strip()
    return {"Authorization": f"Bearer {token}"}

# Dedicated search sub-agent.
# google_search is model-internal grounding and cannot be mixed with FunctionTools,
# so it lives in its own agent which root_agent calls via AgentTool.
search_agent = Agent(
    name="search_agent",
    model=Gemini(model="gemini-flash-latest"),
    description=(
        "Web search specialist. Given a research query, searches the web and "
        "returns comprehensive, sourced findings."
    ),
    instruction=(
        "You are a web research specialist. When given a query:\n"
        "1. Use google_search to find up-to-date, authoritative information.\n"
        "2. Return a detailed summary: key facts, data points, expert opinions, and source context.\n"
        "3. Prioritize accuracy over brevity."
    ),
    tools=[google_search],
)

# Main deep research coordinator.
# Uses gemini_advisor (via MCP) for strategic guidance at key decision points,
# search_agent for web retrieval, and load_web_page for full source content.
root_agent = Agent(
    name="deep_research_agent",
    model=Gemini(
        model="gemini-flash-latest",
        retry_options=types.HttpRetryOptions(attempts=3),
    ),
    description="Deep research agent that produces comprehensive, advisor-guided research reports.",
    instruction=(
        "You are a deep research agent. Your goal is thorough, accurate research reports.\n\n"
        "PROCESS — follow this order:\n"
        "1. Call gemini_advisor with task=<research topic>, executor_model='gemini-flash-latest', "
        "max_advisor_calls=2 to get a strategic plan: which angles to investigate, what to prioritize.\n"
        "2. Execute the plan: use search_agent for each major research angle. Be systematic.\n"
        "3. For the 2-3 most important sources, use load_web_page to read full content.\n"
        "4. Before writing the final report, call gemini_advisor again (same executor_model) "
        "to validate your synthesis and check for gaps.\n"
        "5. Write the final report.\n\n"
        "ADVISOR RULES:\n"
        "- Always pass executor_model='gemini-flash-latest' when calling gemini_advisor.\n"
        "- Give advisor guidance serious weight. Surface conflicts explicitly.\n"
        "- If the advisor and your findings disagree, note the conflict and ask for reconciliation.\n\n"
        "REPORT FORMAT:\n"
        "## Executive Summary\n"
        "## Key Findings  \n"
        "## Detailed Analysis\n"
        "## Sources & Methodology"
    ),
    tools=[
        AgentTool(search_agent),
        load_web_page,
        McpToolset(
            connection_params=SseConnectionParams(
                url=_MCP_SSE_URL,
                headers=_auth_headers(),
                sse_read_timeout=120.0,  # Pro inference can take ~30-60s
            ),
            tool_filter=["gemini_advisor"],
        ),
    ],
)

app = App(
    root_agent=root_agent,
    name="deep_research_app",
)
