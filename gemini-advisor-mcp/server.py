#!/usr/bin/env python3
"""MCP server: caller-supplied executor model with Gemini Pro advisor.

Mirrors the Anthropic advisor-tool pattern:
  1. Executor model (supplied by caller) generates an initial response
  2. Gemini 3.1 Pro reads the full context and returns strategic guidance
  3. Executor refines its response using that guidance
  4. Loop up to max_advisor_calls times; Pro is always consulted

Modes:
  stdio (default): python server.py
  SSE:             python server.py --sse [PORT]   (default port 8765)
"""

import asyncio
import json
import os
import sys
from typing import Any

import mcp.types as mcp_types
from google import genai
from google.genai import types
from mcp.server import Server
from mcp.server.stdio import stdio_server

PRO_MODEL = "gemini-3.1-pro-preview"  # advisor is always Pro; executor is caller-supplied

FLASH_SYSTEM = (
    "You are an efficient executor. Solve tasks concisely and correctly. "
    "When you need strategic guidance on approach or are uncertain how to proceed, "
    "output the exact marker [NEED_ADVISOR] on its own line before your current best attempt."
)

ADVISOR_INSTRUCTION = (
    "You are a strategic advisor reviewing an executor's work. "
    "Provide concise guidance in under 150 words using numbered steps. "
    "Focus on correctness, completeness, and missed edge cases. "
    "Do not produce the final answer yourself — guide the executor."
)

server = Server("gemini-advisor")


def _client() -> genai.Client:
    return genai.Client(
        vertexai=True,
        project=os.environ.get("GOOGLE_CLOUD_PROJECT"),
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
    )


@server.list_tools()
async def list_tools() -> list[mcp_types.Tool]:
    return [
        mcp_types.Tool(
            name="gemini_advisor",
            description=(
                "Process a task using the caller's own model as the executor, with "
                "Gemini 3.1 Pro as a strategic advisor that always reviews the work "
                "and provides guidance. Pass your own model ID as executor_model so "
                "the advisor loop runs the same model you are. Mirrors the Anthropic "
                "advisor-tool pattern."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or question to process.",
                    },
                    "executor_model": {
                        "type": "string",
                        "description": (
                            "Vertex AI model ID of the calling agent, used as the executor "
                            "inside the advisor loop (e.g. 'gemini-flash-latest'). "
                            "Pass your own model name so the loop mirrors your capabilities."
                        ),
                    },
                    "max_advisor_calls": {
                        "type": "integer",
                        "description": "Cap on Pro advisor calls per request. Default 2.",
                        "default": 2,
                    },
                },
                "required": ["task", "executor_model"],
            },
        )
    ]


@server.call_tool()
async def call_tool(
    name: str, arguments: dict[str, Any]
) -> list[mcp_types.TextContent]:
    if name != "gemini_advisor":
        raise ValueError(f"Unknown tool: {name}")

    task: str = arguments["task"]
    executor_model: str = arguments["executor_model"]
    max_advisor_calls: int = arguments.get("max_advisor_calls", 2)

    client = _client()

    # Phase 1: Executor initial pass
    exec_resp = client.models.generate_content(
        model=executor_model,
        contents=task,
        config=types.GenerateContentConfig(system_instruction=FLASH_SYSTEM),
    )
    current_output: str = exec_resp.text

    advisor_calls = 0
    advice_history: list[str] = []

    # Pro is always consulted — no toggle
    while advisor_calls < max_advisor_calls:
        advisor_calls += 1

        # Phase 2: Pro advisor sees full context
        advisor_prompt = (
            f"Task given to executor:\n{task}\n\n"
            f"Executor's current output:\n{current_output}\n\n"
            f"{ADVISOR_INSTRUCTION}"
        )
        pro_resp = client.models.generate_content(
            model=PRO_MODEL,
            contents=advisor_prompt,
        )
        advice: str = pro_resp.text
        advice_history.append(advice)

        # Phase 3: Executor refines using advice
        refinement_prompt = (
            f"Original task:\n{task}\n\n"
            f"Your previous attempt:\n{current_output}\n\n"
            f"Strategic advice from advisor:\n{advice}\n\n"
            "Produce your final response incorporating the advisor's guidance."
        )
        exec_refined = client.models.generate_content(
            model=executor_model,
            contents=refinement_prompt,
            config=types.GenerateContentConfig(system_instruction=FLASH_SYSTEM),
        )
        current_output = exec_refined.text

    result: dict[str, Any] = {
        "response": current_output,
        "executor_model": executor_model,
        "advisor_model": PRO_MODEL,
        "advisor_calls_made": advisor_calls,
    }
    if advice_history:
        result["advisor_guidance"] = advice_history

    return [mcp_types.TextContent(type="text", text=json.dumps(result, indent=2))]


# ── stdio mode ────────────────────────────────────────────────────────────────

async def _run_stdio() -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )


# ── SSE mode ──────────────────────────────────────────────────────────────────

def _run_sse(port: int) -> None:
    import uvicorn
    from mcp.server.sse import SseServerTransport
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.routing import Mount, Route

    transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(
                streams[0],
                streams[1],
                server.create_initialization_options(),
            )

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=transport.handle_post_message),
        ]
    )
    # Cloud Run sets $PORT; fall back to explicit arg or 8765 for local dev.
    host = "0.0.0.0" if os.environ.get("K_SERVICE") else "127.0.0.1"
    uvicorn.run(app, host=host, port=port, log_level="info")


# ── entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--sse":
        port = int(os.environ.get("PORT", sys.argv[2] if len(sys.argv) > 2 else "8765"))
        _run_sse(port)
    else:
        asyncio.run(_run_stdio())
