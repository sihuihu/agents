"""Integration test: run the deep research agent on a sample topic.

Starts the Gemini Advisor MCP server in SSE mode as a background subprocess,
runs the agent, then cleans up.

Usage:
    cd deep-research-agent
    uv run python tests/test_research.py
    uv run python tests/test_research.py "Your custom research topic here"
"""

import asyncio
import os
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

MCP_SERVER_PATH = "/usr/local/google/home/sihuihu/gemini-advisor-mcp/server.py"
MCP_PORT = int(os.environ.get("GEMINI_ADVISOR_PORT", "8765"))

# Ensure Vertex AI location is set before the MCP server subprocess is spawned
# so it inherits the correct location (global has the 3.1 models).
os.environ.setdefault("GOOGLE_CLOUD_LOCATION", "global")

RESEARCH_TOPIC = (
    "What are the most promising recent breakthroughs in solid-state battery technology "
    "for electric vehicles, and what are the main remaining challenges to commercialization?"
)


def _wait_for_server(port: int, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/sse", timeout=1)
        except Exception:
            time.sleep(0.3)
            continue
        return True
    return False


def _start_mcp_server() -> subprocess.Popen:
    # Use the same uv-managed venv python as the test runner
    python = sys.executable
    proc = subprocess.Popen(
        [python, MCP_SERVER_PATH, "--sse", str(MCP_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    # SSE server is ready when the /sse endpoint accepts connections
    if not _wait_for_server(MCP_PORT):
        proc.terminate()
        raise RuntimeError(f"MCP server did not start within 20s on port {MCP_PORT}")
    print(f"[MCP SERVER] Gemini Advisor SSE server ready on port {MCP_PORT}")
    return proc


async def run_research(topic: str) -> str:
    from app.agent import root_agent

    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name="deep_research_test",
        session_service=session_service,
    )
    session = await session_service.create_session(
        app_name="deep_research_test",
        user_id="test_user",
    )

    message = genai_types.Content(
        role="user",
        parts=[genai_types.Part(text=topic)],
    )

    advisor_calls = 0
    search_calls = 0
    final_report = ""

    print(f"\n{'='*60}")
    print(f"RESEARCH TOPIC: {topic}")
    print(f"{'='*60}\n")

    async for event in runner.run_async(
        user_id="test_user",
        session_id=session.id,
        new_message=message,
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool = part.function_call.name
                    if tool == "gemini_advisor":
                        advisor_calls += 1
                        print(f"[ADVISOR CALL #{advisor_calls}] Consulting Gemini 3.1 Pro advisor...")
                    elif tool == "search_agent":
                        search_calls += 1
                        query = part.function_call.args.get("request", "")[:80]
                        print(f"[SEARCH #{search_calls}] {query}...")
                    elif tool == "load_web_page":
                        url = part.function_call.args.get("url", "")[:80]
                        print(f"[WEB PAGE] Loading: {url}...")

        if event.is_final_response() and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    final_report = part.text

    print(f"\n{'='*60}")
    print(f"STATS: advisor_calls={advisor_calls}, search_calls={search_calls}")
    print(f"{'='*60}\n")
    print("FINAL REPORT:")
    print(final_report)

    return final_report


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else RESEARCH_TOPIC

    # Start a local MCP server only when explicitly requested (local dev).
    # Default: use the deployed Cloud Run service (configured in agent.py).
    use_local = os.environ.get("USE_LOCAL_MCP", "").lower() in ("1", "true", "yes")

    mcp_proc = _start_mcp_server() if use_local else None
    if not use_local:
        print("[MCP SERVER] Using deployed Cloud Run service.")
    try:
        asyncio.run(run_research(topic))
    finally:
        if mcp_proc:
            mcp_proc.terminate()
            mcp_proc.wait()
            print("\n[MCP SERVER] Stopped.")
