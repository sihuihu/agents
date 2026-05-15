"""Integration test: run the deep research agent on a sample topic.

Usage:
    cd deep-research-agent
    GEMINI_API_KEY=<key> uv run python tests/test_research.py
"""

import asyncio
import os
import sys

# Make sure the project root is on the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types as genai_types

RESEARCH_TOPIC = (
    "What are the most promising recent breakthroughs in solid-state battery technology "
    "for electric vehicles, and what are the main remaining challenges to commercialization?"
)


async def run_research(topic: str) -> str:
    # Import here so env vars are set first
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
        # Track tool usage
        if event.content and event.content.parts:
            for part in event.content.parts:
                if hasattr(part, "function_call") and part.function_call:
                    tool = part.function_call.name
                    if tool == "gemini_advisor":
                        advisor_calls += 1
                        print(f"[ADVISOR CALL #{advisor_calls}] Consulting Gemini Pro advisor...")
                    elif tool == "search_agent":
                        search_calls += 1
                        query = part.function_call.args.get("request", "")[:80]
                        print(f"[SEARCH #{search_calls}] {query}...")
                    elif tool == "load_web_page":
                        url = part.function_call.args.get("url", "")[:80]
                        print(f"[WEB PAGE] Loading: {url}...")

        # Capture final response
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
    asyncio.run(run_research(topic))
