# DESIGN_SPEC.md — AI Agent Daily Digest

## Overview

An ambient ADK agent that runs on a daily schedule (Cloud Scheduler → Cloud Run) to research
the latest news, blog posts, and research papers related to AI agents. It focuses on
Anthropic, OpenAI, and Google/DeepMind announcements plus technical topics around agent
efficiency, cost control, end-to-end latency, and context management. Results are compiled
into a formatted HTML digest and sent to a recipient email via Gmail SMTP.

## Example Use Cases

- Morning trigger (e.g., 7am) kicks the agent. It runs searches across official blogs and
  arXiv, compiles 5–15 items per topic section, and delivers a structured email by 7:05am.
- User can also trigger it manually: `agents-cli run "run today's agent news digest"`

## Tools Required

| Tool | Source | Auth |
|------|--------|------|
| `google_search` | ADK built-in (`google.adk.tools.google_search_tool`) | Vertex AI grounding — no extra key |
| `send_digest_email` | Custom (`app/tools.py`) | Gmail App Password (`GMAIL_APP_PASSWORD` env var) |

## Key Topics Tracked

- **Labs**: Anthropic (blog, research), OpenAI (blog), Google DeepMind / Google Research
- **Technical**: agent efficiency & throughput, inference cost reduction, end-to-end latency,
  context window management, multi-agent orchestration

## Constraints & Safety Rules

- Agent must NOT use any credentials beyond what is configured via env vars.
- If email credentials are missing, log the digest to stdout rather than silently failing.
- No write access to files, databases, or external services beyond Gmail SMTP.

## Success Criteria

- At least 3 news items found per run (when news exists)
- Digest email received within 5 minutes of trigger
- Sections clearly separated; each item has title, source, date, and summary
- No duplicate items across sections

## Reference Samples

- `ambient-expense-agent` — ambient/scheduled agent pattern with FastAPI trigger endpoint
