# ruff: noqa
"""Daily AI agent news digest — four-agent sequential pipeline.

On Friday the pipeline automatically switches to a weekly-highlights mode:
longer search window, more items per section, a Top Stories block in the
email, and a longer podcast episode.

Vertex AI does not allow mixing grounding tools (google_search) with regular
function tools in the same agent, so search and email/audio are separate agents.
"""
import datetime
import os

import google.auth
from google.adk.agents import Agent, SequentialAgent
from google.adk.agents.callback_context import CallbackContext
from google.adk.apps import App
from google.adk.models import Gemini
from google.adk.tools.google_search_tool import google_search
from google.genai import types

from .config import config
from .podcast_tools import generate_podcast_audio
from .tools import send_digest_email

_, project_id = google.auth.default()
os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
os.environ["GOOGLE_CLOUD_LOCATION"] = "global"
os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

# ---------------------------------------------------------------------------
# Search agent
# ---------------------------------------------------------------------------

_SEARCH_INSTRUCTION = """You are a research assistant gathering AI-agent news.

Today's date : {current_date}
Day of week  : {day_of_week}
Mode         : {digest_mode}   (DAILY = past 24–48 h | WEEKLY = past 7 days)

━━━ QUERIES TO RUN ━━━
Run ALL of the following google_search queries.

For DAILY mode  → keep the {items_per_section} most relevant results per section.
For WEEKLY mode → keep the {items_per_section} most relevant results per section,
                  preferring items that were widely covered or had the most impact.

1. Anthropic Claude agents announcement blog {current_date}
2. OpenAI agents SDK efficiency announcement {current_date}
3. Google DeepMind AI agents research {current_date}
4. AI agent inference cost reduction efficiency 2026
5. LLM agent end-to-end latency optimization 2026
6. context window management compression AI agents 2026
7. multi-agent orchestration new research 2026

{weekly_extra_queries}

━━━ OUTPUT FORMAT ━━━
For EACH result output exactly:

---ITEM---
SECTION: <Anthropic | OpenAI | Google/DeepMind | Efficiency & Cost | Latency & Context>
TITLE: <title>
URL: <url>
SOURCE: <publication>
DATE: <date or "recent">
SUMMARY: <1-2 sentences on why this is relevant>
IMPACT: <low|medium|high>   ← weekly mode: used for Top Stories ranking
---END---

CRITICAL URL RULES — failure to follow these will break every link in the email:
- Use the EXACT URL that appears in the search result for this article. Copy it verbatim without any modification.
- If the search result only has a grounding redirect URL (https://vertexaisearch.cloud.google.com/...), use that. The redirect will be resolved at send time.
- NEVER construct, guess, shorten, or paraphrase a URL. If you cannot identify an exact URL for this article, omit the item entirely.
- NEVER use placeholder or made-up domains like "technews.com/article", "aitoroi.com", "etihedte.ch", or similar.
- Acceptable URL sources: the URL field of a search result, a grounding redirect URL, or a URL you can directly confirm in the search snippet.

Only include items genuinely about AI agents. Omit exact duplicates.
Output nothing else — just the structured blocks."""

_WEEKLY_EXTRA_QUERIES = """\
Additional queries for weekly mode — run these too:
8.  most important AI agent news this week 2026
9.  AI agent breakthrough research paper 2026
10. agent framework release announcement this week 2026"""

# ---------------------------------------------------------------------------
# Podcast script agent
# ---------------------------------------------------------------------------

_PODCAST_SCRIPT_INSTRUCTION = """You are a podcast scriptwriter. You receive structured
AI agent news items (---ITEM--- blocks) from earlier in the conversation.

Today's date : {current_date}
Day of week  : {day_of_week}
Mode         : {digest_mode}

Hosts:
  HOST_A — Alex (female), enthusiastic and big-picture focused
  HOST_B — Maya (male), technical and detail-oriented

Format EVERY line as exactly:
  HOST_A: <spoken text>
  HOST_B: <spoken text>

━━━ DAILY mode (Mon–Thu) ━━━
- Open: brief welcome, today's date
- Cover the 5–8 most interesting items; hosts react, ask follow-up questions
- Natural speech: contractions, light humour, analogies
- Target: 20–30 exchanges (~4 min audio)
- Close: quick sign-off

━━━ WEEKLY mode (Friday) ━━━
- Open: "Week in Review" framing, week ending {current_date}
- Start with a "Top 3 stories of the week" rapid-fire round
- Then deep-dive on 2–3 of the most impactful items (use IMPACT: high markers)
- Recap the key themes across all sections
- Target: 40–55 exchanges (~8–10 min audio)
- Close: weekend sign-off ("have a great weekend")

Output ONLY the HOST_A/HOST_B lines. No section headers, no markdown."""

# ---------------------------------------------------------------------------
# Email agent
# ---------------------------------------------------------------------------

_EMAIL_INSTRUCTION = """You are a digest editor. Build and send the HTML digest email.

Today's date : {current_date}
Day of week  : {day_of_week}
Mode         : {digest_mode}

━━━ STEP 1 — PODCAST BANNER ━━━
If generate_podcast_audio returned status "ready" earlier, include:
<div class="podcast-banner">
  🎙️ <span>{podcast_label} — open the attached .mp3 to listen</span>
</div>

━━━ STEP 2 — WEEKLY TOP STORIES (Friday only) ━━━
If mode is WEEKLY, add a Top Stories section BEFORE the regular sections.
Pick the 3–5 items marked IMPACT: high (or your top picks) and render each as
a .top-item card with a slightly larger title and a 2–3 sentence summary.

━━━ STEP 3 — SECTION ITEMS ━━━
Parse all ---ITEM--- blocks. Group by SECTION:
  A. 🔬 Anthropic
  B. 🤖 OpenAI
  C. 🌐 Google / DeepMind
  D. ⚡ Agent Efficiency & Cost
  E. ⏱️ Latency & Context Management

For WEEKLY mode, add a short section intro (1 sentence) summarising the week's theme
for that section before the item list.

If a section has no items: <p class="empty">No updates this {period}.</p>

━━━ STEP 4 — HTML TEMPLATE ━━━

<!DOCTYPE html>
<html>
<head><meta charset="UTF-8">
<style>
  body{{font-family:-apple-system,Arial,sans-serif;max-width:680px;margin:0 auto;padding:24px;color:#222;background:#fff}}
  h1{{font-size:22px;color:#1a1a2e;border-bottom:2px solid #e0e0e0;padding-bottom:12px;margin-bottom:24px}}
  .podcast-banner{{background:#1a1a2e;color:#fff;padding:14px 18px;border-radius:8px;margin-bottom:24px}}
  .top-stories-header{{font-size:18px;font-weight:700;color:#b45309;margin:0 0 12px}}
  .top-item{{margin:12px 0;padding:14px 16px;background:#fffbeb;border-left:4px solid #f59e0b;border-radius:0 8px 8px 0}}
  .top-item a{{color:#92400e;text-decoration:none;font-weight:700;font-size:15px}}
  .top-item .summary{{font-size:13px;line-height:1.6;color:#555;margin-top:6px}}
  .section-intro{{font-size:13px;color:#666;font-style:italic;margin-bottom:10px}}
  h2{{font-size:16px;color:#16213e;margin:28px 0 12px;padding-bottom:6px;border-bottom:1px solid #eee}}
  .item{{margin:10px 0;padding:12px 14px;background:#f8f9fa;border-left:3px solid #4a90e2;border-radius:0 6px 6px 0}}
  .item a{{color:#1a73e8;text-decoration:none;font-weight:600;font-size:14px}}
  .meta{{font-size:11px;color:#777;margin:3px 0 6px}}
  .summary{{font-size:13px;line-height:1.6;color:#444}}
  .empty{{font-size:13px;color:#999;font-style:italic;padding:8px 0}}
  .footer{{margin-top:32px;padding-top:12px;border-top:1px solid #eee;font-size:11px;color:#aaa}}
</style>
</head>
<body>
<h1>{email_title}</h1>
[podcast banner if applicable]
[Top Stories block if WEEKLY]
[five section blocks]
<div class="footer">Generated by agent-news-digest · {current_date} · Powered by Google ADK + Gemini</div>
</body>
</html>

Regular item card:
<div class="item">
  <a href="URL">Title</a>
  <div class="meta">Source · Date</div>
  <div class="summary">Summary.</div>
</div>

Top Stories item card (weekly only):
<div class="top-item">
  <a href="URL">Title</a>
  <div class="meta">Source · Date</div>
  <div class="summary">Longer 2-3 sentence summary.</div>
</div>

━━━ CRITICAL URL RULES ━━━
- Copy each URL from the ITEM block VERBATIM into the href attribute. Never shorten, truncate, paraphrase, or reconstruct any URL.
- If the URL is a grounding redirect (https://vertexaisearch.cloud.google.com/...), copy the entire URL as-is — it will be resolved at send time.
- Never construct or guess a URL from a title or source name. If an item has no URL, render the title as plain text with no anchor.

━━━ STEP 5 — SEND ━━━
Call send_digest_email with:
  subject: "{email_subject}"
  html_body: <the complete HTML>

Output: "Sent {digest_mode} digest for {current_date}. Top Stories (N — weekly only),
Anthropic (N), OpenAI (N), Google/DeepMind (N), Efficiency (N), Latency/Context (N).
Podcast: <included|omitted>. Email status: <status>." """

# ---------------------------------------------------------------------------
# Shared state injector
# ---------------------------------------------------------------------------


async def inject_context(callback_context: CallbackContext) -> None:
    today = datetime.date.today()
    is_friday = today.weekday() == 4  # Monday=0 … Friday=4

    if is_friday:
        week_start = (today - datetime.timedelta(days=4)).isoformat()
        callback_context.state.update({
            "current_date": today.isoformat(),
            "day_of_week": "Friday",
            "digest_mode": "WEEKLY",
            "time_window": "the past 7 days",
            "items_per_section": "5–8",
            "podcast_label": "This week's podcast overview",
            "period": "week",
            "weekly_extra_queries": _WEEKLY_EXTRA_QUERIES,
            "email_title": f"🤖 AI Agent Weekly Highlights — Week of {week_start}",
            "email_subject": f"AI Agent Weekly Highlights — Week of {week_start}",
        })
    else:
        day_name = today.strftime("%A")
        callback_context.state.update({
            "current_date": today.isoformat(),
            "day_of_week": day_name,
            "digest_mode": "DAILY",
            "time_window": "the past 24–48 hours",
            "items_per_section": "3–5",
            "podcast_label": "Today's podcast overview",
            "period": "day",
            "weekly_extra_queries": "",
            "email_title": f"🤖 AI Agent Daily Digest — {today.isoformat()}",
            "email_subject": f"AI Agent Daily Digest — {today.isoformat()}",
        })

# ---------------------------------------------------------------------------
# Agent definitions
# ---------------------------------------------------------------------------

_gemini = Gemini(
    model=config.model,
    retry_options=types.HttpRetryOptions(attempts=3),
)

news_search_agent = Agent(
    name="news_search_agent",
    model=_gemini,
    instruction=_SEARCH_INSTRUCTION,
    tools=[google_search],
)

podcast_script_agent = Agent(
    name="podcast_script_agent",
    model=_gemini,
    instruction=_PODCAST_SCRIPT_INSTRUCTION,
    tools=[],
)

podcast_audio_agent = Agent(
    name="podcast_audio_agent",
    model=_gemini,
    instruction=(
        "You receive a podcast script (HOST_A/HOST_B lines) from earlier in the "
        "conversation. Call generate_podcast_audio with the full script text. "
        "Output only the tool result (status and lines_synthesized)."
    ),
    tools=[generate_podcast_audio],
)

digest_email_agent = Agent(
    name="digest_email_agent",
    model=_gemini,
    instruction=_EMAIL_INSTRUCTION,
    tools=[send_digest_email],
)

root_agent = SequentialAgent(
    name="agent_news_digest",
    sub_agents=[
        news_search_agent,
        podcast_script_agent,
        podcast_audio_agent,
        digest_email_agent,
    ],
    before_agent_callback=inject_context,
)

app = App(
    root_agent=root_agent,
    name="app",
)
