import re
import smtplib
import ssl
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.mime.audio import MIMEAudio
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from urllib.parse import urlparse

import requests

from .config import config
from .podcast_tools import PODCAST_PATH

# Matches grounding redirect hrefs — quotes are optional since the LLM sometimes omits them
_REDIRECT_RE = re.compile(
    r'''href=["']?(https://vertexaisearch\.cloud\.google\.com/grounding-api-redirect/[^"'>\s]+)["']?'''
)
_REDIRECT_PREFIX = "https://vertexaisearch.cloud.google.com/grounding-api-redirect/"

# Matches <a href="about:invalid">...</a> for delink cleanup
_INVALID_ANCHOR_RE = re.compile(
    r'<a\s+href="about:invalid"[^>]*>(.*?)</a>', re.DOTALL
)

# Matches any http(s) href for domain verification
_HREF_RE = re.compile(r'''href=["']?(https?://[^"'>\s]+)["']?''')
_ANCHOR_RE = re.compile(r'<a\s+href="([^"]+)"([^>]*)>(.*?)</a>', re.DOTALL)

# Well-known domains we trust without verification (saves time, avoids false positives)
_TRUSTED_DOMAINS = {
    "zdnet.com", "techcrunch.com", "theverge.com", "wired.com", "arstechnica.com",
    "openai.com", "anthropic.com", "google.com", "deepmind.google", "blog.google",
    "arxiv.org", "paperswithcode.com", "huggingface.co",
    "techcommunity.microsoft.com", "microsoft.com", "azure.microsoft.com",
    "infoq.com", "sdtimes.com", "venturebeat.com", "thenewstack.io",
    "langchain.com", "langchain.dev", "llamaindex.ai",
    "substack.com", "medium.com", "dev.to",
    "youtube.com", "youtu.be",
    "github.com", "hbr.org", "forbes.com", "wsj.com", "ft.com",
    "reuters.com", "bloomberg.com", "businessinsider.com", "cnbc.com",
    "letsdatascience.com", "aiagentstore.ai", "pminterviewprepclub.substack.com",
}

_SESSION = requests.Session()
_SESSION.max_redirects = 10


def _is_trusted_domain(url: str) -> bool:
    try:
        netloc = urlparse(url).netloc.lower().lstrip("www.")
        if netloc in _TRUSTED_DOMAINS:
            return True
        # e.g. blog.google → google.com, news.ycombinator.com → ycombinator.com
        parts = netloc.split(".")
        if len(parts) >= 2 and ".".join(parts[-2:]) in _TRUSTED_DOMAINS:
            return True
        return False
    except Exception:
        return False


def _follow_one(url: str) -> str:
    """Follow a URL through redirects and return the final destination."""
    try:
        resp = _SESSION.head(url, allow_redirects=True, timeout=5,
                             headers={"User-Agent": "Mozilla/5.0"})
        return resp.url
    except Exception:
        pass
    try:
        resp = _SESSION.get(url, allow_redirects=True, timeout=5, stream=True,
                            headers={"User-Agent": "Mozilla/5.0"})
        dest = resp.url
        resp.close()
        return dest
    except Exception:
        return url


def _resolve_redirect_urls(html: str) -> str:
    """Resolve Vertex AI grounding redirect URLs to actual article URLs.

    Follows up to 3 hops because the LLM sometimes truncates redirect tokens,
    causing the first hop to return another redirect URL before the actual article.
    Unresolvable redirects are replaced with about:invalid for downstream delink.
    """
    seen: dict[str, str] = {}

    def _resolve(match: re.Match) -> str:
        redirect_url = match.group(1)
        if redirect_url in seen:
            return f'href="{seen[redirect_url]}"'

        dest = redirect_url
        for _ in range(3):
            nxt = _follow_one(dest)
            if nxt == dest:
                break
            dest = nxt
            if not dest.startswith(_REDIRECT_PREFIX):
                break

        if dest.startswith(_REDIRECT_PREFIX):
            print(f"[redirect-unresolved] {redirect_url[:80]}", flush=True)
            dest = "about:invalid"
        else:
            print(f"[redirect] {redirect_url[:70]} -> {dest[:100]}", flush=True)

        seen[redirect_url] = dest
        return f'href="{dest}"'

    resolved = _REDIRECT_RE.sub(_resolve, html)
    resolved = _INVALID_ANCHOR_RE.sub(lambda m: f"<strong>{m.group(1)}</strong>", resolved)

    # Delink any redirect URLs still present (e.g. truncated by the LLM and missed by the regex)
    resolved = re.sub(
        r'<a\b[^>]*\bhref=["\']?https://vertexaisearch\.cloud\.google\.com/[^"\'>\s]*["\']?[^>]*>(.*?)</a>',
        lambda m: f"<strong>{m.group(1)}</strong>",
        resolved,
        flags=re.DOTALL,
    )
    return resolved


def _check_domain(url: str) -> tuple[str, bool]:
    """Check whether the domain of a URL is reachable (not the specific article path)."""
    if not url.startswith("http"):
        return url, False
    if _is_trusted_domain(url):
        return url, True
    try:
        parsed = urlparse(url)
        root = f"{parsed.scheme}://{parsed.netloc}/"
        resp = _SESSION.head(root, allow_redirects=True, timeout=4,
                             headers={"User-Agent": "Mozilla/5.0"})
        # Accept anything below 500: 2xx, 3xx, and even 4xx mean the domain exists
        return url, resp.status_code < 500
    except Exception:
        return url, False


def _delink_fake_domains(html: str) -> str:
    """Delink any <a href="URL"> where the domain cannot be reached at all."""
    urls = list(set(_HREF_RE.findall(html)))
    if not urls:
        return html

    reachable: dict[str, bool] = {}
    with ThreadPoolExecutor(max_workers=10) as ex:
        futures = {ex.submit(_check_domain, u): u for u in urls}
        for future in as_completed(futures):
            url, ok = future.result()
            reachable[url] = ok
            if not ok:
                print(f"[fake-domain] {url}", flush=True)

    def _maybe_delink(match: re.Match) -> str:
        url, attrs, text = match.group(1), match.group(2), match.group(3)
        if reachable.get(url, True):
            return match.group(0)
        return f"<strong>{text}</strong>"

    return _ANCHOR_RE.sub(_maybe_delink, html)


def send_digest_email(subject: str, html_body: str) -> dict:
    """Send the AI agent news digest as an HTML email via Gmail SMTP.

    Resolves Vertex AI grounding redirect URLs to permanent source URLs,
    delinks anchors whose domains are unreachable, then attaches
    /tmp/daily_podcast.mp3 if present.

    Args:
        subject: The email subject line.
        html_body: The full HTML content of the digest.

    Returns:
        A dict with 'status', 'recipients', 'links_resolved', and 'podcast_attached'.
    """
    missing = [
        k for k, v in {
            "GMAIL_SENDER": config.gmail_sender,
            "GMAIL_APP_PASSWORD": config.gmail_app_password,
            "GMAIL_RECIPIENT": config.gmail_recipient,
        }.items() if not v
    ]
    if missing:
        return {
            "status": "error",
            "message": f"Missing env vars: {', '.join(missing)}. See .env.example.",
        }

    redirect_count = len(_REDIRECT_RE.findall(html_body))
    resolved_html = _resolve_redirect_urls(html_body)
    resolved_html = _delink_fake_domains(resolved_html)

    # Log final link list for debugging
    final_hrefs = re.findall(r'''href=["']?(https?://[^"'>\s]+)["']?''', resolved_html)
    print(f"[email-links] {len(final_hrefs)} links: {final_hrefs[:10]}", flush=True)

    recipients = [r.strip() for r in config.gmail_recipient.split(",") if r.strip()]

    msg = MIMEMultipart("mixed")
    msg["Subject"] = subject
    msg["From"] = config.gmail_sender
    msg["To"] = ", ".join(recipients)
    msg.attach(MIMEText(resolved_html, "html"))

    podcast_attached = False
    if PODCAST_PATH.exists():
        audio_data = PODCAST_PATH.read_bytes()
        attachment = MIMEAudio(audio_data, "mpeg")
        attachment.add_header(
            "Content-Disposition",
            "attachment",
            filename=f"agent-digest-podcast-{subject[-10:]}.mp3",
        )
        msg.attach(attachment)
        podcast_attached = True

    try:
        ctx = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
            server.login(config.gmail_sender, config.gmail_app_password)
            server.sendmail(config.gmail_sender, recipients, msg.as_string())
        return {
            "status": "sent",
            "recipients": recipients,
            "links_resolved": redirect_count,
            "podcast_attached": podcast_attached,
        }
    except smtplib.SMTPAuthenticationError:
        return {
            "status": "error",
            "message": "Gmail auth failed — check GMAIL_APP_PASSWORD.",
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}
