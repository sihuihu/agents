"""Podcast generation: TTS synthesis to MP3 file.

Parses a two-host script (HOST_A: / HOST_B: lines), synthesizes each line
with a distinct Neural2 voice, concatenates the MP3 segments, and saves the
result to /tmp/daily_podcast.mp3 so send_digest_email can attach it.

No ffmpeg required — TTS returns MP3 directly, and sequential MP3 chunks
from the same codec/bitrate concatenate cleanly.
"""
import re
from pathlib import Path

from google.cloud import texttospeech

_TTS_CLIENT: texttospeech.TextToSpeechClient | None = None

_VOICE_A = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Neural2-F",  # HOST_A — Alex (female)
)
_VOICE_B = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Neural2-J",  # HOST_B — Maya (male)
)
_AUDIO_CFG = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
)

_LINE_RE = re.compile(r"^(HOST_[AB]):\s*(.+)$", re.MULTILINE)
PODCAST_PATH = Path("/tmp/daily_podcast.mp3")


def _tts() -> texttospeech.TextToSpeechClient:
    global _TTS_CLIENT
    if _TTS_CLIENT is None:
        _TTS_CLIENT = texttospeech.TextToSpeechClient()
    return _TTS_CLIENT


def generate_podcast_audio(script: str) -> dict:
    """Synthesize a two-host podcast script to MP3 and save to /tmp.

    The file is saved at /tmp/daily_podcast.mp3. The email tool picks it
    up automatically and attaches it to the digest email.

    Args:
        script: Lines formatted as "HOST_A: <text>" or "HOST_B: <text>".

    Returns:
        A dict with 'status', 'path', and 'lines_synthesized'.
    """
    lines = _LINE_RE.findall(script)
    if not lines:
        return {"status": "error", "message": "No HOST_A/HOST_B lines found in script."}

    mp3_chunks: list[bytes] = []
    for host, text in lines:
        voice = _VOICE_A if host == "HOST_A" else _VOICE_B
        resp = _tts().synthesize_speech(
            input=texttospeech.SynthesisInput(text=text.strip()),
            voice=voice,
            audio_config=_AUDIO_CFG,
        )
        mp3_chunks.append(resp.audio_content)

    PODCAST_PATH.write_bytes(b"".join(mp3_chunks))

    return {
        "status": "ready",
        "path": str(PODCAST_PATH),
        "lines_synthesized": len(lines),
    }
