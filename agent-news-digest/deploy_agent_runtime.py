"""Deploy agent to Agent Runtime, bypassing uv sync (blocked by corp proxy for litellm).

Strategy:
- Pass requirements.txt explicitly → skips uv export / uv lock
- Monkey-patch subprocess.run to replace "uv run python" with the .venv Python
  so the agent introspection step doesn't re-trigger uv sync
"""
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).parent
os.chdir(PROJECT_DIR)
sys.path.insert(0, str(PROJECT_DIR))

VENV_PYTHON = str(PROJECT_DIR / ".venv" / "bin" / "python3")

# Monkey-patch: replace "uv run python <script>" with ".venv/bin/python3 <script>"
_original_subprocess_run = subprocess.run

def _patched_run(args, *rest, **kwargs):
    if isinstance(args, (list, tuple)) and args[:3] == ["uv", "run", "python"]:
        args = [VENV_PYTHON] + list(args[3:])
    return _original_subprocess_run(args, *rest, **kwargs)

subprocess.run = _patched_run

# Also patch inside the agents-cli module after import
import google.agents.cli.deploy.agent_runtime as _ar
_ar.subprocess.run = _patched_run

from google.agents.cli._project import read_project_config
from google.agents.cli.deploy.agent_runtime import deploy_agent_runtime

cfg = read_project_config(".")
req_lines = sum(1 for _ in open("requirements_agent_runtime.txt"))
print(f"Deploying '{cfg.project_name}' to Agent Runtime (Agent Engine)...")
print(f"  project: may-test-358419 | region: us-central1")
print(f"  requirements_agent_runtime.txt: {req_lines} packages")
print(f"  venv python: {VENV_PYTHON}")

result = deploy_agent_runtime(
    cfg=cfg,
    project="may-test-358419",
    location="us-central1",
    source_packages=("./app", "./requirements_agent_runtime.txt"),
    requirements_file="requirements_agent_runtime.txt",
    set_env_vars="GMAIL_SENDER=may.ntusg@gmail.com,MODEL=gemini-flash-latest",
    set_secrets="GMAIL_APP_PASSWORD=gmail-app-password,GMAIL_RECIPIENT=gmail-recipient",
    no_wait=True,
)

print("\nDeployment started (no-wait). Check status with:")
print("  agents-cli deploy --status --project may-test-358419 --no-confirm-project")
