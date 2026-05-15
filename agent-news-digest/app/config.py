import os
from dataclasses import dataclass


@dataclass
class Config:
    model: str
    gmail_sender: str
    gmail_app_password: str
    gmail_recipient: str
    gcs_project: str
    gcs_bucket: str


config = Config(
    model=os.getenv("MODEL", "gemini-flash-latest"),
    gmail_sender=os.getenv("GMAIL_SENDER", ""),
    gmail_app_password=os.getenv("GMAIL_APP_PASSWORD", ""),
    gmail_recipient=os.getenv("GMAIL_RECIPIENT", ""),
    gcs_project=os.getenv("GCS_PROJECT", "may-test-358419"),
    gcs_bucket=os.getenv("GCS_BUCKET", "may-test-358419-agent-podcasts"),
)
