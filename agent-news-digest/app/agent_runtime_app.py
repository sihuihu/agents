import os

from vertexai.agent_engines.templates.adk import AdkApp

from app.agent import app as adk_app

agent_runtime = AdkApp(app=adk_app)
