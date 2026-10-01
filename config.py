"""Loads secrets and settings from the git-ignored .env file next to this module.

Import this before anything that reads environment variables. Variables already set in
the real environment win over .env, so a deployment can override the file.
"""

from pathlib import Path

from dotenv import load_dotenv

ENV_PATH = Path(__file__).parent / ".env"

load_dotenv(ENV_PATH, override=False)
