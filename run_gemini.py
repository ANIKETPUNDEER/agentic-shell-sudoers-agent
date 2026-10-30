"""
Run Agent 25 with Google Gemini instead of Anthropic.

    pip install google-genai
    export GEMINI_API_KEY=...                # free key from Google AI Studio
    sudo -E python3 run_gemini.py "Let alice restart nginx without a password, but nothing else"

Optional: SUDOERS_AGENT_MODEL=<gemini model name>   SUDOERS_AGENT_DIR=/tmp/sudoers_try (safe dry run)
The agent loop, approval prompt and every safety check are the same code as the Anthropic path.
"""
import os
import sys

from gemini_client import GeminiClient
from sudoers_agent import SudoersAgent

if __name__ == "__main__":
    client = GeminiClient()
    agent = SudoersAgent(client=client, model=os.environ.get("SUDOERS_AGENT_MODEL") or client.default_model)
    result = agent.handle(" ".join(sys.argv[1:]) or input("sudoers> "))
    print("\n" + result.message)
    if result.error:
        print(f"[error: {result.error}]")
    print(f"[changed sudoers: {result.changed}]")
