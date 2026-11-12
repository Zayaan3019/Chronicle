"""One-time: copy Chronicle's settings from the local .env into SSM Parameter
Store under /chronicle/ (the two credentials as SecureString). Values are
passed to the AWS CLI through a temp file that is deleted immediately, and
are never printed. Works from PowerShell, cmd or bash:

    .venv\\Scripts\\python deploy\\put_params.py        (Windows)
    .venv/bin/python deploy/put_params.py               (Linux/macOS)
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

env_file = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / ".env"
env = dotenv_values(env_file)
SECURE = {"GROQ_API_KEY", "NEON_DATABASE_URL"}
failed = False

for name in ["LLM_PROVIDER", "GROQ_API_KEY", "NEON_DATABASE_URL", "EDGAR_USER_AGENT"]:
    value = env.get(name)
    if not value:
        print(f"{name}: missing from {env_file}")
        failed = failed or name != "LLM_PROVIDER"
        continue
    fd, path = tempfile.mkstemp(suffix=".json")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"Name": f"/chronicle/{name}", "Value": value,
                       "Type": "SecureString" if name in SECURE else "String", "Overwrite": True}, f)
        r = subprocess.run(
            ["aws", "ssm", "put-parameter", "--region", "us-east-1",
             "--cli-input-json", f"file://{path}", "--query", "Version", "--output", "text"],
            capture_output=True, text=True, shell=(os.name == "nt"),
        )
        if r.returncode == 0:
            print(f"{name}: stored (version {r.stdout.strip()})")
        else:
            failed = True
            print(f"{name}: FAILED {r.stderr.strip()[:200]}")
    finally:
        os.remove(path)

sys.exit(1 if failed else 0)
