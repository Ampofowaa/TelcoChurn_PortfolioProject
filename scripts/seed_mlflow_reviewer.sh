#!/usr/bin/env bash
# Runs ON the EC2 box (shipped by scripts/ssm_run.sh) after deploy_on_box.sh.
# Ensures the MLflow UI's view-only `reviewer` account exists with its
# published password: created if missing, password reset if it was changed.
# Idempotent, so a wiped mlflow_auth database heals itself on the next deploy.
#
# Talks to MLflow inside its own container on localhost, never through Caddy:
# Caddy blocks every user-management write from outside, which is what stops
# a stranger with the public reviewer password from changing it.
set -euo pipefail

cd /opt/telco-churn

# cd.yml's rollback runs this checkout's scripts against the previous commit's
# compose.prod.yml. If that predates MLflow's login there are no user
# endpoints to call, so skip rather than fail the rollback.
if ! grep -q -- '--app-name basic-auth' compose.prod.yml; then
    echo "==> MLflow basic-auth not enabled in compose.prod.yml - nothing to seed"
    exit 0
fi

echo "==> Ensuring the MLflow reviewer account"
docker compose -f compose.prod.yml exec -T mlflow python - <<'PY'
import base64
import json
import os
import time
import urllib.error
import urllib.request

BASE = "http://localhost:5000/mlflow"
REVIEWER = "reviewer"
# Published in README.md on purpose: default_permission is READ and Caddy
# blocks every create and user-management write, so the account is view-only.
REVIEWER_PASSWORD = "telco-reviewer-2026"  # pragma: allowlist secret
ADMIN_AUTH = "Basic " + base64.b64encode(
    f"admin:{os.environ['MLFLOW_ADMIN_PASSWORD']}".encode()
).decode()


def call(method: str, path: str, body: dict | None = None) -> int:
    request = urllib.request.Request(
        BASE + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Authorization": ADMIN_AUTH, "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status
    except urllib.error.HTTPError as error:
        return error.code


deadline = time.monotonic() + 240
while True:
    try:
        urllib.request.urlopen(BASE + "/health", timeout=5)
        break
    except OSError:
        if time.monotonic() > deadline:
            raise SystemExit("mlflow did not become healthy within 240s")
        time.sleep(5)

credentials = {"username": REVIEWER, "password": REVIEWER_PASSWORD}
status = call("GET", f"/api/2.0/mlflow/users/get?username={REVIEWER}")
if status == 404:
    status = call("POST", "/api/2.0/mlflow/users/create", credentials)
    action = "created"
elif status == 200:
    status = call("PATCH", "/api/2.0/mlflow/users/update-password", credentials)
    action = "password reset"
else:
    raise SystemExit(f"looking up {REVIEWER} returned HTTP {status} (admin login rejected?)")

if status != 200:
    raise SystemExit(f"{REVIEWER} not {action}: HTTP {status}")
print(f"{REVIEWER}: {action}")
PY
