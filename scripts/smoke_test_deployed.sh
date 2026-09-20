#!/usr/bin/env bash
# Runs ON the EC2 box (shipped by scripts/ssm_run.sh) and smoke-tests the
# stack through Caddy, exactly as an outside caller reaches it.
#
# On the box rather than on the CI runner so the API key never leaves the box:
# it reads API_KEY/DOMAIN from the same .env refresh-env.sh renders, and
# --resolve pins the domain to loopback so the request still hits Caddy's
# real TLS cert without depending on hairpin routing via the Elastic IP.
#
# (scripts/smoke_test_serving.sh is the separate local docker-compose check.)
set -euo pipefail

set -a
# shellcheck disable=SC1091
source /opt/telco-churn/.env
set +a

BASE_URL="https://${DOMAIN}"
RESOLVE=(--resolve "${DOMAIN}:443:127.0.0.1")
READY_TIMEOUT="${READY_TIMEOUT:-240}"

fail() {
    echo "FAIL: $*"
    docker compose -f /opt/telco-churn/compose.prod.yml logs --tail 100 api || true
    exit 1
}

status() { curl -s -o /dev/null -w '%{http_code}' "${RESOLVE[@]}" "$@"; }

echo "==> /health"
[ "$(status "${BASE_URL}/health")" = "200" ] || fail "/health did not return 200"

# /ready goes 503 -> 200 once the champion has loaded, so poll instead of
# asserting once.
echo "==> /ready (timeout ${READY_TIMEOUT}s)"
elapsed=0
until [ "$(status "${BASE_URL}/ready")" = "200" ]; do
    elapsed=$((elapsed + 5))
    [ "${elapsed}" -lt "${READY_TIMEOUT}" ] || fail "/ready not 200 within ${READY_TIMEOUT}s"
    sleep 5
done

payload='{"customerid":"smoke-test-0001","gender":"Female","seniorcitizen":0,"has_partner":"Yes","dependents":"No","tenure":12,"phoneservice":"Yes","multiplelines":"No","internetservice":"DSL","onlinesecurity":"Yes","onlinebackup":"No","deviceprotection":"No","techsupport":"No","streamingtv":"No","streamingmovies":"No","contract_type":"Month-to-month","paperlessbilling":"Yes","paymentmethod":"Electronic check","monthlycharges":55.5,"totalcharges":650.0,"include_explanation":false}'

echo "==> /predict without a key must be rejected"
code=$(status -X POST "${BASE_URL}/predict" -H "Content-Type: application/json" -d "${payload}")
[ "${code}" = "401" ] || fail "/predict without key returned ${code}, expected 401"

echo "==> /predict with a key returns a probability"
body=$(curl -fsS "${RESOLVE[@]}" -X POST "${BASE_URL}/predict" \
    -H "Content-Type: application/json" -H "X-API-Key: ${API_KEY}" -d "${payload}") \
    || fail "/predict with key errored"
echo "${body}" | jq -e '.probability | (type == "number" and . >= 0 and . <= 1)' >/dev/null \
    || fail "/predict returned no valid probability: ${body}"

echo "==> UI root"
[ "$(status "${BASE_URL}/")" = "200" ] || fail "UI root did not return 200"

echo "==> smoke test passed"
