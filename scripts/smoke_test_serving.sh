#!/usr/bin/env bash
# Smoke-tests the full local stack (Postgres + MLflow + FastAPI + Streamlit)
# via `docker compose up`, then curls /ready, /predict, and /customer/<id>
# against the running API. Invoked as `make smoke-test-serving`.
#
# Assumes a champion has already been promoted (CLAUDE.md's five-step manual
# training cycle: dvc repro calibrate -> register-challenger -> dvc repro
# error_analysis -> review -> register) and customers_raw is populated
# (dvc repro ingest, or the full pipeline) — this script verifies serving
# end to end, it does not seed model/data state itself.
set -euo pipefail

API_URL="${API_URL:-http://localhost:${API_PORT:-8000}}"
READY_TIMEOUT="${READY_TIMEOUT:-180}"
POSTGRES_USER="${POSTGRES_USER:-telco}"
POSTGRES_DB="${POSTGRES_DB:-telco_churn}"

echo "==> docker compose up -d --build"
docker compose up -d --build

echo "==> waiting for ${API_URL}/ready (timeout ${READY_TIMEOUT}s)"
elapsed=0
until ready_body=$(curl -fsS "${API_URL}/ready" 2>/dev/null); do
    elapsed=$((elapsed + 3))
    if [ "${elapsed}" -ge "${READY_TIMEOUT}" ]; then
        echo "FAIL: ${API_URL}/ready did not return 200 within ${READY_TIMEOUT}s"
        docker compose logs fastapi --tail 200
        exit 1
    fi
    sleep 3
done
echo "ready: ${ready_body}"

echo "==> POST ${API_URL}/predict"
predict_payload='{"customerid":"smoke-test-0001","gender":"Female","seniorcitizen":0,"has_partner":"Yes","dependents":"No","tenure":12,"phoneservice":"Yes","multiplelines":"No","internetservice":"DSL","onlinesecurity":"Yes","onlinebackup":"No","deviceprotection":"No","techsupport":"No","streamingtv":"No","streamingmovies":"No","contract_type":"Month-to-month","paperlessbilling":"Yes","paymentmethod":"Electronic check","monthlycharges":55.5,"totalcharges":650.0}'
predict_body=$(curl -fsS -X POST "${API_URL}/predict" -H "Content-Type: application/json" -d "${predict_payload}")
echo "predict: ${predict_body}"

echo "==> GET ${API_URL}/customer/<id>"
customer_id="${CUSTOMER_ID:-}"
if [ -z "${customer_id}" ]; then
    customer_id=$(docker compose exec -T postgres \
        psql -U "${POSTGRES_USER}" -d "${POSTGRES_DB}" -tAc \
        "SELECT customerid FROM customers_raw LIMIT 1;" 2>/dev/null | tr -d '[:space:]')
fi
if [ -z "${customer_id}" ]; then
    echo "SKIP: no customerid found in customers_raw — run \`dvc repro ingest\` first"
else
    customer_body=$(curl -fsS "${API_URL}/customer/${customer_id}")
    echo "customer/${customer_id}: ${customer_body}"
fi

echo "==> smoke test passed"
