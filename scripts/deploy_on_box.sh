#!/usr/bin/env bash
# Runs ON the EC2 box (shipped by scripts/ssm_run.sh via ssm:SendCommand).
# Converges the box onto whatever /telco-churn/{api,ui}-image currently say.
# Idempotent by design: cd.yml's deploy step and its rollback step run this
# exact script, differing only in which SHA the SSM params hold.
#
# `docker image prune` is deliberately absent — it runs only after the smoke
# test passes (scripts/prune_on_box.sh), so a failed deploy never discards
# anything the rollback might want locally.
set -euo pipefail

REGION="us-east-1"
cd /opt/telco-churn

bucket=$(aws ssm get-parameter --name /telco-churn/s3-bucket \
    --query Parameter.Value --output text --region "${REGION}")
aws s3 sync "s3://${bucket}/deploy-config/" /opt/telco-churn/ --region "${REGION}"
chmod +x refresh-env.sh
./refresh-env.sh

# Free space before pulling, not only after a passing smoke test: the root volume
# is 19 GB and each api+ui release is ~2.5 GB, so a pull onto a box that still
# holds old releases fills the disk and the SSM agent dies with "no space left
# on device". `-a` removes every image no container uses; the running release
# (the rollback target) is in use and survives, and anything older is
# re-pullable from ECR, which keeps the last 5 tags.
docker image prune -af
docker compose -f compose.prod.yml pull
# Forward-only: a rollback re-runs this against the old image, and alembic
# is already at head, so it is a no-op there — it never downgrades.
docker compose -f compose.prod.yml run --rm -T api alembic upgrade head < /dev/null
# --build: mlflow is built on the box, not pulled from ECR.
docker compose -f compose.prod.yml up -d --build
