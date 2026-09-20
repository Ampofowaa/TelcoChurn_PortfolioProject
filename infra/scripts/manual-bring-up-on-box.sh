#!/usr/bin/env bash
set -euo pipefail

# Phase E of the Group 9 manual bring-up (see manual-bring-up-local.sh for
# Phases A-D, which must run first). This is a REFERENCE to paste into an
# interactive SSM session on the box itself - not something you scp/execute
# remotely, since the whole point is watching it happen and being able to
# poke around if something's wrong.
#
#   aws ssm start-session --target <instance-id> --region us-east-1
#
# Then paste the commands below (or this whole file) into that session.

REGION="us-east-1"
ACCOUNT_ID="742306314937"
ECR_CRED_HELPER_VERSION="0.12.0"

echo "=== One-time (per box): Amazon ECR credential helper. IAM permission to"
echo "    pull isn't the same as being logged in - without this, docker/"
echo "    docker compose has no ECR credentials at all. Baked into"
echo "    user-data.sh.tpl now for any FUTURE box, but that only runs at a"
echo "    box's first boot - this existing box needs it applied by hand once."
echo "    Skip this block if you've already run it on this box. ==="
# Every line below is a single, self-contained command - no trailing `\`
# continuation, no wrapping `(...)` subshell, no heredoc. Pasting a multi-line
# construct into an interactive SSM session's PTY is fragile (a dropped
# character mid-paste leaves a subshell or heredoc open, and everything
# typed afterwards - including this comment block, if pasted whole - gets
# silently swallowed as "more input" until the shell finally chokes on
# something unbalanced). Hit this for real 2026-09-19: the `(cd ... && ...)`
# form below lost its closing `)` mid-paste and ate the next six lines.
sudo curl -fsSL "https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/$ECR_CRED_HELPER_VERSION/linux-amd64/docker-credential-ecr-login" -o /usr/local/bin/docker-credential-ecr-login
sudo curl -fsSL "https://amazon-ecr-credential-helper-releases.s3.us-east-2.amazonaws.com/$ECR_CRED_HELPER_VERSION/linux-amd64/docker-credential-ecr-login.sha256" -o /tmp/docker-credential-ecr-login.sha256
cd /usr/local/bin && sudo sha256sum -c /tmp/docker-credential-ecr-login.sha256
cd /opt/telco-churn
sudo rm -f /tmp/docker-credential-ecr-login.sha256
sudo chmod +x /usr/local/bin/docker-credential-ecr-login
sudo mkdir -p /root/.docker
echo "{\"credHelpers\":{\"${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com\":\"ecr-login\"}}" | sudo tee /root/.docker/config.json >/dev/null
echo "Done - this also replaces the plaintext basic-auth token the earlier"
echo "manual 'docker login' left in /root/.docker/config.json."

echo "=== Sync the latest deploy-config from S3 (compose.prod.yml, Caddyfile,"
echo "    docker/mlflow/Dockerfile) - the box only pulled this once, at first"
echo "    boot; nothing has re-synced it since. ==="
S3_BUCKET="$(sudo aws ssm get-parameter --name /telco-churn/s3-bucket --query Parameter.Value --output text --region "$REGION")"
sudo aws s3 cp "s3://${S3_BUCKET}/deploy-config/" /opt/telco-churn/ --recursive --region "$REGION"

echo "=== Render a fresh .env from the current SSM params (real api-key,"
echo "    real api-image/ui-image tags, etc.) ==="
sudo /opt/telco-churn/refresh-env.sh

cd /opt/telco-churn

echo "=== Run the one-off Alembic migration against RDS (needs the real"
echo "    api-image from Phase D to already be pullable) ==="
sudo docker compose -f compose.prod.yml run --rm api alembic upgrade head

echo "=== Bring the whole stack up (systemd's own unit does this same"
echo "    sequence on every future boot; this is the first-ever start) ==="
sudo systemctl start telco-churn.service

echo "=== Sanity check from inside the box ==="
sudo docker compose -f compose.prod.yml ps
sudo systemctl status telco-churn.service --no-pager

echo
echo "If anything is unhealthy: sudo docker compose -f compose.prod.yml logs <service> --tail 100"
echo "Now exit this session and run Phase F's verification curls from your own machine."
