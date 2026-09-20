# Source this before any terraform command — Terraform's AWS provider can't
# prompt for MFA itself ("AssumeRoleTokenProvider session option not set"),
# so we assume terraform-admin once via the CLI (which can prompt) and export
# the resulting short-lived session credentials instead.
#
# Usage: source infra/scripts/assume-admin.sh
# (must be sourced, not executed, so the exports land in your current shell)

read -p "MFA code for telco-churn-admin: " MFA_CODE

CREDS=$(aws sts assume-role \
  --profile telco-churn-base \
  --role-arn arn:aws:iam::742306314937:role/terraform-admin \
  --serial-number arn:aws:iam::742306314937:mfa/telco-churn-admin \
  --token-code "$MFA_CODE" \
  --role-session-name terraform-cli \
  --query 'Credentials.[AccessKeyId,SecretAccessKey,SessionToken]' \
  --output text)

if [ -z "$CREDS" ]; then
  echo "assume-role failed — check the MFA code and try again"
else
  export AWS_ACCESS_KEY_ID=$(echo "$CREDS" | cut -f1)
  export AWS_SECRET_ACCESS_KEY=$(echo "$CREDS" | cut -f2)
  export AWS_SESSION_TOKEN=$(echo "$CREDS" | cut -f3)
  export AWS_REGION=us-east-1
  echo "Session active as: $(aws sts get-caller-identity --query Arn --output text)"
fi
