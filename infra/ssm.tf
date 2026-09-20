# Group 7 — SSM Parameter Store entries.
#
# Four ownership models across nine parameters, not two — each mismatch here
# is a real Terraform-fights-reality bug, not a style choice:
#   1. Human-generated  (api-key, grafana-*)      -> ignore_changes on value
#   2. CD-mutated       (api-image, ui-image)     -> ignore_changes on value
#   3. Terraform-derived (rds-url-*, s3-bucket)   -> no ignore_changes, self-corrects
#   4. Static, human-picked literal (domain)      -> plain resource, either works
#
# All nine share the /telco-churn/* path, already covered by iam.tf's
# SsmParams/SsmParamsKms statements on the EC2 instance role — no new IAM
# grant needed here. SecureString values use the default alias/aws/ssm KMS
# key (what kms_ssm_key_arn already grants kms:Decrypt on) — no custom
# key_id. Tier defaults to Standard (free) everywhere; never set Advanced.

# --- Human-generated, ignore_changes required ---------------------------
#
# Terraform creates each with a placeholder; a human overwrites the real
# value afterwards via `aws ssm put-parameter --overwrite`. Without
# ignore_changes, the next unrelated `terraform apply` reverts that
# hand-set value back to the placeholder.

resource "aws_ssm_parameter" "api_key" {
  name        = "/telco-churn/api-key"
  description = "API key for the three data routes (/predict, /predict/batch, /customer/{id}); set by hand with openssl rand -hex 32"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-api-key"
  }
}

resource "aws_ssm_parameter" "grafana_endpoint" {
  name        = "/telco-churn/grafana-endpoint"
  description = "Grafana Cloud remote_write endpoint; set by hand from the Grafana Cloud console"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-grafana-endpoint"
  }
}

resource "aws_ssm_parameter" "grafana_api_key" {
  name        = "/telco-churn/grafana-api-key"
  description = "Grafana Cloud API key for Alloy's remote_write; set by hand from the Grafana Cloud console"
  type        = "SecureString"
  value       = "REPLACE_ME"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-grafana-api-key"
  }
}

# --- CD-mutated, ignore_changes required ---------------------------------
#
# cd.yml (Group 11) overwrites these on every merge to main. Without
# ignore_changes, an unrelated terraform apply (a Group 12b tweak, a stray
# apply after a drift-checking plan) reverts a live deployment back to
# whatever placeholder is hardcoded here.

resource "aws_ssm_parameter" "api_image" {
  name        = "/telco-churn/api-image"
  description = "Full ECR image URI:tag currently deployed for api; mutated by cd.yml"
  type        = "String"
  value       = "${aws_ecr_repository.this["api"].repository_url}:unset"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-api-image"
  }
}

resource "aws_ssm_parameter" "ui_image" {
  name        = "/telco-churn/ui-image"
  description = "Full ECR image URI:tag currently deployed for ui; mutated by cd.yml"
  type        = "String"
  value       = "${aws_ecr_repository.this["ui"].repository_url}:unset"

  lifecycle {
    ignore_changes = [value]
  }

  tags = {
    Name = "${var.project_name}-ui-image"
  }
}

# --- Fully Terraform-derived, no ignore_changes (self-corrects) ---------
#
# Two separate rds-url-* params, not one shared rds-url: a Postgres
# connection string names exactly one database, and this instance holds two
# (telco_churn, mlflow) on the same host/credentials.

resource "aws_ssm_parameter" "rds_url_app" {
  name        = "/telco-churn/rds-url-app"
  description = "Postgres connection string for the app database (customers_raw, customers_crm, prediction_log, ...); renders as POSTGRES_URL"
  type        = "SecureString"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.rds_master.result}@${aws_db_instance.main.address}:${aws_db_instance.main.port}/${aws_db_instance.main.db_name}"

  tags = {
    Name = "${var.project_name}-rds-url-app"
  }
}

resource "aws_ssm_parameter" "rds_url_mlflow" {
  name        = "/telco-churn/rds-url-mlflow"
  description = "Postgres connection string for the mlflow tracking database; renders as MLFLOW_TRACKING_URI (direct DB URI, no MLflow server process)"
  type        = "SecureString"
  value       = "postgresql://${aws_db_instance.main.username}:${random_password.rds_master.result}@${aws_db_instance.main.address}:${aws_db_instance.main.port}/mlflow"

  tags = {
    Name = "${var.project_name}-rds-url-mlflow"
  }
}

resource "aws_ssm_parameter" "s3_bucket" {
  name        = "/telco-churn/s3-bucket"
  description = "MLflow artifact-store bucket name"
  type        = "String"
  value       = aws_s3_bucket.mlflow.id

  tags = {
    Name = "${var.project_name}-s3-bucket"
  }
}

# --- Static, human-picked literal ----------------------------------------

resource "aws_ssm_parameter" "domain" {
  name        = "/telco-churn/domain"
  description = "Public DuckDNS domain serving both api and ui behind Caddy's path-based routing"
  type        = "String"
  value       = "telco-churn.duckdns.org"

  tags = {
    Name = "${var.project_name}-domain"
  }
}
