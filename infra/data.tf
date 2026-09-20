# Group 4 (S3 MLflow artifact store) + Group 6 (RDS Postgres).
#
# Bucket name is suffixed with the account ID for guaranteed global
# uniqueness — same reasoning as the tfstate bucket in infra/bootstrap —
# even though the plan's deliverables section names it without one.

resource "aws_s3_bucket" "mlflow" {
  bucket        = "telco-churn-mlflow-${data.aws_caller_identity.current.account_id}"
  force_destroy = true

  tags = {
    Name = "${var.project_name}-mlflow-artifacts"
  }
}

resource "aws_s3_bucket_versioning" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_lifecycle_configuration" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id

  rule {
    id     = "expire-noncurrent-versions"
    status = "Enabled"
    filter {}

    noncurrent_version_expiration {
      noncurrent_days = 30
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"
    filter {}

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }
}

# Allow-only-the-instance-role, deny-non-TLS. Doesn't explicitly Deny every
# other principal (that would also block the terraform-admin role's own
# ability to inspect/manage the bucket) — the actual verification target per
# the plan is anonymous access getting 403'd, not locking out account admins.
data "aws_iam_policy_document" "mlflow_bucket" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "AWS"
      identifiers = ["*"]
    }

    actions = ["s3:*"]

    resources = [
      aws_s3_bucket.mlflow.arn,
      "${aws_s3_bucket.mlflow.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }

  statement {
    sid    = "InstanceRoleAccess"
    effect = "Allow"

    principals {
      type        = "AWS"
      identifiers = [aws_iam_role.ec2_instance.arn]
    }

    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]

    resources = [
      aws_s3_bucket.mlflow.arn,
      "${aws_s3_bucket.mlflow.arn}/*",
    ]
  }
}

resource "aws_s3_bucket_policy" "mlflow" {
  bucket = aws_s3_bucket.mlflow.id
  policy = data.aws_iam_policy_document.mlflow_bucket.json
}

# --- Group 6: RDS Postgres ----------------------------------------------
#
# One instance, two logical databases: the default database ("telco_churn",
# matching docker-compose.yml/configs/config.yaml's local-dev naming) holds
# the app tables (customers_raw, customers_crm, prediction_log, Optuna's own
# "optuna" schema); "mlflow" is created manually post-apply (Group 6's
# `CREATE DATABASE mlflow;` step) since aws_db_instance only provisions one
# database per instance. Group 7 renders these into two separate SSM params
# (rds-url-app / rds-url-mlflow) rather than one shared URL.

resource "aws_db_subnet_group" "main" {
  name       = "${var.project_name}-rds"
  subnet_ids = aws_subnet.private[*].id

  tags = {
    Name = "${var.project_name}-rds-subnet-group"
  }
}

# rds.force_ssl is dynamic (no reboot required) on the postgres16 family.
resource "aws_db_parameter_group" "main" {
  name   = "${var.project_name}-postgres16"
  family = "postgres16"

  parameter {
    name  = "rds.force_ssl"
    value = "1"
  }

  tags = {
    Name = "${var.project_name}-rds-params"
  }
}

# Master password only, per the plan: no human-typed TF_VAR_, no Secrets
# Manager. Lives in Terraform state (S3 backend, SSE + versioning already
# enabled by infra/bootstrap) and is read back by Group 7's ssm.tf to render
# the two rds-url-* SecureString params.
resource "random_password" "rds_master" {
  length  = 32
  special = false # avoid characters that need URI-encoding in a postgresql:// connection string
}

resource "aws_db_instance" "main" {
  identifier     = "${var.project_name}-db"
  engine         = "postgres"
  engine_version = "16"
  instance_class = "db.t3.micro"

  allocated_storage = 20
  storage_type      = "gp3"
  storage_encrypted = true

  db_name  = "telco_churn"
  username = "telco_admin"
  password = random_password.rds_master.result
  port     = 5432

  db_subnet_group_name   = aws_db_subnet_group.main.name
  vpc_security_group_ids = [aws_security_group.rds.id]
  parameter_group_name   = aws_db_parameter_group.main.name
  publicly_accessible    = false
  multi_az               = false

  # Without this, a password/instance-class change queues silently until the
  # next maintenance window instead of applying — which is how the live RDS
  # password drifted from Terraform state/SSM on 2026-09-18 (see
  # PHASE_12A_TODO.md's Group 6.5 incident note). No production traffic on
  # this single-instance portfolio DB to protect from apply-time blips, so
  # there's no reason to defer.
  apply_immediately = true

  # 7 (the plan's original value) is rejected outright by this account's
  # Free Plan credit restrictions ("FreeTierRestrictionError: backup
  # retention period exceeds the maximum available to free tier customers",
  # hit 2026-09-16) — 1 day is the smallest value that still enables
  # automated backups at all.
  backup_retention_period   = 1
  deletion_protection       = true
  skip_final_snapshot       = false
  final_snapshot_identifier = "${var.project_name}-db-final-snapshot"

  tags = {
    Name = "${var.project_name}-rds"
  }
}
