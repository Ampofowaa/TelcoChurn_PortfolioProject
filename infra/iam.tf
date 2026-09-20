# Group 2 — EC2 instance role/profile, GitHub OIDC provider, CI deploy role.
#
# ECR/S3 ARNs below are interpolated string literals, not resource references —
# ecr.tf (Group 5) and data.tf's bucket (Group 4) don't exist yet at this point
# in the build, so `aws_ecr_repository.*.arn` etc. would fail as undeclared.
# Names are already fixed by the plan, so the ARNs are fully predictable.

data "aws_caller_identity" "current" {}

locals {
  ecr_repo_arns = [
    "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/telco-churn/api",
    "arn:aws:ecr:${var.aws_region}:${data.aws_caller_identity.current.account_id}:repository/telco-churn/ui",
  ]

  mlflow_bucket_arn = "arn:aws:s3:::telco-churn-mlflow-${data.aws_caller_identity.current.account_id}"
  # No slash before the trailing wildcard: ssm:GetParametersByPath evaluates
  # IAM against the queried *path* itself (e.g. "/telco-churn" -> resource
  # "parameter/telco-churn", no trailing slash), which a "parameter/telco-churn/*"
  # pattern doesn't match (requires a literal "/" before the wildcard). AWS's
  # own documented pattern for path-scoped access drops the slash so the
  # zero-width match covers the bare path too. GetParameter/GetParameters
  # still match fine since those check full parameter names ("telco-churn/api-key").
  ssm_param_arn     = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/telco-churn*"
  kms_ssm_key_arn   = "arn:aws:kms:${var.aws_region}:${data.aws_caller_identity.current.account_id}:alias/aws/ssm"
  log_group_arn     = "arn:aws:logs:${var.aws_region}:${data.aws_caller_identity.current.account_id}:log-group:/telco-churn/*"
}

# --- EC2 instance role -------------------------------------------------

resource "aws_iam_role" "ec2_instance" {
  name = "telco-churn-ec2-instance"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Service = "ec2.amazonaws.com" }
        Action    = "sts:AssumeRole"
      }
    ]
  })
}

resource "aws_iam_instance_profile" "ec2_instance" {
  name = "telco-churn-ec2-instance"
  role = aws_iam_role.ec2_instance.name
}

# AWS-published, purpose-built minimal policy for SSM Agent registration +
# Session Manager + Run Command — hand-rolling the equivalent action list
# risks a silently broken shell-access path with no clear error to debug.
resource "aws_iam_role_policy_attachment" "ec2_instance_ssm_core" {
  role       = aws_iam_role.ec2_instance.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_role_policy" "ec2_instance" {
  name = "telco-churn-ec2-instance-permissions"
  role = aws_iam_role.ec2_instance.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "EcrPull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
        ]
        Resource = local.ecr_repo_arns
      },
      {
        Sid    = "MlflowArtifactObjects"
        Effect = "Allow"
        Action = [
          "s3:GetObject",
          "s3:PutObject",
        ]
        Resource = "${local.mlflow_bucket_arn}/*"
      },
      {
        Sid      = "MlflowArtifactBucketList"
        Effect   = "Allow"
        Action   = "s3:ListBucket"
        Resource = local.mlflow_bucket_arn
      },
      {
        Sid    = "SsmParams"
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath",
        ]
        Resource = local.ssm_param_arn
      },
      {
        Sid      = "SsmParamsKms"
        Effect   = "Allow"
        Action   = "kms:Decrypt"
        Resource = local.kms_ssm_key_arn
      },
      {
        Sid      = "CloudWatchMetrics"
        Effect   = "Allow"
        Action   = "cloudwatch:PutMetricData"
        Resource = "*" # CloudWatch metrics have no ARN — AWS requires Resource: "*" here
      },
      {
        Sid    = "CloudWatchLogs"
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents",
        ]
        Resource = "${local.log_group_arn}:*"
      },
    ]
  })
}

# --- GitHub OIDC + CI deploy role --------------------------------------

# Thumbprint is AWS's long-published value for GitHub's OIDC endpoint (AWS
# validates against its own trusted CA chain since 2023 and largely ignores
# this field for well-known providers) — worth a quick cross-check against
# AWS's current GitHub-OIDC docs before the first apply, since getting it
# wrong fails safely (federation just won't work) rather than insecurely.
resource "aws_iam_openid_connect_provider" "github" {
  url             = "https://token.actions.githubusercontent.com"
  client_id_list  = ["sts.amazonaws.com"]
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"] # pragma: allowlist secret - public CA thumbprint
}

resource "aws_iam_role" "ci_deploy" {
  name = "telco-churn-ci-deploy"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect    = "Allow"
        Principal = { Federated = aws_iam_openid_connect_provider.github.arn }
        Action    = "sts:AssumeRoleWithWebIdentity"
        Condition = {
          StringEquals = {
            "token.actions.githubusercontent.com:aud" = "sts.amazonaws.com"
          }
          StringLike = {
            "token.actions.githubusercontent.com:sub" = "repo:Ampofowaa/TelcoChurn_PortfolioProject:ref:refs/heads/main"
          }
        }
      }
    ]
  })
}

resource "aws_iam_role_policy" "ci_deploy" {
  name = "telco-churn-ci-deploy-permissions"
  role = aws_iam_role.ci_deploy.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid      = "EcrAuth"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "EcrPush"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage",
          "ecr:PutImage",
          "ecr:InitiateLayerUpload",
          "ecr:UploadLayerPart",
          "ecr:CompleteLayerUpload",
        ]
        Resource = local.ecr_repo_arns
      },
      {
        # The document side of the ssm:SendCommand permission check — a
        # public AWS document, so its ARN carries no account ID.
        Sid      = "SendCommandDocument"
        Effect   = "Allow"
        Action   = "ssm:SendCommand"
        Resource = "arn:aws:ssm:${var.aws_region}::document/AWS-RunShellScript"
      },
      {
        # The target side of the same check — scoped by the instance's
        # Project tag rather than a specific instance ARN, since compute.tf's
        # instance (Group 8) doesn't exist yet and this way survives the
        # instance being replaced later with no IAM edit.
        Sid      = "SendCommandTarget"
        Effect   = "Allow"
        Action   = "ssm:SendCommand"
        Resource = "arn:aws:ec2:${var.aws_region}:${data.aws_caller_identity.current.account_id}:instance/*"
        Condition = {
          StringEquals = {
            "ssm:resourceTag/Project" = "telco-churn"
          }
        }
      },
      {
        Sid      = "GetCommandInvocation"
        Effect   = "Allow"
        Action   = ["ssm:GetCommandInvocation", "ssm:ListCommandInvocations"]
        Resource = "*" # no resource-level permission support for these actions
      },
      {
        # `aws s3 sync` lists the destination to work out what differs, so
        # PutObject alone 403s on ListObjectsV2. Bucket-level action, scoped
        # to the same prefix via s3:prefix so the rest of the bucket's keys
        # stay unlistable.
        Sid       = "DeployConfigList"
        Effect    = "Allow"
        Action    = "s3:ListBucket"
        Resource  = local.mlflow_bucket_arn
        Condition = {
          StringLike = {
            "s3:prefix" = ["deploy-config", "deploy-config/*"]
          }
        }
      },
      {
        # cd.yml syncs infra/deploy/ + docker/mlflow/Dockerfile here before
        # the SSM deploy command. This one prefix only — never model
        # artifacts or anything else in the bucket.
        Sid      = "DeployConfigUpload"
        Effect   = "Allow"
        Action   = "s3:PutObject"
        Resource = "${local.mlflow_bucket_arn}/deploy-config/*"
      },
      {
        # cd.yml overwrites these two on every deploy, and reads them first
        # so a failed smoke test can put the previous SHA back (rollback).
        # Nothing else under /telco-churn/* — never api-key or the RDS URLs.
        Sid    = "ImageParams"
        Effect = "Allow"
        Action = ["ssm:GetParameter", "ssm:PutParameter"]
        Resource = [
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/telco-churn/api-image",
          "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/telco-churn/ui-image",
        ]
      },
    ]
  })
}
