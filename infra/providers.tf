provider "aws" {
  region = var.aws_region

  default_tags {
    tags = {
      Project     = "telco-churn"
      Environment = "prod"
      ManagedBy   = "terraform"
    }
  }
}
