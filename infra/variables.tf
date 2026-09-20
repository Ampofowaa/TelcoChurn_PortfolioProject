variable "aws_region" {
  description = "AWS region for every resource this project creates"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Short name used as a prefix/tag across all resources"
  type        = string
  default     = "telco-churn"
}
