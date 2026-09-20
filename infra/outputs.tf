# Populated as each group's resources land (ALB/Caddy domain, ECR repo URLs,
# etc.) — outputs are for human/CLI visibility only; other .tf files in this
# same root module reference resources directly, not via these.

output "vpc_id" {
  value = aws_vpc.main.id
}

output "public_subnet_ids" {
  value = aws_subnet.public[*].id
}

output "private_subnet_ids" {
  value = aws_subnet.private[*].id
}

output "ec2_security_group_id" {
  value = aws_security_group.ec2.id
}

output "rds_security_group_id" {
  value = aws_security_group.rds.id
}

output "elastic_ip" {
  value = aws_eip.instance.public_ip
}

output "mlflow_bucket" {
  value = aws_s3_bucket.mlflow.bucket
}

output "ecr_repository_urls" {
  value = { for k, v in aws_ecr_repository.this : k => v.repository_url }
}

output "rds_address" {
  value = aws_db_instance.main.address
}

output "rds_port" {
  value = aws_db_instance.main.port
}

output "instance_id" {
  value = aws_instance.main.id
}

# sensitive = true: never printed by a bare `terraform output`/`apply`, only
# by explicit `terraform output -raw rds_master_password`. Group 6.5's manual
# CREATE DATABASE/alembic steps need it directly - Group 7's SSM params
# already consume it via direct resource reference, not this output.
output "rds_master_password" {
  value     = random_password.rds_master.result
  sensitive = true
}
