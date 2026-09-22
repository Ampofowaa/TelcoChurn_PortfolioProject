# Group 12 — CloudWatch log groups/alarms, SNS topic, billing alarms + AWS Budgets.
#
# Billing guardrails land ahead of the rest of Group 12 because they protect the
# credit balance while Groups 1–11 create billable resources. The SNS topic is
# shared: Group 12's operational alarms attach to aws_sns_topic.alerts too.
#
# AWS/Billing metrics exist only in us-east-1, which is this project's region.

resource "aws_sns_topic" "alerts" {
  name = "${var.project_name}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

resource "aws_cloudwatch_metric_alarm" "billing" {
  for_each = var.billing_alarm_thresholds_usd

  alarm_name          = "${var.project_name}-estimated-charges-${each.key}usd"
  alarm_description   = "Month-to-date EstimatedCharges exceeded ${each.key} USD"
  namespace           = "AWS/Billing"
  metric_name         = "EstimatedCharges"
  dimensions          = { Currency = "USD" }
  statistic           = "Maximum"
  period              = 21600 # billing metric updates a few times a day
  evaluation_periods  = 1
  threshold           = each.value
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
}

# include_credit = false so the budget tracks gross usage; with credits netted
# off it would read ~$0 for the whole credit lifetime and never alert.
resource "aws_budgets_budget" "monthly" {
  name         = "${var.project_name}-monthly"
  budget_type  = "COST"
  limit_amount = var.monthly_budget_usd
  limit_unit   = "USD"
  time_unit    = "MONTHLY"

  cost_types {
    include_credit = false
    include_refund = false
  }

  dynamic "notification" {
    for_each = [50, 80, 100]

    content {
      comparison_operator        = "GREATER_THAN"
      threshold                  = notification.value
      threshold_type             = "PERCENTAGE"
      notification_type          = "ACTUAL"
      subscriber_email_addresses = [var.alert_email]
    }
  }
}
