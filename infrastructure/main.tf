# 1. Configure the AWS Provider (Pointing to our LocalStack emulator)
provider "aws" {
  region                      = "us-east-1"
  access_key                  = "mock_key"
  secret_key                  = "mock_secret"
  skip_credentials_validation = true
  skip_metadata_api_check     = true
  skip_requesting_account_id  = true

  # Tell Terraform to build resources locally instead of in the real cloud
  endpoints {
    sqs    = "http://localhost:4566"
    lambda = "http://localhost:4566"
    iam    = "http://localhost:4566"
  }
}

# 2. Define the SQS Queues
resource "aws_sqs_queue" "task_queue" {
  name                       = "leviathan-prod-task-queue"
  visibility_timeout_seconds = 30 # Time before a task is reassigned if a worker fails
  message_retention_seconds  = 86400 # 1 Day
}

resource "aws_sqs_queue" "response_queue" {
  name                       = "leviathan-prod-response-queue"
  visibility_timeout_seconds = 30
}

# 3. Output the physical URLs of the created queues
output "task_queue_url" {
  value = aws_sqs_queue.task_queue.url
}

output "response_queue_url" {
  value = aws_sqs_queue.response_queue.url
}