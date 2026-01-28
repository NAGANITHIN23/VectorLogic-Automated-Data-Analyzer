variable "aws_region" {
  description = "The AWS region to deploy the data analyzer"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  type    = string
  default = "VectorLogic"
}
