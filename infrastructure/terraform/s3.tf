resource "aws_s3_bucket" "data_storage" {
  bucket = "vectorlogic-automated-data-analyzer" # Must be globally unique
  force_destroy = true # Useful for deleting the project easily later
}

resource "aws_s3_bucket_public_access_block" "block_public" {
  bucket = aws_s3_bucket.data_storage.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}
