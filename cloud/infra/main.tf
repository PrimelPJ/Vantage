# ---- hard-case dataset bucket -------------------------------------------------
resource "aws_s3_bucket" "hard_cases" {
  bucket        = "${var.project}-hard-cases-${data.aws_caller_identity.me.account_id}"
  force_destroy = true
}

resource "aws_s3_bucket_public_access_block" "hard_cases" {
  bucket                  = aws_s3_bucket.hard_cases.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_server_side_encryption_configuration" "hard_cases" {
  bucket = aws_s3_bucket.hard_cases.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

# Expire raw hard cases after 90 days; by then they are in a curated dataset.
resource "aws_s3_bucket_lifecycle_configuration" "hard_cases" {
  bucket = aws_s3_bucket.hard_cases.id
  rule {
    id     = "expire-raw"
    status = "Enabled"
    filter { prefix = "hard-cases/" }
    expiration { days = 90 }
  }
}

# ---- SageMaker execution role -------------------------------------------------
resource "aws_iam_role" "sagemaker_exec" {
  name = "${var.project}-sagemaker-exec"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "sagemaker.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy_attachment" "sagemaker_full" {
  role       = aws_iam_role.sagemaker_exec.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSageMakerFullAccess"
}

# ---- robot role: invoke the endpoint + write hard cases -----------------------
# On real hardware a robot assumes this via the IoT Core credentials provider
# (X.509 cert exchanged for temporary IAM creds), so no static keys live on the
# robot. See docs/architecture.md.
resource "aws_iam_role" "robot" {
  name = "${var.project}-robot"
  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [{
      Effect    = "Allow"
      Principal = { Service = "credentials.iot.amazonaws.com" }
      Action    = "sts:AssumeRole"
    }]
  })
}

resource "aws_iam_role_policy" "robot" {
  role = aws_iam_role.robot.id
  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect   = "Allow"
        Action   = ["sagemaker:InvokeEndpoint"]
        Resource = "arn:aws:sagemaker:${var.region}:${data.aws_caller_identity.me.account_id}:endpoint/${var.project}-*"
      },
      {
        Effect   = "Allow"
        Action   = ["s3:PutObject"]
        Resource = "${aws_s3_bucket.hard_cases.arn}/*"
      }
    ]
  })
}
