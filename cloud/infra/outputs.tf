output "hard_case_bucket"    { value = aws_s3_bucket.hard_cases.bucket }
output "sagemaker_role_arn"  { value = aws_iam_role.sagemaker_exec.arn }
output "robot_role_arn"      { value = aws_iam_role.robot.arn }
output "deploy_hint" {
  value = "deploy the endpoint: python cloud/sagemaker/deploy_endpoint.py --role <sagemaker_role_arn> --bucket <a model bucket> --endpoint vantage-detector"
}
