
import boto3
from botocore.exceptions import NoCredentialsError, PartialCredentialsError, ClientError

def get_s3_client():
    """Create and return an S3 client."""
    try:
        return boto3.client("s3")
    except (NoCredentialsError, PartialCredentialsError):
        # Handle missing credentials gracefully
        return None

def list_s3_buckets():
    """List all S3 buckets."""
    s3_client = get_s3_client()
    if not s3_client:
        return {"error": "AWS credentials not configured."}
    try:
        return s3_client.list_buckets()
    except ClientError as e:
        return {"error": str(e)}

def list_s3_objects(bucket_name: str):
    """List all objects in a given S3 bucket."""
    s3_client = get_s3_client()
    if not s3_client:
        return {"error": "AWS credentials not configured."}
    try:
        return s3_client.list_objects_v2(Bucket=bucket_name)
    except ClientError as e:
        return {"error": str(e)}

def download_s3_file(bucket_name: str, object_key: str):
    """Download a file from S3."""
    s3_client = get_s3_client()
    if not s3_client:
        return {"error": "AWS credentials not configured."}
    try:
        response = s3_client.get_object(Bucket=bucket_name, Key=object_key)
        return response["Body"].read()
    except ClientError as e:
        return {"error": str(e)}
