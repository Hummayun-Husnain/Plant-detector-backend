"""
S3 helper functions used by the Cerebrium inference app.

Credentials come from environment variables, which are injected by Cerebrium
at runtime from secrets you set with:

    cerebrium secret set AWS_ACCESS_KEY_ID <value>
    cerebrium secret set AWS_SECRET_ACCESS_KEY <value>
    cerebrium secret set AWS_REGION <value>
    cerebrium secret set AWS_BUCKET_NAME <value>

Adapted from the original download_testing.py, but keys are now passed in
explicitly (the Next.js backend decides the key naming scheme) rather than
generated here, so the frontend and backend always agree on where a file
lives.
"""

import os
import boto3
from botocore.exceptions import ClientError, NoCredentialsError

AWS_ACCESS_KEY_ID = os.environ.get("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.environ.get("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")
BUCKET_NAME = os.environ.get("AWS_BUCKET_NAME")

_s3 = None


def get_client():
    """Lazily create a single boto3 S3 client per container."""
    global _s3
    if _s3 is None:
        if not all([AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, BUCKET_NAME]):
            raise RuntimeError(
                "Missing AWS env vars. Set AWS_ACCESS_KEY_ID, "
                "AWS_SECRET_ACCESS_KEY, AWS_REGION and AWS_BUCKET_NAME as "
                "Cerebrium secrets."
            )
        _s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION,
        )
    return _s3


def download_file(s3_key: str, local_path: str) -> bool:
    s3 = get_client()
    try:
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        s3.download_file(BUCKET_NAME, s3_key, local_path)
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"[s3_utils] download_file failed for {s3_key}: {e}")
        raise


def upload_file(local_path: str, s3_key: str, content_type: str | None = None) -> str:
    s3 = get_client()
    extra_args = {"ContentType": content_type} if content_type else {}
    try:
        s3.upload_file(local_path, BUCKET_NAME, s3_key, ExtraArgs=extra_args)
        return s3_key
    except ClientError as e:
        print(f"[s3_utils] upload_file failed for {s3_key}: {e}")
        raise


def upload_bytes(data: bytes, s3_key: str, content_type: str = "application/json") -> str:
    s3 = get_client()
    try:
        s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=data, ContentType=content_type)
        return s3_key
    except ClientError as e:
        print(f"[s3_utils] upload_bytes failed for {s3_key}: {e}")
        raise
