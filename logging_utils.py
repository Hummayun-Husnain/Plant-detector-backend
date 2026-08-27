"""
Application logging utilities for the Plant Counter backend.

Logs important events without exposing sensitive information like
passwords, tokens, AWS credentials, etc.
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("plant-counter")


def log_user_registration(user_id: str, email: str) -> None:
    """Log user registration event."""
    logger.info(f"[USER_REGISTRATION] user_id={user_id}, email={email}")


def log_user_login(user_id: str, email: str) -> None:
    """Log user login event."""
    logger.info(f"[USER_LOGIN] user_id={user_id}, email={email}")


def log_user_logout(user_id: str, email: str) -> None:
    """Log user logout event."""
    logger.info(f"[USER_LOGOUT] user_id={user_id}, email={email}")


def log_video_upload(job_id: str, user_id: str, file_name: str, file_size: int) -> None:
    """Log video upload event."""
    logger.info(
        f"[VIDEO_UPLOAD] job_id={job_id}, user_id={user_id}, "
        f"file_name={file_name}, file_size={file_size}"
    )


def log_processing_started(job_id: str, user_id: str, input_key: str) -> None:
    """Log processing start event."""
    logger.info(
        f"[PROCESSING_STARTED] job_id={job_id}, user_id={user_id}, "
        f"input_key={input_key}"
    )


def log_processing_completed(
    job_id: str,
    user_id: str,
    plant_count: int,
    processing_time: float,
    output_key: str,
) -> None:
    """Log processing completion event."""
    logger.info(
        f"[PROCESSING_COMPLETED] job_id={job_id}, user_id={user_id}, "
        f"plant_count={plant_count}, processing_time={processing_time:.2f}s, "
        f"output_key={output_key}"
    )


def log_processing_failed(
    job_id: str,
    user_id: str,
    error_message: str,
    error_type: str = "unknown",
) -> None:
    """Log processing failure event."""
    logger.error(
        f"[PROCESSING_FAILED] job_id={job_id}, user_id={user_id}, "
        f"error_type={error_type}, error_message={error_message}"
    )


def log_s3_upload(key: str, file_size: int, content_type: str) -> None:
    """Log S3 upload event."""
    logger.info(
        f"[S3_UPLOAD] key={key}, file_size={file_size}, "
        f"content_type={content_type}"
    )


def log_s3_upload_failed(key: str, error_message: str) -> None:
    """Log S3 upload failure event."""
    logger.error(f"[S3_UPLOAD_FAILED] key={key}, error={error_message}")


def log_s3_url_generated(key: str, url_type: str = "presigned") -> None:
    """Log S3 URL generation event."""
    logger.info(f"[S3_URL_GENERATED] key={key}, url_type={url_type}")


def log_ffmpeg_conversion(input_path: str, output_path: str, success: bool) -> None:
    """Log FFmpeg conversion event."""
    status = "success" if success else "failed"
    logger.info(
        f"[FFMPEG_CONVERSION] status={status}, "
        f"input={input_path}, output={output_path}"
    )


def log_inference_error(job_id: str, error_type: str, error_message: str) -> None:
    """Log inference error event."""
    logger.error(
        f"[INFERENCE_ERROR] job_id={job_id}, error_type={error_type}, "
        f"error_message={error_message}"
    )


def log_database_error(operation: str, error_message: str) -> None:
    """Log database error event."""
    logger.error(
        f"[DATABASE_ERROR] operation={operation}, error={error_message}"
    )


def log_api_request(
    action: str,
    user_id: Optional[str] = None,
    job_id: Optional[str] = None,
) -> None:
    """Log API request event."""
    logger.info(
        f"[API_REQUEST] action={action}, "
        f"user_id={user_id or 'anonymous'}, "
        f"job_id={job_id or 'none'}"
    )


def log_api_error(
    action: str,
    error_message: str,
    user_id: Optional[str] = None,
) -> None:
    """Log API error event."""
    logger.error(
        f"[API_ERROR] action={action}, user_id={user_id or 'anonymous'}, "
        f"error={error_message}"
    )


def log_metrics(job_id: str, metrics: dict) -> None:
    """Log processing metrics."""
    safe_metrics = {
        k: v
        for k, v in metrics.items()
        if k not in ["password", "token", "secret", "key"]
    }
    logger.info(
        f"[METRICS] job_id={job_id}, metrics={json.dumps(safe_metrics)}"
    )
