"""
Cerebrium entrypoint.

The frontend no longer uses AWS credentials. All S3 operations (upload input,
generate presigned download URLs, check status, list jobs) are handled by this
Cerebrium app. The Next.js frontend only proxies calls to Cerebrium.

Routes are selected with the `action` field in the request body:

    action="upload"  -> receive a base64 video, upload to S3, run inference,
                        upload output + metadata (call with ?async=true)
    action="status"  -> check whether metadata JSON exists in S3
    action="download" -> generate a temporary S3 GET URL
    action="list"    -> list finished jobs
    action="predict" or no action -> original inference on input_key/output_key
"""

import base64
import json
import os
import tempfile
import traceback
import uuid

from inference import run_inference
from agronomy_reporter import generate_agronomic_report
from s3_utils import (
    download_file,
    upload_file,
    upload_bytes,
    generate_presigned_url,
    object_exists,
    get_json_object,
    list_job_keys,
    get_public_url,
)
from logging_utils import (
    log_video_upload,
    log_processing_started,
    log_processing_completed,
    log_processing_failed,
    log_s3_upload,
    log_s3_upload_failed,
    log_s3_url_generated,
    log_api_request,
    log_api_error,
    log_metrics,
)


def _job_id_from_meta(meta_key: str) -> str:
    return meta_key.replace("outputs/", "").replace(".json", "").split("/")[-1]


def _job_id() -> str:
    return uuid.uuid4().hex


def _run_pipeline(input_key: str, output_key: str, meta_key: str, config: dict) -> dict:
    job_id = _job_id_from_meta(meta_key)

    print(f"[predict] job_id={job_id}", flush=True)
    print(f"[predict] input_key={input_key}", flush=True)
    print(f"[predict] output_key={output_key}", flush=True)
    print(f"[predict] meta_key={meta_key}", flush=True)
    print(f"[predict] config={config}", flush=True)
    
    log_processing_started(job_id, "unknown", input_key)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_input = os.path.join(tmp_dir, "input.mp4")
        local_output = os.path.join(tmp_dir, "output.mp4")

        print(f"[predict] tmp_dir={tmp_dir}", flush=True)

        try:
            print("[predict] downloading input...", flush=True)
            download_file(input_key, local_input)

            print(
                f"[predict] download complete, size={os.path.getsize(local_input)} bytes",
                flush=True,
            )

            print("[predict] starting run_inference...", flush=True)

            result = run_inference(
                local_input,
                local_output,
                config=config,
            )

            print("[predict] run_inference completed", flush=True)
            print(f"[predict] result={result}", flush=True)

            print("[predict] generating agronomic report...", flush=True)
            report = {}
            try:
                report = generate_agronomic_report(
                    local_input, result["count"], job_id
                )
                print("[predict] agronomic report generated", flush=True)
            except Exception as report_err:
                print(f"[predict] agronomic report error: {report_err}", flush=True)

            print("[predict] uploading output...", flush=True)

            upload_file(
                local_output,
                output_key,
                content_type="video/mp4",
            )

            print("[predict] output upload completed", flush=True)
            log_s3_upload(output_key, os.path.getsize(local_output), "video/mp4")

            meta = {
                "job_id": job_id,
                "status": "done",
                "count": result["count"],
                "output_key": output_key,
                "input_key": input_key,
                "metrics": result["metrics"],
                "experiment_benchmarks": result["experiment_benchmarks"],
                "pdf_report_url": report.get("report_url"),
                "insights_summary": (
                    (report.get("insights") or {}).get("executive_summary")
                    if report.get("insights")
                    else None
                ),
                "processing_status": report.get(
                    "processing_status", "done"
                ),
                "agronomic_insights": report.get("insights"),
            }

            print("[predict] uploading metadata...", flush=True)

            upload_bytes(
                json.dumps(meta).encode("utf-8"),
                meta_key,
            )

            print("[predict] metadata upload completed", flush=True)
            print("[predict] SUCCESS", flush=True)

            log_processing_completed(
                job_id,
                "unknown",
                result["count"],
                result["metrics"]["processing_time"],
                output_key,
            )
            log_metrics(job_id, result["metrics"])

            return meta

        except Exception as e:
            print("[predict] !!! ERROR !!!", flush=True)
            print(f"[predict] {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

            log_processing_failed(job_id, "unknown", str(e), type(e).__name__)

            error_meta = {
                "job_id": job_id,
                "status": "error",
                "error": str(e),
                "trace": traceback.format_exc(),
            }

            try:
                print("[predict] writing error metadata...", flush=True)

                upload_bytes(
                    json.dumps(error_meta).encode("utf-8"),
                    meta_key,
                )

                print("[predict] error metadata written", flush=True)

            except Exception as inner:
                print(
                    f"[predict] failed to write error meta_key: {inner}",
                    flush=True,
                )
                log_s3_upload_failed(meta_key, str(inner))

            raise


def _handle_upload(
    job_id: str,
    video_b64: str,
    file_name: str,
    ext: str,
    config: dict,
    file_type: str | None = None,
) -> dict:
    ext = (ext or file_name.split(".")[-1] or "mp4").lower()

    if not job_id:
        job_id = _job_id()

    input_key = f"uploads/{job_id}.{ext}"
    output_key = f"outputs/{job_id}.mp4"
    meta_key = f"outputs/{job_id}.json"

    print(f"[upload] job_id={job_id}", flush=True)
    print(f"[upload] file_name={file_name}", flush=True)
    print(f"[upload] input_key={input_key}", flush=True)

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_input = os.path.join(tmp_dir, f"input.{ext}")

        print(f"[upload] decoding video, b64_length={len(video_b64)}", flush=True)

        with open(local_input, "wb") as f:
            f.write(base64.b64decode(video_b64))

        file_size = os.path.getsize(local_input)
        print(f"[upload] decoded, file_size={file_size}", flush=True)

        log_video_upload(job_id, "unknown", file_name, file_size)

        upload_file(local_input, input_key, content_type=file_type or f"video/{ext}")
        log_s3_upload(input_key, file_size, file_type or f"video/{ext}")

        print("[upload] input uploaded, starting inference...", flush=True)

        return _run_pipeline(input_key, output_key, meta_key, config)


def _status(job_id: str) -> dict:
    if not job_id:
        raise ValueError("job_id is required")

    meta_key = f"outputs/{job_id}.json"

    print(f"[status] job_id={job_id}", flush=True)

    exists = object_exists(meta_key)
    if not exists:
        return {"status": "processing"}

    meta = get_json_object(meta_key)

    if meta.get("status") == "done":
        output_key = meta.get("output_key")
        if not output_key:
            return {
                "status": "error",
                "error": "Job completed but no output video key was found.",
                "job_id": job_id,
            }

        try:
            if not object_exists(output_key):
                return {
                    "status": "error",
                    "error": "Output video not found in S3.",
                    "job_id": job_id,
                }
            
            # Generate both presigned URL (for direct download) and public URL (for playback)
            presigned_url = generate_presigned_url(
                output_key,
                "get_object",
                3600,
                content_disposition='attachment; filename="plant-count-result.mp4"',
                content_type="video/mp4",
            )
            
            # Public URL for direct browser playback (if bucket is public)
            public_url = get_public_url(output_key)
            log_s3_url_generated(output_key, "presigned")
            log_s3_url_generated(output_key, "public")
            
        except Exception as e:
            print(f"[status] Failed to generate URLs for {output_key}: {e}", flush=True)
            log_api_error("status", str(e), job_id)
            return {
                "status": "error",
                "error": "Failed to generate output video URL.",
                "job_id": job_id,
            }

        return {
            "status": "done",
            "count": meta.get("count", 0),
            "outputKey": output_key,
            "inputKey": meta.get("input_key"),
            "video_url": presigned_url,  # For backwards compatibility
            "public_url": public_url,  # Public URL for direct playback
            "presigned_url": presigned_url,  # Explicit presigned URL
            "metrics": meta.get("metrics"),
            "experiment_benchmarks": meta.get("experiment_benchmarks", {}),
            "pdf_report_url": meta.get("pdf_report_url"),
            "insights_summary": meta.get("insights_summary"),
            "processing_status": meta.get("processing_status", "done"),
            "agronomic_insights": meta.get("agronomic_insights"),
        }

    if meta.get("status") == "error":
        return {
            "status": "error",
            "error": meta.get("error", "Inference failed."),
            "job_id": job_id,
        }

    return {"status": "processing"}


def _download_url(key: str) -> dict:
    if not key:
        raise ValueError("key is required")

    print(f"[download] key={key}", flush=True)

    return {
        "url": generate_presigned_url(key, "get_object", 900),
    }


def _list_jobs() -> dict:
    print("[list] listing jobs", flush=True)

    keys = list_job_keys("outputs/", 100)
    jobs = []

    for key in keys:
        try:
            raw = get_json_object(key)
            if raw.get("job_id"):
                jobs.append(raw)
            elif raw.get("status") in ("done", "error"):
                job_id = key.replace("outputs/", "").replace(".json", "").split("/")[-1]
                jobs.append({
                    "job_id": job_id,
                    "status": "completed" if raw["status"] == "done" else "failed",
                    "metrics": raw.get("metrics", {
                        "plant_count": raw.get("count", 0),
                        "video_duration": "00:00:00",
                        "frames_processed": 0,
                        "processing_time": 0,
                        "average_processing_fps": 0,
                        "unique_track_ids": 0,
                        "total_detections": 0,
                        "average_detection_confidence": 0,
                        "model_version": "unknown",
                        "tracker_configuration": {
                            "tracker_name": "unknown",
                            "version": "",
                            "config": {},
                        },
                        "counting_line_configuration": {
                            "line_coordinates": [[0, 0], [100, 100]],
                            "direction": "bidirectional",
                        },
                    }),
                    "experiment_benchmarks": raw.get(
                        "experiment_benchmarks",
                        {"ground_truth_count": 0, "error_rate_percentage": 0},
                    ),
                    "output_key": raw.get("output_key"),
                    "input_key": raw.get("input_key"),
                    "error": raw.get("error"),
                    "pdf_report_url": raw.get("pdf_report_url"),
                    "insights_summary": raw.get("insights_summary"),
                    "processing_status": raw.get("processing_status", "done"),
                    "agronomic_insights": raw.get("agronomic_insights"),
                })
        except Exception:
            # skip unparseable entries
            continue

    return {"jobs": jobs}


def predict(input_key=None, output_key=None, meta_key=None, action=None, **kwargs):
    print(f"[predict] action={action}", flush=True)

    if action == "upload" or "video_b64" in kwargs:
        job_id = kwargs.get("job_id")
        video_b64 = kwargs.get("video_b64")
        file_name = kwargs.get("file_name", "input.mp4")
        ext = kwargs.get("ext") or file_name.split(".")[-1] or "mp4"
        config = kwargs.get("config") or {}

        if not video_b64:
            raise ValueError("video_b64 is required for upload")

        file_type = kwargs.get("file_type")
        return _handle_upload(job_id, video_b64, file_name, ext, config, file_type)

    if action == "status":
        return _status(kwargs.get("job_id"))

    if action == "download":
        return _download_url(kwargs.get("key"))

    if action == "list":
        return _list_jobs()

    if not input_key or not output_key or not meta_key:
        raise ValueError("input_key, output_key, meta_key are required for predict")

    return _run_pipeline(input_key, output_key, meta_key, kwargs.get("config") or {})
