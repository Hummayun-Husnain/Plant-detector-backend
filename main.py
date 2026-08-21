"""
Cerebrium entrypoint.

Deployed as a normal Cerebrium app. Called by the Next.js backend as:

    POST /v4/<project-id>/<app-name>/predict?async=true
    {
        "input_key": "uploads/<jobId>.mp4",
        "output_key": "outputs/<jobId>.mp4",
        "meta_key": "outputs/<jobId>.json",
        "config": { ... }
    }

Because it is invoked with async=true, Cerebrium returns a 202 immediately
with a run_id and keeps running this function in the background. The
frontend never talks to Cerebrium directly after kicking off the job - it
polls its own /api/status route, which checks whether meta_key exists in S3.
"""

import json
import os
import tempfile
import traceback

from inference import run_inference
from s3_utils import download_file, upload_file, upload_bytes


def _job_id_from_meta(meta_key: str) -> str:
    return meta_key.replace("outputs/", "").replace(".json", "").split("/")[-1]


def predict(input_key: str, output_key: str, meta_key: str, **kwargs):
    print("[predict] START", flush=True)

    config = kwargs.get("config") or {}
    job_id = _job_id_from_meta(meta_key)

    print(f"[predict] job_id={job_id}", flush=True)
    print(f"[predict] input_key={input_key}", flush=True)
    print(f"[predict] output_key={output_key}", flush=True)
    print(f"[predict] meta_key={meta_key}", flush=True)
    print(f"[predict] config={config}", flush=True)

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

            print("[predict] uploading output...", flush=True)

            upload_file(
                local_output,
                output_key,
                content_type="video/mp4",
            )

            print("[predict] output upload completed", flush=True)

            meta = {
                "job_id": job_id,
                "status": "done",
                "count": result["count"],
                "output_key": output_key,
                "input_key": input_key,
                "metrics": result["metrics"],
                "experiment_benchmarks": result["experiment_benchmarks"],
            }

            print("[predict] uploading metadata...", flush=True)

            upload_bytes(
                json.dumps(meta).encode("utf-8"),
                meta_key,
            )

            print("[predict] metadata upload completed", flush=True)
            print("[predict] SUCCESS", flush=True)

            return meta

        except Exception as e:
            print("[predict] !!! ERROR !!!", flush=True)
            print(f"[predict] {type(e).__name__}: {e}", flush=True)
            print(traceback.format_exc(), flush=True)

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

            raise