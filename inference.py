"""
Plant counting inference with full instrumentation.

Adapted from the original count_plants.py.
"""

import math
import os
import subprocess
import time
from datetime import timedelta

import cv2
import tqdm
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

CLASS_NAMES = ["plant"]
FRAME_SIZE = (1024, 1024)
# Full-width line through the middle of the (1024x1024) frame. Every plant
# that travels through the video passes this line; the old y=900 line sat at
# the bottom edge where plants were often already out of frame.
DEFAULT_LINE = [[0, 512], [1024, 512]]
# Minimum half-height of the counting band in pixels. The band grows with the
# plant's box so a large plant moving fast cannot jump over it between frames.
MIN_BAND_HALF_HEIGHT = 15
CONF_THRESHOLD = 0.3

TRACKER_CONFIG = {
    "max_age": 20,
    "n_init": 3,
    "nms_max_overlap": 1.0,
    "max_cosine_distance": 0.2,
    "nn_budget": None,
    "override_track_class": None,
    "embedder": "mobilenet",
    "half": True,
    "bgr": True,
}


def _get_model():
    model_path = os.environ.get("MODEL_PATH", "plant-counter-model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"YOLO weights not found at {model_path}. Either bundle the .pt "
            f"file in the deployment or upload it to persistent storage with "
            f"`cerebrium cp <local_file> plant-counter-model.pt` and set "
            f"MODEL_PATH accordingly."
        )
    return YOLO(model_path)


def _format_duration(seconds: float) -> str:
    return str(timedelta(seconds=int(seconds))) or "00:00:00"


def _resolve_counting_line(cfg: dict) -> list:
    """Pixel line used for the band check.

    The app sends `lineCoordinates` as fractions of the frame, for example
    [[0.25, 0.5], [0.75, 0.5]]. Values already in pixels are left as they are.
    """
    raw = cfg.get("line_coordinates") or cfg.get("lineCoordinates") or DEFAULT_LINE
    (x1, y1), (x2, y2) = raw
    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
    if max(abs(x1), abs(y1), abs(x2), abs(y2)) <= 1.0:
        width, height = FRAME_SIZE
        x1, x2 = x1 * width, x2 * width
        y1, y2 = y1 * height, y2 * height
    return [[x1, y1], [x2, y2]]


def _convert_to_mp4_ffmpeg(input_path: str, output_path: str) -> bool:
    """
    Convert video to MP4 using FFmpeg with H.264 codec for maximum browser compatibility.
    Returns True if successful, False otherwise.
    """
    try:
        print(f"[ffmpeg] Converting {input_path} to MP4 with H.264 codec...", flush=True)
        cmd = [
            "ffmpeg",
            "-i", input_path,
            "-c:v", "libx264",  # H.264 video codec
            "-preset", "fast",  # Encoding speed (fast is good for real-time)
            "-crf", "23",  # Quality (0-51, lower is better, 23 is default)
            "-c:a", "aac",  # AAC audio codec
            "-b:a", "128k",  # Audio bitrate
            "-movflags", "+faststart",  # Enable streaming from the start
            "-y",  # Overwrite output file
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            print(f"[ffmpeg] Successfully converted to MP4", flush=True)
            return True
        else:
            print(f"[ffmpeg] Conversion failed: {result.stderr}", flush=True)
            return False
    except Exception as e:
        print(f"[ffmpeg] Error during conversion: {e}", flush=True)
        return False


def run_inference(
    input_path: str,
    output_path: str,
    config: dict | None = None,
    on_progress=None,
) -> dict:
    """
    Run YOLO + DeepSORT plant counting on input_path, write an annotated
    video to output_path, and return full instrumentation metrics.
    """
    cfg = config or {}
    line = _resolve_counting_line(cfg)
    direction = cfg.get("direction") or "bidirectional"
    ground_truth = int(cfg.get("ground_truth_count") or 0)
    model_version = cfg.get("modelVersion") or os.environ.get("MODEL_VERSION") or "plant-counter-model"
    tracker_name = cfg.get("tracker") or "DeepSORT"

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    model = _get_model()
    tracker = DeepSort(**TRACKER_CONFIG)

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # type: ignore
    writer = cv2.VideoWriter(output_path, fourcc, src_fps, FRAME_SIZE)

    start_time = time.time()
    crossed_ids = set()
    all_track_ids = set()
    all_detections = 0
    confidences = []
    frames_processed = 0
    last_progress_at = 0.0
    pbar = tqdm.tqdm(total=frame_count)

    def _emit_progress(stage: str, force: bool = False) -> None:
        """Tell the caller how far counting has got. Throttled so a long
        video does not write a status update on every frame."""
        nonlocal last_progress_at
        if on_progress is None:
            return
        now = time.time()
        if not force and (now - last_progress_at) < 2.0:
            return
        last_progress_at = now
        elapsed = now - start_time
        # Unknown length stays 0 while counting, so the page keeps its slow
        # fallback instead of treating every frame as "finished".
        total = frame_count
        if stage != "counting" and total <= 0:
            total = frames_processed
        try:
            on_progress({
                "stage": stage,
                "frames_processed": frames_processed,
                "total_frames": total,
                "count": len(crossed_ids),
                "average_processing_fps": round(frames_processed / elapsed, 2) if elapsed > 0 else 0.0,
            })
        except Exception as e:
            print(f"[inference] progress callback failed: {e}", flush=True)

    x1, y1 = map(int, line[0])
    x2, y2 = map(int, line[1])

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.resize(img, FRAME_SIZE)
        results = model(img, stream=True)

        detections = []
        for r in results:
            for box in r.boxes:
                bx1, by1, bx2, by2 = box.xyxy[0]
                bx1, by1, bx2, by2 = int(bx1), int(by1), int(bx2), int(by2)
                w, h = bx2 - bx1, by2 - by1
                conf = math.ceil(box.conf[0] * 100) / 100
                cls = int(box.cls[0])
                current_class = CLASS_NAMES[cls]

                if conf > CONF_THRESHOLD:
                    detections.append((([bx1, by1, w, h]), conf, current_class))
                    confidences.append(conf)

        tracks = tracker.update_tracks(detections, frame=img)
        all_detections += len(detections)
        frames_processed += 1

        cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 2)

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = int(track.track_id)
            all_track_ids.add(track_id)
            ltrb = track.to_ltrb()
            tx1, ty1, tx2, ty2 = int(ltrb[0]), int(ltrb[1]), int(ltrb[2]), int(ltrb[3])
            w, h = tx2 - tx1, ty2 - ty1
            cx, cy = tx1 + w // 2, ty1 + h // 2

            # A plant is "on the line" when the line passes through its box
            # (at least a 30px band for very small plants).
            band = max(MIN_BAND_HALF_HEIGHT, h // 2)
            crossed = x1 <= cx <= x2 and y1 - band < cy < y2 + band
            if crossed and track_id not in crossed_ids:
                crossed_ids.add(track_id)

            box_color = (0, 200, 0) if crossed else (255, 180, 0)
            cv2.rectangle(img, (tx1, ty1), (tx2, ty2), box_color, 2)
            cv2.putText(
                img,
                f"ID {track_id}",
                (tx1, max(0, ty1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                box_color,
                2,
            )

        cv2.putText(
            img,
            f"Count: {len(crossed_ids)}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (0, 0, 255),
            3,
        )

        writer.write(img)
        pbar.update(1)
        _emit_progress("counting", force=(frame_count > 0 and frames_processed >= frame_count))

    pbar.close()
    _emit_progress("encoding", force=True)
    cap.release()
    writer.release()

    # Convert to MP4 using FFmpeg for maximum browser compatibility
    # This ensures H.264 codec and proper streaming support
    temp_output = output_path + ".tmp.mp4"
    if os.path.exists(output_path):
        os.rename(output_path, temp_output)
        if _convert_to_mp4_ffmpeg(temp_output, output_path):
            try:
                os.remove(temp_output)
            except Exception as e:
                print(f"[inference] Failed to clean up temp file: {e}", flush=True)
        else:
            # If FFmpeg conversion fails, try to use the original file
            print("[inference] FFmpeg conversion failed, attempting to use original output", flush=True)
            if os.path.exists(temp_output):
                try:
                    os.remove(output_path)
                    os.rename(temp_output, output_path)
                except Exception as e:
                    print(f"[inference] Failed to restore temp file: {e}", flush=True)

    processing_time = time.time() - start_time
    average_fps = frames_processed / processing_time if processing_time > 0 else 0.0
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
    duration_seconds = frames_processed / src_fps if src_fps else 0

    count = len(crossed_ids)
    error_rate = 0.0
    if ground_truth > 0:
        error_rate = abs(count - ground_truth) / ground_truth * 100

    return {
        "count": count,
        "metrics": {
            "plant_count": count,
            "video_duration": _format_duration(duration_seconds),
            "frames_processed": frames_processed,
            "processing_time": round(processing_time, 2),
            "average_processing_fps": round(average_fps, 2),
            "unique_track_ids": len(all_track_ids),
            "total_detections": all_detections,
            "average_detection_confidence": round(avg_confidence, 4),
            "model_version": model_version,
            "tracker_configuration": {
                "tracker_name": tracker_name,
                "version": "deep-sort-realtime",
                "config": TRACKER_CONFIG,
            },
            "counting_line_configuration": {
                "line_coordinates": line,
                "direction": direction,
            },
        },
        "experiment_benchmarks": {
            "ground_truth_count": ground_truth,
            "error_rate_percentage": round(error_rate, 2),
        },
    }
