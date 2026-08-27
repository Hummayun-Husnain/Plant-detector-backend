"""Agronomic reporting pipeline: keyframe extraction, Groq Vision, PDF, S3."""

import base64
import io
import json
import os
import shutil
import tempfile
from datetime import datetime
from typing import List

import cv2
from groq import Groq
from PIL import Image
from pydantic import BaseModel, Field

from pdf_generator import create_pdf
from s3_utils import upload_file, generate_presigned_url

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_VISION_MODEL", "llama-3.2-11b-vision-preview")


class AgronomicInsights(BaseModel):
    executive_summary: str = Field(..., description="High-level agronomic overview and context.")
    plant_health_assessment: str = Field(..., description="Observed plant health observations.")
    nutrient_deficiencies: str = Field(..., description="Signs of nutrient deficiencies.")
    stress_and_pest_indicators: str = Field(..., description="Disease, pest, drought, or other stress signs.")
    actionable_recommendations: str = Field(..., description="Practical next steps and treatment recommendations.")


DEFAULT_INSIGHTS = {
    "executive_summary": (
        "Automated plant counting completed successfully. The AI agronomic vision analysis "
        "could not be completed due to an API or rate-limiting issue. The PDF contains the "
        "keyframe summary and the detected plant count."
    ),
    "plant_health_assessment": "Unable to assess plant health at this time.",
    "nutrient_deficiencies": "No nutrient deficiency analysis available.",
    "stress_and_pest_indicators": "No stress or pest indicators available.",
    "actionable_recommendations": (
        "Review the keyframes and plant count manually. Re-run the report once the "
        "vision service is available."
    ),
}


SYSTEM_PROMPT = (
    "You are an expert agricultural nutritionist and agronomist. Based on these representative plant images and the detected plant count, provide structured and actionable insights about plant health, growth, possible nutrient deficiencies, stress indicators, disease/pest signs, and recommendations."
)


def extract_keyframes(video_path: str, num_frames: int = 5) -> List[Image.Image]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    if num_frames <= 1:
        indices = [0]
    else:
        step = (total_frames - 1) / (num_frames - 1)
        indices = [int(round(i * step)) for i in range(num_frames)]

    frames: List[Image.Image] = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(Image.fromarray(frame))

    cap.release()
    return frames


def _pil_to_base64(img: Image.Image, quality: int = 85) -> str:
    rgb = img.convert("RGB")
    rgb.thumbnail((1024, 1024), Image.LANCZOS)
    buffer = io.BytesIO()
    rgb.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _build_messages(images_b64: List[str], plant_count: int):
    user_text = (
        f"Total detected plant count: {plant_count}.\n\n"
        "Analyze the attached video keyframes and return a single JSON object "
        "with exactly the following five keys. Each value must be a detailed "
        "descriptive paragraph suitable for a professional agronomic PDF report.\n\n"
        "Required JSON schema:\n"
        '{\n'
        '  "executive_summary": "Concise high-level agronomic overview and key takeaways.",\n'
        '  "plant_health_assessment": "Overall plant health, vigor, canopy, and stand evaluation.",\n'
        '  "nutrient_deficiencies": "Observed or suspected nutrient deficiencies and visual symptoms.",\n'
        '  "stress_and_pest_indicators": "Disease, pest pressure, drought, lodging, or other stress signs.",\n'
        '  "actionable_recommendations": "Specific, practical next steps for the grower."\n'
        "}\n\n"
        "Do not include any text outside the JSON object."
    )

    image_parts = [
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}
        for b64 in images_b64
    ]

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [{"type": "text", "text": user_text}] + image_parts,
        },
    ]


def call_groq_vision(images: List[Image.Image], plant_count: int) -> dict:
    if not GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY is not set.")

    client = Groq(api_key=GROQ_API_KEY)
    images_b64 = [_pil_to_base64(img) for img in images]
    messages = _build_messages(images_b64, plant_count)

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        response_format={"type": "json_object"},
        temperature=0.2,
        max_completion_tokens=2048,
    )

    content = response.choices[0].message.content
    print(f"[agronomy_reporter] Groq raw response: {content[:500]}", flush=True)
    if not content:
        raise RuntimeError("Groq returned empty content.")

    data = json.loads(content)
    insights = AgronomicInsights(**data)
    return insights.model_dump()


def _generate_report_pdf(
    job_id: str,
    plant_count: int,
    insights: dict,
    keyframes: List[Image.Image],
) -> str:
    report_dir = tempfile.mkdtemp()
    pdf_path = os.path.join(report_dir, f"agronomic_report_{job_id}.pdf")
    create_pdf(pdf_path, job_id, plant_count, insights, keyframes)
    return pdf_path


def generate_agronomic_report(
    video_path: str,
    plant_count: int,
    job_id: str,
    num_frames: int = 5,
) -> dict:
    keyframes = extract_keyframes(video_path, num_frames)
    if not keyframes:
        raise RuntimeError(f"No keyframes could be extracted from {video_path}")

    try:
        insights = call_groq_vision(keyframes, plant_count)
        processing_status = "done"
    except Exception as e:
        print(f"[agronomy_reporter] Groq Vision failed: {e}", flush=True)
        insights = dict(DEFAULT_INSIGHTS)
        processing_status = "report_failed"

    pdf_path = _generate_report_pdf(job_id, plant_count, insights, keyframes)
    report_key = f"reports/{job_id}.pdf"
    report_url = None

    try:
        upload_file(pdf_path, report_key, content_type="application/pdf")
        report_url = generate_presigned_url(
            report_key,
            "get_object",
            604800,
            content_type="application/pdf",
        )
    except Exception as e:
        print(f"[agronomy_reporter] PDF upload failed: {e}", flush=True)
        report_key = None
    finally:
        try:
            shutil.rmtree(os.path.dirname(pdf_path), ignore_errors=True)
        except Exception:
            pass

    return {
        "report_key": report_key,
        "report_url": report_url,
        "insights": insights,
        "processing_status": processing_status,
    }
