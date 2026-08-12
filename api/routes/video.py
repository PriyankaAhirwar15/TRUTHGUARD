"""
api/routes/video.py
--------------------
FastAPI router for POST /detect/video.

Accepts a video file upload, extracts frames, runs the dual-stream
VideoDeepfakeDetector, and returns per-frame scores + a video-level verdict
with a confidence timeline chart.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from api.schemas import VideoDetectionResponse
from utils.file_utils import VIDEO_EXTS

router = APIRouter()


@router.post(
    "/video",
    response_model=VideoDetectionResponse,
    summary="Video Deepfake Detection",
    description=(
        "Upload a video file (MP4, AVI, MOV, MKV, WEBM). "
        "Returns a video-level verdict, per-frame forgery scores, "
        "and a timeline bar chart (base-64 PNG)."
    ),
    tags=["Detection"],
)
async def detect_video(
    file: UploadFile = File(
        ..., description="Video file to analyse (MP4 / AVI / MOV / MKV / WEBM)."
    ),
):
    """
    Endpoint: POST /detect/video

    Pipeline
    --------
    1. Validate the uploaded file type.
    2. Save to a temporary file.
    3. Run VideoPredictor.predict() →
          { verdict, confidence, per_frame_scores, timeline_b64, ... }
    4. Delete the temp file.
    5. Return the VideoDetectionResponse.
    """
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(VIDEO_EXTS)}",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="tg_vid_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        from api.main import video_predictor

        result = video_predictor.predict(tmp_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return VideoDetectionResponse(**result)
