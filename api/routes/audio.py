"""
api/routes/audio.py
--------------------
FastAPI router for POST /detect/audio.

Accepts an audio file upload, computes the mel-spectrogram, runs the
dual-stream AudioDeepfakeDetector, and returns verdict + attention heatmap.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from api.schemas import AudioDetectionResponse
from utils.file_utils import AUDIO_EXTS

router = APIRouter()


@router.post(
    "/audio",
    response_model=AudioDetectionResponse,
    summary="Audio Deepfake / Voice-Clone Detection",
    description=(
        "Upload an audio file (WAV, MP3, FLAC, OGG, M4A, AAC). "
        "Returns a real/fake verdict and a mel-spectrogram with "
        "LSTM attention weights overlaid (base-64 PNG)."
    ),
    tags=["Detection"],
)
async def detect_audio(
    file: UploadFile = File(
        ..., description="Audio file to analyse (WAV / MP3 / FLAC / OGG / M4A / AAC)."
    ),
):
    """
    Endpoint: POST /detect/audio

    Pipeline
    --------
    1. Validate the uploaded file type.
    2. Save to a temporary file.
    3. Run AudioPredictor.predict() →
          { verdict, confidence, attention_weights, heatmap_b64 }
    4. Delete the temp file.
    5. Return the AudioDetectionResponse.
    """
    suffix = Path(file.filename or "upload.wav").suffix.lower()
    if suffix not in AUDIO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(AUDIO_EXTS)}",
        )

    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="tg_aud_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        from api.main import audio_predictor

        result = audio_predictor.predict(tmp_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return AudioDetectionResponse(**result)
