"""
api/routes/combined.py
-----------------------
FastAPI router for POST /detect/combined.

This is the flagship endpoint of TRUTHGUARD.

It accepts a video file that contains an embedded audio track (e.g. a video
call recording or social-media video), runs BOTH the video and audio modules
independently, then passes their results to the cross-modal fusion layer.

Key behaviour: if the video verdict and the audio verdict DISAGREE
(one says real, the other says fake), the fusion layer raises a
"HIGH RISK — MODALITY MISMATCH" flag.  This pattern is a strong indicator
of a targeted deepfake attack where only one modality was manipulated.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from api.schemas import CombinedDetectionResponse, VideoDetectionResponse, AudioDetectionResponse
from utils.file_utils import VIDEO_EXTS
from fusion import fuse

router = APIRouter()


@router.post(
    "/combined",
    response_model=CombinedDetectionResponse,
    summary="Cross-Modal Fusion: Video + Audio Deepfake Detection",
    description=(
        "Upload a video file with an embedded audio track. "
        "Runs independent video and audio deepfake analysis, then applies "
        "the cross-modal fusion decision rules.  "
        "Raises a 'HIGH RISK — MODALITY MISMATCH' flag if the visual and "
        "audio authenticity signals disagree — a hallmark of targeted deepfake attacks."
    ),
    tags=["Detection", "Fusion"],
)
async def detect_combined(
    file: UploadFile = File(
        ...,
        description=(
            "Video file containing an audio track to analyse "
            "(MP4 / AVI / MOV / MKV / WEBM).  "
            "The audio track is extracted automatically."
        ),
    ),
):
    """
    Endpoint: POST /detect/combined

    Pipeline
    --------
    1. Validate the uploaded file type (must be video).
    2. Save video to a temporary file.
    3. Extract audio from the video using ffmpeg → separate temp WAV file.
    4. Run VideoPredictor.predict()  → video_result dict.
    5. Run AudioPredictor.predict()  → audio_result dict.
    6. Call fusion.fuse() with both verdicts → FusionResult.
    7. Assemble and return CombinedDetectionResponse.
    8. Clean up all temp files.

    Fusion Rules (short form — see fusion/fusion.py for full documentation)
    -----------------------------------------------------------------------
    - Both real  → verdict = real,  risk = LOW / MEDIUM
    - Both fake  → verdict = fake,  risk = HIGH
    - Mismatch   → verdict = fake,  risk = HIGH RISK — MODALITY MISMATCH
    """
    # ── Validate file type ────────────────────────────────────────────────
    suffix = Path(file.filename or "upload.mp4").suffix.lower()
    if suffix not in VIDEO_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. This endpoint requires a video file.",
        )

    tmp_video = tmp_audio = None

    try:
        # ── Save video upload ─────────────────────────────────────────────
        fd, tmp_video = tempfile.mkstemp(suffix=suffix, prefix="tg_comb_")
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        # ── Extract audio track → WAV ─────────────────────────────────────
        _, tmp_audio = tempfile.mkstemp(suffix=".wav", prefix="tg_audio_")
        _extract_audio(tmp_video, tmp_audio)

        # ── Run both predictors ───────────────────────────────────────────
        from api.main import video_predictor, audio_predictor

        video_result = video_predictor.predict(tmp_video)
        audio_result = audio_predictor.predict(tmp_audio)

        # ── Cross-modal fusion ────────────────────────────────────────────
        fusion_result = fuse(
            video_verdict    = video_result["verdict"],
            video_confidence = video_result["confidence"],
            audio_verdict    = audio_result["verdict"],
            audio_confidence = audio_result["confidence"],
        )

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        for p in [tmp_video, tmp_audio]:
            if p and os.path.exists(p):
                os.remove(p)

    return CombinedDetectionResponse(
        **fusion_result.to_dict(),
        video_result=VideoDetectionResponse(**video_result),
        audio_result=AudioDetectionResponse(**audio_result),
    )


# ---------------------------------------------------------------------------
# Audio extraction helper
# ---------------------------------------------------------------------------


def _extract_audio(video_path: str, output_wav: str) -> None:
    """
    Extract the audio track from *video_path* and save as a 16 kHz mono WAV
    at *output_wav* using ffmpeg.

    Falls back to a silent WAV if ffmpeg is not available (so the system
    still runs on environments without ffmpeg — audio result will be
    low-confidence but the endpoint won't crash).
    """
    import subprocess
    import shutil

    if shutil.which("ffmpeg") is None:
        # ffmpeg not available — write a 4-second silent WAV
        _write_silent_wav(output_wav, duration=4, sr=16_000)
        return

    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn",                    # no video
        "-acodec", "pcm_s16le",   # 16-bit PCM
        "-ar", "16000",           # 16 kHz
        "-ac", "1",               # mono
        output_wav,
    ]
    try:
        subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=60,
        )
    except subprocess.CalledProcessError:
        # Video may have no audio — write silent WAV
        _write_silent_wav(output_wav, duration=4, sr=16_000)


def _write_silent_wav(path: str, duration: float = 4.0, sr: int = 16_000) -> None:
    """Write a silent PCM WAV file to *path* (fallback when no audio track)."""
    import numpy as np
    import soundfile as sf

    silence = np.zeros(int(sr * duration), dtype=np.float32)
    sf.write(path, silence, sr)
