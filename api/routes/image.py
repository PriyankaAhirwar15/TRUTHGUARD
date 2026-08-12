"""
api/routes/image.py
--------------------
FastAPI router for POST /detect/image.

Accepts a single image file upload, runs it through the ImagePredictor,
and returns a verdict, confidence scores, and a GradCAM heatmap overlay.
"""

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, UploadFile, HTTPException

from api.schemas import ImageDetectionResponse
from utils.file_utils import detect_file_type, IMAGE_EXTS

router = APIRouter()


@router.post(
    "/image",
    response_model=ImageDetectionResponse,
    summary="Image Deepfake Detection",
    description=(
        "Upload an image file (JPEG, PNG, BMP, WEBP, TIFF). "
        "Returns a real/fake verdict, per-class probabilities, "
        "and a GradCAM heatmap PNG overlaid on the original image "
        "(base-64 encoded)."
    ),
    tags=["Detection"],
)
async def detect_image(
    file: UploadFile = File(
        ..., description="Image file to analyse (JPEG / PNG / BMP / WEBP / TIFF)."
    ),
):
    """
    Endpoint: POST /detect/image

    Pipeline
    --------
    1. Validate the uploaded file type.
    2. Save to a temporary file.
    3. Run ImagePredictor.predict() → { verdict, confidence, heatmap_b64 }.
    4. Delete the temp file.
    5. Return the ImageDetectionResponse.
    """
    # ── Validate file type ────────────────────────────────────────────────
    suffix = Path(file.filename or "upload.jpg").suffix.lower()
    if suffix not in IMAGE_EXTS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Supported: {sorted(IMAGE_EXTS)}",
        )

    # ── Save upload to a temp file ────────────────────────────────────────
    fd, tmp_path = tempfile.mkstemp(suffix=suffix, prefix="tg_img_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(await file.read())

        # ── Import predictor (lazy — avoids loading weights at import time) ──
        from api.main import image_predictor

        result = image_predictor.predict(tmp_path)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return ImageDetectionResponse(**result)
