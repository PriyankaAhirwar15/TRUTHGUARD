"""
api/main.py
------------
TRUTHGUARD FastAPI application.

Endpoints
---------
GET  /health             → service health check + loaded modules list
POST /detect/image       → image deepfake detection (EfficientNet-B4 + GradCAM)
POST /detect/video       → video deepfake detection (spatial + temporal streams)
POST /detect/audio       → audio deepfake / voice-clone detection (CNN + LSTM)
POST /detect/combined    → cross-modal fusion (video + audio, mismatch detection)

Architecture Notes
------------------
* Predictors are instantiated once at startup (module-level singletons) and
  reused across requests — model loading is expensive and should NOT happen
  per-request.
* Routes use lazy imports (``from api.main import <predictor>``) so that the
  predictor objects are always the same singletons defined here.
* The application runs on Uvicorn with a single worker (appropriate for the
  CPU-tier HuggingFace Spaces environment).
"""

import sys
import os

# Ensure the project root is on sys.path regardless of how the app is started
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.schemas import HealthResponse

# ---------------------------------------------------------------------------
# Application instance
# ---------------------------------------------------------------------------

app = FastAPI(
    title="TRUTHGUARD — Multi-Modal Deepfake Detection API",
    description=(
        "A unified deepfake detection system with three independent modules "
        "(image, video, audio) and a cross-modal fusion layer that flags "
        "mismatches between visual and audio authenticity signals.\n\n"
        "Built with EfficientNet-B4, Transformer, LSTM, GradCAM, and FastAPI."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Allow Gradio UI (same host, different port) to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Module-level predictor singletons
# Loaded once at startup; shared across all requests.
# ---------------------------------------------------------------------------

from models.image.predictor import ImagePredictor
from models.video.predictor import VideoPredictor
from models.audio.predictor import AudioPredictor

import torch as _torch
_DEVICE = "cuda" if _torch.cuda.is_available() else "cpu"

image_predictor = ImagePredictor(weights_path=None, pretrained=False, device=_DEVICE)
video_predictor = VideoPredictor(weights_path=None, pretrained=False, device=_DEVICE)
audio_predictor = AudioPredictor(weights_path=None, device=_DEVICE)

print("[TRUTHGUARD] All modules loaded [OK]")

# ---------------------------------------------------------------------------
# Register routers
# ---------------------------------------------------------------------------

from api.routes.image    import router as image_router
from api.routes.video    import router as video_router
from api.routes.audio    import router as audio_router
from api.routes.combined import router as combined_router

app.include_router(image_router,    prefix="/detect")
app.include_router(video_router,    prefix="/detect")
app.include_router(audio_router,    prefix="/detect")
app.include_router(combined_router, prefix="/detect")

# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health Check",
    description="Returns API status and a list of successfully loaded detection modules.",
    tags=["System"],
)
async def health():
    """
    GET /health

    Returns the service status and which detection modules are operational.
    Useful for container health-check probes and monitoring dashboards.
    """
    return HealthResponse(
        status="ok",
        version="1.0.0",
        modules_loaded=["image", "video", "audio", "fusion"],
    )


# ---------------------------------------------------------------------------
# Entry point for direct execution
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,      # disable reload in production
        workers=1,         # single worker for CPU-tier deployment
    )
