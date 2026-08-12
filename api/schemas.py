"""
api/schemas.py
--------------
Pydantic request/response models for all TRUTHGUARD API endpoints.

Using Pydantic v2 syntax (FastAPI >= 0.100 ships with Pydantic v2 support).
All response models include full docstrings so the auto-generated /docs page
is self-explanatory.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared sub-models
# ---------------------------------------------------------------------------


class BaseDetectionResponse(BaseModel):
    """Fields common to every detection response."""

    verdict: str = Field(
        ...,
        description="Final verdict: 'real' or 'fake'.",
        examples=["fake"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the predicted verdict (0 = uncertain, 1 = certain).",
        examples=[0.9312],
    )
    real_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw probability assigned to the 'real' class.",
    )
    fake_prob: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Raw probability assigned to the 'fake' class.",
    )


# ---------------------------------------------------------------------------
# Image endpoint
# ---------------------------------------------------------------------------


class ImageDetectionResponse(BaseDetectionResponse):
    """
    Response from POST /detect/image.

    Contains the binary verdict, confidence scores, and a GradCAM heatmap
    overlaid on the original image encoded as base-64 PNG.
    """

    heatmap_b64: str = Field(
        ...,
        description=(
            "Base-64-encoded PNG of the GradCAM activation heatmap blended "
            "onto the original image.  Decode with base64.b64decode() or "
            "embed directly in an <img src='data:image/png;base64,...'> tag."
        ),
    )


# ---------------------------------------------------------------------------
# Video endpoint
# ---------------------------------------------------------------------------


class VideoDetectionResponse(BaseDetectionResponse):
    """
    Response from POST /detect/video.

    Includes both a video-level verdict and per-frame forgery scores,
    together with a timeline chart PNG.
    """

    per_frame_scores: list[float] = Field(
        ...,
        description=(
            "List of fake-confidence probabilities, one per sampled frame. "
            "Values close to 1.0 indicate suspected forged frames."
        ),
    )
    timeline_b64: str = Field(
        ...,
        description=(
            "Base-64-encoded PNG bar chart of per-frame fake-confidence scores. "
            "Red bars = suspected fake frames; green = suspected real."
        ),
    )
    num_frames: int = Field(
        ...,
        description="Number of frames sampled from the video for analysis.",
    )
    fake_frame_count: int = Field(
        ...,
        description="Number of sampled frames classified as fake.",
    )


# ---------------------------------------------------------------------------
# Audio endpoint
# ---------------------------------------------------------------------------


class AudioDetectionResponse(BaseDetectionResponse):
    """
    Response from POST /detect/audio.

    Includes the binary verdict and an attention heatmap overlaid on
    the mel-spectrogram.
    """

    attention_weights: list[float] = Field(
        ...,
        description=(
            "Per-timestep LSTM attention weights after normalisation to [0,1]. "
            "High values indicate time regions the model focused on most."
        ),
    )
    heatmap_b64: str = Field(
        ...,
        description=(
            "Base-64-encoded PNG showing the mel-spectrogram with attention "
            "weights overlaid as a heatmap row."
        ),
    )


# ---------------------------------------------------------------------------
# Combined (fusion) endpoint
# ---------------------------------------------------------------------------


class CombinedDetectionResponse(BaseModel):
    """
    Response from POST /detect/combined.

    The fusion layer independently runs both the video and audio modules,
    then applies explicit decision rules to produce a combined verdict.
    A MODALITY MISMATCH flag is raised when the two streams disagree —
    this pattern is a strong indicator of a targeted deepfake attack.
    """

    verdict: str = Field(
        ...,
        description="Combined verdict: 'real' or 'fake'.",
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Combined confidence score.",
    )
    risk_level: str = Field(
        ...,
        description=(
            "Categorical risk: 'LOW' | 'MEDIUM' | 'HIGH' | "
            "'HIGH RISK — MODALITY MISMATCH'."
        ),
    )
    mismatch: bool = Field(
        ...,
        description=(
            "True if the video and audio modules produced conflicting verdicts. "
            "This is a HIGH-RISK signal even if individual confidences are moderate."
        ),
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of the fusion decision.",
    )
    video_verdict: str = Field(..., description="Raw verdict from the video module.")
    video_confidence: float = Field(..., description="Raw confidence from the video module.")
    audio_verdict: str = Field(..., description="Raw verdict from the audio module.")
    audio_confidence: float = Field(..., description="Raw confidence from the audio module.")
    video_result: VideoDetectionResponse = Field(
        ...,
        description="Full video module result.",
    )
    audio_result: AudioDetectionResponse = Field(
        ...,
        description="Full audio module result.",
    )


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class HealthResponse(BaseModel):
    """Response from GET /health."""

    status: str = Field(default="ok")
    version: str = Field(default="1.0.0")
    modules_loaded: list[str] = Field(
        default_factory=list,
        description="List of successfully loaded detection modules.",
    )
