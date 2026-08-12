"""
fusion/fusion.py
-----------------
Cross-Modal Fusion Layer — the decision engine of TRUTHGUARD.

This module compares the independent verdicts from the video and audio
modules and produces a combined risk assessment.

Design Philosophy
-----------------
The fusion layer is intentionally a RULE-BASED decision system, NOT a
learned model.  This choice is deliberate:

  1. Transparency — the logic is auditable and explainable in plain English.
  2. Robustness   — no additional training data is required.
  3. Sensitivity  — a modality mismatch is treated as a HIGH-RISK signal
                    on its own, regardless of individual confidence scores.
                    This is because real deepfake attacks often target only
                    one modality (e.g. voice-cloned audio over real video).

Fusion Rules (documented as code comments)
-------------------------------------------
Case A: Both modules agree on "real"
    → verdict = "real", risk = "LOW", confidence = weighted average.

Case B: Both modules agree on "fake"
    → verdict = "fake", risk = "HIGH", confidence = weighted average.

Case C: Video says "real", Audio says "fake"  (or vice versa)
    → verdict = "fake"  (conservative — always flag when unsure)
    → risk    = "HIGH RISK — MODALITY MISMATCH"
    → This mismatch pattern is a strong fraud indicator:
      an attacker may have cloned the voice but not the face (or vice versa).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

Verdict    = Literal["real", "fake"]
RiskLevel  = Literal[
    "LOW",
    "MEDIUM",
    "HIGH",
    "HIGH RISK — MODALITY MISMATCH",
]


@dataclass
class FusionResult:
    """
    The unified output of the cross-modal fusion layer.

    Attributes
    ----------
    verdict          : final combined verdict
    confidence       : combined confidence score in [0, 1]
    risk_level       : categorical risk assessment
    video_verdict    : raw verdict from the video module
    video_confidence : raw confidence from the video module
    audio_verdict    : raw verdict from the audio module
    audio_confidence : raw confidence from the audio module
    explanation      : human-readable explanation of the fusion decision
    mismatch         : True if the two modalities disagreed
    """
    verdict:          Verdict
    confidence:       float
    risk_level:       RiskLevel
    video_verdict:    Verdict
    video_confidence: float
    audio_verdict:    Verdict
    audio_confidence: float
    explanation:      str
    mismatch:         bool

    def to_dict(self) -> dict:
        return {
            "verdict":          self.verdict,
            "confidence":       round(self.confidence, 4),
            "risk_level":       self.risk_level,
            "mismatch":         self.mismatch,
            "explanation":      self.explanation,
            "video_verdict":    self.video_verdict,
            "video_confidence": round(self.video_confidence, 4),
            "audio_verdict":    self.audio_verdict,
            "audio_confidence": round(self.audio_confidence, 4),
        }


# ---------------------------------------------------------------------------
# Weights for confidence averaging
# ---------------------------------------------------------------------------

# The video module generally has higher fidelity on face-swap attacks, while
# audio is more reliable for voice-clone detection.  These weights balance
# the two when verdicts agree.  They are tunable without retraining.
VIDEO_WEIGHT = 0.55
AUDIO_WEIGHT = 0.45


def _weighted_confidence(
    video_conf: float,
    audio_conf: float,
    video_w: float = VIDEO_WEIGHT,
    audio_w: float = AUDIO_WEIGHT,
) -> float:
    """Return the weighted average of two confidence scores."""
    return video_w * video_conf + audio_w * audio_conf


# ---------------------------------------------------------------------------
# Main fusion function
# ---------------------------------------------------------------------------


def fuse(
    video_verdict:    Verdict,
    video_confidence: float,
    audio_verdict:    Verdict,
    audio_confidence: float,
) -> FusionResult:
    """
    Apply the cross-modal fusion decision rules.

    Parameters
    ----------
    video_verdict    : "real" | "fake" — verdict from the video module
    video_confidence : float in [0, 1] — confidence from the video module
    audio_verdict    : "real" | "fake" — verdict from the audio module
    audio_confidence : float in [0, 1] — confidence from the audio module

    Returns
    -------
    FusionResult dataclass (see above for field descriptions)
    """
    mismatch = video_verdict != audio_verdict

    # ------------------------------------------------------------------
    # CASE A — Both agree: REAL
    # ------------------------------------------------------------------
    if not mismatch and video_verdict == "real":
        combined_conf = _weighted_confidence(video_confidence, audio_confidence)
        # Downgrade confidence slightly if either module was uncertain
        risk = "LOW" if combined_conf >= 0.70 else "MEDIUM"
        return FusionResult(
            verdict          = "real",
            confidence       = combined_conf,
            risk_level       = risk,
            video_verdict    = video_verdict,
            video_confidence = video_confidence,
            audio_verdict    = audio_verdict,
            audio_confidence = audio_confidence,
            explanation      = (
                f"Both the visual stream and the audio stream independently classify "
                f"this content as REAL (video confidence: {video_confidence:.0%}, "
                f"audio confidence: {audio_confidence:.0%}). "
                f"Combined weighted confidence: {combined_conf:.0%}."
            ),
            mismatch = False,
        )

    # ------------------------------------------------------------------
    # CASE B — Both agree: FAKE
    # ------------------------------------------------------------------
    if not mismatch and video_verdict == "fake":
        combined_conf = _weighted_confidence(video_confidence, audio_confidence)
        return FusionResult(
            verdict          = "fake",
            confidence       = combined_conf,
            risk_level       = "HIGH",
            video_verdict    = video_verdict,
            video_confidence = video_confidence,
            audio_verdict    = audio_verdict,
            audio_confidence = audio_confidence,
            explanation      = (
                f"Both the visual stream and the audio stream independently classify "
                f"this content as FAKE (video confidence: {video_confidence:.0%}, "
                f"audio confidence: {audio_confidence:.0%}). "
                f"Combined weighted confidence: {combined_conf:.0%}. "
                f"High-confidence multi-modal forgery detected."
            ),
            mismatch = False,
        )

    # ------------------------------------------------------------------
    # CASE C — MODALITY MISMATCH (the most dangerous case)
    # ------------------------------------------------------------------
    #
    # One modality looks real while the other looks fake.
    # This is a strong indicator of a targeted deepfake attack where
    # only one modality was manipulated (e.g. voice-clone over real video,
    # or face-swap with original audio).
    #
    # Decision: ALWAYS return "fake" (conservative) and raise maximum alert.
    # Confidence = the higher of the two module confidences.
    #
    combined_conf = max(video_confidence, audio_confidence)

    if video_verdict == "fake":
        # Video flagged fake, audio said real
        detail = (
            f"The visual analysis flagged the video as FAKE "
            f"(confidence {video_confidence:.0%}), but the audio analysis "
            f"classified the voice as REAL (confidence {audio_confidence:.0%})."
        )
    else:
        # Audio flagged fake, video said real
        detail = (
            f"The audio analysis flagged the voice as FAKE "
            f"(confidence {audio_confidence:.0%}), but the visual analysis "
            f"classified the video as REAL (confidence {video_confidence:.0%})."
        )

    return FusionResult(
        verdict          = "fake",
        confidence       = combined_conf,
        risk_level       = "HIGH RISK — MODALITY MISMATCH",
        video_verdict    = video_verdict,
        video_confidence = video_confidence,
        audio_verdict    = audio_verdict,
        audio_confidence = audio_confidence,
        explanation      = (
            f"⚠️  MODALITY MISMATCH DETECTED. {detail} "
            f"When visual and audio authenticity signals disagree, this is a "
            f"strong indicator of a sophisticated deepfake attack where only "
            f"one modality was manipulated. TRUTHGUARD conservatively classifies "
            f"the content as FAKE. Do NOT trust this content."
        ),
        mismatch = True,
    )
