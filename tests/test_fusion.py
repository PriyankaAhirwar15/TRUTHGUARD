"""
tests/test_fusion.py
---------------------
Unit tests for Module 4 — Cross-Modal Fusion Layer.

Tests cover all three fusion decision rules:
  Case A: Both real -> combined real (LOW/MEDIUM risk)
  Case B: Both fake -> combined fake (HIGH risk)
  Case C: Mismatch  -> combined fake (HIGH RISK — MODALITY MISMATCH)
"""

import os
import sys
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from fusion import fuse


def test_both_real_high_confidence():
    """When both modules say REAL with high confidence -> combined REAL, LOW risk."""
    res = fuse(
        video_verdict="real", video_confidence=0.95,
        audio_verdict="real", audio_confidence=0.90
    )
    assert res.verdict == "real"
    assert res.risk_level == "LOW"
    assert res.mismatch is False
    assert 0.90 <= res.confidence <= 0.95


def test_both_real_low_confidence():
    """When both say REAL but low confidence -> combined REAL, MEDIUM risk."""
    res = fuse(
        video_verdict="real", video_confidence=0.60,
        audio_verdict="real", audio_confidence=0.55
    )
    assert res.verdict == "real"
    assert res.risk_level == "MEDIUM"
    assert res.mismatch is False


def test_both_fake():
    """When both say FAKE -> combined FAKE, HIGH risk."""
    res = fuse(
        video_verdict="fake", video_confidence=0.85,
        audio_verdict="fake", audio_confidence=0.88
    )
    assert res.verdict == "fake"
    assert res.risk_level == "HIGH"
    assert res.mismatch is False


def test_mismatch_video_fake_audio_real():
    """Video FAKE + Audio REAL -> FAKE with HIGH RISK — MODALITY MISMATCH."""
    res = fuse(
        video_verdict="fake", video_confidence=0.80,
        audio_verdict="real", audio_confidence=0.92
    )
    assert res.verdict == "fake"
    assert res.risk_level == "HIGH RISK — MODALITY MISMATCH"
    assert res.mismatch is True
    assert res.confidence == 0.92  # max confidence
    assert "MODALITY MISMATCH" in res.explanation


def test_mismatch_video_real_audio_fake():
    """Video REAL + Audio FAKE -> FAKE with HIGH RISK — MODALITY MISMATCH."""
    res = fuse(
        video_verdict="real", video_confidence=0.91,
        audio_verdict="fake", audio_confidence=0.84
    )
    assert res.verdict == "fake"
    assert res.risk_level == "HIGH RISK — MODALITY MISMATCH"
    assert res.mismatch is True
    assert res.confidence == 0.91  # max confidence
    assert "MODALITY MISMATCH" in res.explanation


def test_to_dict_structure():
    """Check dictionary serialization."""
    res = fuse("real", 0.9, "real", 0.9)
    d = res.to_dict()
    assert d["verdict"] == "real"
    assert "video_confidence" in d
    assert "audio_confidence" in d
