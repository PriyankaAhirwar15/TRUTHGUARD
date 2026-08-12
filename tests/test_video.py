"""
tests/test_video.py
--------------------
Unit tests for the video deepfake detection module.

Tests cover:
  1. VideoDeepfakeDetector architecture — forward pass shapes
  2. VideoPredictor — output keys, types, and per-frame score count
     (uses a synthetic video written with OpenCV, no real weights)
"""

import os
import sys
import tempfile
import numpy as np
import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import cv2


# ---------------------------------------------------------------------------
# Helper — write a tiny synthetic video
# ---------------------------------------------------------------------------


def _make_test_video(path: str, n_frames: int = 20, w: int = 64, h: int = 64) -> None:
    """Write a small coloured-noise video to *path* using OpenCV."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, 25.0, (w, h))
    for _ in range(n_frames):
        frame = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
        out.write(frame)
    out.release()


# ---------------------------------------------------------------------------
# 1. Model architecture tests
# ---------------------------------------------------------------------------


class TestVideoModel:
    def setup_method(self):
        from models.video.model import VideoDeepfakeDetector
        self.ModelClass = VideoDeepfakeDetector

    def test_video_logits_shape(self):
        """video_logits should be (B, 2)."""
        model = self.ModelClass(pretrained=False, num_frames=4, d_model=64, nhead=4)
        model.eval()
        dummy = torch.zeros(2, 4, 3, 224, 224)
        with torch.no_grad():
            v_logits, f_logits = model(dummy)
        assert v_logits.shape == (2, 2), f"video_logits shape: {v_logits.shape}"

    def test_frame_logits_shape(self):
        """frame_logits should be (B, T, 2)."""
        model = self.ModelClass(pretrained=False, num_frames=4, d_model=64, nhead=4)
        model.eval()
        dummy = torch.zeros(1, 4, 3, 224, 224)
        with torch.no_grad():
            v_logits, f_logits = model(dummy)
        assert f_logits.shape == (1, 4, 2), f"frame_logits shape: {f_logits.shape}"

    def test_single_frame_no_crash(self):
        """Model should handle T=1 (edge case)."""
        model = self.ModelClass(pretrained=False, num_frames=1, d_model=64, nhead=4)
        model.eval()
        dummy = torch.zeros(1, 1, 3, 224, 224)
        with torch.no_grad():
            v_logits, f_logits = model(dummy)
        assert v_logits.shape == (1, 2)

    def test_output_is_finite(self):
        model = self.ModelClass(pretrained=False, num_frames=4, d_model=64, nhead=4)
        model.eval()
        dummy = torch.randn(1, 4, 3, 224, 224)
        with torch.no_grad():
            v_logits, f_logits = model(dummy)
        assert torch.isfinite(v_logits).all()
        assert torch.isfinite(f_logits).all()


# ---------------------------------------------------------------------------
# 2. VideoPredictor end-to-end tests
# ---------------------------------------------------------------------------


class TestVideoPredictor:
    @pytest.fixture(autouse=True)
    def _create_temp_video(self, tmp_path):
        self.video_path = str(tmp_path / "test_video.mp4")
        _make_test_video(self.video_path, n_frames=30)

    def test_predict_returns_required_keys(self):
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        required = {
            "verdict", "confidence", "real_prob", "fake_prob",
            "per_frame_scores", "timeline_b64", "num_frames", "fake_frame_count"
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_per_frame_scores_length(self):
        """per_frame_scores should have exactly num_frames entries."""
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        assert len(result["per_frame_scores"]) == 4, (
            f"Expected 4 scores, got {len(result['per_frame_scores'])}"
        )

    def test_per_frame_scores_in_range(self):
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        for i, s in enumerate(result["per_frame_scores"]):
            assert 0.0 <= s <= 1.0, f"Frame {i} score out of range: {s}"

    def test_fake_frame_count_consistent(self):
        """fake_frame_count should equal count of scores >= 0.5."""
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        expected = sum(1 for s in result["per_frame_scores"] if s >= 0.5)
        assert result["fake_frame_count"] == expected

    def test_verdict_is_valid(self):
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        assert result["verdict"] in ("real", "fake")

    def test_probabilities_sum_to_one(self):
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        total = result["real_prob"] + result["fake_prob"]
        assert abs(total - 1.0) < 0.01, f"Probs sum to {total}"

    def test_timeline_b64_is_valid_png(self):
        import base64, io
        from PIL import Image
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        result = pred.predict(self.video_path)
        img = Image.open(io.BytesIO(base64.b64decode(result["timeline_b64"])))
        assert img.size[0] > 0

    def test_missing_video_raises(self):
        from models.video.predictor import VideoPredictor
        pred = VideoPredictor(pretrained=False, num_frames=4)
        with pytest.raises((FileNotFoundError, Exception)):
            pred.predict("/nonexistent/path/video.mp4")
