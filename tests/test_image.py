"""
tests/test_image.py
--------------------
Unit tests for the image deepfake detection module.

Tests cover:
  1. Model architecture — forward pass shape, GroupNorm replacement
  2. GradCAM — heatmap shape and value range
  3. ImagePredictor — output keys, types, and value ranges
     (uses a real temporary image, no weights loaded)
"""

import os
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image

# ── Ensure project root is on path ───────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Model architecture tests
# ---------------------------------------------------------------------------


class TestImageModel:
    def setup_method(self):
        from models.image.model import ImageDeepfakeDetector, _replace_bn_with_gn
        self.ModelClass = ImageDeepfakeDetector
        self.replace_fn = _replace_bn_with_gn

    def test_forward_shape(self):
        """Model should output (1, 2) logits for a single image."""
        model = self.ModelClass(pretrained=False)
        model.eval()
        dummy = torch.zeros(1, 3, 380, 380)
        with torch.no_grad():
            out = model(dummy)
        assert out.shape == (1, 2), f"Expected (1, 2), got {out.shape}"

    def test_batch_size_1_stability(self):
        """With GroupNorm replacing BatchNorm, batch-size-1 should not raise."""
        model = self.ModelClass(pretrained=False)
        model.eval()
        dummy = torch.randn(1, 3, 380, 380)
        with torch.no_grad():
            out = model(dummy)
        # Should not raise; check output is finite
        assert torch.isfinite(out).all(), "Model output contains NaN or Inf"

    def test_no_batchnorm_in_backbone(self):
        """All BatchNorm2d layers should have been replaced with GroupNorm."""
        model = self.ModelClass(pretrained=False)
        for name, module in model.backbone.named_modules():
            assert not isinstance(module, nn.BatchNorm2d), (
                f"BatchNorm2d still present at: {name}"
            )

    def test_output_logits_not_probabilities(self):
        """Raw output should NOT sum to 1 (they are logits, not softmaxed)."""
        model = self.ModelClass(pretrained=False)
        model.eval()
        dummy = torch.randn(2, 3, 380, 380)
        with torch.no_grad():
            out = model(dummy)
        row_sums = out.sum(dim=1)
        # If these were softmax probabilities they'd sum to 1; logits won't
        assert not torch.allclose(row_sums, torch.ones(2), atol=0.01), (
            "Output looks like softmax probabilities, expected raw logits"
        )


# ---------------------------------------------------------------------------
# 2. GradCAM tests
# ---------------------------------------------------------------------------


class TestGradCAM:
    def setup_method(self):
        from models.image.model import ImageDeepfakeDetector
        from models.image.gradcam import GradCAM
        self.model = ImageDeepfakeDetector(pretrained=False)
        self.model.eval()
        self.GradCAM = GradCAM

    def test_heatmap_shape(self):
        """GradCAM should return a 2-D float array."""
        target = self.model.backbone.blocks[-1]
        gradcam = self.GradCAM(self.model, target)
        dummy = torch.randn(1, 3, 380, 380)
        heatmap = gradcam.generate(dummy, target_class=1)
        assert heatmap.ndim == 2, f"Expected 2-D heatmap, got {heatmap.ndim}-D"
        gradcam.remove_hooks()

    def test_heatmap_range(self):
        """GradCAM heatmap should be normalised to [0, 1]."""
        target = self.model.backbone.blocks[-1]
        gradcam = self.GradCAM(self.model, target)
        dummy = torch.randn(1, 3, 380, 380)
        heatmap = gradcam.generate(dummy, target_class=1)
        assert heatmap.min() >= 0.0 - 1e-6, "Heatmap min is below 0"
        assert heatmap.max() <= 1.0 + 1e-6, "Heatmap max is above 1"
        gradcam.remove_hooks()

    def test_overlay_returns_uint8_rgb(self):
        """overlay() should return an HxWx3 uint8 array."""
        target = self.model.backbone.blocks[-1]
        gradcam = self.GradCAM(self.model, target)
        original = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        dummy = torch.randn(1, 3, 380, 380)
        blended = gradcam.overlay(original, dummy)
        assert blended.dtype == np.uint8, "Expected uint8 output"
        assert blended.shape == (224, 224, 3), f"Wrong shape: {blended.shape}"
        gradcam.remove_hooks()


# ---------------------------------------------------------------------------
# 3. ImagePredictor end-to-end test
# ---------------------------------------------------------------------------


class TestImagePredictor:
    @pytest.fixture(autouse=True)
    def _create_temp_image(self, tmp_path):
        """Write a small random RGB PNG to disk for testing."""
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        self.img_path = str(tmp_path / "test_image.jpg")
        img.save(self.img_path)

    def test_predict_returns_required_keys(self):
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        required = {"verdict", "confidence", "real_prob", "fake_prob", "heatmap_b64"}
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_verdict_is_valid(self):
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        assert result["verdict"] in ("real", "fake"), (
            f"Unexpected verdict: {result['verdict']}"
        )

    def test_probabilities_sum_to_one(self):
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        total = result["real_prob"] + result["fake_prob"]
        assert abs(total - 1.0) < 0.01, f"Probabilities sum to {total}, expected ~1.0"

    def test_confidence_in_range(self):
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        assert 0.0 <= result["confidence"] <= 1.0, (
            f"Confidence out of range: {result['confidence']}"
        )

    def test_heatmap_b64_is_nonempty_string(self):
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        assert isinstance(result["heatmap_b64"], str), "heatmap_b64 should be a string"
        assert len(result["heatmap_b64"]) > 100, "heatmap_b64 looks empty"

    def test_heatmap_b64_is_valid_png(self):
        """Decoded base-64 should be a valid PIL-readable PNG."""
        import base64
        from models.image.predictor import ImagePredictor
        pred = ImagePredictor(pretrained=False)
        result = pred.predict(self.img_path)
        img_bytes = base64.b64decode(result["heatmap_b64"])
        img = Image.open(__import__("io").BytesIO(img_bytes))
        assert img.mode in ("RGB", "RGBA"), f"Unexpected image mode: {img.mode}"
