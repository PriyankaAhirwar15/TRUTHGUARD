"""
tests/test_audio.py
--------------------
Unit tests for the audio deepfake detection module.

Tests cover:
  1. AudioDeepfakeDetector architecture — forward pass shapes and attention weights
  2. Spectrogram generation & normalization
  3. AudioPredictor end-to-end inference (uses a synthetic WAV file)
"""

import os
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch


# ---------------------------------------------------------------------------
# Helper — write a synthetic WAV audio file
# ---------------------------------------------------------------------------


def _make_test_wav(path: str, duration_sec: float = 3.0, sr: int = 16000) -> None:
    """Generate a sine wave + noise audio file for testing."""
    t = np.linspace(0, duration_sec, int(sr * duration_sec), endpoint=False)
    # Sine wave (440 Hz) + random noise
    audio = 0.5 * np.sin(2 * np.pi * 440 * t) + 0.1 * np.random.randn(len(t))
    sf.write(path, audio.astype(np.float32), sr)


# ---------------------------------------------------------------------------
# 1. Model architecture tests
# ---------------------------------------------------------------------------


class TestAudioModel:
    def setup_method(self):
        from models.audio.model import AudioDeepfakeDetector
        self.ModelClass = AudioDeepfakeDetector

    def test_forward_logits_and_attention_shape(self):
        """Model should return (B, 2) logits and (B, T') attention weights."""
        model = self.ModelClass(n_mels=128, cnn_dim=128, lstm_hidden=128)
        model.eval()
        dummy_spec = torch.randn(2, 1, 128, 125)  # (B, 1, n_mels, T)
        with torch.no_grad():
            logits, attn_w = model(dummy_spec)
        assert logits.shape == (2, 2), f"Logits shape error: {logits.shape}"
        assert attn_w.shape[0] == 2, f"Batch size mismatch in attn: {attn_w.shape}"
        assert attn_w.ndim == 2, f"Attn weights should be 2D, got {attn_w.ndim}D"

    def test_attention_weights_sum_to_one(self):
        """Attention weights across time steps should sum to 1.0 (softmax)."""
        model = self.ModelClass(n_mels=128, cnn_dim=128, lstm_hidden=128)
        model.eval()
        dummy_spec = torch.randn(1, 1, 128, 125)
        with torch.no_grad():
            _, attn_w = model(dummy_spec)
        sums = attn_w.sum(dim=1)
        assert torch.allclose(sums, torch.ones(1), atol=1e-4), f"Attn weights sum: {sums}"

    def test_output_is_finite(self):
        model = self.ModelClass(n_mels=128, cnn_dim=128, lstm_hidden=128)
        model.eval()
        dummy_spec = torch.randn(1, 1, 128, 125)
        with torch.no_grad():
            logits, attn_w = model(dummy_spec)
        assert torch.isfinite(logits).all()
        assert torch.isfinite(attn_w).all()


# ---------------------------------------------------------------------------
# 2. Spectrogram module tests
# ---------------------------------------------------------------------------


class TestSpectrogram:
    @pytest.fixture(autouse=True)
    def _create_temp_wav(self, tmp_path):
        self.wav_path = str(tmp_path / "test_audio.wav")
        _make_test_wav(self.wav_path, duration_sec=2.0)

    def test_load_spectrogram_tensor_shape(self):
        from models.audio.spectrogram import load_spectrogram_tensor
        tensor, spec = load_spectrogram_tensor(self.wav_path, duration=4.0)
        assert tensor.ndim == 4, f"Tensor shape error: {tensor.shape}"
        assert tensor.shape[:3] == (1, 1, 128), f"Tensor dims error: {tensor.shape}"
        assert spec.ndim == 2, f"Spec shape error: {spec.shape}"
        assert spec.shape[0] == 128, f"Spec n_mels error: {spec.shape}"


# ---------------------------------------------------------------------------
# 3. AudioPredictor end-to-end tests
# ---------------------------------------------------------------------------


class TestAudioPredictor:
    @pytest.fixture(autouse=True)
    def _create_temp_wav(self, tmp_path):
        self.wav_path = str(tmp_path / "test_audio.wav")
        _make_test_wav(self.wav_path, duration_sec=3.0)

    def test_predict_returns_required_keys(self):
        from models.audio.predictor import AudioPredictor
        pred = AudioPredictor()
        result = pred.predict(self.wav_path)
        required = {
            "verdict", "confidence", "real_prob", "fake_prob",
            "attention_weights", "heatmap_b64"
        }
        assert required.issubset(result.keys()), (
            f"Missing keys: {required - result.keys()}"
        )

    def test_verdict_is_valid(self):
        from models.audio.predictor import AudioPredictor
        pred = AudioPredictor()
        result = pred.predict(self.wav_path)
        assert result["verdict"] in ("real", "fake")

    def test_probabilities_sum_to_one(self):
        from models.audio.predictor import AudioPredictor
        pred = AudioPredictor()
        result = pred.predict(self.wav_path)
        total = result["real_prob"] + result["fake_prob"]
        assert abs(total - 1.0) < 0.01

    def test_attention_weights_nonempty(self):
        from models.audio.predictor import AudioPredictor
        pred = AudioPredictor()
        result = pred.predict(self.wav_path)
        assert len(result["attention_weights"]) > 0

    def test_heatmap_b64_is_valid_png(self):
        import base64, io
        from PIL import Image
        from models.audio.predictor import AudioPredictor
        pred = AudioPredictor()
        result = pred.predict(self.wav_path)
        img = Image.open(io.BytesIO(base64.b64decode(result["heatmap_b64"])))
        assert img.size[0] > 0
