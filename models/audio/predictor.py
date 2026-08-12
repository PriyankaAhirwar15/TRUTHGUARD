"""
models/audio/predictor.py
--------------------------
High-level inference wrapper for the audio deepfake detector.

This is the only class that the API layer imports from this module.
It handles:
  - Mel-spectrogram computation
  - Model forward pass
  - Attention-weight extraction
  - Spectrogram + attention heatmap generation
  - Result packaging
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .model import AudioDeepfakeDetector
from .spectrogram import load_spectrogram_tensor
from utils.visualization import spectrogram_heatmap


class AudioPredictor:
    """
    End-to-end audio deepfake detection predictor.

    Parameters
    ----------
    weights_path : optional path to a .pth checkpoint
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        self.model = AudioDeepfakeDetector()
        self.model.eval()
        self.model.to(self.device)

        if weights_path is not None:
            self.load_weights(weights_path)

    # ------------------------------------------------------------------

    def load_weights(self, path: str | Path) -> None:
        state = torch.load(str(path), map_location=self.device)
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        print(f"[AudioPredictor] Loaded weights from {path}")

    # ------------------------------------------------------------------

    def predict(self, audio_path: str | Path) -> dict[str, Any]:
        """
        Run full inference on an audio file.

        Parameters
        ----------
        audio_path : path to the input audio file (.wav / .mp3 / .flac / ...)

        Returns
        -------
        dict with keys:
          - verdict           : "real" | "fake"
          - confidence        : float in [0, 1]
          - real_prob         : float
          - fake_prob         : float
          - attention_weights : list[float] — per-timestep attention values
          - heatmap_b64       : base-64 PNG of spectrogram + attention overlay
        """
        # Load and compute spectrogram
        tensor, spec = load_spectrogram_tensor(audio_path)
        tensor = tensor.to(self.device)   # (1, 1, n_mels, T)

        with torch.no_grad():
            logits, attn_w = self.model(tensor)   # (1,2), (1,T')

        probs     = F.softmax(logits, dim=1)[0]  # (2,)
        real_prob = probs[0].item()
        fake_prob = probs[1].item()
        verdict   = "fake" if fake_prob >= 0.5 else "real"
        confidence = fake_prob if verdict == "fake" else real_prob

        # Attention weights: detach and normalise to [0,1]
        attn_np = attn_w[0].cpu().numpy()          # (T',)
        if attn_np.max() > attn_np.min():
            attn_np = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min())

        # Generate spectrogram + attention heatmap
        heatmap_b64 = spectrogram_heatmap(
            spectrogram=spec,
            attention=attn_np,
            title="Audio Attention Heatmap (TRUTHGUARD)",
        )

        return {
            "verdict":           verdict,
            "confidence":        round(confidence, 4),
            "real_prob":         round(real_prob, 4),
            "fake_prob":         round(fake_prob, 4),
            "attention_weights": attn_np.tolist(),
            "heatmap_b64":       heatmap_b64,
        }
