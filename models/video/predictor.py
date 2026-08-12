"""
models/video/predictor.py
--------------------------
High-level inference wrapper for the video deepfake detector.

Responsibilities
----------------
* Extract frames from a video file using OpenCV
* Preprocess frames with the same ImageNet pipeline as the image module
* Run the dual-stream VideoDeepfakeDetector
* Return per-frame scores, video-level verdict, and a confidence timeline PNG
"""

from __future__ import annotations

import cv2
import numpy as np
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .model import VideoDeepfakeDetector
from utils.visualization import plot_timeline


# ---------------------------------------------------------------------------
# Frame preprocessing (same statistics as image module)
# ---------------------------------------------------------------------------

FRAME_PREPROCESS = T.Compose([
    T.Resize((380, 380)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class VideoPredictor:
    """
    End-to-end video deepfake detection predictor.

    Parameters
    ----------
    weights_path : optional path to a .pth checkpoint with fine-tuned weights
    num_frames   : number of frames to sample uniformly from the video (default 16)
    device       : 'cpu' or 'cuda'
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        num_frames: int = 16,
        pretrained: bool = False,
        device: str = "cpu",
    ):
        self.device     = torch.device(device)
        self.num_frames = num_frames

        self.model = VideoDeepfakeDetector(
            pretrained=pretrained, num_frames=num_frames
        )
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
        print(f"[VideoPredictor] Loaded weights from {path}")

    # ------------------------------------------------------------------
    # Frame extraction
    # ------------------------------------------------------------------

    def _extract_frames(
        self, video_path: str | Path
    ) -> list[np.ndarray]:
        """
        Uniformly sample self.num_frames frames from the video.

        Returns a list of HxWx3 uint8 BGR arrays (OpenCV convention).
        Raises FileNotFoundError if the video cannot be opened.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = self.num_frames  # fallback for live streams

        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        frames  = []
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(frame)

        cap.release()

        # If fewer frames were read (short video), pad by repeating last
        while len(frames) < self.num_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))

        return frames[:self.num_frames]

    def _preprocess_frames(
        self, bgr_frames: list[np.ndarray]
    ) -> torch.Tensor:
        """
        Convert a list of BGR frames to a (1, T, 3, H, W) float tensor.
        """
        tensors = []
        for bgr in bgr_frames:
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(rgb)
            tensors.append(FRAME_PREPROCESS(pil))

        # Stack → (T, 3, H, W)
        clip = torch.stack(tensors, dim=0)
        return clip.unsqueeze(0).to(self.device)   # (1, T, 3, H, W)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, video_path: str | Path) -> dict[str, Any]:
        """
        Run full inference on a video file.

        Returns
        -------
        dict with keys:
          - verdict           : "real" | "fake"
          - confidence        : float — video-level confidence of predicted class
          - real_prob         : float
          - fake_prob         : float
          - per_frame_scores  : list[float] — fake-probability per frame
          - timeline_b64      : base-64 PNG of the per-frame timeline chart
          - num_frames        : int — number of frames analysed
          - fake_frame_count  : int — frames classified as fake
        """
        bgr_frames = self._extract_frames(video_path)
        tensor     = self._preprocess_frames(bgr_frames)  # (1, T, 3, H, W)

        with torch.no_grad():
            video_logits, frame_logits = self.model(tensor)

        # Video-level prediction
        video_probs = F.softmax(video_logits, dim=1)[0]   # (2,)
        real_prob   = video_probs[0].item()
        fake_prob   = video_probs[1].item()
        verdict     = "fake" if fake_prob >= 0.5 else "real"
        confidence  = fake_prob if verdict == "fake" else real_prob

        # Per-frame predictions
        frame_probs  = F.softmax(frame_logits, dim=2)[0]  # (T, 2)
        per_frame_fake = frame_probs[:, 1].tolist()        # fake prob per frame
        fake_count     = sum(1 for s in per_frame_fake if s >= 0.5)

        # Timeline visualisation
        timeline_b64 = plot_timeline(
            per_frame_fake,
            threshold=0.5,
            title="Per-Frame Forgery Confidence",
        )

        return {
            "verdict":          verdict,
            "confidence":       round(confidence, 4),
            "real_prob":        round(real_prob, 4),
            "fake_prob":        round(fake_prob, 4),
            "per_frame_scores": [round(s, 4) for s in per_frame_fake],
            "timeline_b64":     timeline_b64,
            "num_frames":       len(bgr_frames),
            "fake_frame_count": fake_count,
        }
