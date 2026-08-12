"""
models/image/predictor.py
--------------------------
High-level inference wrapper for the image deepfake detector.

This is the only class the API layer needs to import from this module.
It handles:
  - Image loading and preprocessing
  - Model forward pass
  - GradCAM heatmap generation
  - Packaging results into a plain dictionary
"""

from __future__ import annotations

import numpy as np
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

from .model import ImageDeepfakeDetector
from .gradcam import GradCAM
from utils.visualization import array_to_b64


# ---------------------------------------------------------------------------
# Image preprocessing pipeline (ImageNet statistics)
# ---------------------------------------------------------------------------

PREPROCESS = T.Compose([
    T.Resize((380, 380)),           # EfficientNet-B4 native resolution
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


class ImagePredictor:
    """
    End-to-end image deepfake detection predictor.

    Parameters
    ----------
    weights_path : optional path to a .pth file with fine-tuned weights.
                   If None, randomly-initialised weights are used (for demo).
    device       : 'cpu' or 'cuda' — defaults to CPU for deployment.
    """

    def __init__(
        self,
        weights_path: str | Path | None = None,
        pretrained: bool = False,
        device: str = "cpu",
    ):
        self.device = torch.device(device)

        # Build model (EfficientNet-B4 backbone)
        self.model = ImageDeepfakeDetector(pretrained=pretrained)
        self.model.eval()
        self.model.to(self.device)

        # Load fine-tuned deepfake weights if provided
        if weights_path is not None:
            self.load_weights(weights_path)

        # GradCAM targetting the last block of EfficientNet-B4
        target_layer = self.model.backbone.blocks[-1]
        self.gradcam = GradCAM(self.model, target_layer)

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def load_weights(self, path: str | Path) -> None:
        """
        Load a .pth checkpoint into the model.
        The checkpoint should be a state-dict saved with torch.save().
        """
        state = torch.load(str(path), map_location=self.device)
        # Accept both bare state-dicts and {"model": state_dict} wrappers
        if "model" in state:
            state = state["model"]
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        print(f"[ImagePredictor] Loaded weights from {path}")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    def _load_and_preprocess(self, image_path: str | Path) -> tuple[np.ndarray, torch.Tensor]:
        """
        Load an image file, return:
          - original_array : HxWx3 uint8 NumPy array
          - tensor         : (1, 3, 380, 380) float tensor on self.device
        """
        pil_img = Image.open(str(image_path)).convert("RGB")
        original_array = np.array(pil_img)
        tensor = PREPROCESS(pil_img).unsqueeze(0).to(self.device)
        return original_array, tensor

    # ------------------------------------------------------------------
    # Main predict method
    # ------------------------------------------------------------------

    def predict(self, image_path: str | Path) -> dict[str, Any]:
        """
        Run full inference on an image file.

        Parameters
        ----------
        image_path : path to the input image

        Returns
        -------
        dict with keys:
          - verdict      : "real" | "fake"
          - confidence   : float in [0, 1] — probability of predicted class
          - real_prob    : float — raw probability of 'real' class
          - fake_prob    : float — raw probability of 'fake' class
          - heatmap_b64  : base-64 PNG of the GradCAM overlay
        """
        original, tensor = self._load_and_preprocess(image_path)

        with torch.enable_grad():              # needed for GradCAM backprop
            # Forward pass
            logits = self.model(tensor)        # (1, 2)
            probs  = F.softmax(logits, dim=1)  # (1, 2)

        real_prob = probs[0, 0].item()
        fake_prob = probs[0, 1].item()

        predicted_class = int(probs[0].argmax())
        verdict     = "fake" if predicted_class == 1 else "real"
        confidence  = fake_prob if verdict == "fake" else real_prob

        # Generate GradCAM heatmap (explain the 'fake' class)
        blended = self.gradcam.overlay(original, tensor, target_class=1)
        heatmap_b64 = array_to_b64(blended)

        return {
            "verdict":     verdict,
            "confidence":  round(confidence, 4),
            "real_prob":   round(real_prob, 4),
            "fake_prob":   round(fake_prob, 4),
            "heatmap_b64": heatmap_b64,
        }
