"""
models/image/gradcam.py
------------------------
Gradient-weighted Class Activation Map (GradCAM) for the image deepfake
detector.

How it works
------------
1. Register a forward hook on the last convolutional block of EfficientNet-B4
   to capture the activation map A (C x H' x W').
2. Register a backward hook on the same layer to capture the gradients
   dScore/dA, then average-pool over the spatial dims → weights α_c.
3. Compute heatmap = ReLU(Σ_c  α_c * A_c) and normalise to [0, 1].
4. Return the heatmap together with an RGB overlay image.

Reference: Selvaraju et al., "Grad-CAM", ICCV 2017.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from utils.visualization import overlay_heatmap


class GradCAM:
    """
    Gradient-weighted Class Activation Maps for EfficientNet-B4.

    Parameters
    ----------
    model      : ImageDeepfakeDetector instance (or any nn.Module)
    target_layer : the nn.Module whose activations to visualise.
                   For EfficientNet-B4 via timm, use
                   ``model.backbone.blocks[-1]``.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    # ------------------------------------------------------------------
    # Hooks
    # ------------------------------------------------------------------

    def _save_activation(self, module, input, output):
        """Forward hook — captures the feature map tensor."""
        self._activations = output.detach()  # (B, C, H', W')

    def _save_gradient(self, module, grad_input, grad_output):
        """Backward hook — captures the gradient tensor."""
        self._gradients = grad_output[0].detach()  # (B, C, H', W')

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: int = 1,          # 1 = 'fake'
    ) -> np.ndarray:
        """
        Compute the GradCAM heatmap for *target_class*.

        Parameters
        ----------
        input_tensor : (1, 3, H, W) preprocessed image tensor
        target_class : class index to explain (default 1 = fake)

        Returns
        -------
        heatmap : (H, W) float32 array in [0, 1]
        """
        self.model.eval()

        # Enable gradients for this call
        input_tensor = input_tensor.requires_grad_(True)

        # Forward pass
        logits = self.model(input_tensor)          # (1, 2)
        score = logits[0, target_class]            # scalar

        # Backward pass — compute gradients w.r.t. activations
        self.model.zero_grad()
        score.backward()

        # Retrieve hooked tensors
        activations = self._activations[0]         # (C, H', W')
        gradients   = self._gradients[0]           # (C, H', W')

        # Global-average-pool the gradients → importance weights per channel
        weights = gradients.mean(dim=(1, 2))       # (C,)

        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU (keep only positive influence on the target class)
        cam = F.relu(cam)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)

        return cam.numpy()

    def overlay(
        self,
        original_image: np.ndarray,
        input_tensor: torch.Tensor,
        target_class: int = 1,
    ) -> np.ndarray:
        """
        Return an RGB image with the GradCAM heatmap blended in.

        Parameters
        ----------
        original_image : (H, W, 3) uint8 array (the unprocessed image)
        input_tensor   : (1, 3, H, W) preprocessed tensor
        target_class   : class to explain

        Returns
        -------
        blended : (H, W, 3) uint8 array
        """
        heatmap = self.generate(input_tensor, target_class)
        return overlay_heatmap(original_image, heatmap)

    def remove_hooks(self):
        """Deregister the forward / backward hooks (call when done)."""
        self._fwd_hook.remove()
        self._bwd_hook.remove()
