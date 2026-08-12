"""
models/image/model.py
----------------------
EfficientNet-B4-based binary classifier for image deepfake detection.

Key design decisions
--------------------
* Backbone  : timm's EfficientNet-B4 pretrained on ImageNet.
* BatchNorm fix : All BatchNorm2d layers are converted to GroupNorm so that
  batch-size-1 inference is numerically stable (no running-stat issues).
* Head      : global-average-pool → Dropout(0.3) → Linear(1792 → 2).
* Output    : raw logits; call F.softmax(logits, dim=-1) to get probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


# ---------------------------------------------------------------------------
# BatchNorm → GroupNorm conversion
# ---------------------------------------------------------------------------

def _replace_bn_with_gn(module: nn.Module, num_groups: int = 32) -> nn.Module:
    """
    Recursively replace every BatchNorm2d in *module* with a GroupNorm that
    has the same number of channels.  This makes the model stable at
    batch-size = 1 (GroupNorm statistics are computed over spatial dims only).

    Parameters
    ----------
    num_groups : number of groups for GroupNorm.  Must divide every channel
                 count in the network.  32 is safe for EfficientNet-B4.
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            # GroupNorm requires num_channels % num_groups == 0
            # Use min(num_groups, child.num_features) to be safe
            groups = min(num_groups, child.num_features)
            # Ensure divisibility
            while child.num_features % groups != 0 and groups > 1:
                groups -= 1
            gn = nn.GroupNorm(
                num_groups=groups,
                num_channels=child.num_features,
                eps=child.eps,
                affine=child.affine,
            )
            setattr(module, name, gn)
        else:
            _replace_bn_with_gn(child, num_groups)
    return module


# ---------------------------------------------------------------------------
# Model definition
# ---------------------------------------------------------------------------


class ImageDeepfakeDetector(nn.Module):
    """
    Binary deepfake classifier built on EfficientNet-B4.

    Forward pass
    ------------
    Input  : (B, 3, H, W) float tensor, values in [0, 1] or normalised.
    Output : (B, 2) raw logits  [logit_real, logit_fake]

    Usage
    -----
    >>> model = ImageDeepfakeDetector(pretrained=True)
    >>> model.eval()
    >>> logits = model(image_tensor)           # (1, 2)
    >>> probs  = torch.softmax(logits, dim=1)  # (1, 2)
    """

    def __init__(self, pretrained: bool = True, dropout: float = 0.3):
        super().__init__()

        # Load EfficientNet-B4 without the default classifier head
        self.backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,          # remove original FC layer
            global_pool="avg",      # global average pool → feature vector
        )

        # Replace all BatchNorm2d layers with GroupNorm for batch-size-1 stability
        _replace_bn_with_gn(self.backbone)

        # Get the feature dimension from the backbone
        feature_dim = self.backbone.num_features  # 1792 for EfficientNet-B4

        # Custom classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, 2),   # 2 classes: real / fake
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 3, H, W) — preprocessed image tensor

        Returns
        -------
        logits : (B, 2) — [real_logit, fake_logit]
        """
        features = self.backbone(x)        # (B, 1792)
        logits = self.classifier(features) # (B, 2)
        return logits

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the backbone feature map (before global pool) for GradCAM.
        Used by the GradCAM module to register hooks on the last conv block.
        """
        # timm's forward_features returns the feature map before pooling
        return self.backbone.forward_features(x)
