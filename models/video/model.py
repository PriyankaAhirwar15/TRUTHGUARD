"""
models/video/model.py
----------------------
Dual-stream deepfake detector for video.

Architecture
------------
Stream 1 — Spatial (per-frame):  
    EfficientNet-B4 (with GroupNorm replacing BatchNorm) extracts a 1792-D
    feature vector per frame.

Stream 2 — Temporal (across frames):  
    The sequence of per-frame features is passed through an 8-head Transformer
    encoder (2 layers).  A [CLS] token aggregates the sequence into a single
    video-level descriptor.

Head:
    Linear(d_model → 2) classifies the [CLS] token → real / fake logits.
    Simultaneously, a per-frame Linear(d_model → 2) head produces frame-level
    forgery scores.
"""

import torch
import torch.nn as nn
import timm

# Reuse the GroupNorm conversion from the image module
from models.image.model import _replace_bn_with_gn


class VideoDeepfakeDetector(nn.Module):
    """
    Dual-stream (spatial + temporal) video deepfake classifier.

    Parameters
    ----------
    pretrained    : load ImageNet-pretrained EfficientNet-B4 backbone
    num_frames    : expected sequence length (default 16)
    d_model       : Transformer hidden dimension (projected from 1792)
    nhead         : number of attention heads in Transformer
    num_layers    : number of Transformer encoder layers
    dropout       : dropout rate
    """

    def __init__(
        self,
        pretrained: bool = True,
        num_frames: int = 16,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.num_frames = num_frames
        self.d_model    = d_model

        # ── Spatial stream ──────────────────────────────────────────────
        self.spatial_backbone = timm.create_model(
            "efficientnet_b4",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
        )
        _replace_bn_with_gn(self.spatial_backbone)  # batch-size-1 stability

        spatial_dim = self.spatial_backbone.num_features  # 1792

        # Project spatial features into Transformer dimension
        self.input_proj = nn.Sequential(
            nn.Linear(spatial_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # ── Temporal stream ─────────────────────────────────────────────
        # Learnable [CLS] token prepended to the frame sequence
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        # Learnable positional embeddings (num_frames + 1 for [CLS])
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_frames + 1, d_model)
        )
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Classification heads ─────────────────────────────────────────
        # Video-level head (operates on [CLS] token)
        self.video_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 2),
        )

        # Per-frame head (operates on each frame token)
        self.frame_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(d_model, 2),
        )

    # ------------------------------------------------------------------

    def forward(
        self, frames: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        frames : (B, T, 3, H, W) — batch of T-frame clips

        Returns
        -------
        video_logits : (B, 2)    — video-level real/fake logits
        frame_logits : (B, T, 2) — per-frame real/fake logits
        """
        B, T, C, H, W = frames.shape

        # ── Spatial: extract per-frame features ─────────────────────────
        # Merge batch and time dims so the CNN sees (B*T, C, H, W)
        flat_frames = frames.view(B * T, C, H, W)
        spatial_feats = self.spatial_backbone(flat_frames)  # (B*T, 1792)
        spatial_feats = spatial_feats.view(B, T, -1)         # (B, T, 1792)

        # Project to d_model
        tokens = self.input_proj(spatial_feats)              # (B, T, d_model)

        # ── Temporal: prepend [CLS] and encode ──────────────────────────
        cls_tokens = self.cls_token.expand(B, -1, -1)        # (B, 1, d_model)
        tokens     = torch.cat([cls_tokens, tokens], dim=1)  # (B, T+1, d_model)
        tokens     = tokens + self.pos_embed[:, :T + 1, :]   # add positional embedding

        encoded    = self.temporal_encoder(tokens)           # (B, T+1, d_model)

        # ── Heads ────────────────────────────────────────────────────────
        cls_out      = encoded[:, 0, :]        # (B, d_model)  — [CLS] token
        frame_tokens = encoded[:, 1:, :]      # (B, T, d_model)

        video_logits = self.video_head(cls_out)             # (B, 2)
        frame_logits = self.frame_head(frame_tokens)        # (B, T, 2)

        return video_logits, frame_logits
