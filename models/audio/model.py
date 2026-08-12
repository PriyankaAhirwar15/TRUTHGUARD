"""
models/audio/model.py
----------------------
Dual-stream audio deepfake / voice-clone detector.

Architecture
------------
Stream 1 — Spectral (CNN over mel-spectrogram patches):
    A small CNN extracts local spectral texture features from the
    mel-spectrogram (similar to how the image module treats images).

Stream 2 — Temporal (LSTM with attention over time):
    The CNN patch features are fed as a time-series into a 2-layer
    bidirectional LSTM.  A learnable attention mechanism produces
    per-timestep weights (used for explainability) and a context vector.

Fusion:
    Concatenate CNN global-pool features + LSTM context → Linear head.

Output:
    - logits        : (B, 2) real/fake logits
    - attention_w   : (B, T') per-timestep attention weights (for heatmap)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Spectral CNN
# ---------------------------------------------------------------------------

class SpectralCNN(nn.Module):
    """
    Lightweight CNN that maps a mel-spectrogram (B, 1, n_mels, T) to a
    sequence of patch features (B, T', cnn_dim).

    The spatial (mel-bin) dimension is fully collapsed via strided conv +
    adaptive pool, leaving only the time dimension for the LSTM.
    """

    def __init__(self, n_mels: int = 128, cnn_dim: int = 256):
        super().__init__()
        self.n_mels  = n_mels
        self.cnn_dim = cnn_dim

        # Three conv blocks; each halves the time dimension
        self.blocks = nn.Sequential(
            # Block 1: (B,1,128,T) → (B,32,64,T//2)
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            # Block 2: (B,32,64,T//2) → (B,64,32,T//4)
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            # Block 3: (B,64,32,T//4) → (B,128,16,T//8)
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 1), padding=(1, 1)),
            nn.GroupNorm(8, 128),
            nn.GELU(),
        )

        # Collapse mel-bin dimension → (B, cnn_dim, T')
        self.freq_pool = nn.AdaptiveAvgPool2d((1, None))   # (B,128,1,T')
        self.proj      = nn.Conv1d(128, cnn_dim, kernel_size=1)

    def forward(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        spectrogram : (B, 1, n_mels, T) float tensor

        Returns
        -------
        patch_features : (B, T', cnn_dim)
        """
        x = self.blocks(spectrogram)       # (B, 128, n_mels//8, T')
        x = self.freq_pool(x).squeeze(2)   # (B, 128, T')
        x = self.proj(x)                   # (B, cnn_dim, T')
        return x.permute(0, 2, 1)          # (B, T', cnn_dim)


# ---------------------------------------------------------------------------
# Attention mechanism
# ---------------------------------------------------------------------------

class AdditiveAttention(nn.Module):
    """
    Bahdanau-style additive attention over a sequence.

    Input  : (B, T, hidden_dim)
    Output : context (B, hidden_dim), weights (B, T)
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score_fn = nn.Linear(hidden_dim, 1, bias=True)

    def forward(
        self, hidden_seq: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scores  = self.score_fn(hidden_seq).squeeze(-1)   # (B, T)
        weights = F.softmax(scores, dim=-1)                # (B, T)
        context = (weights.unsqueeze(-1) * hidden_seq).sum(dim=1)  # (B, H)
        return context, weights


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------


class AudioDeepfakeDetector(nn.Module):
    """
    Dual-stream (spectral CNN + temporal LSTM-with-attention) audio deepfake
    detector.

    Parameters
    ----------
    n_mels      : mel-spectrogram bins (default 128)
    cnn_dim     : CNN output channels
    lstm_hidden : LSTM hidden dimension (per direction)
    lstm_layers : number of LSTM layers
    dropout     : dropout rate
    """

    def __init__(
        self,
        n_mels:      int = 128,
        cnn_dim:     int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout:     float = 0.3,
    ):
        super().__init__()

        # Spectral stream
        self.cnn = SpectralCNN(n_mels=n_mels, cnn_dim=cnn_dim)

        # Temporal stream
        self.lstm = nn.LSTM(
            input_size=cnn_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )

        lstm_out_dim = lstm_hidden * 2   # bidirectional

        # Attention
        self.attention = AdditiveAttention(lstm_out_dim)

        # Fusion + classification head
        # concat(cnn_global_pool, lstm_context) → 2
        self.classifier = nn.Sequential(
            nn.Linear(cnn_dim + lstm_out_dim, 256),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, 2),
        )

    def forward(
        self, spectrogram: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        spectrogram : (B, 1, n_mels, T) float tensor (log-mel)

        Returns
        -------
        logits      : (B, 2) — real/fake logits
        attn_weights: (B, T') — per-timestep attention weights
        """
        # Spectral stream
        patch_feats = self.cnn(spectrogram)    # (B, T', cnn_dim)

        # Global-pool spectral features for fusion
        cnn_global  = patch_feats.mean(dim=1)  # (B, cnn_dim)

        # Temporal stream
        lstm_out, _ = self.lstm(patch_feats)   # (B, T', lstm_hidden*2)
        context, attn_w = self.attention(lstm_out)  # (B, lstm_h*2), (B, T')

        # Fusion
        fused  = torch.cat([cnn_global, context], dim=-1)  # (B, cnn_dim + lstm_h*2)
        logits = self.classifier(fused)                     # (B, 2)

        return logits, attn_w
