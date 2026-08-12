"""
utils/visualization.py
-----------------------
Shared visualization helpers used across all TRUTHGUARD modules.

Provides:
  - overlay_heatmap()   : blend a GradCAM heatmap onto an RGB image
  - plot_timeline()     : bar-chart of per-frame confidence scores
  - spectrogram_heatmap(): overlay attention weights on a mel-spectrogram
  - array_to_b64()      : encode a NumPy image array as a base-64 PNG string
"""

import io
import base64

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (safe for servers)
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def array_to_b64(img_array: np.ndarray) -> str:
    """
    Convert an HxWx3 uint8 NumPy array to a base-64-encoded PNG string.
    The returned string can be embedded directly in an HTML <img src="..."> tag.
    """
    pil_img = Image.fromarray(img_array.astype(np.uint8))
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def fig_to_b64(fig: plt.Figure) -> str:
    """Convert a Matplotlib Figure to a base-64-encoded PNG string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


# ---------------------------------------------------------------------------
# Public visualization functions
# ---------------------------------------------------------------------------


def overlay_heatmap(
    original_image: np.ndarray,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: str = "jet",
) -> np.ndarray:
    """
    Blend a single-channel heatmap onto an RGB image using alpha compositing.

    Parameters
    ----------
    original_image : HxWx3 uint8 array
    heatmap        : HxW float32 array, values in [0, 1] (normalised)
    alpha          : opacity of the heatmap overlay (0 = invisible, 1 = opaque)
    colormap       : matplotlib colormap name

    Returns
    -------
    HxWx3 uint8 blended image
    """
    h, w = original_image.shape[:2]

    # Resize heatmap to match image if needed
    if heatmap.shape != (h, w):
        heatmap_pil = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_pil = heatmap_pil.resize((w, h), Image.BILINEAR)
        heatmap = np.array(heatmap_pil) / 255.0

    # Apply colormap → RGBA, then extract RGB
    cmap = plt.get_cmap(colormap)
    colored = cmap(heatmap)[..., :3]          # HxWx3 float [0,1]
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Alpha blend
    blended = (
        (1 - alpha) * original_image.astype(np.float32)
        + alpha * colored_uint8.astype(np.float32)
    ).clip(0, 255).astype(np.uint8)

    return blended


def plot_timeline(
    per_frame_scores: list[float],
    threshold: float = 0.5,
    title: str = "Per-Frame Forgery Confidence",
) -> str:
    """
    Render a bar chart of per-frame fake-confidence scores.

    Bars above *threshold* are coloured red (suspected fake frames);
    bars below are coloured green (suspected real).

    Returns a base-64 PNG string.
    """
    n = len(per_frame_scores)
    colors = ["#e74c3c" if s >= threshold else "#2ecc71" for s in per_frame_scores]

    fig, ax = plt.subplots(figsize=(max(6, n * 0.4), 3))
    ax.bar(range(n), per_frame_scores, color=colors, edgecolor="none", zorder=2)
    ax.axhline(threshold, color="#f39c12", linestyle="--", linewidth=1.2,
               label=f"Threshold ({threshold})")
    ax.set_ylim(0, 1)
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Fake confidence")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    ax.spines[:].set_color("#444")

    return fig_to_b64(fig)


def spectrogram_heatmap(
    spectrogram: np.ndarray,
    attention: np.ndarray,
    sr: int = 22050,
    hop_length: int = 512,
    title: str = "Audio Attention Heatmap",
) -> str:
    """
    Overlay attention weights on a mel-spectrogram and return a base-64 PNG.

    Parameters
    ----------
    spectrogram : (n_mels, T) float array — log-power mel-spectrogram
    attention   : (T,) float array — per-time-step attention weight [0,1]
    """
    n_mels, T = spectrogram.shape

    fig, axes = plt.subplots(2, 1, figsize=(10, 5),
                             gridspec_kw={"height_ratios": [4, 1]})

    # Top: spectrogram
    axes[0].imshow(
        spectrogram, aspect="auto", origin="lower",
        cmap="magma", interpolation="nearest"
    )
    axes[0].set_title(title, color="white")
    axes[0].set_ylabel("Mel bin", color="white")
    axes[0].tick_params(colors="white")

    # Bottom: attention bar
    attention_2d = attention[np.newaxis, :]           # 1 x T
    axes[1].imshow(
        attention_2d, aspect="auto", cmap="hot",
        vmin=0, vmax=1, interpolation="nearest"
    )
    axes[1].set_ylabel("Attn", color="white", fontsize=8)
    axes[1].set_yticks([])
    axes[1].tick_params(colors="white")
    axes[1].set_xlabel("Time frames", color="white")

    for ax in axes:
        ax.set_facecolor("#1a1a2e")
    fig.patch.set_facecolor("#16213e")
    fig.tight_layout()

    return fig_to_b64(fig)
