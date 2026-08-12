"""
models/audio/spectrogram.py
----------------------------
Mel-spectrogram computation and preprocessing for the audio module.

All parameters are chosen to be compatible with the CNN input size
expected by AudioDeepfakeDetector.
"""

from __future__ import annotations

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path


# ---------------------------------------------------------------------------
# Default spectrogram parameters
# ---------------------------------------------------------------------------

DEFAULT_SR      = 16_000   # resample all audio to 16 kHz
N_MELS          = 128      # mel filter-bank bins
N_FFT           = 1024     # FFT window size  (~64 ms at 16 kHz)
HOP_LENGTH      = 512      # hop size         (~32 ms at 16 kHz)
FMAX            = 8_000    # maximum frequency (Hz)
DURATION_SEC    = 4.0      # clip duration — pad/truncate to this
AMPLITUDE_TO_DB = True     # convert power spectrogram to decibels


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def load_audio(path: str | Path, sr: int = DEFAULT_SR) -> np.ndarray:
    """
    Load an audio file, convert to mono, and resample to *sr* Hz.

    Supports all formats handled by soundfile / libsndfile (.wav, .flac,
    .ogg) plus .mp3 via librosa's audioread fallback.

    Returns a 1-D float32 NumPy array in [-1, 1].
    """
    try:
        audio, orig_sr = sf.read(str(path), always_2d=False)
        # Convert to mono if stereo
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if orig_sr != sr:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=sr)
    except Exception:
        # Fallback: let librosa handle the format (slower, needs audioread)
        audio, _ = librosa.load(str(path), sr=sr, mono=True)

    return audio


def pad_or_truncate(audio: np.ndarray, sr: int, duration: float) -> np.ndarray:
    """
    Pad (with zeros) or truncate *audio* to exactly *duration* seconds.
    """
    target_len = int(sr * duration)
    if len(audio) >= target_len:
        return audio[:target_len]
    # Pad with zeros
    return np.pad(audio, (0, target_len - len(audio)), mode="constant")


def compute_melspectrogram(
    audio: np.ndarray,
    sr: int      = DEFAULT_SR,
    n_mels: int  = N_MELS,
    n_fft: int   = N_FFT,
    hop_length: int = HOP_LENGTH,
    fmax: float  = FMAX,
    amplitude_to_db: bool = AMPLITUDE_TO_DB,
) -> np.ndarray:
    """
    Compute a log-power mel-spectrogram.

    Returns
    -------
    spec : (n_mels, T) float32 NumPy array
           T = ceil(len(audio) / hop_length)
    """
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels,
        fmax=fmax,
    )
    if amplitude_to_db:
        mel = librosa.power_to_db(mel, ref=np.max)

    return mel.astype(np.float32)


def spectrogram_to_tensor(
    spec: np.ndarray,
    normalise: bool = True,
) -> torch.Tensor:
    """
    Convert a (n_mels, T) spectrogram to a (1, 1, n_mels, T) float tensor.

    Optionally normalises the spectrogram to zero mean / unit variance.
    """
    if normalise:
        mean, std = spec.mean(), spec.std()
        if std > 1e-6:
            spec = (spec - mean) / std

    tensor = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0)  # (1, 1, n_mels, T)
    return tensor


def load_spectrogram_tensor(
    path: str | Path,
    sr: int         = DEFAULT_SR,
    duration: float = DURATION_SEC,
    n_mels: int     = N_MELS,
    n_fft: int      = N_FFT,
    hop_length: int = HOP_LENGTH,
) -> tuple[torch.Tensor, np.ndarray]:
    """
    End-to-end: load audio → compute mel-spectrogram → tensor.

    Returns
    -------
    tensor : (1, 1, n_mels, T) — ready for model input
    spec   : (n_mels, T)       — raw spectrogram (for visualisation)
    """
    audio = load_audio(path, sr=sr)
    audio = pad_or_truncate(audio, sr=sr, duration=duration)
    spec  = compute_melspectrogram(audio, sr=sr, n_mels=n_mels,
                                   n_fft=n_fft, hop_length=hop_length)
    tensor = spectrogram_to_tensor(spec)
    return tensor, spec
