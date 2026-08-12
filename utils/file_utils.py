"""
utils/file_utils.py
-------------------
Shared file-handling utilities used across all TRUTHGUARD modules.

Responsibilities:
  - Detect the type of an uploaded file (image / video / audio)
  - Provide safe temp-file context managers
  - Validate file sizes and formats before sending to a module
"""

import os
import tempfile
import mimetypes
from contextlib import contextmanager
from pathlib import Path
from typing import Literal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Supported MIME prefixes / suffixes per modality
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

MAX_IMAGE_MB = 20
MAX_VIDEO_MB = 200
MAX_AUDIO_MB = 50

FileType = Literal["image", "video", "audio", "unknown"]


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def detect_file_type(filepath: str | Path) -> FileType:
    """
    Infer the modality of *filepath* from its extension.

    Returns one of: "image", "video", "audio", "unknown".
    """
    ext = Path(filepath).suffix.lower()
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    if ext in AUDIO_EXTS:
        return "audio"
    # Fall back to MIME sniffing
    mime, _ = mimetypes.guess_type(str(filepath))
    if mime:
        if mime.startswith("image/"):
            return "image"
        if mime.startswith("video/"):
            return "video"
        if mime.startswith("audio/"):
            return "audio"
    return "unknown"


def validate_file_size(filepath: str | Path, file_type: FileType) -> None:
    """
    Raise *ValueError* if the file exceeds the per-modality size limit.
    """
    size_mb = Path(filepath).stat().st_size / (1024 * 1024)
    limits = {"image": MAX_IMAGE_MB, "video": MAX_VIDEO_MB, "audio": MAX_AUDIO_MB}
    limit = limits.get(file_type, 200)
    if size_mb > limit:
        raise ValueError(
            f"File size {size_mb:.1f} MB exceeds the {limit} MB limit for {file_type} files."
        )


@contextmanager
def temp_file(suffix: str = "", prefix: str = "truthguard_"):
    """
    Context manager that yields a temporary file path and removes it on exit.

    Usage::

        with temp_file(suffix=".mp4") as path:
            # write to / read from path
            ...
        # file is deleted here
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix=prefix)
    try:
        os.close(fd)
        yield path
    finally:
        if os.path.exists(path):
            os.remove(path)


def save_upload(file_bytes: bytes, suffix: str) -> str:
    """
    Persist *file_bytes* to a temp file with the given *suffix* and return
    the absolute path.  The caller is responsible for deleting the file.
    """
    fd, path = tempfile.mkstemp(suffix=suffix, prefix="truthguard_")
    with os.fdopen(fd, "wb") as f:
        f.write(file_bytes)
    return path
