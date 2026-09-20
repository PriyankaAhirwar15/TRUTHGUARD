"""
ui/app.py
----------
TRUTHGUARD — Gradio Dashboard (single entry point for the UI).

Layout
------
The interface has four tabs, one per detection modality:

  Tab 1 — Image    : upload image → GradCAM heatmap + verdict
  Tab 2 — Video    : upload video → per-frame timeline + verdict
  Tab 3 — Audio    : upload audio → spectrogram attention heatmap + verdict
  Tab 4 — Combined : upload video (with audio) → fusion verdict + mismatch flag

How it works
------------
The Gradio app calls the FastAPI backend directly via httpx (same process or
localhost:8000).  This keeps the UI and API decoupled: the UI is purely a
presentation layer that formats API responses.

Running
-------
  python ui/app.py                   # standalone Gradio on port 7860
  OR
  python app.py                      # root entry-point for HuggingFace Spaces
"""

from __future__ import annotations

import sys
import os
import base64
import io

# Ensure project root is on the path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx
import gradio as gr
from PIL import Image
import numpy as np

# ZeroGPU support — required on Hugging Face Spaces with ZeroGPU hardware
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    # Running locally — create a no-op decorator so the code works unchanged
    class _NoopSpaces:
        @staticmethod
        def GPU(fn=None, duration=60):
            if fn is not None:
                return fn
            def decorator(f):
                return f
            return decorator
    spaces = _NoopSpaces()
    HAS_SPACES = False

# ---------------------------------------------------------------------------
# Backend URL
# ---------------------------------------------------------------------------

API_BASE = os.environ.get("TRUTHGUARD_API_URL", "http://127.0.0.1:8000")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _b64_to_pil(b64_str: str) -> Image.Image:
    """Decode a base-64 PNG string into a PIL Image."""
    return Image.open(io.BytesIO(base64.b64decode(b64_str)))


def _verdict_badge(verdict: str, confidence: float, risk_level: str | None = None) -> str:
    """
    Return an HTML badge string with colour-coded verdict.
    Green = real, Red = fake, Orange = mismatch.
    """
    if risk_level and "MISMATCH" in risk_level:
        colour = "#f39c12"
        icon   = "⚠️"
    elif verdict == "fake":
        colour = "#e74c3c"
        icon   = "🔴"
    else:
        colour = "#2ecc71"
        icon   = "🟢"

    label = risk_level or verdict.upper()
    return (
        f"<div style='"
        f"background:{colour};color:#fff;padding:12px 20px;"
        f"border-radius:8px;font-size:1.1rem;font-weight:700;"
        f"text-align:center;margin:8px 0'>"
        f"{icon} {label} &nbsp;|&nbsp; Confidence: {confidence:.1%}"
        f"</div>"
    )


def _call_api(endpoint: str, file_path: str, mime_type: str) -> dict:
    """POST a file to the given API endpoint and return the JSON response."""
    with open(file_path, "rb") as f:
        files = {"file": (os.path.basename(file_path), f, mime_type)}
        resp  = httpx.post(f"{API_BASE}/detect/{endpoint}", files=files, timeout=120)
    resp.raise_for_status()
    return resp.json()


# ---------------------------------------------------------------------------
# Tab 1 — Image
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def analyse_image(image_path: str) -> tuple:
    """
    Send the uploaded image to /detect/image and return:
      (verdict_html, heatmap_pil, details_text)
    """
    if image_path is None:
        return "<p>Please upload an image.</p>", None, ""

    try:
        data = _call_api("image", image_path, "image/jpeg")
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", None, ""

    badge   = _verdict_badge(data["verdict"], data["confidence"])
    heatmap = _b64_to_pil(data["heatmap_b64"])
    details = (
        f"Real probability : {data['real_prob']:.4f}\n"
        f"Fake probability : {data['fake_prob']:.4f}\n"
        f"Verdict          : {data['verdict'].upper()}\n"
        f"Confidence       : {data['confidence']:.4f}"
    )
    return badge, heatmap, details


# ---------------------------------------------------------------------------
# Tab 2 — Video
# ---------------------------------------------------------------------------


@spaces.GPU(duration=120)
def analyse_video(video_path: str) -> tuple:
    """
    Send the uploaded video to /detect/video and return:
      (verdict_html, timeline_pil, details_text)
    """
    if video_path is None:
        return "<p>Please upload a video.</p>", None, ""

    try:
        data = _call_api("video", video_path, "video/mp4")
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", None, ""

    badge    = _verdict_badge(data["verdict"], data["confidence"])
    timeline = _b64_to_pil(data["timeline_b64"])
    details  = (
        f"Frames analysed  : {data['num_frames']}\n"
        f"Fake frames      : {data['fake_frame_count']} / {data['num_frames']}\n"
        f"Real probability : {data['real_prob']:.4f}\n"
        f"Fake probability : {data['fake_prob']:.4f}\n"
        f"Verdict          : {data['verdict'].upper()}\n"
        f"Confidence       : {data['confidence']:.4f}\n\n"
        f"Per-frame scores :\n{data['per_frame_scores']}"
    )
    return badge, timeline, details


# ---------------------------------------------------------------------------
# Tab 3 — Audio
# ---------------------------------------------------------------------------


@spaces.GPU(duration=60)
def analyse_audio(audio_path: str) -> tuple:
    """
    Send the uploaded audio to /detect/audio and return:
      (verdict_html, heatmap_pil, details_text)
    """
    if audio_path is None:
        return "<p>Please upload an audio file.</p>", None, ""

    try:
        data = _call_api("audio", audio_path, "audio/wav")
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", None, ""

    badge   = _verdict_badge(data["verdict"], data["confidence"])
    heatmap = _b64_to_pil(data["heatmap_b64"])
    details = (
        f"Real probability : {data['real_prob']:.4f}\n"
        f"Fake probability : {data['fake_prob']:.4f}\n"
        f"Verdict          : {data['verdict'].upper()}\n"
        f"Confidence       : {data['confidence']:.4f}\n\n"
        f"Attention weight preview (first 20):\n"
        f"{[round(w, 3) for w in data['attention_weights'][:20]]}"
    )
    return badge, heatmap, details


# ---------------------------------------------------------------------------
# Tab 4 — Combined (Cross-Modal Fusion)
# ---------------------------------------------------------------------------


@spaces.GPU(duration=180)
def analyse_combined(video_path: str) -> tuple:
    """
    Send the uploaded video to /detect/combined and return:
      (fusion_verdict_html, video_timeline_pil, audio_heatmap_pil, details_text)
    """
    if video_path is None:
        return "<p>Please upload a video file with audio.</p>", None, None, ""

    try:
        data = _call_api("combined", video_path, "video/mp4")
    except Exception as e:
        return f"<p style='color:red'>Error: {e}</p>", None, None, ""

    badge   = _verdict_badge(data["verdict"], data["confidence"], data["risk_level"])
    v_img   = _b64_to_pil(data["video_result"]["timeline_b64"])
    a_img   = _b64_to_pil(data["audio_result"]["heatmap_b64"])

    mismatch_warning = ""
    if data["mismatch"]:
        mismatch_warning = (
            "\n⚠️  MODALITY MISMATCH DETECTED — "
            "visual and audio authenticity signals DISAGREE.\n"
            "This is a strong fraud indicator.\n"
        )

    details = (
        f"=== FUSION RESULT ===\n"
        f"Verdict          : {data['verdict'].upper()}\n"
        f"Risk Level       : {data['risk_level']}\n"
        f"Confidence       : {data['confidence']:.4f}\n"
        f"Mismatch         : {data['mismatch']}\n"
        f"{mismatch_warning}\n"
        f"Explanation:\n{data['explanation']}\n\n"
        f"=== VIDEO MODULE ===\n"
        f"Verdict    : {data['video_verdict'].upper()}\n"
        f"Confidence : {data['video_confidence']:.4f}\n\n"
        f"=== AUDIO MODULE ===\n"
        f"Verdict    : {data['audio_verdict'].upper()}\n"
        f"Confidence : {data['audio_confidence']:.4f}"
    )
    return badge, v_img, a_img, details


# ---------------------------------------------------------------------------
# Gradio Interface
# ---------------------------------------------------------------------------

CSS = """
body { font-family: 'Inter', sans-serif; background: #0f0f1a; color: #e0e0e0; }
.gradio-container { max-width: 960px; margin: auto; }
h1 { background: linear-gradient(135deg, #6c63ff, #e94560);
     -webkit-background-clip: text; -webkit-text-fill-color: transparent;
     font-size: 2.4rem; font-weight: 800; text-align: center; margin-bottom: 4px; }
.tab-nav button { font-weight: 600; }
"""

TITLE = """
<h1>🛡️ TRUTHGUARD</h1>
<p style='text-align:center;color:#aaa;margin-top:0'>
  Multi-Modal Deepfake Detection · Image · Video · Audio · Cross-Modal Fusion
</p>
"""

with gr.Blocks(css=CSS, title="TRUTHGUARD — Deepfake Detector") as demo:
    gr.HTML(TITLE)

    with gr.Tabs():

        # ── Tab 1: Image ─────────────────────────────────────────────────
        with gr.Tab("🖼️  Image"):
            gr.Markdown("### Upload an image to detect facial deepfakes")
            with gr.Row():
                img_input  = gr.Image(type="filepath", label="Input Image")
                img_heatmap = gr.Image(type="pil",      label="GradCAM Heatmap Overlay")
            img_badge   = gr.HTML(label="Verdict")
            img_details = gr.Textbox(label="Detailed Scores", lines=6, interactive=False)
            img_btn     = gr.Button("🔍 Analyse Image", variant="primary")
            img_btn.click(
                analyse_image,
                inputs=[img_input],
                outputs=[img_badge, img_heatmap, img_details],
            )

        # ── Tab 2: Video ─────────────────────────────────────────────────
        with gr.Tab("🎬  Video"):
            gr.Markdown("### Upload a video to detect frame-level deepfakes")
            with gr.Row():
                vid_input    = gr.Video(label="Input Video")
                vid_timeline = gr.Image(type="pil", label="Per-Frame Confidence Timeline")
            vid_badge   = gr.HTML(label="Verdict")
            vid_details = gr.Textbox(label="Detailed Scores", lines=10, interactive=False)
            vid_btn     = gr.Button("🔍 Analyse Video", variant="primary")
            vid_btn.click(
                analyse_video,
                inputs=[vid_input],
                outputs=[vid_badge, vid_timeline, vid_details],
            )

        # ── Tab 3: Audio ─────────────────────────────────────────────────
        with gr.Tab("🎙️  Audio"):
            gr.Markdown("### Upload audio to detect voice clones / synthetic speech")
            with gr.Row():
                aud_input   = gr.Audio(type="filepath", label="Input Audio")
                aud_heatmap = gr.Image(type="pil", label="Spectrogram + Attention Heatmap")
            aud_badge   = gr.HTML(label="Verdict")
            aud_details = gr.Textbox(label="Detailed Scores", lines=8, interactive=False)
            aud_btn     = gr.Button("🔍 Analyse Audio", variant="primary")
            aud_btn.click(
                analyse_audio,
                inputs=[aud_input],
                outputs=[aud_badge, aud_heatmap, aud_details],
            )

        # ── Tab 4: Combined ───────────────────────────────────────────────
        with gr.Tab("🔗  Combined (Fusion)"):
            gr.Markdown(
                "### Upload a video with audio track\n"
                "Runs independent video & audio analysis, then applies the "
                "**cross-modal fusion layer**.  "
                "A ⚠️ **MODALITY MISMATCH** flag is raised if visual and "
                "audio signals disagree — a strong deepfake indicator."
            )
            comb_input  = gr.Video(label="Input Video (with audio track)")
            comb_badge  = gr.HTML(label="Fusion Verdict")
            with gr.Row():
                comb_v_img = gr.Image(type="pil", label="Video — Frame Timeline")
                comb_a_img = gr.Image(type="pil", label="Audio — Attention Heatmap")
            comb_details = gr.Textbox(label="Full Fusion Report", lines=20, interactive=False)
            comb_btn     = gr.Button("🔍 Analyse (Fusion)", variant="primary")
            comb_btn.click(
                analyse_combined,
                inputs=[comb_input],
                outputs=[comb_badge, comb_v_img, comb_a_img, comb_details],
            )

    gr.Markdown(
        "<p style='text-align:center;color:#555;font-size:0.8rem'>"
        "TRUTHGUARD v1.0 · EfficientNet-B4 · Transformer · LSTM · GradCAM"
        "</p>"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Start the FastAPI backend in a background thread, then launch Gradio
    import threading
    import uvicorn
    import time

    def _start_api():
        # Import here to trigger model loading before Gradio starts
        from api.main import app as fastapi_app
        uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

    api_thread = threading.Thread(target=_start_api, daemon=True)
    api_thread.start()

    # Give the API a moment to start before Gradio tries to call it
    time.sleep(3)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
