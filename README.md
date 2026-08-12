---
title: TRUTHGUARD — Multi-Modal Deepfake Detection System
emoji: 🛡️
colorFrom: indigo
colorTo: red
sdk: gradio
sdk_version: 4.31.5
app_file: app.py
pinned: false
license: mit
short_description: TRUTHGUARD is a unified multi-modal deepfake detection framework analyzing images, video, and audio for AI manipulation.
---

# 🛡️ TRUTHGUARD — Unified Multi-Modal Deepfake Detection System

TRUTHGUARD is a multi-modal AI deepfake detection framework that independently analyzes **images**, **video**, and **audio** for synthetic manipulation, orchestrating decisions through a **Cross-Modal Fusion Layer**.

---

## 🌟 Key Features & Architecture

TRUTHGUARD strictly decouples each modality into an independently testable and swappable module:

```
                      ┌─────────────────────────┐
                      │    Gradio Dashboard     │
                      └────────────┬────────────┘
                                   │ HTTP
                      ┌────────────▼────────────┐
                      │     FastAPI Backend     │
                      └────┬───────┬───────┬────┘
                           │       │       │
      ┌────────────────────┘       │       └────────────────────┐
      │                            │                            │
┌─────▼──────────┐         ┌───────▼────────┐          ┌────────▼─────────┐
│  Image Module  │         │  Video Module  │          │   Audio Module   │
│ EfficientNet-B4│         │ Dual-Stream    │          │ CNN + BiLSTM     │
│ + GroupNorm    │         │ (Spatial/Temp) │          │ + Attention      │
└────────┬───────┘         └───────┬────────┘          └────────┬─────────┘
         │                         │                            │
         │ GradCAM                 │ Timeline                   │ Heatmap
         ▼                         ▼                            ▼
  Image Verdict             Video Verdict                Audio Verdict
                                   │                            │
                                   └──────────────┬─────────────┘
                                                  │
                                       ┌──────────▼──────────┐
                                       │ Cross-Modal Fusion  │
                                       │ (Mismatch Detector) │
                                       └─────────────────────┘
```

### Module 1: Image Deepfake Detection
- **Backbone**: `timm` EfficientNet-B4 pretrained on ImageNet.
- **BatchNorm Fix**: Converted all `BatchNorm2d` layers to `GroupNorm` (32 groups) to eliminate numerical instability during batch-size-1 inference.
- **Explainability**: GradCAM hooks generate pixel-level heatmaps highlighting manipulated facial regions.

### Module 2: Video Deepfake Detection
- **Dual-Stream Architecture**:
  - **Spatial Stream**: EfficientNet-B4 extracts 1792-D feature vectors per frame.
  - **Temporal Stream**: 8-head Transformer encoder with a learnable `[CLS]` token captures inter-frame forgery patterns across sampled frames.
- **Outputs**: Video-level verdict + per-frame forgery probabilities + interactive timeline visualization chart.

### Module 3: Audio Deepfake / Voice-Clone Detection
- **Preprocessing**: Converts audio to log-power mel-spectrograms (128 mel bins, 16 kHz).
- **Dual-Stream Architecture**:
  - **Spectral Stream**: Lightweight CNN extracts local time-frequency texture patches.
  - **Temporal Stream**: 2-layer Bidirectional LSTM with Additive Attention models speech prosody.
- **Explainability**: Normalized attention weights overlaid onto the mel-spectrogram.

### Module 4: Cross-Modal Fusion Layer (Mismatch Detection)
A transparent, rule-based decision engine that compares independent visual and audio verdicts:
1. **Agreement (Real/Real or Fake/Fake)**: Computes a weighted average confidence score (`0.55 * video + 0.45 * audio`).
2. **Disagreement (Modality Mismatch)**: If face looks real but voice is flagged fake (or vice versa), TRUTHGUARD conservatively reports **FAKE** with a **`HIGH RISK — MODALITY MISMATCH`** warning.
   - *Rationale*: Attackers frequently manipulate only one modality (e.g. voice-cloning over authentic video). A mismatch between visual and acoustic signals is itself a primary indicator of fraud.

---

## 📁 Repository Structure

```
TRUTHGUARD/
├── models/
│   ├── image/        # EfficientNet-B4, GradCAM, ImagePredictor
│   ├── video/        # Dual-stream EfficientNet-B4 + Transformer, VideoPredictor
│   └── audio/        # Spectral CNN + BiLSTM + Attention, AudioPredictor
├── fusion/           # Cross-modal fusion decision logic
├── api/              # FastAPI application, Pydantic schemas, and routes
├── ui/               # Gradio dashboard interface
├── utils/            # Shared file detection and visualization utilities
├── tests/            # Pytest suite for models, API, and fusion logic
├── app.py            # Entry point for Hugging Face Spaces
├── Dockerfile        # Container build definition (Python 3.11 CPU)
├── docker-compose.yml# Local multi-port deployment configuration
└── requirements.txt  # Project dependencies
```

---

## 🚀 Running Locally

### Option 1: Native Python Environment

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Application** (Starts both FastAPI backend on port 8000 and Gradio UI on port 7860):
   ```bash
   python app.py
   ```
   Open your browser to `http://localhost:7860`.

3. **Run API Only**:
   ```bash
   python api/main.py
   ```
   Interactive Swagger docs are available at `http://localhost:8000/docs`.

### Option 2: Docker / Docker Compose

```bash
docker-compose up --build
```
Access Gradio UI at `http://localhost:7860` and FastAPI docs at `http://localhost:8000/docs`.

---

## 🧪 Running Unit Tests

Run the full pytest suite:

```bash
pytest tests/ -v
```

---

## 🐳 Deployment to Hugging Face Spaces

1. Create a new **Docker** Space on Hugging Face Spaces (CPU Basic tier).
2. Push this repository to your Space repository.
3. Hugging Face Spaces will build the `Dockerfile` and launch `app.py` on port `7860`.

---

## 📄 License
MIT License.
