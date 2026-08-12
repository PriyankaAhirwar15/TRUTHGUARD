"""
app.py
------
Hugging Face Spaces Entry Point for TRUTHGUARD.

Hugging Face Spaces expects `app.py` in the root directory.
This script launches both the FastAPI backend (in a background thread)
and the Gradio dashboard interface on port 7860.
"""

import sys
import os
import time
import threading
import uvicorn

# Add project root to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from api.main import app as fastapi_app
from ui.app import demo

def start_backend():
    print("[TRUTHGUARD] Starting FastAPI backend on port 8000...")
    uvicorn.run(fastapi_app, host="127.0.0.1", port=8000, log_level="warning")

if __name__ == "__main__":
    # Start FastAPI server in background thread
    t = threading.Thread(target=start_backend, daemon=True)
    t.start()

    # Wait for API to initialize models
    time.sleep(3)

    # Launch Gradio UI on Hugging Face default port (7860)
    print("[TRUTHGUARD] Launching Gradio UI on port 7860...")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
    )
