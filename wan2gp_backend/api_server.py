"""
FastAPI Server — Wan2GP Prompt Enhancer
=======================================
REST + SSE API for the Flutter desktop client.

Endpoints:
    GET  /api/status   — current model state
    GET  /api/models   — available model list
    GET  /api/modes    — available enhancement modes
    POST /api/load     — load a model  (SSE stream of progress lines)
    POST /api/unload   — unload current model
    POST /api/enhance  — run prompt enhancement (multipart: prompt + optional image/video)

Usage:
    python api_server.py
    python api_server.py --port 7860 --models-dir ckpts
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Wan2GP API Server for Flutter client")
_parser.add_argument("--models-dir", type=str, default="ckpts",
                     help="Directory where models are stored / downloaded (default: ckpts)")
_parser.add_argument("--port", type=int, default=7860, help="Port for the API server")
_parser.add_argument("--host", type=str, default="127.0.0.1",
                     help="Server bind address (default: 127.0.0.1)")
_args, _unknown_args = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Initialise backend service
# ---------------------------------------------------------------------------
import backend_service as svc

svc.set_models_dir(_args.models_dir)

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI(title="Wan2GP Prompt Enhancer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# GET /api/status
# ---------------------------------------------------------------------------
@app.get("/api/status")
def get_status():
    return svc.get_status()


# ---------------------------------------------------------------------------
# GET /api/models
# ---------------------------------------------------------------------------
@app.get("/api/models")
def get_models():
    return [
        {
            "no": no,
            "label": label,
            "backends": svc.BACKEND_CHOICES if no >= 3 else ["quanto_int8"],
        }
        for label, no in svc.MODEL_CHOICES
    ]


# ---------------------------------------------------------------------------
# GET /api/modes
# ---------------------------------------------------------------------------
@app.get("/api/modes")
def get_modes():
    result = []
    for label, (needs_image, video_prompt, text_prompt, _sys, needs_video) in svc.MODES.items():
        key = label.split("—")[0].strip()
        result.append({
            "key": key,
            "label": label,
            "needs_image": needs_image,
            "needs_video": needs_video,
            "video_prompt": video_prompt,
            "text_prompt": text_prompt,
        })
    return result


# ---------------------------------------------------------------------------
# POST /api/load  — SSE stream
# ---------------------------------------------------------------------------
@app.post("/api/load")
def load_model(body: dict):
    model_no: int = body.get("model_no", 3)
    backend: str = body.get("backend", "gguf")

    def _stream():
        for event in svc.load_model_gen(model_no, backend):
            # event is now a dict; serialise to JSON for the SSE payload
            if not isinstance(event, dict):
                event = {"type": "status", "status": str(event)}
            payload = json.dumps(event)
            yield f"data: {payload}\n\n"

    return StreamingResponse(_stream(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# GET /api/check_models
# ---------------------------------------------------------------------------
@app.get("/api/check_models")
def check_models(model_no: int = 3, backend: str = "gguf"):
    return svc.check_models(model_no, backend)


# ---------------------------------------------------------------------------
# POST /api/unload
# ---------------------------------------------------------------------------
@app.post("/api/unload")
def unload_model():
    msg = svc.unload_model()
    return {"status": msg}


# ---------------------------------------------------------------------------
# POST /api/enhance  — multipart form
# ---------------------------------------------------------------------------
@app.post("/api/enhance")
async def enhance(
    prompt: str = Form(""),
    mode: str = Form("T2V — Text → Video prompt"),
    thinking: bool = Form(False),
    max_tokens: int = Form(512),
    temperature: float = Form(0.6),
    top_p: float = Form(0.9),
    seed: int = Form(-1),
    custom_system_prompt: str = Form(""),
    image: UploadFile | None = File(None),
    video: UploadFile | None = File(None),
):
    pil_image = None
    video_path = None

    try:
        # Process uploaded image → PIL.Image
        if image is not None and image.filename:
            image_bytes = await image.read()
            if image_bytes:
                pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process uploaded video → temp file path
        if video is not None and video.filename:
            video_bytes = await video.read()
            if video_bytes:
                suffix = os.path.splitext(video.filename)[1] or ".mp4"
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                tmp.write(video_bytes)
                tmp.close()
                video_path = tmp.name

        enhanced, error = svc.enhance_prompt(
            prompt=prompt,
            image=pil_image,
            video_path=video_path,
            mode_label=mode,
            thinking_enabled=thinking,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            user_system_prompt=custom_system_prompt or None,
        )

        if error:
            return {"enhanced_text": "", "error": error}
        return {"enhanced_text": enhanced, "error": ""}

    finally:
        # Clean up temp video file
        if video_path and os.path.exists(video_path):
            try:
                os.unlink(video_path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=_args.host, port=_args.port)
