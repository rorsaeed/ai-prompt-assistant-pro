"""
Standalone Prompt Enhancer App — Wan2GP
=======================================
A Gradio 5 chat-style interface for enhancing text/image prompts using the
same models and inference code as the main Wan2GP application.

Supports all four enhancer model configurations:
  1 — Florence2 + LLaMA 3.2 (INT8)
  2 — Florence2 + LLaMA JoyCaption (INT8)
  3 — Qwen3.5-4B Abliterated (vision + text)
  4 — Qwen3.5-9B Abliterated (vision + text)

Models are downloaded automatically from HuggingFace into the ckpts/ folder.

Usage:
    python prompt_enhancer_app.py
    python prompt_enhancer_app.py --models-dir /path/to/models
    python prompt_enhancer_app.py --port 7861 --share
"""

from __future__ import annotations

import argparse
import os
import queue
import secrets
import sys
import threading
from typing import Optional

# ---------------------------------------------------------------------------
# Path bootstrap — must happen before any shared.* imports
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# CLI args (parsed early so we can set the models dir before importing fl)
# ---------------------------------------------------------------------------
_parser = argparse.ArgumentParser(description="Wan2GP Standalone Prompt Enhancer")
_parser.add_argument("--models-dir", type=str, default="ckpts",
                     help="Directory where models are stored / downloaded (default: ckpts)")
_parser.add_argument("--port", type=int, default=7860, help="Port for the Gradio server")
_parser.add_argument("--share", action="store_true", help="Create a public Gradio share link")
_parser.add_argument("--server-name", type=str, default="127.0.0.1",
                     help="Server bind address (default: 127.0.0.1)")
_args, _unknown_args = _parser.parse_known_args()

# ---------------------------------------------------------------------------
# Resolve the models directory before setting up files_locator
# ---------------------------------------------------------------------------
_models_dir = _args.models_dir
if not os.path.isabs(_models_dir):
    _models_dir = os.path.join(_ROOT, _models_dir)
os.makedirs(_models_dir, exist_ok=True)

# ---------------------------------------------------------------------------
# Core imports
# ---------------------------------------------------------------------------
import gradio as gr
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download

from shared.utils import files_locator as fl

# Tell the file locator where to look for (and download to) model files
fl.set_checkpoints_paths([_models_dir])

# Register GGUF file extension handler with mmgp so .gguf checkpoints are
# loaded via the custom reader instead of falling back to torch.load().
import shared.qtypes.gguf as _gguf_module
from mmgp.quant_router import register_file_extension as _register_ext, register_handler as _register_handler
_register_ext("gguf", _gguf_module)
_register_handler(_gguf_module)

from shared.prompt_enhancer.loader import (
    load_prompt_enhancer_runtime,
    unload_prompt_enhancer_models,
)
from shared.prompt_enhancer.audio_understanding import (
    analyze_videos_audio,
    unload_audio_understanding_models,
)
from shared.prompt_enhancer.prompt_enhance_utils import (
    generate_cinematic_prompt,
    T2V_CINEMATIC_PROMPT,
    T2I_VISUAL_PROMPT,
    T2T_TEXT_PROMPT,
    IT2V_CINEMATIC_PROMPT,
    IT2I_VISUAL_PROMPT,
    I2V_CINEMATIC_PROMPT,
    I2I_VISUAL_PROMPT,
    VT2V_CINEMATIC_PROMPT,
    V2V_CINEMATIC_PROMPT,
    VT2I_VISUAL_PROMPT,
    V2I_VISUAL_PROMPT,
)

try:
    from mmgp import offload as _offload
    _MMGP_AVAILABLE = True
except ImportError:
    _MMGP_AVAILABLE = False
    _offload = None

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MODEL_CHOICES = [
    ("Florence2 + LLaMA 3.2  (mode 1)", 1),
    ("Florence2 + LLaMA JoyCaption  (mode 2)", 2),
    ("Qwen3.5-4B Abliterated  (mode 3)", 3),
    ("Qwen3.5-9B Abliterated  (mode 4)", 4),
]
MODEL_CHOICE_LABELS = [label for label, _ in MODEL_CHOICES]
MODEL_NO_FROM_LABEL = {label: no for label, no in MODEL_CHOICES}

BACKEND_CHOICES = ["quanto_int8", "gguf"]

# Enhancement modes: (needs_image, video_prompt, text_prompt, custom_system_prompt, needs_video)
MODES = {
    "T2V — Text → Video prompt":        (False, True,  False, None,                False),
    "T2I — Text → Image prompt":        (False, False, False, None,                False),
    "T2T — Text → Speech/script":       (False, False, True,  None,                False),
    "IT2V — Image + Text → Video":      (True,  True,  False, None,                False),
    "IT2I — Image + Text → Image":      (True,  False, False, None,                False),
    "I2V — Image only → Video":         (True,  True,  False, I2V_CINEMATIC_PROMPT, False),
    "I2I — Image only → Image":         (True,  False, False, I2I_VISUAL_PROMPT,   False),
    "VT2V — Video + Text → Video":      (False, True,  False, None,                True),
    "VT2I — Video + Text → Image":      (False, False, False, None,                True),
    "V2V — Video only → Video":         (False, True,  False, V2V_CINEMATIC_PROMPT, True),
    "V2I — Video only → Image":         (False, False, False, V2I_VISUAL_PROMPT,   True),
}
MODE_LABELS = list(MODES.keys())

# ---------------------------------------------------------------------------
# Global runtime state (protected by a lock)
# ---------------------------------------------------------------------------
_state_lock = threading.Lock()
_runtime = None          # PromptEnhancerRuntime
_offloadobj = None       # mmgp offload object
_loaded_model_no: int = 0
_loaded_backend: str = ""

# Set by load_model_gen so _process_files_def can report progress to the UI
_download_status_callback: Optional[callable] = None


# ---------------------------------------------------------------------------
# Download helper — mirrors process_files_def in wgp.py
# ---------------------------------------------------------------------------
def _process_files_def(repoId=None, sourceFolderList=None, fileList=None, targetFolderList=None):
    """Download model files from HuggingFace if they are not already present."""
    if targetFolderList is None:
        targetFolderList = [None] * len(sourceFolderList)

    def _notify(msg: str):
        print(f"[Downloader] {msg}")
        if _download_status_callback is not None:
            try:
                _download_status_callback(msg)
            except Exception:
                pass

    for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList, fileList):
        if targetFolder is not None and len(targetFolder) == 0:
            targetFolder = None

        explicit_target = targetFolder if targetFolder is not None else (sourceFolder if len(sourceFolder) > 0 else None)
        targetRoot = fl.get_smart_download_root(explicit_target)
        local_dir = os.path.join(targetRoot, targetFolder) if targetFolder is not None else targetRoot

        if len(files) == 0:
            lookup = sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder)
            if fl.locate_folder(lookup, error_if_none=False) is None:
                _notify(f"Downloading folder  {repoId}/{sourceFolder} …")
                snapshot_download(repo_id=repoId, allow_patterns=sourceFolder + "/*", local_dir=local_dir)
                _notify(f"Done  {repoId}/{sourceFolder}")
        else:
            for onefile in files:
                if len(sourceFolder) > 0:
                    lookup = (sourceFolder + "/" + onefile) if targetFolder is None else os.path.join(targetFolder, sourceFolder, onefile)
                    if fl.locate_file(lookup, error_if_none=False) is None:
                        _notify(f"Downloading  {repoId}/{sourceFolder}/{onefile} …")
                        hf_hub_download(repo_id=repoId, filename=onefile, local_dir=local_dir, subfolder=sourceFolder)
                        _notify(f"Done  {onefile}")
                    else:
                        _notify(f"Already present  {onefile}")
                else:
                    lookup = onefile if targetFolder is None else os.path.join(targetFolder, onefile)
                    if fl.locate_file(lookup, error_if_none=False) is None:
                        _notify(f"Downloading  {repoId}/{onefile} …")
                        hf_hub_download(repo_id=repoId, filename=onefile, local_dir=local_dir)
                        _notify(f"Done  {onefile}")
                    else:
                        _notify(f"Already present  {onefile}")


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def _is_loaded(model_no: int, backend: str) -> bool:
    return _runtime is not None and _loaded_model_no == model_no and _loaded_backend == backend


def _load_model_internal(model_no: int, backend: str, status_cb: callable) -> str:
    """
    The actual synchronous load logic.
    status_cb(msg) is called for each progress step.
    Returns a final status string.
    """
    global _runtime, _offloadobj, _loaded_model_no, _loaded_backend, _download_status_callback

    with _state_lock:
        if _is_loaded(model_no, backend):
            return f"Model already loaded: {MODEL_CHOICE_LABELS[model_no - 1]}"

        _unload_locked()

        status_cb(f"Preparing to load model {MODEL_CHOICE_LABELS[model_no - 1]} …")
        _download_status_callback = status_cb
        try:
            runtime = load_prompt_enhancer_runtime(
                _process_files_def,
                enhancer_enabled=model_no,
                lm_decoder_engine="",
                qwen_backend=backend,
            )
        except Exception as exc:
            return f"Error: {exc}"
        finally:
            _download_status_callback = None

        _runtime = runtime
        _loaded_model_no = model_no
        _loaded_backend = backend

        status_cb("Setting up VRAM offloading …")
        if _MMGP_AVAILABLE and runtime.pipe_models:
            pipe = dict(runtime.pipe_models)
            budgets = dict(runtime.budgets) if runtime.budgets else {}
            budgets.setdefault("*", 3000)
            try:
                _offloadobj = _offload.profile(pipe, profile_no=4, budgets=budgets)
            except Exception as exc:
                print(f"[Enhancer] mmgp offload.profile failed ({exc}); models will remain fully in memory.")
                _offloadobj = None

        label = MODEL_CHOICE_LABELS[model_no - 1]
        return f"Loaded: {label}"


def load_model_gen(model_no: int, backend: str):
    """
    Generator version of load_model that streams progress strings.
    Runs the actual loading in a background thread and yields each
    status message as it arrives.
    """
    msg_queue: queue.Queue = queue.Queue()
    done_event = threading.Event()
    result_holder: list = []

    def _cb(msg: str):
        msg_queue.put(msg)

    def _worker():
        result = _load_model_internal(model_no, backend, _cb)
        result_holder.append(result)
        done_event.set()
        msg_queue.put(None)  # sentinel

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        msg = msg_queue.get()
        if msg is None:
            break
        yield msg

    final = result_holder[0] if result_holder else "Unknown error"
    yield final


def load_model(model_no: int, backend: str) -> str:
    """Synchronous convenience wrapper. Returns the final status string."""
    last = ""
    for msg in load_model_gen(model_no, backend):
        last = msg
    return last


def unload_model() -> str:
    """Unload the current model and free VRAM. Returns a status string."""
    with _state_lock:
        if _runtime is None:
            return "No model is currently loaded."
        _unload_locked()
        return "Model unloaded."


def _unload_locked():
    """Unload without acquiring the lock (caller must hold _state_lock)."""
    global _runtime, _offloadobj, _loaded_model_no, _loaded_backend
    if _runtime is not None:
        unload_audio_understanding_models(_runtime)
        unload_prompt_enhancer_models(_runtime.image_caption_model, _runtime.llm_model)
        _runtime = None
    if _offloadobj is not None:
        try:
            _offloadobj.unload_all()
        except Exception:
            pass
        _offloadobj = None
    _loaded_model_no = 0
    _loaded_backend = ""


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def enhance_prompt(
    prompt: str,
    image: Optional[Image.Image],
    video_path: Optional[str],
    mode_label: str,
    thinking_enabled: bool,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    seed: int,
) -> tuple[str, str]:
    """
    Run prompt enhancement.

    Returns (enhanced_text, error_message).
    On success error_message is "".
    On failure enhanced_text is "" and error_message contains the message.
    """
    with _state_lock:
        if _runtime is None:
            return "", "No model loaded. Click 'Load Model' first."

        needs_image, video_prompt, text_prompt, custom_system_prompt, needs_video = MODES[mode_label]

        if needs_image and image is None:
            return "", f"Mode '{mode_label}' requires an image. Please upload one."

        if needs_video and not video_path:
            return "", f"Mode '{mode_label}' requires a video. Please upload one."

        if needs_video and _loaded_model_no < 3:
            return "", "Video modes are only supported with Qwen3.5 models (modes 3 & 4). Please load a Qwen3.5 model."

        images: Optional[list] = None
        videos: Optional[list] = None
        prompt_text = prompt.strip()

        if needs_video:
            videos = [video_path]
            # Pure-video modes use a dummy user prompt
            if custom_system_prompt is not None:
                prompt_text = "a video" if not prompt_text else prompt_text
        elif needs_image:
            images = [image]
            # Pure-image modes use a dummy user prompt
            if custom_system_prompt is not None:
                prompt_text = "an image" if not prompt_text else prompt_text

        if not prompt_text and not needs_image and not needs_video:
            return "", "Please enter a prompt."

        try:
            results = generate_cinematic_prompt(
                image_caption_model=_runtime.image_caption_model,
                image_caption_processor=_runtime.image_caption_processor,
                prompt_enhancer_model=_runtime.llm_model,
                prompt_enhancer_tokenizer=_runtime.llm_tokenizer,
                prompt=[prompt_text],
                images=images,
                videos=videos,
                video_prompt=video_prompt,
                text_prompt=text_prompt,
                max_new_tokens=max_new_tokens,
                prompt_enhancer_instructions=custom_system_prompt,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=None,
                seed=seed if seed >= 0 else secrets.randbits(31),
                post_image_caption_hook=None,
                video_audio_analyzer=(lambda paths: analyze_videos_audio(_runtime, paths)) if needs_video else None,
                thinking_enabled=thinking_enabled,
            )
        except Exception as exc:
            import traceback
            traceback.print_exc()
            return "", f"Inference error: {exc}"

        if _offloadobj is not None:
            try:
                _offloadobj.unload_all()
            except Exception:
                pass

        enhanced = results[0] if results else ""
        return enhanced, ""


# ---------------------------------------------------------------------------
# Gradio UI helpers
# ---------------------------------------------------------------------------
def _backend_visibility(model_label: str) -> gr.update:
    """Show quantization backend selector only for Qwen3.5 models (3 & 4)."""
    model_no = MODEL_NO_FROM_LABEL.get(model_label, 1)
    return gr.update(visible=model_no >= 3)


def _mode_image_required(mode_label: str) -> bool:
    return MODES[mode_label][0]


def _mode_video_required(mode_label: str) -> bool:
    return MODES[mode_label][4]


def _mode_visibility_updates(mode_label: str):
    """Return visibility updates for image and video upload components."""
    needs_image = MODES[mode_label][0]
    needs_video = MODES[mode_label][4]
    return gr.update(visible=not needs_video), gr.update(visible=needs_video)


# ---------------------------------------------------------------------------
# Gradio app
# ---------------------------------------------------------------------------
def build_app() -> gr.Blocks:
    with gr.Blocks(title="Wan2GP Prompt Enhancer") as demo:
        gr.Markdown(
            "# Wan2GP Prompt Enhancer\n"
            "Enhance text and image prompts for video/image generation using LLM models. "
            "Select a model below, click **Load Model**, then type your prompt and click **Enhance**."
        )

        with gr.Row():
            # ---- Left sidebar ----
            with gr.Column(scale=1, min_width=280):
                gr.Markdown("### Model")
                model_dropdown = gr.Dropdown(
                    choices=MODEL_CHOICE_LABELS,
                    value=MODEL_CHOICE_LABELS[0],
                    label="Enhancer model",
                    interactive=True,
                )
                backend_radio = gr.Radio(
                    choices=BACKEND_CHOICES,
                    value="quanto_int8",
                    label="Qwen3.5 quantization backend",
                    visible=False,
                    interactive=True,
                )
                with gr.Row():
                    load_btn = gr.Button("Load Model", variant="primary")
                    unload_btn = gr.Button("Unload", variant="secondary")
                status_md = gr.Markdown("*No model loaded.*")
                download_log = gr.Textbox(
                    label="Download / load progress",
                    lines=6,
                    max_lines=6,
                    interactive=False,
                    visible=False,
                )

                gr.Markdown("---")
                gr.Markdown("### Parameters")
                mode_radio = gr.Radio(
                    choices=MODE_LABELS,
                    value=MODE_LABELS[0],
                    label="Enhancement mode",
                    interactive=True,
                )
                thinking_check = gr.Checkbox(
                    label="Enable thinking mode (Qwen3.5 only)",
                    value=False,
                    interactive=True,
                )
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=512, step=32,
                    label="Max new tokens",
                )
                temperature_slider = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.6, step=0.05,
                    label="Temperature",
                )
                top_p_slider = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.9, step=0.05,
                    label="Top-p",
                )
                seed_number = gr.Number(
                    value=-1, label="Seed  (-1 = random)", precision=0, minimum=-1,
                )

            # ---- Main chat area ----
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    label="Enhancement history",
                    height=520,
                )
                image_upload = gr.Image(
                    label="Input image (optional — required for I2V / I2I / IT2V / IT2I modes)",
                    type="pil",
                    height=200,
                    sources=["upload", "clipboard"],
                )
                video_upload = gr.Video(
                    label="Input video (required for VT2V / VT2I / V2V / V2I modes — Qwen3.5 only)",
                    height=200,
                    sources=["upload"],
                    visible=False,
                )
                prompt_box = gr.Textbox(
                    label="Your prompt",
                    placeholder="Type your prompt here…",
                    lines=3,
                )
                with gr.Row():
                    enhance_btn = gr.Button("✨ Enhance", variant="primary", scale=3)
                    clear_btn = gr.Button("🗑 Clear chat", variant="secondary", scale=1)

        # ---- Event handlers ----

        # Show/hide backend selector depending on chosen model
        model_dropdown.change(
            fn=_backend_visibility,
            inputs=model_dropdown,
            outputs=backend_radio,
        )

        # Show/hide image vs video upload depending on chosen mode
        mode_radio.change(
            fn=_mode_visibility_updates,
            inputs=mode_radio,
            outputs=[image_upload, video_upload],
        )

        # Load model button — streams download/load progress line-by-line
        def _on_load(model_label, backend):
            model_no = MODEL_NO_FROM_LABEL[model_label]
            log_lines: list[str] = []
            # Show the log box while loading
            yield (
                gr.update(value="**Status:** Loading …"),
                gr.update(value="", visible=True),
            )
            for msg in load_model_gen(model_no, backend):
                log_lines.append(msg)
                # Keep last 50 lines to avoid unbounded growth
                if len(log_lines) > 50:
                    log_lines = log_lines[-50:]
                log_text = "\n".join(log_lines)
                yield (
                    gr.update(value=f"**Status:** {msg}"),
                    gr.update(value=log_text, visible=True),
                )
            # Final state: hide the log after a moment by leaving it visible
            # (user can see the complete trace; it stays until next load)

        load_btn.click(
            fn=_on_load,
            inputs=[model_dropdown, backend_radio],
            outputs=[status_md, download_log],
        )

        # Unload model button
        def _on_unload():
            msg = unload_model()
            return gr.update(value=f"**Status:** {msg}")

        unload_btn.click(
            fn=_on_unload,
            inputs=[],
            outputs=status_md,
        )

        # Enhance button
        def _on_enhance(
            history,
            prompt,
            image,
            video,
            mode_label,
            thinking,
            max_tokens,
            temperature,
            top_p,
            seed,
        ):
            if not prompt.strip() and not _mode_image_required(mode_label) and not _mode_video_required(mode_label):
                gr.Warning("Please enter a prompt before enhancing.")
                return history, prompt

            enhanced, error = enhance_prompt(
                prompt=prompt,
                image=image,
                video_path=video,
                mode_label=mode_label,
                thinking_enabled=thinking,
                max_new_tokens=int(max_tokens),
                temperature=float(temperature),
                top_p=float(top_p),
                seed=int(seed),
            )

            if error:
                gr.Warning(error)
                return history, prompt

            # Build a rich user message showing mode + media indicator
            media_indicator = " 🎥" if video else (" 🖼" if image is not None else "")
            user_msg = f"**[{mode_label.split('—')[0].strip()}]**{media_indicator}  {prompt.strip()}" if prompt.strip() else f"**[{mode_label.split('—')[0].strip()}]**{media_indicator}"
            assistant_msg = enhanced

            history = history or []
            history = history + [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": assistant_msg},
            ]
            return history, enhanced

        enhance_btn.click(
            fn=_on_enhance,
            inputs=[
                chatbot,
                prompt_box,
                image_upload,
                video_upload,
                mode_radio,
                thinking_check,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                seed_number,
            ],
            outputs=[chatbot, prompt_box],
        )

        # Allow pressing Enter in the prompt box to trigger enhance
        prompt_box.submit(
            fn=_on_enhance,
            inputs=[
                chatbot,
                prompt_box,
                image_upload,
                video_upload,
                mode_radio,
                thinking_check,
                max_tokens_slider,
                temperature_slider,
                top_p_slider,
                seed_number,
            ],
            outputs=[chatbot, prompt_box],
        )

        # Clear chat
        clear_btn.click(fn=lambda: [], inputs=[], outputs=chatbot)

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = build_app()
    app.launch(
        server_name=_args.server_name,
        server_port=_args.port,
        share=_args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )
