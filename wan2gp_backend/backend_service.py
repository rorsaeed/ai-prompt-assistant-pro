"""
Backend Service — Wan2GP Prompt Enhancer
========================================
Extracted business logic from prompt_enhancer_app.py so it can be shared
between the Gradio UI and the FastAPI server (for the Flutter desktop client).

This module is import-safe: it bootstraps sys.path and initialises the model
loaders exactly once, regardless of which entry point loads it.
"""

from __future__ import annotations

import os
import queue
import requests
import secrets
import sys
import threading
import time
from typing import Optional

from PIL import Image

# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---------------------------------------------------------------------------
# Models directory — can be overridden before first use via set_models_dir()
# ---------------------------------------------------------------------------
_models_dir: str = os.path.join(_ROOT, "ckpts")


def set_models_dir(path: str) -> None:
    """Set the directory where models are stored / downloaded."""
    global _models_dir
    _models_dir = path if os.path.isabs(path) else os.path.join(_ROOT, path)
    os.makedirs(_models_dir, exist_ok=True)
    _setup_locator()


def _setup_locator() -> None:
    from shared.utils import files_locator as fl
    fl.set_checkpoints_paths([_models_dir])


# ---------------------------------------------------------------------------
# Lazy one-time heavy imports (called after models dir is set)
# ---------------------------------------------------------------------------
_initialised = False


def initialise() -> None:
    """Run heavy imports and register GGUF handler. Idempotent."""
    global _initialised
    if _initialised:
        return
    os.makedirs(_models_dir, exist_ok=True)
    _setup_locator()

    import shared.qtypes.gguf as _gguf_module
    from mmgp.quant_router import (
        register_file_extension as _register_ext,
        register_handler as _register_handler,
    )
    _register_ext("gguf", _gguf_module)
    _register_handler(_gguf_module)
    _initialised = True


# ---------------------------------------------------------------------------
# Lazy imports that depend on initialise()
# ---------------------------------------------------------------------------
def _get_loader():
    from shared.prompt_enhancer.loader import (
        load_prompt_enhancer_runtime,
        unload_prompt_enhancer_models,
    )
    return load_prompt_enhancer_runtime, unload_prompt_enhancer_models


def _get_generate():
    from shared.prompt_enhancer.prompt_enhance_utils import generate_cinematic_prompt
    return generate_cinematic_prompt


def _get_audio_helpers():
    from shared.prompt_enhancer.audio_understanding import (
        analyze_videos_audio,
        unload_audio_understanding_models,
    )
    return analyze_videos_audio, unload_audio_understanding_models


def _get_offload():
    try:
        from mmgp import offload
        return offload
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Prompt imports — always safe (pure constants)
# ---------------------------------------------------------------------------
from shared.prompt_enhancer.prompt_enhance_utils import (
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
_download_status_callback: Optional[callable] = None

# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------
from huggingface_hub import hf_hub_download, snapshot_download
from shared.utils import files_locator as fl

# ---------------------------------------------------------------------------
# Approximate file sizes (bytes) — fallback when HF HEAD request fails
# ---------------------------------------------------------------------------
_APPROX_FILE_SIZES: dict = {
    # Qwen3.5-4B
    "Qwen3.5-4B-Abliterated-text-Q4_K_M.gguf": 2_500_000_000,
    "Qwen3.5-4B-vision_bf16.safetensors": 700_000_000,
    "Qwen3.5-4B-Abliterated_quanto_bf16_int8.safetensors": 5_000_000_000,
    # Qwen3.5-9B
    "Qwen3.5-9B-Abliterated-text-Q4_K_M.gguf": 5_500_000_000,
    "Qwen3.5-9B-vision_bf16.safetensors": 1_400_000_000,
    "Qwen3.5-9B-Abliterated_quanto_bf16_int8.safetensors": 10_000_000_000,
    # Florence2 + LLaMA / JoyCaption
    "model.safetensors": 770_000_000,
    "Llama3_2_quanto_bf16_int8.safetensors": 3_000_000_000,
    "llama_joycaption_quanto_bf16_int8.safetensors": 9_000_000_000,
    # Tokenizer / misc (small but included for totals)
    "tokenizer.json": 11_000_000,
    "merges.txt": 456_000,
    "vocab.json": 798_000,
    "normalizer.json": 3_000_000,
    "preprocessor_config.json": 2_000,
    "special_tokens_map.json": 3_000,
    "pytorch_model.bin": 1_500_000_000,
}
_APPROX_FILE_SIZES_BY_FOLDER: dict[tuple[str, str], int] = {
    ("Whisper_Large_V3_Turbo", "model.safetensors"): 1_650_000_000,
}


def _approx_file_size(folder: str, filename: str) -> int:
    return _APPROX_FILE_SIZES_BY_FOLDER.get((folder, filename), _APPROX_FILE_SIZES.get(filename, 0))

_QWEN_AUDIO_MODEL_SPECS = [
    {"folder": "Whisper_Large_V3_Turbo", "repo": "openai/whisper-large-v3-turbo", "files": [
        "config.json", "generation_config.json", "merges.txt", "model.safetensors",
        "normalizer.json", "preprocessor_config.json", "special_tokens_map.json",
        "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]},
    {"folder": "Clap_Htsat_Unfused", "repo": "laion/clap-htsat-unfused", "files": [
        "config.json", "merges.txt", "preprocessor_config.json", "pytorch_model.bin",
        "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json", "vocab.json",
    ]},
]


def _hf_file_size(repo_id: str, subfolder: str, filename: str) -> int:
    """Get the real file size via a HuggingFace Hub HEAD request.
    Falls back to _APPROX_FILE_SIZES on any error."""
    try:
        from huggingface_hub import hf_hub_url
        path = f"{subfolder}/{filename}" if subfolder else filename
        url = hf_hub_url(repo_id=repo_id, filename=path)
        r = requests.head(url, allow_redirects=True, timeout=10)
        cl = r.headers.get("Content-Length")
        if cl:
            return int(cl)
    except Exception:
        pass
    return _APPROX_FILE_SIZES.get(filename, 0)

# File specs used by check_models — avoids importing heavy ML/torch modules.
_FLORENCE2_FILES = [
    "config.json", "configuration_florence2.py", "model.safetensors",
    "preprocessor_config.json", "tokenizer.json", "tokenizer_config.json",
]
_MODEL_FILE_SPECS: dict = {
    1: [
        {"folder": "Florence2",  "repo": "DeepBeepMeep/LTX_Video", "files": _FLORENCE2_FILES},
        {"folder": "Llama3_2",   "repo": "DeepBeepMeep/LTX_Video", "files": [
            "config.json", "generation_config.json",
            "Llama3_2_quanto_bf16_int8.safetensors",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        ]},
    ],
    2: [
        {"folder": "Florence2",  "repo": "DeepBeepMeep/LTX_Video", "files": _FLORENCE2_FILES},
        {"folder": "llama-joycaption-beta-one-hf-llava", "repo": "DeepBeepMeep/LTX_Video", "files": [
            "config.json", "llama_config.json",
            "llama_joycaption_quanto_bf16_int8.safetensors",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
        ]},
    ],
    3: {
        "gguf": [{"folder": "Qwen3_5_4B_Abliterated", "repo": "DeepBeepMeep/Wan2.1", "files": [
            "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
            "Qwen3.5-4B-Abliterated-text-Q4_K_M.gguf", "Qwen3.5-4B-vision_bf16.safetensors",
        ]}, *_QWEN_AUDIO_MODEL_SPECS],
        "quanto_int8": [{"folder": "Qwen3_5_4B_Abliterated", "repo": "DeepBeepMeep/Wan2.1", "files": [
            "chat_template.jinja", "config.json", "generation_config.json", "merges.txt",
            "tokenizer.json", "tokenizer_config.json", "video_preprocessor_config.json", "vocab.json",
            "Qwen3.5-4B-Abliterated_quanto_bf16_int8.safetensors", "Qwen3.5-4B-vision_bf16.safetensors",
        ]}, *_QWEN_AUDIO_MODEL_SPECS],
    },
    4: {
        "gguf": [{"folder": "Qwen3_5_9B_Abliterated", "repo": "DeepBeepMeep/Wan2.1", "files": [
            "chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json",
            "video_preprocessor_config.json", "vocab.json",
            "Qwen3.5-9B-Abliterated-text-Q4_K_M.gguf", "Qwen3.5-9B-vision_bf16.safetensors",
        ]}, *_QWEN_AUDIO_MODEL_SPECS],
        "quanto_int8": [{"folder": "Qwen3_5_9B_Abliterated", "repo": "DeepBeepMeep/Wan2.1", "files": [
            "chat_template.jinja", "config.json", "tokenizer.json", "tokenizer_config.json",
            "video_preprocessor_config.json", "vocab.json",
            "Qwen3.5-9B-Abliterated_quanto_bf16_int8.safetensors", "Qwen3.5-9B-vision_bf16.safetensors",
        ]}, *_QWEN_AUDIO_MODEL_SPECS],
    },
}


def _process_files_def(repoId=None, sourceFolderList=None, fileList=None, targetFolderList=None):
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


def _download_with_requests(
    repo_id: str, filename: str, subfolder: str, local_dir: str, on_progress
) -> None:
    """Download a single HF Hub file via streaming requests, calling on_progress(done, total, speed_bps)."""
    from huggingface_hub import hf_hub_url
    url = hf_hub_url(repo_id=repo_id, filename=filename,
                     subfolder=subfolder if subfolder else None)
    dest_dir = os.path.join(local_dir, subfolder) if subfolder else local_dir
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, filename)
    with requests.get(url, stream=True, allow_redirects=True, timeout=300) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        last_t = time.monotonic()
        last_b = 0
        speed = 0
        with open(dest_path, "wb") as fh:
            for chunk in resp.iter_content(chunk_size=512 * 1024):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    now = time.monotonic()
                    dt = now - last_t
                    if dt >= 0.4 and on_progress:
                        speed = max(0, (done - last_b) / dt)
                        last_t = now
                        last_b = done
                        on_progress(done, total, int(speed))
    if on_progress:
        on_progress(done, total, 0)


def _make_process_files_def(progress_cb):
    """Return a process_files_def that streams structured progress events via progress_cb."""

    def _emit(event: dict):
        print(f"[Downloader] {event.get('status', '')}")
        if progress_cb is not None:
            try:
                progress_cb(event)
            except Exception:
                pass

    def _def(repoId=None, sourceFolderList=None, fileList=None, targetFolderList=None):
        if targetFolderList is None:
            targetFolderList = [None] * len(sourceFolderList)

        for targetFolder, sourceFolder, files in zip(targetFolderList, sourceFolderList, fileList):
            if targetFolder is not None and len(targetFolder) == 0:
                targetFolder = None

            explicit_target = targetFolder if targetFolder is not None else (sourceFolder if len(sourceFolder) > 0 else None)
            targetRoot = fl.get_smart_download_root(explicit_target)
            local_dir = os.path.join(targetRoot, targetFolder) if targetFolder is not None else targetRoot

            if len(files) == 0:
                lookup = sourceFolder if targetFolder is None else os.path.join(targetFolder, sourceFolder)
                if fl.locate_folder(lookup, error_if_none=False) is None:
                    _emit({"type": "status", "status": f"Downloading folder {repoId}/{sourceFolder}…"})
                    snapshot_download(repo_id=repoId, allow_patterns=sourceFolder + "/*", local_dir=local_dir)
                    _emit({"type": "status", "status": f"Done {sourceFolder}"})
            else:
                for onefile in files:
                    if len(sourceFolder) > 0:
                        lookup = (sourceFolder + "/" + onefile) if targetFolder is None else os.path.join(targetFolder, sourceFolder, onefile)
                    else:
                        lookup = onefile if targetFolder is None else os.path.join(targetFolder, onefile)

                    if fl.locate_file(lookup, error_if_none=False) is not None:
                        _emit({"type": "file_present", "status": f"Already present: {onefile}", "file": onefile})
                        continue

                    approx = _approx_file_size(targetFolder or sourceFolder or "", onefile)
                    _emit({"type": "download_start", "status": f"Downloading {onefile}…",
                           "file": onefile, "size_bytes": approx})

                    def _on_prog(done, total, spd, _f=onefile, _a=approx):
                        _emit({
                            "type": "download_progress",
                            "status": f"Downloading {_f}: {done // 1_000_000} MB / {max(total, done) // 1_000_000} MB",
                            "file": _f,
                            "bytes_done": done,
                            "bytes_total": total if total > 0 else _a,
                            "speed_bps": spd,
                        })

                    _download_with_requests(
                        repo_id=repoId,
                        filename=onefile,
                        subfolder=sourceFolder,
                        local_dir=local_dir,
                        on_progress=_on_prog,
                    )
                    _emit({"type": "download_done", "status": f"Downloaded: {onefile}", "file": onefile})

    return _def


# ---------------------------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------------------------
def _is_loaded(model_no: int, backend: str) -> bool:
    return _runtime is not None and _loaded_model_no == model_no and _loaded_backend == backend


def _load_model_internal(model_no: int, backend: str, status_cb: callable) -> str:
    global _runtime, _offloadobj, _loaded_model_no, _loaded_backend

    load_fn, _ = _get_loader()
    _offload = _get_offload()

    with _state_lock:
        if _is_loaded(model_no, backend):
            return f"Model already loaded: {MODEL_CHOICE_LABELS[model_no - 1]}"
        _unload_locked()

        status_cb({"type": "status", "status": f"Preparing to load model {MODEL_CHOICE_LABELS[model_no - 1]} …"})
        try:
            runtime = load_fn(
                _make_process_files_def(status_cb),
                enhancer_enabled=model_no,
                lm_decoder_engine="",
                qwen_backend=backend,
            )
        except Exception as exc:
            return f"Error: {exc}"

        _runtime = runtime
        _loaded_model_no = model_no
        _loaded_backend = backend

        status_cb("Setting up VRAM offloading …")
        if _offload is not None and runtime.pipe_models:
            pipe = dict(runtime.pipe_models)
            budgets = dict(runtime.budgets) if runtime.budgets else {}
            budgets.setdefault("*", 3000)
            try:
                _offloadobj = _offload.profile(pipe, profile_no=4, budgets=budgets)
            except Exception as exc:
                print(f"[Enhancer] mmgp offload.profile failed ({exc}); models stay in memory.")
                _offloadobj = None

        label = MODEL_CHOICE_LABELS[model_no - 1]
        return f"Loaded: {label}"


def load_model_gen(model_no: int, backend: str):
    """Generator that yields structured progress dicts while loading/downloading."""
    initialise()
    msg_queue: queue.Queue = queue.Queue()
    result_holder: list = []

    def _cb(event):
        # Accept both plain strings (legacy) and dicts
        if isinstance(event, str):
            event = {"type": "status", "status": event}
        msg_queue.put(event)

    def _worker():
        result = _load_model_internal(model_no, backend, _cb)
        result_holder.append(result)
        msg_queue.put(None)

    t = threading.Thread(target=_worker, daemon=True)
    t.start()

    while True:
        event = msg_queue.get()
        if event is None:
            break
        yield event

    final_status = result_holder[0] if result_holder else "Unknown error"
    yield {"type": "complete", "status": final_status}


def load_model(model_no: int, backend: str) -> str:
    last = ""
    for event in load_model_gen(model_no, backend):
        if isinstance(event, dict):
            last = event.get("status", last)
        else:
            last = str(event)
    return last


def check_models(model_no: int, backend: str) -> dict:
    """Return which files are present/missing for the given model without loading anything."""
    initialise()
    model_no = int(model_no)
    backend = str(backend or "gguf")

    spec = _MODEL_FILE_SPECS.get(model_no)
    if spec is None:
        return {"error": f"Unknown model_no {model_no}", "all_present": False,
                "files": [], "total_missing_bytes": 0}

    # Models 3 & 4 have per-backend specs
    if isinstance(spec, dict):
        folder_specs = spec.get(backend) or spec.get("gguf", [])
    else:
        folder_specs = spec

    all_files = []
    for fs in folder_specs:
        folder = fs["folder"]
        repo = fs["repo"]
        for fname in fs["files"]:
            lookup = f"{folder}/{fname}" if folder else fname
            present = fl.locate_file(lookup, error_if_none=False) is not None
            all_files.append({
                "name": fname,
                "folder": folder,
                "_repo": repo,
                "present": present,
                "size_bytes": _approx_file_size(folder, fname),
            })

    missing = [f for f in all_files if not f["present"]]

    # Fetch real sizes from HuggingFace for missing files (parallel HEAD requests)
    if missing:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=min(8, len(missing))) as pool:
            futs = {
                pool.submit(_hf_file_size, f["_repo"], f["folder"], f["name"]): f
                for f in missing
            }
            for fut in as_completed(futs):
                try:
                    futs[fut]["size_bytes"] = fut.result()
                except Exception:
                    pass

    # Strip internal _repo key before returning
    for f in all_files:
        f.pop("_repo", None)

    model_label = (MODEL_CHOICE_LABELS[model_no - 1]
                   if 1 <= model_no <= len(MODEL_CHOICE_LABELS) else f"Model {model_no}")
    return {
        "all_present": len(missing) == 0,
        "missing_count": len(missing),
        "total_missing_bytes": sum(f["size_bytes"] for f in missing),
        "files": all_files,
        "model_label": model_label,
    }


def unload_model() -> str:
    with _state_lock:
        if _runtime is None:
            return "No model is currently loaded."
        _unload_locked()
        return "Model unloaded."


def _unload_locked():
    global _runtime, _offloadobj, _loaded_model_no, _loaded_backend
    _, unload_fn = _get_loader()
    _, unload_audio_models = _get_audio_helpers()
    if _runtime is not None:
        unload_audio_models(_runtime)
        unload_fn(_runtime.image_caption_model, _runtime.llm_model)
        _runtime = None
    if _offloadobj is not None:
        try:
            _offloadobj.unload_all()
        except Exception:
            pass
        _offloadobj = None
    _loaded_model_no = 0
    _loaded_backend = ""


def get_status() -> dict:
    """Return current model state as a serialisable dict."""
    return {
        "loaded": _runtime is not None,
        "model_no": _loaded_model_no,
        "backend": _loaded_backend,
        "model_label": MODEL_CHOICE_LABELS[_loaded_model_no - 1] if _loaded_model_no > 0 else "",
    }


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
    user_system_prompt: Optional[str] = None,
) -> tuple[str, str]:
    generate_cinematic_prompt = _get_generate()
    analyze_videos_audio, _ = _get_audio_helpers()

    with _state_lock:
        if _runtime is None:
            return "", "No model loaded. Click 'Load Model' first."

        needs_image, video_prompt, text_prompt, custom_system_prompt, needs_video = MODES[mode_label]

        # User-supplied system prompt overrides the mode default.
        if user_system_prompt:
            custom_system_prompt = user_system_prompt

        if needs_image and image is None:
            return "", f"Mode '{mode_label}' requires an image. Please upload one."
        if needs_video and not video_path:
            return "", f"Mode '{mode_label}' requires a video. Please upload one."
        if needs_video and _loaded_model_no < 3:
            return "", "Video modes are only supported with Qwen3.5 models (modes 3 & 4)."

        images: Optional[list] = None
        videos: Optional[list] = None
        prompt_text = prompt.strip()

        if needs_video:
            videos = [video_path]
            if custom_system_prompt is not None:
                prompt_text = "a video" if not prompt_text else prompt_text
        elif needs_image:
            images = [image]
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
