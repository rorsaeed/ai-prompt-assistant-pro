from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import os
from typing import Any, Iterable

import av
import numpy as np
import torch
from transformers import (
    AutoModelForSpeechSeq2Seq,
    AutoProcessor,
    ClapModel,
    ClapProcessor,
    pipeline,
)

from shared.utils import files_locator as fl


logger = logging.getLogger(__name__)

WHISPER_REPO_ID = "openai/whisper-large-v3-turbo"
CLAP_REPO_ID = "laion/clap-htsat-unfused"
WHISPER_ASSETS_DIR = "Whisper_Large_V3_Turbo"
CLAP_ASSETS_DIR = "Clap_Htsat_Unfused"
TARGET_SAMPLE_RATE = 16_000
CLAP_CHUNK_SECONDS = 10
CLAP_TOP_K_PER_CHUNK = 5
CLAP_TOP_K_FINAL = 8
MAX_TRANSCRIPT_CHARS = 1400

WHISPER_REQUIRED_FILES = (
    "config.json",
    "generation_config.json",
    "merges.txt",
    "model.safetensors",
    "normalizer.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

CLAP_REQUIRED_FILES = (
    "config.json",
    "merges.txt",
    "preprocessor_config.json",
    "pytorch_model.bin",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
)

AUDIO_EVENT_LABELS = (
    "speech",
    "conversation",
    "narration",
    "whispering",
    "singing",
    "chanting",
    "laughter",
    "crying",
    "cheering",
    "applause",
    "crowd noise",
    "people talking",
    "silence",
    "room tone",
    "indoor ambience",
    "outdoor ambience",
    "city traffic",
    "car engine",
    "train",
    "airplane",
    "wind",
    "rain",
    "thunder",
    "ocean waves",
    "river water",
    "fire crackling",
    "footsteps",
    "door closing",
    "glass breaking",
    "alarm",
    "siren",
    "phone ringing",
    "keyboard typing",
    "camera shutter",
    "gunshot",
    "explosion",
    "machinery",
    "drilling",
    "construction noise",
    "dog barking",
    "cat meowing",
    "birdsong",
    "horse",
    "music",
    "background music",
    "orchestral music",
    "electronic music",
    "rock music",
    "hip hop music",
    "jazz music",
    "piano music",
    "guitar music",
    "drum beats",
)

_NON_DESCRIPTIVE_EVENT_LABELS = {
    "speech",
    "conversation",
    "people talking",
    "music",
    "background music",
    "silence",
}


@dataclass(slots=True)
class VideoAudioAnalysis:
    has_audio: bool
    speech_transcript: str = ""
    audio_events: tuple[str, ...] = ()
    audio_summary: str = ""


def ensure_audio_understanding_assets(process_files_def) -> None:
    process_files_def(
        repoId=WHISPER_REPO_ID,
        sourceFolderList=[""],
        fileList=[list(WHISPER_REQUIRED_FILES)],
        targetFolderList=[WHISPER_ASSETS_DIR],
    )
    process_files_def(
        repoId=CLAP_REPO_ID,
        sourceFolderList=[""],
        fileList=[list(CLAP_REQUIRED_FILES)],
        targetFolderList=[CLAP_ASSETS_DIR],
    )


def get_audio_assets_dirs() -> tuple[str, str]:
    whisper_dir = fl.locate_folder(WHISPER_ASSETS_DIR, error_if_none=False) or fl.get_download_location(WHISPER_ASSETS_DIR)
    clap_dir = fl.locate_folder(CLAP_ASSETS_DIR, error_if_none=False) or fl.get_download_location(CLAP_ASSETS_DIR)
    if whisper_dir is None or not os.path.isdir(whisper_dir):
        raise FileNotFoundError(f"Missing Whisper assets folder '{WHISPER_ASSETS_DIR}'")
    if clap_dir is None or not os.path.isdir(clap_dir):
        raise FileNotFoundError(f"Missing CLAP assets folder '{CLAP_ASSETS_DIR}'")
    return whisper_dir, clap_dir


def unload_audio_understanding_models(runtime: Any) -> None:
    for attr in (
        "audio_transcriber_pipeline",
        "audio_classifier_pipeline",
        "audio_transcriber_processor",
        "audio_classifier_processor",
        "audio_transcriber_model",
        "audio_classifier_model",
    ):
        if hasattr(runtime, attr):
            setattr(runtime, attr, None)
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def analyze_videos_audio(runtime: Any, video_paths: list[str]) -> list[VideoAudioAnalysis]:
    if not video_paths:
        return []
    _ensure_audio_models_loaded(runtime)
    return [_analyze_single_video(runtime, video_path) for video_path in video_paths]


def _candidate_device_specs() -> Iterable[tuple[str, torch.dtype, int]]:
    if torch.cuda.is_available():
        yield "cuda", torch.float16, 0
    yield "cpu", torch.float32, -1


def _ensure_audio_models_loaded(runtime: Any) -> None:
    if getattr(runtime, "audio_transcriber_pipeline", None) is not None and getattr(runtime, "audio_classifier_pipeline", None) is not None:
        return

    whisper_dir, clap_dir = get_audio_assets_dirs()
    last_error: Exception | None = None
    for device_name, dtype, pipeline_device in _candidate_device_specs():
        try:
            whisper_processor = AutoProcessor.from_pretrained(whisper_dir, local_files_only=True)
            whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
                whisper_dir,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                local_files_only=True,
            )
            whisper_model.to(torch.device(device_name))
            whisper_model.eval()
            whisper_pipe = pipeline(
                task="automatic-speech-recognition",
                model=whisper_model,
                tokenizer=whisper_processor.tokenizer,
                feature_extractor=whisper_processor.feature_extractor,
                device=pipeline_device,
                torch_dtype=dtype,
            )

            clap_processor = ClapProcessor.from_pretrained(clap_dir, local_files_only=True)
            clap_model = ClapModel.from_pretrained(clap_dir, local_files_only=True)
            clap_model.to(torch.device(device_name))
            clap_model.eval()
            clap_pipe = pipeline(
                task="zero-shot-audio-classification",
                model=clap_model,
                tokenizer=clap_processor.tokenizer,
                feature_extractor=clap_processor.feature_extractor,
                device=pipeline_device,
                torch_dtype=dtype,
            )

            runtime.audio_transcriber_model = whisper_model
            runtime.audio_transcriber_processor = whisper_processor
            runtime.audio_transcriber_pipeline = whisper_pipe
            runtime.audio_classifier_model = clap_model
            runtime.audio_classifier_processor = clap_processor
            runtime.audio_classifier_pipeline = clap_pipe
            runtime.audio_model_device = device_name
            return
        except Exception as exc:
            last_error = exc
            logger.warning("Audio model load failed on %s: %s", device_name, exc)
            unload_audio_understanding_models(runtime)
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
    if last_error is not None:
        raise last_error


def _analyze_single_video(runtime: Any, video_path: str) -> VideoAudioAnalysis:
    try:
        decoded = decode_audio_samples(video_path)
        if decoded is None:
            return VideoAudioAnalysis(has_audio=False)
        samples, sample_rate = decoded
        if samples.size == 0:
            return VideoAudioAnalysis(has_audio=False)
    except Exception as exc:
        logger.exception("Failed to decode audio from video '%s': %s", video_path, exc)
        raise

    try:
        transcript = _transcribe_audio(runtime.audio_transcriber_pipeline, samples, sample_rate)
    except Exception as exc:
        logger.exception("Failed to transcribe audio from video '%s': %s", video_path, exc)
        raise

    try:
        events = _classify_audio_events(runtime.audio_classifier_pipeline, samples, sample_rate)
    except Exception as exc:
        logger.exception("Failed to classify audio events from video '%s': %s", video_path, exc)
        raise

    summary = build_audio_summary(transcript, events)
    return VideoAudioAnalysis(
        has_audio=True,
        speech_transcript=transcript,
        audio_events=tuple(events),
        audio_summary=summary,
    )


def decode_audio_samples(video_path: str) -> tuple[np.ndarray, int] | None:
    with av.open(video_path) as container:
        audio_stream = next((stream for stream in container.streams if stream.type == "audio"), None)
        if audio_stream is None:
            return None
        resampler = av.audio.resampler.AudioResampler(
            format="s16",
            layout="mono",
            rate=TARGET_SAMPLE_RATE,
        )
        chunks: list[np.ndarray] = []
        # Pass the actual stream object instead of its global container index.
        # In mixed video+audio containers, `audio_stream.index` can point past
        # the audio-only stream tuple that PyAV uses for `decode(audio=...)`.
        for frame in container.decode(audio_stream):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            frame_list = resampled if isinstance(resampled, list) else [resampled]
            for one_frame in frame_list:
                ndarray = one_frame.to_ndarray()
                if ndarray.size == 0:
                    continue
                chunks.append(np.asarray(ndarray).reshape(-1))
        flushed = resampler.resample(None)
        frame_list = flushed if isinstance(flushed, list) else ([flushed] if flushed is not None else [])
        for one_frame in frame_list:
            ndarray = one_frame.to_ndarray()
            if ndarray.size == 0:
                continue
            chunks.append(np.asarray(ndarray).reshape(-1))
    if not chunks:
        return None
    audio = np.concatenate(chunks).astype(np.float32) / 32768.0
    audio = np.clip(audio, -1.0, 1.0)
    return audio, TARGET_SAMPLE_RATE


def _transcribe_audio(asr_pipeline, samples: np.ndarray, sample_rate: int) -> str:
    result = asr_pipeline(
        {"raw": samples, "sampling_rate": sample_rate},
        chunk_length_s=30,
        batch_size=8,
        return_timestamps=False,
        generate_kwargs={"task": "transcribe"},
    )
    text = ""
    if isinstance(result, dict):
        text = str(result.get("text", "") or "")
    elif isinstance(result, str):
        text = result
    return normalize_transcript_text(text)


def _classify_audio_events(audio_pipeline, samples: np.ndarray, sample_rate: int) -> list[str]:
    target_sample_rate = int(getattr(getattr(audio_pipeline, "feature_extractor", None), "sampling_rate", sample_rate) or sample_rate)
    classifier_samples = _resample_audio(samples, sample_rate, target_sample_rate)
    chunk_size = TARGET_SAMPLE_RATE * CLAP_CHUNK_SECONDS
    if chunk_size <= 0:
        return []

    aggregated: dict[str, float] = {}
    target_chunk_size = target_sample_rate * CLAP_CHUNK_SECONDS
    for start in range(0, len(classifier_samples), target_chunk_size):
        chunk = classifier_samples[start:start + target_chunk_size]
        if chunk.size < target_sample_rate:
            continue
        results = audio_pipeline(
            chunk,
            candidate_labels=list(AUDIO_EVENT_LABELS),
            top_k=CLAP_TOP_K_PER_CHUNK,
        )
        for item in results or []:
            label = str(item.get("label", "") or "").strip()
            if not label:
                continue
            score = float(item.get("score", 0.0) or 0.0)
            aggregated[label] = max(score, aggregated.get(label, 0.0))
    return select_stable_audio_events(aggregated)


def select_stable_audio_events(label_scores: dict[str, float]) -> list[str]:
    ordered = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
    selected: list[str] = []
    for label, score in ordered:
        if score < 0.18:
            continue
        if label in selected:
            continue
        selected.append(label)
        if len(selected) >= CLAP_TOP_K_FINAL:
            break
    return selected


def build_audio_summary(transcript: str, audio_events: Iterable[str]) -> str:
    transcript = normalize_transcript_text(transcript)
    filtered_events = [event for event in audio_events if event not in _NON_DESCRIPTIVE_EVENT_LABELS]
    if transcript and filtered_events:
        return f"Contains spoken dialogue with prominent audio such as {', '.join(filtered_events)}."
    if transcript:
        return "Contains spoken dialogue."
    if filtered_events:
        return f"Prominent audio includes {', '.join(filtered_events)}."
    return ""


def normalize_transcript_text(text: str) -> str:
    normalized = " ".join(str(text or "").split())
    if len(normalized) <= MAX_TRANSCRIPT_CHARS:
        return normalized
    truncated = normalized[:MAX_TRANSCRIPT_CHARS].rsplit(" ", 1)[0].strip()
    return f"{truncated}…" if truncated else normalized[:MAX_TRANSCRIPT_CHARS]


def _resample_audio(samples: np.ndarray, from_rate: int, to_rate: int) -> np.ndarray:
    if from_rate == to_rate:
        return np.asarray(samples, dtype=np.float32)
    if samples.size == 0:
        return np.asarray(samples, dtype=np.float32)
    duration = samples.shape[0] / float(from_rate)
    target_length = max(1, int(round(duration * to_rate)))
    source_positions = np.linspace(0.0, duration, num=samples.shape[0], endpoint=False, dtype=np.float64)
    target_positions = np.linspace(0.0, duration, num=target_length, endpoint=False, dtype=np.float64)
    resampled = np.interp(target_positions, source_positions, samples.astype(np.float32, copy=False))
    return np.asarray(resampled, dtype=np.float32)


def summarize_audio_debug(samples: np.ndarray) -> dict[str, float]:
    if samples.size == 0:
        return {"duration_s": 0.0, "rms": 0.0}
    rms = float(np.sqrt(np.mean(np.square(samples))))
    return {
        "duration_s": round(samples.size / TARGET_SAMPLE_RATE, 3),
        "rms": round(rms, 6),
        "dbfs": round(20.0 * math.log10(max(rms, 1e-8)), 3),
    }
