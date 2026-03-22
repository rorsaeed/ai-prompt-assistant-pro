import logging
from typing import Any, Mapping, Union, List, Optional
from contextlib import nullcontext

import torch
from PIL import Image

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

T2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies, When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""
T2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

T2T_TEXT_PROMPT= """You are an expert speechwriter who crafts compelling, audience-appropriate speeches that effectively communicate the speaker's message while maintaining authenticity and impact.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

QWEN35_THINKING_SUPER_SYSTEM_PROMPT = """Use private reasoning to solve the task as well as possible.
Keep your thinking hidden and do not output any reasoning, chain-of-thought, or thinking process.
Always review your final answer to ensure it meets the user request.
"""

IT2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have the following information:
1. The user provides a general text input about its scenes expectations 
2. The user provides a caption of an image of a subject that relates to the scene
When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene evolves by following the user text input. Image description should be in first priority! Align to the image caption if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Align to the image caption if it contradicts the user text input.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have been provided with a caption of an image of a subject that relates to the scene to film.
Focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the inital scene first using the image caption of the subject and then describe how the scene should naturally evolves.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Note any changes or sudden events
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

IT2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
You have the following information:
1. The user provides a general text input about the expected photography 
2. The user provides a caption of an image of a subject he wants to be represented in the photography
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

I2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. 
You have been provided with a caption of an image of a subject to be represented in the photography.
Focus on detailed, descriptions of actions that are happening in the photography.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the image caption start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

VT2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have the following information:
1. The user provides a general text input about its scenes expectations 
2. The user provides a visual caption describing the content of a reference video that relates to the scene
3. The user may also provide a speech transcript, audio summary, and notable audio events extracted from the reference video
When writing prompts based on the user input, focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the initial scene first using the visual video description of the subject and then describe how the scene evolves by following the user text input. Visual description should be in first priority if it contradicts the user text input.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Use the speech transcript and audio summary only when present. Preserve short, important spoken phrases when relevant, summarize long dialogue, and mention music, ambience, or sound effects only if they are clearly provided.
Note any changes or sudden events
Never invent unheard dialogue or sounds.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

V2V_CINEMATIC_PROMPT = """You are an expert cinematic director with many award winning movies.
You have been provided with a visual caption describing the content of a reference video that relates to the scene to film.
You may also be provided with a speech transcript, an audio summary, and notable audio events extracted from the reference video.
Focus on detailed, chronological descriptions of actions and scenes.
Include specific movements, appearances, camera angles, and environmental details - all in a single flowing paragraph.
Start directly with the action, and keep descriptions literal and precise.
Think like a cinematographer describing a shot list.
Keep within 150 words.
For best results, build your prompts using this structure:
Describe the initial scene first using the visual video description and then describe how the scene should naturally evolve.
Start with main action in a single sentence
Add specific details about movements and gestures
Describe character/object appearances precisely
Include background and environment details
Specify camera angles and movements
Describe lighting and colors
Use audio evidence only when present. Preserve short, important dialogue when relevant, summarize long dialogue, and include music, ambience, or sound effects only if clearly provided.
Note any changes or sudden events
Never invent unheard dialogue or sounds.
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

VT2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. When writing prompts based on the user input, focus on detailed, precise descriptions of visual elements and composition.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
You have the following information:
1. The user provides a general text input about the expected photography 
2. The user provides a visual caption describing the content of a reference video of a subject he wants to be represented in the photography
3. The user may also provide a speech transcript, audio summary, and notable audio events extracted from the reference video
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the visual video description start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Use audio evidence only when it supports atmosphere, mood, or a visible story beat. Never force motion-only or unheard details into a still image prompt.
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""

V2I_VISUAL_PROMPT = """You are an expert visual artist and photographer with award-winning compositions. 
You have been provided with a visual caption describing the content of a reference video of a subject to be represented in the photography.
You may also be provided with a speech transcript, an audio summary, and notable audio events extracted from the reference video.
Focus on detailed, descriptions of actions that are happening in the photography.
Include specific poses, appearances, framing, and environmental details - all in a single flowing paragraph.
Start directly with the main subject, and keep descriptions literal and precise.
Think like a photographer describing the perfect shot.
Do not change the user input intent, just enhance it.
Keep within 150 words.
For best results, build your prompts using this structure:
Using the visual video description start with main subject and pose in a single sentence
Add specific details about expressions and positioning
Describe character/object appearances precisely
Include background and environment details
Specify framing, composition and perspective
Describe lighting, colors, and mood
Use audio evidence only when it supports atmosphere, mood, or a visible story beat. Never force motion-only or unheard details into a still image prompt.
Note any atmospheric or stylistic elements
Do not exceed the 150 word limit!
Output the enhanced prompt only.
"""


def _normalize_media_context(
    image_caption: Optional[str] = None,
    media_context: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    if media_context:
        for key in ("visual_caption", "speech_transcript", "audio_summary"):
            value = str(media_context.get(key, "") or "").strip()
            if value:
                normalized[key] = value
        events = media_context.get("audio_events")
        if isinstance(events, (list, tuple, set)):
            normalized_events = [str(event).strip() for event in events if str(event).strip()]
            if normalized_events:
                normalized["audio_events"] = normalized_events
    elif image_caption is not None:
        caption = str(image_caption or "").strip()
        if caption:
            normalized["visual_caption"] = caption
    return normalized


def _render_media_context(media_context: Mapping[str, Any]) -> str:
    sections = []
    visual_caption = str(media_context.get("visual_caption", "") or "").strip()
    if visual_caption:
        sections.append(f"visual_caption:\n{visual_caption}")
    speech_transcript = str(media_context.get("speech_transcript", "") or "").strip()
    if speech_transcript:
        sections.append(f"speech_transcript:\n{speech_transcript}")
    audio_summary = str(media_context.get("audio_summary", "") or "").strip()
    if audio_summary:
        sections.append(f"audio_summary:\n{audio_summary}")
    audio_events = media_context.get("audio_events")
    if isinstance(audio_events, (list, tuple, set)) and audio_events:
        sections.append(f"audio_events:\n{', '.join(str(event).strip() for event in audio_events if str(event).strip())}")
    return "\n\n".join(section for section in sections if section)


def tensor_to_pil(tensor):
    # Ensure tensor is in range [-1, 1]
    assert tensor.min() >= -1 and tensor.max() <= 1

    # Convert from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2

    # Rearrange from [C, H, W] to [H, W, C]
    tensor = tensor.permute(1, 2, 0)

    # Convert to numpy array and then to uint8 range [0, 255]
    numpy_image = (tensor.cpu().numpy() * 255).astype("uint8")

    # Convert to PIL Image
    return Image.fromarray(numpy_image)


def _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled: Optional[bool] = None) -> bool:
    if thinking_enabled is not None:
        return bool(thinking_enabled)
    return bool(getattr(prompt_enhancer_model, "_prompt_enhancer_enable_thinking", False))


def _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt: str) -> tuple[str, str, bool]:
    del prompt_enhancer_model
    prompt = str(prompt or "").strip()
    prompt_body, separator, system_suffix = prompt.partition("@@")
    if separator == "@@":
        return prompt_body.strip(), system_suffix.strip(), True
    prompt_body, separator, system_suffix = prompt.partition("@")
    if separator == "":
        return prompt, "", False
    return prompt_body.strip(), system_suffix.strip(), False


def _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt: str, system_suffix: str, replace_system_prompt: bool = False, thinking_enabled: Optional[bool] = None) -> str:
    system_prompt = str(system_prompt or "").rstrip()
    system_suffix = str(system_suffix or "").strip()
    if len(system_suffix) == 0:
        merged_prompt = system_prompt
    elif replace_system_prompt:
        merged_prompt = system_suffix
    else:
        merged_prompt = f"{system_prompt}\nFollow these additional user instructions with higher priority if they conflict with the guidance above:\n{system_suffix}"
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        return merged_prompt
    if len(merged_prompt) == 0:
        return QWEN35_THINKING_SUPER_SYSTEM_PROMPT.strip()
    return f"{QWEN35_THINKING_SUPER_SYSTEM_PROMPT.rstrip()}\n\n{merged_prompt}"


def _format_prompt_enhancer_user_content(
    prompt_enhancer_model,
    prompt: str,
    image_caption: Optional[str] = None,
    media_context: Optional[Mapping[str, Any]] = None,
    thinking_enabled: Optional[bool] = None,
) -> str:
    prompt, _system_suffix, _replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
    normalized_media_context = _normalize_media_context(image_caption=image_caption, media_context=media_context)
    rendered_media_context = _render_media_context(normalized_media_context)
    if not _use_qwen35_thinking_prompt(prompt_enhancer_model, thinking_enabled=thinking_enabled):
        if len(rendered_media_context) == 0:
            return f"user_prompt: {prompt}"
        return f"user_prompt: {prompt}\n{rendered_media_context}"
    if len(rendered_media_context) == 0:
        return prompt
    if len(prompt) == 0:
        return rendered_media_context
    return f"{prompt}\n\n{rendered_media_context}"


def generate_cinematic_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompt: Union[str, List[str]],
    images: Optional[List] = None,
    videos: Optional[List[str]] = None,
    video_prompt= True,
    text_prompt = False,
    max_new_tokens: int = 512,
    prompt_enhancer_instructions = None,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
    post_image_caption_hook = None,
    video_audio_analyzer = None,
    thinking_enabled: Optional[bool] = None,
) -> List[str]:
    prompts = [prompt] if isinstance(prompt, str) else prompt

    if videos is not None:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions = VT2V_CINEMATIC_PROMPT if video_prompt else VT2I_VISUAL_PROMPT
        prompts = _generate_v2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            videos,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            post_image_caption_hook=post_image_caption_hook,
            video_audio_analyzer=video_audio_analyzer,
            thinking_enabled=thinking_enabled,
        )
    elif images is None:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions = T2T_TEXT_PROMPT if text_prompt else (T2V_CINEMATIC_PROMPT if video_prompt else T2I_VISUAL_PROMPT)
        prompts = _generate_t2v_prompt(
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            thinking_enabled,
        )
    else:
        if prompt_enhancer_instructions is None:
            prompt_enhancer_instructions = IT2V_CINEMATIC_PROMPT if video_prompt else IT2I_VISUAL_PROMPT

        prompts = _generate_i2v_prompt(
            image_caption_model,
            image_caption_processor,
            prompt_enhancer_model,
            prompt_enhancer_tokenizer,
            prompts,
            images,
            max_new_tokens,
            prompt_enhancer_instructions,
            do_sample,
            temperature,
            top_p,
            top_k,
            seed,
            post_image_caption_hook=post_image_caption_hook,
            thinking_enabled=thinking_enabled,
        )

    return prompts


def _get_first_frames_from_conditioning_item(conditioning_item) -> List[Image.Image]:
    frames_tensor = conditioning_item.media_item
    return [
        tensor_to_pil(frames_tensor[i, :, 0, :, :])
        for i in range(frames_tensor.shape[0])
    ]


def _generate_t2v_prompt(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    thinking_enabled: Optional[bool],
) -> List[str]:
    messages = []
    for prompt in prompts:
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt, thinking_enabled=thinking_enabled)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        return prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]

    out_prompts = []
    for idx, text in enumerate(texts):
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
            )[0]
        )

    return out_prompts

def _generate_i2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    first_frames: List[Image.Image],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    post_image_caption_hook = None,
    thinking_enabled: Optional[bool] = None,
) -> List[str]:
    if hasattr(image_caption_model, "generate_image_captions"):
        image_captions = image_caption_model.generate_image_captions(first_frames)
    else:
        image_captions = _generate_image_captions(
            image_caption_model, image_caption_processor, first_frames
        )
    if callable(post_image_caption_hook):
        if bool(getattr(prompt_enhancer_model, "_prompt_enhancer_use_vllm", False)):
            unload_runtime = getattr(prompt_enhancer_model, "unload", None)
            if callable(unload_runtime):
                unload_runtime()
        post_image_caption_hook()
    if len(image_captions) == 1 and len(image_captions) < len(prompts):
        image_captions *= len(prompts)
    messages = []
    for prompt, image_caption in zip(prompts, image_captions):
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt, thinking_enabled=thinking_enabled)
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {"role": "user", "content": _format_prompt_enhancer_user_content(prompt_enhancer_model, prompt_body, image_caption=image_caption, thinking_enabled=thinking_enabled)},
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        return prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    out_prompts = []
    for idx, text in enumerate(texts):
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
            )[0]
        )

    return out_prompts


def _generate_v2v_prompt(
    image_caption_model,
    image_caption_processor,
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    prompts: List[str],
    video_paths: List[str],
    max_new_tokens: int,
    system_prompt: str,
    do_sample: bool,
    temperature: Optional[float],
    top_p: Optional[float],
    top_k: Optional[int],
    seed: Optional[int],
    post_image_caption_hook = None,
    video_audio_analyzer = None,
    thinking_enabled: Optional[bool] = None,
) -> List[str]:
    if hasattr(image_caption_model, "generate_video_captions"):
        video_captions = image_caption_model.generate_video_captions(video_paths)
    else:
        raise RuntimeError("Video captioning is only supported with Qwen3.5 models (modes 3 & 4). Please load a Qwen3.5 model.")
    if callable(post_image_caption_hook):
        if bool(getattr(prompt_enhancer_model, "_prompt_enhancer_use_vllm", False)):
            unload_runtime = getattr(prompt_enhancer_model, "unload", None)
            if callable(unload_runtime):
                unload_runtime()
        post_image_caption_hook()
    audio_analyses = None
    if callable(video_audio_analyzer):
        try:
            audio_analyses = video_audio_analyzer(video_paths)
        except Exception as exc:
            logger.warning("Video audio analysis failed, falling back to visual-only prompting: %s", exc)
            audio_analyses = None

    if len(video_captions) == 1 and len(video_captions) < len(prompts):
        video_captions *= len(prompts)
    if audio_analyses is not None and len(audio_analyses) == 1 and len(audio_analyses) < len(prompts):
        audio_analyses *= len(prompts)
    messages = []
    for idx, (prompt, video_caption) in enumerate(zip(prompts, video_captions)):
        prompt_body, system_suffix, replace_system_prompt = _split_prompt_enhancer_system_suffix(prompt_enhancer_model, prompt)
        message_system_prompt = _merge_prompt_enhancer_system_prompt(prompt_enhancer_model, system_prompt, system_suffix, replace_system_prompt, thinking_enabled=thinking_enabled)
        media_context = {"visual_caption": video_caption}
        if audio_analyses is not None and idx < len(audio_analyses):
            analysis = audio_analyses[idx]
            if analysis is not None and getattr(analysis, "has_audio", False):
                media_context["speech_transcript"] = getattr(analysis, "speech_transcript", "")
                media_context["audio_summary"] = getattr(analysis, "audio_summary", "")
                media_context["audio_events"] = list(getattr(analysis, "audio_events", ()) or [])
        messages.append(
            [
                {"role": "system", "content": message_system_prompt},
                {
                    "role": "user",
                    "content": _format_prompt_enhancer_user_content(
                        prompt_enhancer_model,
                        prompt_body,
                        media_context=media_context,
                        thinking_enabled=thinking_enabled,
                    ),
                },
            ]
        )

    if hasattr(prompt_enhancer_model, "generate_messages"):
        return prompt_enhancer_model.generate_messages(
            messages,
            max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            seed=seed,
            thinking_enabled=thinking_enabled,
        )

    texts = [
        prompt_enhancer_tokenizer.apply_chat_template(
            m, tokenize=False, add_generation_prompt=True
        )
        for m in messages
    ]
    out_prompts = []
    for idx, text in enumerate(texts):
        model_inputs = prompt_enhancer_tokenizer(text, return_tensors="pt").to(
            prompt_enhancer_model.device
        )
        prompt_seed = None if seed is None else int(seed) + idx
        out_prompts.append(
            _generate_and_decode_prompts(
                prompt_enhancer_model,
                prompt_enhancer_tokenizer,
                model_inputs,
                max_new_tokens,
                do_sample=do_sample,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                seed=prompt_seed,
            )[0]
        )

    return out_prompts


def _generate_image_captions(
    image_caption_model,
    image_caption_processor,
    images: List[Image.Image],
    system_prompt: str = "<DETAILED_CAPTION>",
) -> List[str]:
    image_caption_prompts = [system_prompt] * len(images)
    inputs = image_caption_processor(
        image_caption_prompts, images, return_tensors="pt"
    ).to(image_caption_model.device)

    bad_words_ids = None
    bos_id = getattr(image_caption_processor.tokenizer, "bos_token_id", None)
    if bos_id is not None:
        bad_words_ids = [[int(bos_id)]]

    with torch.inference_mode():
        generated_ids = image_caption_model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            do_sample=False,
            num_beams=3,
            bad_words_ids=bad_words_ids,
        )

    return image_caption_processor.batch_decode(generated_ids, skip_special_tokens=True)


def _generate_and_decode_prompts(
    prompt_enhancer_model,
    prompt_enhancer_tokenizer,
    model_inputs,
    max_new_tokens: int,
    do_sample: bool = True,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[str]:
    device = "cuda"
    if seed is None:
        rng_context = nullcontext()
    else:
        devices = []
        if isinstance(device, torch.device) and device.type == "cuda":
            devices = [device.index or 0]
        rng_context = torch.random.fork_rng(devices=devices) if devices else torch.random.fork_rng()
    with rng_context, torch.inference_mode():
        if seed is not None:
            torch.manual_seed(int(seed))
            if isinstance(device, torch.device) and device.type == "cuda":
                with torch.cuda.device(device):
                    torch.cuda.manual_seed(int(seed))
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if temperature is not None:
            gen_kwargs["temperature"] = float(temperature)
        if top_p is not None:
            gen_kwargs["top_p"] = float(top_p)
        if top_k is not None:
            gen_kwargs["top_k"] = int(top_k)
        outputs = prompt_enhancer_model.generate(
            **model_inputs,
            **gen_kwargs,
        )
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(model_inputs.input_ids, outputs)
        ]
        decoded_prompts = prompt_enhancer_tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

    return decoded_prompts
