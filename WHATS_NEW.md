# What's New in v1.2.0

## Highlights
### Free Provider (g4f.space)
A new **Free** provider that connects to the public [g4f.space](https://g4f.space) relay — no API key, no account, no setup required.

- **5 Routes** — Groq, Ollama, Pollinations, Nvidia NIM, and Gemini; switch with a single dropdown in the sidebar
- **Searchable Model Picker** — type to filter the full model list returned by the selected route; tap a row to select (single-select radio style)
- **Image Support** — images are sent as standard OpenAI multipart content; Gemini and Pollinations routes generally have the best multimodal support
- **Default Provider** — selected automatically for new installs so users can start chatting with zero configuration
- **Route Info Banner** — a contextual note below the route dropdown explains that not all models or routes support image analysis and suggests switching routes (e.g., Gemini or Pollinations) if a request fails
- **Availability Note** — as a public free relay, individual routes and models may be rate-limited or temporarily unavailable

### Local Enhancer Gemma 4 Support
- Added **Gemma 4 E4B**
- Added **Gemma 4 26B A4B**
- Gemma model loading, status reporting, and download checks are now integrated into the same Local Enhancer flow as the existing models

### PromptFill Template Studio
- Added native PromptFill browsing, editing, and variable-filling workflows inside the desktop app
- Included imported PromptFill categories, banks, and templates with inline variable editing and AI Smart Terms support
- Added PromptFill media preview support for template images and videos

---

# What's New in v1.1.0
## New Features

### Local Enhancer (New Provider)
A fully self-contained LLM prompt enhancer for Wan2.1 — no Ollama, no Python, no third-party tools required. The backend is bundled inside the app.

- **No Setup Required** — backend starts automatically when you select the provider and shuts down when you switch away
- **Auto Mode Detection** — picks the right enhancement mode (T2V, I2V, V2V, I2I, etc.) based on your attached media and chosen output type
- **Auto System Prompt** — global toggle that selects the optimal system prompt per mode, or lets you use a custom one
- **Generation Output Type** — choose Image or Video; the enhancer adjusts its prompt style accordingly
- **11 Enhancement Modes** — T2V, T2I, T2T, I2V, IT2V, I2I, IT2I, V2V, VT2V, V2I, VT2I
- **Quantization Backends** — GGUF and Quanto INT8 for flexible VRAM usage
- **Configurable LLM Parameters** — adjust max tokens, temperature, top-p, and seed from the Local Enhancer Settings dialog
- **Audio Understanding for Video Modes** — in Qwen video modes (V2V, VT2V, V2I, VT2I), locally analyses video audio with Whisper + CLAP and incorporates dialogue, ambience, music, and sound effects into the rewritten prompt
- **Graceful Fallback** — automatically falls back to visual-only prompting if a video has no audio or if audio analysis fails

---

### SVG Generator (New Tab)
Generate fully self-contained SVG vector graphics from text descriptions or reference images.

- **Text to SVG** — describe any object, icon, or scene
- **Animated SVG** — toggle to Animated mode for CSS `@keyframes` looping effects
- **Reference Image** — attach an image; the AI recreates it in vector format
- **Multi-Provider** — works with whichever API provider is selected in the sidebar (Google Gemini, Ollama, LM Studio, Koboldcpp)
- **Export Options** — download as SVG, PNG, GIF, Animated PNG (APNG), MP4 (H.264), or MOV (lossless)
- **Browser Preview** — open animated SVGs in the system browser for full CSS animation playback

---

### Prompt to JSON Pipeline (New)
A two-step AI pipeline that converts simple text prompts into highly structured JSON payloads with a synthesized `master_prompt`.

- **Dynamic Field Selection** — automatically determines which fields are relevant (e.g., `camera_movement`, `lighting`, `color_palette`)
- **Master Prompt Generation** — synthesizes all selected variables into a cohesive, highly descriptive paragraph
- **Provider Agnostic** — runs on whichever model and API provider is currently selected
- **Generation Integration** — accessible via the **JSON Enhance** button in the Veo Video tab

---

## Enhancements

### Veo Video Generation
- **Audio-Aware Video Prompting** — Local Enhancer video modes can now incorporate speech, ambience, music, and sound effects from attached videos to generate richer prompts

### Core Capabilities
- **Auto-Update Checker** — startup check for new app versions with release notes and a download link

### System Prompt Builder
- Predefined prompts expanded from **46 to 57**
- New **Wan2GP Modes** category covering all 11 enhancement modes used by Local Enhancer
- Prompt names updated to reflect current model naming (Wan2.1, LTX-2)

---

## Troubleshooting Additions

### Slow generation
- Local Enhancer video modes with audio enabled are slower than visual-only prompting because they run local speech transcription and audio tagging before final prompt generation

### Local Enhancer video prompt has no audio details
- Audio understanding only runs for **Local Enhancer Qwen video modes** (`V2V`, `VT2V`, `V2I`, `VT2I`)
- Use model **3** or **4** in the Local Enhancer provider
- Retry with a short clip that has clear, loud speech or obvious background audio
- Check the Python API log for `Failed to decode audio`, `Failed to transcribe audio`, or `Failed to classify audio events`
- On first use, wait for the one-time Whisper / CLAP downloads to finish
