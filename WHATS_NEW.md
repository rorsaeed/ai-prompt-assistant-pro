# What's New in v1.4.0

## Highlights

### Interactive Proactive Co-Creator
An intelligent prompt co-creation interface inspired by proactive co-creator research. Rather than just running standard chat, this mode helps you decompose, analyze, and enrich your creative ideas.
- **Belief Graph Decomposition** — Automatically parses your prompt into structured entities, visual attributes, and relationships.
- **Atomic Clarifying Questions** — Dynamically generates 3 focused, single-topic questions with multiple choice options to resolve ambiguities.
- **Proactiveness Levels** — Configure the AI's creativity level (Low, Medium, High). Higher levels infer details and suggest creative additions.
- **Global Context & Diversity** — The engine avoids assuming Western norms and actively suggests diverse ethnicities, visual styles, and cultural details.
- **Reference Image Grounding** — Uses attached images as a "source of truth", mapping visible details and asking clarifying questions on intended modifications.
- **Interactive UI Cards** — Adjust entities, select alternative values, and answer clarifications directly via a rich interactive panel inside the chat bubble.

### Unified API Settings Studio
Manage all AI providers from a centralized, beautiful configuration interface.
- **Centralized Dashboard** — Configure API keys, base URLs, and parameters for all local and cloud-based LLM backends.
- **Expandable Cards & Visibility Toggles** — Clean layout with expandable cards and toggles to show/hide providers from the sidebar.
- **Comprehensive Provider List** — Native support for Google Gemini, Vertex AI, OpenAI, Anthropic, Mistral, OpenRouter, Groq, Together, SwiftRouter, NVIDIA NIM, Ollama, LM Studio, Koboldcpp, and the Free Provider.

### Prompt Library & Local Seed Service
The prompt library is now more robust, performant, and self-contained.
- **Offline Seeding** — Bundles and initializes trending prompt libraries (such as Nano Banana, Seedance 2.0, Grok Imagine) directly from local assets on first run.
- **Media Download Support** — Easily download preview videos and images from prompt galleries straight to your local drive.
- **Persistent Storage & Search** — Fast local search, sorting, and persistent storage support for prompt library entries.

### Agent Workflow & Plan Panel Enhancements
- **Rerun from Task** — Re-run agent plan execution starting from a specific task when a plan is paused or completed.
- **Session Model Picker Override** — Honorable per-session model overrides via the Skills toolbar picker during plan execution.
- **Retry Failed Commands** — Offers a quick "Retry command" option in the Agent Turn card to rerun failed shell commands.
- **Skill Detail Panel** — Hover over any skill to open a detailed documentation dialog, rendering its `SKILL.md` and allowing you to click on example prompts to prefill chat.

### Model Additions & Chat Tweaks
- **Gemini 3.5 Flash** — Added support under Google Vertex AI.
- **Message Editing & Regeneration** — Hover controls in chat bubbles for editing user messages and regenerating assistant responses.
- **Auto-Sync and Dependency Updates** — Core backend improvements, upgraded dependencies, and robust async sync operations.

---

# What's New in v1.3.2


## Highlights
### Skills Screen
The new **Skills** screen is the easiest way to run agent skills in AI Prompt Assistant. Open the screen, choose a skill, create a session, and describe the task. The app handles the skill instructions, workspace, approvals, attachments, and prerequisite checks for you.

- **Zero Setup Required** - Built-in skills are already available, and required skill assets are installed automatically. No manual skill installation is needed.
- **Built-In Skills** - HyperFrames, GSAP, HyperFrames CLI, HyperFrames Registry, Website to HyperFrames, and Remotion are included.
- **Free to Run** - Skills can use the Free provider routes, so they can run without paid API keys when a suitable free model is selected.
- **Local Model Friendly** - Skills can also run through local providers such as Ollama, LM Studio, Koboldcpp, or other OpenAI-compatible local endpoints.
- **Add Your Own Skills** - Use **Add skill** in the Skills panel and paste an `owner/repo` value or a full GitHub URL. The app finds the skill bundle and installs it into your local skills library.
- **Agent Workflow Tools** - Sessions get their own working folders, file attachments, requirement checks, command approval controls, safe-command auto approval, and plan-and-execute mode.

### Libraries Tab
The prompt-library experience has been upgraded from a single Nano Banana gallery into a unified **Libraries** tab that supports multiple prompt sources in one place.

- **Multi-Source Browser** — switch between Nano Banana Pro, Seedance 2.0, GPT Image 1.5, SeeDream 4.5, Gemini 3, and Grok Imagine from the same screen

### Free Provider
A new **Free** provider that connects to a public relay — no API key, no account, no setup required.

- **5 Routes** — Groq, Ollama, Pollinations, Nvidia NIM, and Gemini; switch with a single dropdown in the sidebar
- **Searchable Model Picker** — type to filter the full model list returned by the selected route; tap a row to select (single-select radio style)
- **Image Support** — images are sent as standard OpenAI multipart content; Gemini and Pollinations routes generally have the best multimodal support
- **Default Provider** — selected automatically for new installs so users can start chatting with zero configuration

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
