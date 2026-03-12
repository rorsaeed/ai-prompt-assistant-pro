# AI Prompt Assistant - Flutter Desktop App

A powerful Flutter desktop application for analyzing images and videos using multiple AI vision models.

![AI Prompt Assistant – Chat Interface](docs/screenshots/chat_main.jpg)

## Features

### Multi-Provider Support
- **Ollama** - Local model hosting with keep-alive configuration
- **LM Studio** - Local OpenAI-compatible API with model unloading
- **Koboldcpp** - Local inference server
- **Google Gemini** - Cloud API with image and video support
- **Veo Video Generation** - Powered by Google Video FX for professional cinematic results
- **Image Studio** - High-quality image generation using Gemini 3 and Imagen 4 models

### Image Studio (Generation & Editing)
- **Text to Image** - Create stunning visuals from descriptive prompts
- **Image to Image** - Use reference images to guide style, composition, and content
- **"Surprise Me"** - One-click creative prompt generation for both text-to-image and editing workflows
- **"Use as Reference"** - Instantly use any generated image as a starting point for further variations
- **Advanced Resolution** - Select between 1K, 2K, and 4K output (model dependent)
- **Aspect Ratio Control** - Standard 1:1, Landscape 16:9, or Portrait 9:16 support
- **Model Support** - Integrated Gemini 3 Pro, Gemini 2.5 Flash, and Imagen 4 models

### Image Studio Generation

1. **Switch to Image Studio Tab** in the sidebar
2. **Select Model**: Choose between Imagen 4, Gemini 3 Pro (preview), or Gemini 2.5 Flash
3. **Configure Settings**: Select Aspect Ratio and Resolution (1K/2K/4K for Pro models)
4. **Choose Mode**:
   - *Text to Image*: Enter a prompt or use 🎲 **Surprise Me** for inspiration
   - *Image to Image*: Attach a reference image or use ↺ **Use as Reference** on a previously generated image
5. **Improve Prompt**: Use the ✨ wand icon to have an LLM enhance your base prompt for better results
6. **Generate**: Click the Send icon to start generation

![Image Studio – Text to Image Generation](docs/screenshots/image_studio.jpg)

### Veo Video Generation
- **Text to Video** - Generate high-quality cinematic videos from text prompts
- **Image to Video** - Use start and end images to guide video generation
- **Extend Video** - Automatically extend existing videos by extracting the last frame and generating a continuation, then seamlessly merging them with FFmpeg
- **Prompt Enhancement** - Built-in LLM-powered rewriter that uses attached images and video frames to create highly detailed cinematic prompts
- **Advanced Controls** - Configure aspect ratio (16:9, 9:16) and resolution (720p, 1080p, 4K)
- **FFmpeg Integration** - Automatic downloading and configuration of FFmpeg for complex video operations (Windows auto-download)

### Core Capabilities
- **Multi-Model Execution** - Run queries against multiple models simultaneously
- **Image Analysis** - Upload and analyze multiple images with drag-and-drop support
- **Video Analysis** - Full Google Files API integration with resumable uploads (Google only)
- **Chat Interface** - Conversation-based interaction with streaming responses
- **Bulk Analysis** - Batch process entire folders of images
- **System Prompt Builder** - Generate prompts with 11 caption types, 30 length options, and 25 extra options
- **Prompt Director Pro** - AI image/video prompt writing helper with model-aware dropdowns for style, camera, lighting, composition, and video movement
- **Conversation Management** - Save, load, rename, delete, and move conversations to folders
- **Conversation Folders** - Organize chats in a nested folder tree with subfolder support
- **Conversation Search** - Real-time search with debounce across all saved conversations
- **Export** - Export conversations to TXT or JSON format
- **Nano Banana Prompt Library** - Curated prompt gallery with search, category filters, image thumbnails, and one-click copy or send-to-Image-Studio
- **Theme Customization** - Multiple color palettes with light / dark / system mode toggle

### System Prompt Builder
- **11 Caption Types**: Descriptive, Stable Diffusion, MidJourney, Danbooru tags, Art Critic, Product Listing, Social Media, and more
- **30 Length Options**: From "Very Short" (20-40 words) to "260 words", plus custom word counts
- **25 Extra Options**: Control ethnicity/gender, lighting, camera details, watermarks, aesthetic quality, and more
- **46 Predefined Prompts**: Built-in prompts for video formats (wan2, ltx-2), image editing (FLUX, Qwen), tagging (Danbooru, PonyXL), and various photography styles

### Prompt Director Pro
A built-in prompt writing helper (inspired by AILTC Prompt Director) accessible from the chat input area via the ✨ magic wand button. Supports 9 AI models across image and video generation:

- **9 AI Models**: Flux, Midjourney 7, Nano Banana, SeeDream 4, Z-Image, Qwen, Wan 2.2/2.1 Video, LTX-2 Video
- **Model-Aware Sections**: Each model shows only its relevant controls — image models show camera/composition, video models add movement/pacing
- **6 Control Sections**: Style & Look (art style, film look, color palette, texture), World & Environment, Camera Gear (body, focal length, format, lens, aperture), Composition (shot size, angle), Lighting & Mood, Video Movement (relation, camera movement, pacing)
- **Randomize**: One-click randomization of all visible settings for creative inspiration
- **Live Preview**: See the assembled prompt in real-time before inserting it into the chat
- **Direct Integration**: Generated prompts are inserted directly into the message input box

![Prompt Director Pro Dialog](docs/screenshots/prompt_director.jpg)

## Installation

### Prerequisites
- Flutter SDK 3.0 or later
- Windows, macOS, or Linux desktop platform
- For local models: Ollama, LM Studio, or Koboldcpp installed and running
- For Google Gemini: API key from [Google AI Studio](https://aistudio.google.com/app/api-keys)

### Setup

1. **Clone the repository**
   ```bash
   cd flutter_app/ai_prompt_assistant
   ```

2. **Install dependencies**
   ```bash
   flutter pub get
   ```

3. **Run the app**
   ```bash
   # Windows
   flutter run -d windows

   # macOS
   flutter run -d macos

   # Linux
   flutter run -d linux
   ```

4. **Build release version**
   ```bash
   flutter build windows
   flutter build macos
   flutter build linux
   ```

## Usage

### First Time Setup

1. **Open the sidebar** (hamburger menu icon)
2. **Select API Provider** (Ollama, LM Studio, Koboldcpp, or Google)
3. **Configure provider settings**:
   - For local providers: Set API base URL (default ports: Ollama=11434, LM Studio=1234, Koboldcpp=5001)
   - For Google: Enter API key
4. **Fetch models** using the "Fetch Models" button
5. **Select one or more models** from the available list
6. **Choose or create a system prompt**

![Sidebar – Provider Configuration](docs/screenshots/sidebar_config.jpg)

### Chat Interface

1. **Upload images** (optional): Click "Add Images" or drag-and-drop
2. **Upload videos** (Google only): Click "Add Videos"
3. **Enter your message** or click "Analyze Image(s)" for media-only analysis
4. **View streaming responses** from all selected models simultaneously
6. **Open Prompt Director** (✨ wand icon next to message box) to build image/video prompts with guided dropdowns
7. **Regenerate** any response by clicking the refresh icon
6. **Delete** messages using the × button

![Chat Interface – Multi-Model Responses](docs/screenshots/chat_interface.jpg)

### Veo Video Generation

1. **Switch to Veo Tab** in the sidebar
2. **Select Generation Mode**: Text to Video, Frame to Video, or Extend Video
3. **Configure Settings**: Choose aspect ratio (Landscape/Portrait) and Resolution
4. **Attach Media**: Click "Attach Media" or drag-and-drop images/videos.
   - *Frame to Video*: Add a Start Frame and/or End Frame image.
   - *Extend Video*: Add an Input Video.
5. **Enhance Prompt**: Click the ✨ wand icon in the input bar to have an LLM rewrite your prompt based on your text and any attached media (extracted frames are used for videos).
6. **Generate**: Click the Send icon to start the generation process.
7. **Extend Existing Videos**: Use the "Extend" button on any generated video bubble to automatically load it into the input for continuation.

![Veo Video Generation](docs/screenshots/veo_generation.jpg)

### Bulk Analysis

1. **Select a folder** containing images (.png, .jpg, .jpeg, .webp)
2. **Choose system prompt** from the sidebar
3. **Enable "Save prompts to text file"** to save results alongside images
4. **Click "Analyze All Images"** to process the entire folder
5. **View progress** and results in the grid layout

![Bulk Analysis – Image Grid](docs/screenshots/bulk_analysis.jpg)

### System Prompt Builder

1. **Open the sidebar** and expand "System Prompt Builder"
2. **Select caption type** (e.g., "Stable Diffusion Prompt")
3. **Choose caption length** (e.g., "Long" or "150 words")
4. **Enable extra options** (e.g., "Include lighting info", "No ethnicity")
5. **If using NAME_OPTION**: Enter the specific name for people in images
6. **Click "Generate and Apply Prompt"** to build and apply the prompt
7. **Save custom prompts** with a unique name for later reuse

![System Prompt Builder](docs/screenshots/prompt_builder.jpg)

### Conversation Management

- **New Chat**: Click "New Chat" button to start fresh
- **Load Chat**: Select from the conversation tree in the sidebar
- **Rename Chat**: Right-click (⋮ menu) on a conversation → Rename
- **Delete Chat**: Right-click (⋮ menu) on a conversation → Delete
- **Move Chat**: Right-click (⋮ menu) on a conversation → Move to Folder
- **Export**: Use "to .txt" or "to .json" buttons to export current conversation
- **Search**: Use the search box at the top of the sidebar to filter conversations

### Conversation Folders

- **Create Folder**: Click the folder icon in the sidebar
- **Subfolders**: Right-click a folder → New Subfolder for nested organization
- **Rename / Delete Folder**: Right-click (⋮ menu) on any folder
- **Expand / Collapse**: Click a folder to toggle its contents

### Nano Banana Prompt Library

1. **Switch to the Prompts Tab** in the top navigation bar
2. **Browse or Search** prompts by title, description, or content using the search bar
3. **Filter by Category** using the group chips (e.g., Portrait, Landscape, Abstract)
4. **Copy Prompt**: Click **Copy Prompt** on any card to copy it to clipboard
5. **Use Image**: Click **Use Image** to download the reference image and open it directly in Image Studio with the prompt pre-filled
6. **Preview**: Click any card to open a full detail dialog with zoom support

![Nano Banana Prompt Library](docs/screenshots/nano_banana.jpg)

## Configuration

### Config File Location
- **Windows**: `C:\Users\<username>\Documents\ai_prompt_assistant\data\config.json`
- **macOS**: `~/Documents/ai_prompt_assistant/data/config.json`
- **Linux**: `~/Documents/ai_prompt_assistant/data/config.json`

### Data Storage Structure
```
ai_prompt_assistant/
├── data/
│   ├── config.json                    # User settings
│   ├── system_prompts.json            # Custom prompts
│   └── conversations/                 # Saved chats
│       └── *.json
├── temp_images/                       # Uploaded images
└── temp_videos/                       # Uploaded videos
```

## Recommended Models

### Local Models (Ollama/LM Studio)
- **Llama JoyCaption Alpha One** (12GB VRAM) - Best for system prompt builder
- **Gemma 3 27B** (24GB VRAM) - High quality, complex scenes
- **Gemma 3 12B** (8GB VRAM) - Balanced performance
- **Qwen2.5-VL-7B** (8GB VRAM) - Excellent detail and instruction following
- **LLaVA 1.6** (8GB VRAM) - Popular open-source option

### Google Gemini & Imagen Models (Cloud)
- **imagen-4.0-generate-001** - Latest Imagen 4 model for photorealistic results
- **gemini-3-pro-image-preview** - High-quality reasoning + image generation
- **gemini-3-flash-image-preview** - Fast, lightweight image generation
- **gemini-2.5-flash-image** - Efficient and reliable image creation
- **gemini-3-flash-preview** - Fast text/vision analysis
- **gemini-3-pro-preview** - Best quality text/vision analysis

## Advanced Features

### Ollama Keep-Alive
- **-1**: Use server default
- **0**: Unload immediately after response
- **Positive number**: Keep loaded for N seconds

### LM Studio Model Unloading
Enable "Unload model after response" to automatically free VRAM after each generation (requires `lms` CLI tool in PATH).

### Video Processing (Google)
1. Video is uploaded to Google Files API with resumable protocol
2. Processing status polled every 2 seconds (5-minute timeout)
3. Once ACTIVE, video URI is included in generation request
4. Supports: mp4, avi, mov, mkv, webm, flv, wmv, m4v

### Streaming Responses
- Real-time token-by-token display
- Automatic `<think>...</think>` tag filtering
- Concurrent streams for multiple models
- Auto-scroll to latest content

### FFmpeg Engine & Video FX
- **Automatic Setup**: Automatically downloads `ffmpeg.exe` for Windows on first run to the application support directory.
- **Last Frame Extraction**: Uses FFmpeg to accurately extract the final frame of a video for use as a starting point for extensions.
- **Seamless Merging**: Concatenates multiple video segments into a single file while preserving audio and ensuring consistent timestamps.
- **Audio Preservation**: Intelligently handles audio tracks during video extensions to maintain professional quality.

## Troubleshooting

### "Connection refused" error
- Verify the API provider is running and accessible
- Check the base URL and port number
- Test with `curl http://localhost:PORT/api/tags` (Ollama) or `/v1/models` (others)

### "No models available"
- Click "Fetch Models" after starting the API server
- For LM Studio: Load at least one model in the UI first
- For Koboldcpp: Launch with `--port 5001 --usecublas` or similar

### Video upload fails
- Google only: Verify API key is correct
- Check video file size (large files may exceed quota)
- Ensure video format is supported
- Check Google Cloud console for quota limits

### Slow generation
- Local models: Reduce model size or enable GPU acceleration
- Google: Check API rate limits and billing
- Try selecting fewer models for concurrent execution

### Images not loading
- Verify file permissions in temp_images directory
- Check supported formats: png, jpg, jpeg, webp
- Try re-uploading the image



### Run Tests
```bash
flutter test
```

### Code Generation
```bash
# Generate json_serializable code
flutter pub run build_runner build

# Watch mode for development
flutter pub run build_runner watch
```

## Architecture

- **State Management**: Provider pattern with ChangeNotifier
- **HTTP Client**: Dio for streaming SSE responses and multipart uploads
- **Video Processing**: FFmpeg (via `ffmpeg_kit_flutter` and raw CLI) for frame extraction and concatenation
- **Media Playback**: `media_kit` for cross-platform video preview and thumbnails
- **File I/O**: path_provider for cross-platform directories
- **Serialization**: json_serializable for type-safe JSON
- **Desktop Integration**: window_manager for window control
- **Theming**: Dynamic Material 3 color schemes generated from multiple seed palettes, with light / dark / system mode support

## License

This application is **partially open-source**.

- The core framework, UI components, and general integrations are open-source.
- Certain advanced features, custom AI prompt configurations, and proprietary modules may be closed-source or subject to specific usage restrictions.

Please refer to individual file headers or contact the repository owner for detailed licensing information, commercial usage, and distribution rights.

## Support

For issues, feature requests, or contributions, please visit the GitHub repository.
