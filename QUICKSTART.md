# Quick Start Guide

## Prerequisites

1. **Install Flutter**
   - Download from: https://flutter.dev/docs/get-started/install
   - Add Flutter to your PATH
   - Run `flutter doctor` to verify installation

2. **Install a Local Model Provider** (choose one or more):
   - **Ollama**: https://ollama.ai
   - **LM Studio**: https://lmstudio.ai
   - **Koboldcpp**: https://github.com/LostRuins/koboldcpp

   OR

   - **Google Gemini API Key**: https://aistudio.google.com/app/api-keys

## Setup & Run

### Option 1: Run in Development Mode

```bash
cd flutter_app/ai_prompt_assistant
flutter pub get
flutter run -d windows  # or macos/linux
```

### Option 2: Build Release Version

**Windows:**
```bash
# Double-click build_windows.bat
# OR run manually:
flutter build windows --release
```

**macOS:**
```bash
flutter build macos --release
```

**Linux:**
```bash
flutter build linux --release
```

## First Use

1. **Start your local model server**
   ```bash
   # Ollama (in separate terminal)
   ollama serve

   # LM Studio
   # Open LM Studio, load a vision model, enable API server

   # Koboldcpp
   koboldcpp --port 5001 your_model.gguf
   ```

2. **Download a vision model**
   ```bash
   # Ollama examples:
   ollama pull llava:latest
   ollama pull llama-joycaption-alpha-one-hf-llava
   ollama pull qwen2.5-vl:7b

   # LM Studio: Browse and download from the UI
   ```

3. **Launch AI Prompt Assistant**
   - Run the app using one of the methods above
   - Open sidebar (≡ menu icon)
   - Select API provider
   - Click "Fetch Models"
   - Select one or more models
   - Upload an image
   - Click "Analyze Image(s)" or type a message

## Example Workflow

### Basic Image Analysis
1. Open sidebar → Select "Ollama" → Set URL: `http://localhost:11434`
2. Click "Fetch Models" → Select "llava:latest"
3. Select "Default Image-to-Prompt" from system prompts
4. Click "Add Images" → Select your image
5. Click "Analyze Image(s)"
6. View the generated description

### Using System Prompt Builder
1. Open sidebar → Expand "System Prompt Builder"
2. Caption Type: "Stable Diffusion Prompt"
3. Caption Length: "Long"
4. Enable extra options:
   - ✓ Include lighting info
   - ✓ Include camera angle  
   - ✓ No ethnicity
5. Click "Generate and Apply Prompt"
6. Upload an image and analyze

### Bulk Analysis
1. Go to "Bulk Analysis" tab
2. Click "Browse" → Select folder with images
3. Enable "Save prompts to text file"
4. Click "Analyze All Images"
5. Wait for progress bar to complete
6. Prompts are saved as .txt files next to each image

5. Regenerate specific responses if needed

### Video Generation (Veo)
1. Go to "Veo" tab
2. Choose "Text to Video" or "Frame to Video"
3. Enter a prompt like "A cinematic shot of a sunset over the ocean"
4. Click ✨ icon to enhance the prompt
5. Click Send and wait for generation (approx. 20-30s)
6. Tap "Extend" on the generated video to continue the scene

## Keyboard Shortcuts

- **Enter**: Send message (in chat input)
- **Ctrl+N**: New chat (when sidebar is open)
- **Esc**: Close image preview dialog

## Troubleshooting

**"Failed to fetch models"**
- Check that the API server is running
- Verify the base URL and port
- Test with: `curl http://localhost:11434/api/tags` (Ollama)

**"No images found in folder"**
- Ensure folder contains .png, .jpg, .jpeg, or .webp files
- Check file permissions

**App won't start**
- Run `flutter doctor` to check for issues
- Run `flutter clean && flutter pub get`
- Try `flutter run -v` for verbose output

**Slow generation**
- Use smaller models (4B/7B instead of 27B)
- Enable GPU acceleration in your model server
- Select fewer models for concurrent execution

## Tips

- **Save frequently used prompts**: Use the save prompt feature
- **Organize conversations**: Rename chats with descriptive titles
- **Export for backup**: Use "Export to .json" before major changes
- **Test different models**: Each model has different strengths
- **Adjust prompt length**: Longer prompts = more detail but slower
- **Use extra options**: Fine-tune output style with 25+ options

## Next Steps

- Explore the 46 predefined prompts for different use cases
- Try different caption types (Descriptive, MidJourney, Danbooru tags)
- Experiment with multi-model execution to compare outputs
- Use bulk analysis for large image datasets
- Create custom system prompts for your specific workflow

## Support

For issues or questions:
- Check the full README.md
- Review error messages in the app
- Test with `curl` commands to verify API connectivity
- Check model server logs for errors
