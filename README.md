# DubbLM (formerly AIBorch)

![DubbLM Logo](logo.png)

An intelligent video dubbing system that uses AI to create natural, context-aware translations and high-quality speech synthesis for video content.

## Important Notice

**DubbLM is designed for high-quality, context-aware translation processing, not real-time applications.** This system prioritizes translation accuracy, context understanding, and content adaptation over processing speed. The AI thoroughly analyzes conversations, adapts content for specific audiences, and ensures natural speech patterns - processes that require significant computational time. Expect processing times that are considerably longer than the original video duration.

## How It Works

The DubbLM process consists of several AI-powered stages:

1. **Audio Extraction & Speaker Diarization** - Separates speakers and identifies who speaks when
2. **Transcription** - Converts speech to text using advanced models (Whisper, OpenAI, AssemblyAI)
3. **Context-Aware Translation** - Uses LLM to translate with full context understanding
4. **Translation Refinement** - Applies persona-specific refinement for natural speech patterns
5. **Voice Synthesis** - Generates dubbed audio using TTS systems (OpenAI, Gemini, Coqui)
6. **Audio/Video Integration** - Combines translated audio with original video

## Why LLM Translation is Superior

Traditional translators work sentence-by-sentence without context. Our LLM approach:
- **Understands full conversation context** - maintains coherence across dialogue
- **Preserves speaker personalities** - adapts tone and style per character
- **Handles technical terminology** - maintains consistency with domain-specific terms
- **Creates natural speech patterns** - optimized for audio dubbing, not just text

## Refinement Personas

The `refinement_persona` feature adapts translations for specific audiences:

- **`normal`** - Standard, natural translation preserving all details
- **`casual_manager`** - Simplifies technical content for business audiences
- **`child`** - Transforms complex topics into child-friendly stories
- **`housewife`** - Makes content relatable to household managers and families
- **`science_popularizer`** - Engaging explanations for general audiences
- **`it_buddy`** - Casual IT jargon for developer audiences
- **`ai_buddy`** - Clear, professional language for AI practitioners

## TTS Model Comparison

### Gemini TTS
- **Quality**: Highest natural speech quality
- **Speed**: ~0.25x video speed (slower processing)
- **Cost**: Higher pricing
- **Best for**: Premium productions requiring top quality

### OpenAI TTS
- **Quality**: Good, reliable speech synthesis
- **Speed**: ~0.5x video speed (moderate processing)
- **Cost**: More affordable
- **Best for**: Balanced quality/cost projects

### Voice Selection
- **Automatic**: AI matches voices to speaker characteristics
- **Manual**: Assign specific voices per speaker from config
**Note**: Voice cloning is not yet implemented

### Background Audio Processing
- **`keep_background: true`** - Preserves original background music and ambient sounds
- **Memory Warning**: Background separation requires significant RAM usage. Avoid using this feature on long videos (>30 minutes) as it may cause memory issues on systems with limited RAM

## Installation

### Requirements
- Python 3.12+
- FFmpeg
- CUDA (optional, for GPU acceleration when WhisperX enabled)

### Setup
```bash
# Install system dependencies
# Ubuntu/Debian:
sudo apt update && sudo apt install ffmpeg python3.12 python3.12-venv
# macOS:
brew install ffmpeg python@3.12
# Windows: Download FFmpeg from https://ffmpeg.org/download.html

# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# Clone and install dependencies
git clone https://github.com/ArteusAI/DubbLM.git
cd DubbLM
python3.12 -m venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (see API Keys section below)
```

### API Keys Configuration

Different features require different API keys. Add these to your `.env` file:

**Required for basic functionality:**
- `OPENAI_API_KEY` - For OpenAI TTS and transcription services
- `GOOGLE_API_KEY` - For Gemini TTS and LLM translation services

**Optional (depending on chosen services):**

*For Transcription:*
- `ASSEMBLYAI_API_KEY` - If using `transcription_system: "assemblyai"`
- `OPENAI_API_KEY` - If using `transcription_system: "openai"`

*For Translation:*
- `GOOGLE_API_KEY` - If using `llm_provider: "gemini"` (default)
- `OPENROUTER_API_KEY` - If using `llm_provider: "openrouter"`

*For Text-to-Speech:*
- `OPENAI_API_KEY` - If using `tts_system: "openai"`
- `GOOGLE_API_KEY` - If using `tts_system: "gemini"`
- No API key needed for `tts_system: "coqui"` (local TTS)

**Example .env file:**
```env
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-api-key-here
ASSEMBLYAI_API_KEY=your-assemblyai-key-here
OPENROUTER_API_KEY=your-openrouter-key-here
```

**Minimum setup:** You need at least `GOOGLE_API_KEY` for default Gemini-based translation and TTS.

## Usage Examples

### Basic Dubbing
```bash
python dubblm_cli.py --input video.mp4 --source_language en --target_language es
```

### With Configuration File
```bash
python dubblm_cli.py --config my_config.yml --input video.mp4
```

### Advanced Options
```bash
# High-quality Gemini TTS with specific persona
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language fr \
  --tts_system gemini \
  --refinement_persona casual_manager \
  --save_translated_subtitles

# Multiple TTS systems per speaker
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language de \
  --tts_system_mapping '{"SPEAKER_00": "gemini", "SPEAKER_01": "openai"}'
```

### Speaker Analysis
```bash
# Generate speaker report before dubbing
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --generate_speaker_report
```

### Debug Mode
```bash
# Create debug video with speaker labels
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language de \
  --debug_info \
  --debug_diarize_only
```

## Configuration

Create `dubbing_config.yml` to set default parameters:

```yaml
source_language: "en"
target_language: "es"
tts_system: "gemini"
refinement_persona: "normal"
voice_auto_selection: true
save_translated_subtitles: true
remove_pauses: true
use_two_pass_encoding: true

# Per-speaker voice mapping
voice_name:
  SPEAKER_A: "alloy"
  SPEAKER_B: "nova"

# Per-speaker TTS systems
tts_system_mapping:
  SPEAKER_00: "gemini"
  SPEAKER_01: "openai"
```

## Output Files

The tool generates:
- `{input}_{target_lang}.mp4` - Dubbed video
- `{input}_{target_lang}.srt` - Translated subtitles (optional)
- `artifacts/` - Debug files, transcriptions, and intermediate audio

## Demo Video

[![DubbLM Demo](example.png)](https://youtu.be/UADjkgMXQCY)


## Performance Tips

- Use `--no_cache` to force fresh processing
- Enable `--remove_pauses` to optimize timing
- Use GPU for faster processing when available
- Consider `--start_time` and `--duration` for testing on video segments
- **Avoid `--keep_background` on long videos** - Background audio separation consumes significant RAM and may cause memory issues on videos longer than 30 minutes

## Roadmap

**Upcoming features and improvements:**

- **🚀 Lightweight Setup** - Reduce installation size by making optional packages (NVIDIA CUBLAS, etc.) truly optional and installable only when needed
- **📺 Smart Ad Removal** - Automatic detection and removal of native advertisements from video content during processing
- **⏱️ Dialogue Pace Control** - Advanced controls for managing conversation tempo and speech timing across different speakers
- **🔧 Code Refactoring** - Ongoing improvements to code structure, performance optimizations, and maintainability

## About us

This project is open to use and fork for everyone and developed by IT engineers of [Arteus](https://arteus.io/) - a company specializing in adaptive AI systems for business automation, sales, and customer service.

## You are talented

Want to contribute, ask http://t.me/pavelfedortsov

## License

MIT 
