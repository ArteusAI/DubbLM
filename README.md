# DubbLM (formerly AIBorch)

![DubbLM Logo](logo.png)

An intelligent video dubbing system that uses AI to create natural, context-aware translations and high-quality speech synthesis for video content.

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
- **Manual**: Assign specific voices per speaker
- **Note**: Voice cloning is not yet implemented

## Installation

### Requirements
- Python 3.12+
- FFmpeg
- CUDA (optional, for GPU acceleration when WhisperX enabled)

### Setup
```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install dependencies
git clone <repository-url>
cd DubbLM
uv pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys (OpenAI, Gemini, etc.)
```

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

## Performance Tips

- Use `--no_cache` to force fresh processing
- Enable `--remove_pauses` to optimize timing
- Use GPU for faster processing when available
- Consider `--start_time` and `--duration` for testing on video segments

## About us

This project is open to use and fork for everyone and developed by IT engineers of [Arteus](https://arteus.io/) - a company specializing in adaptive AI systems for business automation, sales, and customer service.

## License

MIT 