# DubbLM

![DubbLM Logo](logo.png)

DubbLM is an AI video dubbing system for high-quality, context-aware translation, speech synthesis, and video assembly. It can be used through a React UI, a FastAPI backend, a one-shot external API, or the original CLI workflow.

## Important Notice

**DubbLM is optimized for quality, not real-time processing.** The system analyzes speaker context, translates dialogue with LLMs, refines wording, synthesizes speech, adjusts timing, and rebuilds the final video. Processing can take significantly longer than the source video duration, especially with premium TTS, background preservation, LLM editor passes, or high-resolution video settings.

## What It Does

- **Context-aware translation** with Gemini or OpenRouter-backed LLMs.
- **React UI workflow** for project upload, presets, transcription, segment editing, dubbing, and result download.
- **FastAPI + Celery backend** for persistent projects, async processing, progress streaming, and external integrations.
- **Segment editor** for translation edits, speaker renaming, voice assignment, TTS previews, rephrasing, muting, and cache resets.
- **Multiple TTS providers** including Gemini, OpenAI, MiniMax, and local/F5/Coqui paths where configured.
- **Speaker-aware controls** including diarization, voice matching, per-speaker prompts, and gender inference/overrides.
- **Timing and video controls** for background audio, original-audio ranges, segment stretching, volume, subtitles, and quality presets.
- **Reports and cost tracking** with per-stage API usage, downloadable artifacts, and summary reports.

## How It Works

The DubbLM pipeline is split into async stages:

1. **Upload and project setup** - Store source video and project config.
2. **Audio extraction and diarization** - Identify speakers and speech ranges.
3. **Transcription** - Convert speech to text using AssemblyAI, OpenAI + PyAnnote, or WhisperX.
4. **Context analysis and translation** - Translate with full dialogue context and optional glossary/prompt guidance.
5. **Refinement and editor pass** - Apply persona/style refinement and optional LLM editor improvements.
6. **Segment review** - Edit text, speakers, voices, prompts, and preview TTS before final dubbing.
7. **Speech synthesis** - Generate per-segment speech with selected TTS providers and voice mappings.
8. **Audio/video assembly** - Align speech, preserve background if requested, generate subtitles, and encode the final video.
9. **Report generation** - Store cost, timing, artifacts, and downloadable summary data.

## Interfaces

### Web UI

The UI is the main workflow for interactive use:

- Project list with persistent project state.
- Batch upload and background upload progress.
- Presets: `fast`, `hq`, and `ultra`.
- Server-side API key management.
- Live processing logs and progress via SSE.
- Segment editor after transcription.
- Final result view with video streaming, downloads, subtitles, report, and API cost panel.

### API

The backend exposes `FastAPI` routes under `/api/v1`:

- Project workflow: create projects, upload videos, update config, transcribe, edit segments, dub, download.
- One-shot workflow: `POST /api/v1/translate` uploads a video and starts the full pipeline.
- Status: polling through job status endpoints or live SSE through `/projects/{projectId}/status`.
- Resources: voices, personas, languages, presets, settings, thumbnails, frames, reports, and artifacts.

Swagger UI is available at:

```text
http://localhost:8000/api/v1/docs
```

See [examples/external_translate_api.md](examples/external_translate_api.md) for detailed external API documentation.

### CLI

The CLI remains useful for scripts, local experiments, and batch jobs:

```bash
python dubblm_cli.py --input video.mp4 --source_language en --target_language ru
```

CLI options are loaded from `dubbing_config.yml` by default and can be overridden with flags.

## Installation

### Requirements

- Python 3.12+
- FFmpeg and FFprobe
- Redis for the API/Celery workflow
- Node.js 20+ for the frontend
- CUDA is optional, mainly for local WhisperX/GPU-heavy paths

### Python Setup

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg redis-server python3.12 python3.12-venv

# macOS
brew install ffmpeg redis python@3.12
brew services start redis

# Clone and install
git clone https://github.com/ArteusAI/DubbLM.git
cd DubbLM
python3.12 -m venv .venv
source .venv/bin/activate
pip install uv
uv pip install -r requirements.txt

# Configure environment
cp env.example .env
```

Edit `.env` with the API keys you need.

## API Keys

DubbLM can read keys from `.env` or from server-side settings saved by the UI.

Common keys:

- `GOOGLE_API_KEY` - Gemini LLM/TTS and Gemini-based enrichment.
- `GEMINI_API_KEY` - Stored by the UI settings layer; keep aligned with `GOOGLE_API_KEY` if needed by your deployment.
- `OPENAI_API_KEY` - OpenAI TTS and OpenAI transcription paths.
- `OPENROUTER_API_KEY` - OpenRouter LLM/refinement/editor models.
- `ASSEMBLYAI_API_KEY` - AssemblyAI transcription and diarization.
- `HF_TOKEN` - PyAnnote/Hugging Face access for local diarization/gender inference paths.
- `MINIMAX_API_KEY` - MiniMax TTS.
- `MINIMAX_GROUP_ID` - Optional MiniMax group/account identifier for MiniMax TTS.

Minimum practical setup for the default UI presets is usually:

```env
GOOGLE_API_KEY=your-google-key
ASSEMBLYAI_API_KEY=your-assemblyai-key
OPENAI_API_KEY=your-openai-key
```

Add `OPENROUTER_API_KEY` for OpenRouter editor/refinement models and `MINIMAX_API_KEY` for MiniMax voices.

## Running the Web App

Start Redis first if it is not already running:

```bash
redis-server
```

Start the backend API:

```bash
source .venv/bin/activate
python run_api.py
```

Start the Celery worker in a second terminal:

```bash
source .venv/bin/activate
python run_worker.py
```

Start the frontend in a third terminal:

```bash
cd src/frontend
npm install
API_URL=http://localhost:8000 npm run dev
```

Open:

```text
http://localhost:3000
```

The frontend proxies `/api` to `API_URL`; the backend defaults to port `8000`.

## UI Workflow

1. Open the project dashboard.
2. Upload one or more videos.
3. Choose source/target language and a preset:
   - `fast` - Faster, simpler TTS, 720p output.
   - `hq` - Balanced default, Gemini TTS, 1080p output.
   - `ultra` - Highest quality, editor/refinement options, background preservation, original quality.
4. Configure advanced options if needed: provider/model, TTS style, voice mappings, prompt prefixes, segment merge/stretch settings, background/original audio ranges, subtitles, and workers.
5. Start transcription.
6. Review and edit segments in the editor.
7. Preview TTS for individual segments and adjust voices/prompts.
8. Start final dubbing.
9. Download video, subtitles, report, and inspect cost/timing statistics.

## Segment Editor

After transcription, the editor supports:

- Editing translated segment text.
- Muting segments.
- Rephrasing a segment with an LLM prompt.
- Generating and replaying TTS previews.
- Renaming speakers globally.
- Changing a speaker voice across all matching segments.
- Setting per-segment or per-speaker TTS prompts.
- Overriding inferred speaker gender, then re-running translation when required.
- Resetting only TTS cache while keeping transcription and translation.

If gender overrides change after translation, the backend blocks final dubbing until translation is refreshed so grammatical gender can stay consistent.

## Presets and Configuration

`dubbing_config.yml` is the main configuration file for CLI defaults and UI/API preset defaults.

Important configuration areas:

- `default_preset` and `presets.fast|hq|ultra`
- `source_language`, `target_language`
- `transcription_system`, `whisper_model`, `speakers_expected`
- `llm_provider`, `llm_model_name`, `refinement_*`, `editor_*`
- `tts_system`, `tts_model`, `tts_fallback_model`, `tts_prompt_prefix`
- `voice_auto_selection`, `voice_name`, `voice_prompt`, `tts_system_mapping`
- `keep_background`, `keep_original_audio_ranges`, `dubbed_volume`, `background_volume`
- `segment_stretch`, `segments_optimization`
- `save_original_subtitles`, `save_translated_subtitles`
- `pricing` for cost estimation/reporting

## TTS Providers and Voices

Supported provider paths include:

- **Gemini TTS** - Highest quality path, supports Gemini voice catalog, prompt styles, fallback model, emotion enrichment, and long-form segment handling.
- **OpenAI TTS** - Reliable and faster for balanced jobs.
- **MiniMax TTS** - Additional voice model support via `MINIMAX_API_KEY`.
- **F5/Coqui/local paths** - Available for local or experimental setups when dependencies and references are configured.

Voice selection can be automatic or manually mapped per speaker. The UI exposes provider voices and preview samples when sample files are available.

TTS style presets include:

- `podcast`
- `lecture`
- `gothic`
- `news`
- `custom`
- `auto`

## Translation Personas

Personas are loaded from files in `src/translation/personas/`. Current built-in personas include:

- `none`
- `normal`
- `casual_manager`
- `child`
- `housewife`
- `science_popularizer`
- `it_buddy`
- `ai_buddy`
- `ai_visioner`
- `pedantic`
- `poet`
- `pushkin_style`
- `tractorman`
- `informal`
- `adhd_clarity`
- `product_demo`

The UI can list available personas through `/api/v1/resources/personas`.

## External One-Shot API

Use `POST /api/v1/translate` when an external client wants to upload a video and run the full pipeline in one request.

Example:

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -F "file=@demo.mp4" \
  -F "targetLang=ru" \
  -F "preset=hq" \
  -F "sourceLang=auto" \
  -F "minimalDiarizationMerge=true" \
  -F "speakerCount=1"
```

The response contains a `projectId`, `jobId`, `pollUrl`, and `downloadUrl`.

For a full guide, see [examples/external_translate_api.md](examples/external_translate_api.md). For Python examples, see [examples/api_usage_example.py](examples/api_usage_example.py).

## CLI Usage

### Basic Dubbing

```bash
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language ru
```

### Config File

```bash
python dubblm_cli.py \
  --config dubbing_config.yml \
  --input video.mp4
```

### Cost Estimation

```bash
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language ru \
  --estimate_cost
```

### Segment Testing

```bash
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language ru \
  --start_time 60 \
  --duration 120
```

### Speaker Report

```bash
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language ru \
  --generate_speaker_report
```

### Batch With Glob

```bash
python dubblm_cli.py \
  --input "sources/**/*.mp4" \
  --source_language en \
  --target_language ru
```

For all flags:

```bash
python dubblm_cli.py --help
```

## Output Files

Depending on interface and configuration, DubbLM generates:

- Final dubbed MP4.
- Source and translated subtitles in SRT/VTT.
- Per-project artifacts under `projects/<project_id>/`.
- Intermediate audio chunks, previews, cache files, debug logs, reports, and cost ledgers.
- `report.md` and `report.json` for completed API/UI jobs.

The API result view can stream the dubbed video with HTTP range support and download subtitles/reports.

## Performance Tips

- Use `fast` for quick iteration and `hq`/`ultra` for final outputs.
- Test with `start_time` and `duration` before processing long videos.
- Keep `max_workers` moderate; high values can increase API pressure and memory use.
- Use `--estimate_cost` before long CLI jobs.
- Avoid `keep_background` on very long videos unless the machine has enough RAM.
- Use `reset-tts-cache` from the UI/API when only TTS needs to be regenerated.
- Preserve original audio ranges for sections such as music, ads, intros, or untranslatable clips.

## Demo Video

[![DubbLM Demo](example.png)](https://youtu.be/UADjkgMXQCY)

## Roadmap

- Lighter optional dependency installation for GPU/local-heavy features.
- Better production packaging for API, worker, frontend, Redis, and storage.
- More robust provider fallback and cost prediction.
- Expanded voice catalogs, voice samples, and validation.
- Continued video timing, segment optimization, and editor improvements.

## About Us

This project is open to use and fork. It is developed by IT engineers of [Arteus](https://arteus.io/), a company specializing in adaptive AI systems for business automation, sales, and customer service.

## Contributing

Want to contribute or ask a question: http://t.me/pavelfedortsov

## Give Us a Star

<picture>
  <source
    media="(prefers-color-scheme: dark)"
    srcset="
      https://api.star-history.com/svg?repos=ArteusAI/DubbLM&type=Date&theme=dark
    "
  />
  <source
    media="(prefers-color-scheme: light)"
    srcset="
      https://api.star-history.com/svg?repos=ArteusAI/DubbLM&type=Date
    "
  />
  <img
    alt="Star History Chart"
    src="https://api.star-history.com/svg?repos=ArteusAI/DubbLM&type=Date"
  />
</picture>

## License

MIT

## Changelog Since Last README Update

This changelog covers commits after the last README-touching commit, `4a7d7aa` from 2025-09-19, through `a6e66ac` on 2026-06-06.

### UI and Backend Workflow

- Added the FastAPI/Celery project workflow and connected it to the frontend: project persistence, uploads, config updates, processing jobs, and worker-backed transcription/dubbing (`800c5fd`, `63709ca`).
- Polished the frontend workflow with project dashboard improvements, processing progress, presets, video thumbnails/frames, and Gemini TTS synthesis integration (`7cae81e`, `e74a918`).
- Preserved segment data across project transitions and improved editor/result continuity (`c204bbf`).
- Added one-shot external translation API and API examples for third-party clients (`a6e66ac`).

### Editor and Translation Refinement

- Added editor mode with optional LLM editor pass, model/provider controls, reasoning effort, and text adjustment controls (`a8f5d42`, `f6bb6ab`).
- Improved segment rephrasing and translation prompt handling (`1b02e22`).
- Moved refinement personas into files and expanded persona coverage, including ADHD clarity and demo/product-oriented prompting (`c979213`, `98f3376`, `36bf524`).

### TTS, Voices, and Speaker Controls

- Added per-speaker TTS prompt support and UI/backend config plumbing (`fd89c52`).
- Added automatic speaker gender inference, speaker metadata, gender overrides, and stale-translation checks before dubbing (`66badce`).
- Improved voice segment normalization, volume normalization, long synthesis handling, and Gemini/OpenAI voice matching (`2d4759b`, `7aa4177`).
- Added MiniMax/new TTS model support, voice samples/catalog updates, and TTS provider expansion (`a6e66ac`).
- Reverted default Gemini TTS selection back to the 2.5 TTS model line after experimentation (`ce120dc`).

### Cost Tracking and Reports

- Added cost analysis, adaptive voice segment length estimation, and cost estimation before dubbing (`4e1e954`, `734c2b5`, `76341b4`).
- Added better reporting for estimation misses and more relaxed logging around cost/debug output (`4eae375`, `d2ba822`).
- Added report/cost infrastructure used by result views and downloadable artifacts (`a6e66ac`).

### Segment and Video Timing

- Added segment optimization and better segment stretching for speech/video alignment (`c218d47`, `ec64085`).
- Removed the older pause-removal speedup path and replaced it with newer stretch controls (`20b7383`, `ac343b9`, `3476441`).
- Improved presets and video stretching controls in UI/backend configuration (`e74a918`).

### Defaults, Tests, and Maintenance

- Fixed default parameters after the previous README update (`345bf53`).
- Added tests for important pipeline, frontend/backend, cost, TTS, project, route, and timing behavior (`fdf4634`).
- Continued frontend/backend normalization and integration cleanup across project config and processing paths (`63709ca`, `800c5fd`).
