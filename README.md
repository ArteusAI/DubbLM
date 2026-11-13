# DubbLM

![DubbLM Logo](logo.png)

DubbLM is an AI video dubbing system for context-aware translation, expressive
speech synthesis, speaker diarization, and automatic video assembly. It can be
used through a React UI, a FastAPI backend, a one-shot API, or the CLI.

## Important Notice

**DubbLM is optimized for quality, not real-time processing.** The pipeline
analyzes speaker context, translates and refines dialogue, synthesizes speech,
adjusts timing, and rebuilds the video. Processing can take significantly
longer than the source video duration, especially with premium TTS, background
preservation, LLM editor passes, or high-resolution video.

## Features

- **OpenRouter-first defaults** — one `OPENROUTER_API_KEY` covers transcription,
  diarization, translation, refinement, TTS, and emotion enrichment.
- **Speaker-aware workflow** — word-level diarization, speaker metadata, voice
  matching, per-speaker prompts, and gender overrides.
- **Context-aware translation** with OpenRouter-backed LLMs (Gemini, GPT,
  Claude, and others).
- **High-quality TTS** with Gemini 3.8 Flash via OpenRouter, plus optional
  OpenRouter Qwen / Fish Audio / Grok voices.
- **React UI** for uploads, URL downloads, presets, editing, TTS previews, and
  result downloads.
- **FastAPI + Celery** backend with persistent projects, async jobs, SSE
  progress, reports, cost tracking, and an external API.
- **Video controls** for background audio, original-audio ranges, segment
  stretching, volume, subtitles, and quality.
- **Optional ИИ Борщ publishing** from the result view when
  `BORSCH_PUBLISH_ENABLED=true`.

## How It Works

1. Upload a video or download it from a supported URL.
2. Extract audio and run transcription + speaker diarization.
3. Translate with full dialogue context and optional glossary/persona guidance.
4. Refine wording and apply an optional LLM editor pass.
5. Review and edit segments, speakers, voices, and TTS prompts.
6. Synthesize speech per segment with timing and voice validation.
7. Assemble audio/video, generate subtitles, and write reports and cost data.

## Quick Start with Docker Compose

Requirements: Docker with the Compose plugin.

```bash
git clone https://github.com/ArteusAI/DubbLM.git
cd DubbLM

cp .env.example .env
cp settings.example.yml settings.yml

docker compose up -d --build
```

Open the web UI:

```text
http://localhost:3007
```

On first launch the setup wizard asks only for an OpenRouter key. It validates
the key and writes it to `.env` and `settings.yml`. The API is available at
`http://localhost:8007`; Swagger UI is at
`http://localhost:8007/api/v1/docs`.

Useful commands:

```bash
docker compose ps
docker compose logs -f api worker
docker compose restart api worker
docker compose down
docker compose up -d --build
```

The compose stack runs:

| Service | Purpose | Port |
|---|---|---|
| `frontend` | React UI + nginx proxy | `3007` |
| `api` | FastAPI backend | `8007` |
| `worker` | Celery transcription/dubbing workers | internal |
| `redis` | Celery broker/result backend | internal |

Persistent data is stored in `projects/`, `cache/`, `dubblm.db`, `settings.yml`,
and `.env`. Back these up before reinstalling. They are intentionally excluded
from git.

## Local Python Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

redis-server
```

Start the backend and worker in separate terminals:

```bash
python run_api.py
python run_worker.py
```

Start the frontend:

```bash
cd src/frontend
npm install
API_URL=http://localhost:8000 npm run dev
```

The frontend runs at `http://localhost:3000` and proxies `/api` to the backend.

Defaults live in `dubbing_config.yml`; translation personas are files under
`src/translation/personas/`.

## API Keys and First-Run Setup

The default `fast`, `hq`, and `ultra` profiles need one key:

```env
OPENROUTER_API_KEY=sk-or-...
```

The first-run web wizard validates the key and persists it. Optional legacy
providers are not required for the default profiles:

- `OPENAI_API_KEY` — legacy OpenAI TTS/transcription.
- `GOOGLE_API_KEY` / `GEMINI_API_KEY` — legacy direct Gemini paths.
- `ASSEMBLYAI_API_KEY` — legacy direct AssemblyAI transcription.
- `HF_TOKEN` — optional Hugging Face access for local models.
- `MINIMAX_API_KEY` / `MINIMAX_GROUP_ID` — legacy MiniMax TTS.

If a Docker-mounted `.env` is not writable, the key is still persisted in
`settings.yml` and applied to the runtime environment.

## Web UI Workflow

1. Open the dashboard and upload one or more videos, or paste a video URL.
2. Choose source/target languages and a profile: `fast`, `hq`, or `ultra`
   (`ultra` is the default).
3. Start transcription and review segments in the editor.
4. Adjust translations, speaker names, voices, prompts, or mute segments.
5. Preview individual TTS segments when needed.
6. Start final dubbing and download the result, subtitles, and report.

The result view supports video streaming with HTTP range requests, downloads,
subtitles, cost reports, and optional ИИ Борщ publishing.

## External One-Shot API

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -F "file=@demo.mp4" \
  -F "targetLang=ru" \
  -F "preset=ultra" \
  -F "sourceLang=auto"
```

The response contains `projectId`, `jobId`, `pollUrl`, and `downloadUrl`.
See [examples/external_translate_api.md](examples/external_translate_api.md)
and [examples/api_usage_example.py](examples/api_usage_example.py).

## CLI

```bash
python dubblm_cli.py \
  --input video.mp4 \
  --source_language en \
  --target_language ru
```

Configuration is loaded from `dubbing_config.yml` and can be overridden with
flags. Useful options:

```bash
# Cost estimate
python dubblm_cli.py --input video.mp4 --estimate_cost

# Process a time window
python dubblm_cli.py --input video.mp4 --start_time 60 --duration 120

# Speaker report
python dubblm_cli.py --input video.mp4 --generate_speaker_report

# All flags
python dubblm_cli.py --help
```

## Output Files

- Final dubbed MP4.
- Source/translated subtitles in SRT/VTT.
- Project artifacts under `projects/<project_id>/`.
- TTS previews, audio chunks, cache files, and logs.
- `report.md` and `report.json` with timing, costs, and API usage.

## Performance Tips

- Use `fast` for quick iteration and `hq`/`ultra` for final output.
- Test with `--start_time` and `--duration` before long videos.
- Keep `max_workers` moderate; high values increase API pressure and memory use.
- Use `--estimate_cost` before long CLI jobs.
- Avoid `keep_background` on very long videos unless the machine has enough RAM.
- Use `reset-tts-cache` when only TTS needs to be regenerated.

## Dubbing Example: ИИ Борщ

A real Russian dub produced with DubbLM and published on the **ИИ Борщ** channel:

[**Эндрю Ын: где на самом деле скрыты главные возможности ИИ**](https://ai-borsch.ru/watch/fjn1sIdA6JE)

- Source: Andrew Ng interview, English
- Target: Russian
- Duration: ~35:40
- Pipeline: OpenRouter MAI diarization → context-aware translation/refinement → Gemini 3.8 Flash TTS via OpenRouter

### Translation sample

**Original:**

> There's been a lot of misinformation about AI.

**Russian dub:**

> Вокруг искусственного интеллекта, или ИИ, сейчас много дезинформации.

**Original:**

> So a handful of leading AI companies have been very loud voices fear mongering around AI to try to get regulations passed. This drumbeat of fear based messaging has skewed societal's perception to be really negative on AI.

**Russian dub:**

> Несколько ведущих ИИ-компаний активно раздувают страхи вокруг этой технологии, чтобы добиться её законодательного регулирования. [short pause] Из-за этого постоянного потока пугающих заявлений общество стало относиться к ИИ гораздо хуже.

`[short pause]` is a delivery instruction for the TTS model; it is not spoken
in the final audio.

## Roadmap

- Lighter optional dependency installation for GPU/local-heavy features.
- Better production packaging for API, worker, frontend, Redis, and storage.
- More robust provider fallback and cost prediction.
- Expanded voice catalogs, voice samples, and validation.
- Continued video timing, segment optimization, and editor improvements.

## About Us

Developed by IT engineers at [Arteus](https://arteus.io/), a company
specializing in adaptive AI systems for business automation, sales, and
customer service.

## Contributing

Questions and contributions: http://t.me/pavelfedortsov

If DubbLM is useful, please give it a star:
https://github.com/ArteusAI/DubbLM

## License

MIT
