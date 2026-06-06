# DubbLM External Translate API

Brief documentation for external clients: one-shot video translation without the multi-step project workflow.

**Base URL:** `http://<host>:<port>/api/v1`

**OpenAPI (Swagger JSON):** `GET /api/v1/openapi.json`  
**Swagger UI:** `GET /api/v1/docs`

---

## Quick Start

1. Get supported languages -> `GET /resources/languages`
2. Submit a video -> `POST /translate`
3. Poll progress -> `GET /projects/{projectId}/jobs/{jobId}/status`
4. Download the result -> `GET /projects/{projectId}/download/video`

---

## 1. Starting Translation

### `POST /translate`

Uploads a video and immediately starts the full pipeline: transcription -> translation -> dubbing -> MP4 assembly.

**Content-Type:** `multipart/form-data`

| Field | Required | Default | Description |
|------|--------------|--------------|----------|
| `file` | yes | - | Video: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v` |
| `targetLang` | yes | - | Target language (BCP-47 code, e.g. `ru`) |
| `preset` | no | `hq` | Quality: `fast`, `hq`, `ultra` |
| `sourceLang` | no | `auto` | Source language; `auto` = auto-detect |
| `minimalDiarizationMerge` | no | `false` | Minimal post-diarization block merging (for screen recordings) |
| `keepBackground` | no | `false` | Keep the source background track (music, ambient audio) |
| `enableLlmEditor` | no | `false` | LLM editor pass after translation (`ultra` enables it, but the external API overrides it) |
| `speakerCount` | no | `1` | Expected number of speakers for diarization (AssemblyAI) |
| `personaId` | no | from preset | Persona refinement: `normal`, `informal`, `tractorman`, `none`, ... full list: `GET /resources/personas` |
| `name` | no | auto | Project name in the system |

By default, **`keepBackground=false`**: the final video contains only dubbed speech. To mix it with the source background, pass `keepBackground=true`.

By default, **`enableLlmEditor=false`**: no additional LLM editor pass (faster and cheaper). To enable it, pass `enableLlmEditor=true`.

**`202` response:**

```json
{
  "projectId": "proj_a1b2c3d4e5f6",
  "jobId": "job_transcribe_abc12345",
  "status": "processing",
  "pollUrl": "/api/v1/projects/proj_a1b2c3d4e5f6/jobs/job_transcribe_abc12345/status",
  "downloadUrl": "/api/v1/projects/proj_a1b2c3d4e5f6/download/video"
}
```

Save `projectId` and `jobId`. `pollUrl` and `downloadUrl` are relative paths from the base URL.

### Example (curl)

```bash
curl -X POST "http://localhost:8000/api/v1/translate" \
  -F "file=@demo.mp4" \
  -F "targetLang=ru" \
  -F "preset=hq" \
  -F "sourceLang=auto" \
  -F "minimalDiarizationMerge=true" \
  -F "speakerCount=1" \
  -F "personaId=tractorman"
```

### Example (Python)

```python
import requests
import time

BASE = "http://localhost:8000/api/v1"

with open("demo.mp4", "rb") as f:
    resp = requests.post(
        f"{BASE}/translate",
        files={"file": ("demo.mp4", f, "video/mp4")},
        data={"targetLang": "ru", "preset": "hq", "minimalDiarizationMerge": "true"},
    )
resp.raise_for_status()
job = resp.json()

poll_url = f"{BASE}{job['pollUrl']}"
while True:
    status = requests.get(poll_url).json()
    print(status["progress"], status["currentStep"], status["status"])
    if status["status"] in ("completed", "failed", "cancelled"):
        break
    time.sleep(10)

if status["status"] == "completed":
    video = requests.get(f"{BASE}{job['downloadUrl']}")
    open("result.mp4", "wb").write(video.content)
```

### `minimalDiarizationMerge`

If `true`, the following values are written to the project config:

- `postDiarizationMergeGap = 0` - almost do not merge neighboring blocks from the same speaker after diarization
- `repairSpeakerFragmentation = false` - do not "stitch" short fragments of speaker labels

Useful for screen recordings where speech must stay synchronized with on-screen actions.

---

## 2. Available Languages

### `GET /resources/languages`

Returns the list of languages accepted by the UI and API in `targetLang` / `sourceLang`:

```json
[
  {"code": "en", "name": "English"},
  {"code": "ru", "name": "Russian"},
  {"code": "es", "name": "Spanish"}
]
```

The full list is returned by the endpoint. Use the **`code`** field in requests.

**Source language `auto`:** if the video's language is unknown, pass `sourceLang=auto` (the default). The transcriber will try to detect the language automatically.

**Target language:** specified explicitly (`targetLang=ru`, `de`, `en`, ...). The code must be from `/resources/languages` or be a BCP-47-compatible code supported by the backend.

### Quality Presets

```http
GET /settings/presets
GET /settings/presets/hq
```

| preset | Purpose |
|--------|------------|
| `fast` | Faster, simpler TTS, 720p |
| `hq` | Quality/speed balance (default) |
| `ultra` | Maximum quality, LLM editor, original video quality |

---

## 3. Translation Progress

A one-shot request creates one job with `autoProcess=true`: the same `jobId` runs through transcription **and** dubbing until completion.

### Option A: Polling (recommended for simple clients)

```http
GET /projects/{projectId}/jobs/{jobId}/status
```

**Example response:**

```json
{
  "jobId": "job_transcribe_abc12345",
  "projectId": "proj_a1b2c3d4e5f6",
  "type": "transcribe",
  "status": "processing",
  "progress": 48,
  "currentStep": "speech_synthesis",
  "logs": [
    {"id": "...", "message": "Translating chunk 2/5", "type": "info", "timestamp": "..."}
  ],
  "errorMessage": null,
  "createdAt": "2026-05-31T12:00:00+00:00",
  "startedAt": "2026-05-31T12:00:01+00:00",
  "completedAt": null
}
```

**Fields:**

| Field | Values | Meaning |
|------|----------|-------|
| `status` | `pending`, `processing`, `completed`, `failed`, `cancelled` | Job state |
| `progress` | `0`-`100` | Overall progress |
| `currentStep` | see below | Current phase |
| `errorMessage` | string / null | Error text when `failed` |
| `logs` | array | Latest pipeline messages |

**Typical `currentStep` values:**

| step | Stage |
|------|------|
| `initialization` | Start |
| `audio_extraction` | Audio extraction |
| `diarization` | Diarization |
| `speaker_analysis` | Speaker analysis |
| `translation` | Translation |
| `handoff` | Handoff to dubbing |
| `speech_synthesis` | TTS |
| `background_audio` | Background audio |
| `video_combine` | Video assembly |
| `complete` | Done |

**When to treat the job as finished:**

- `status == "completed"` -> video can be downloaded
- `status == "failed"` -> check `errorMessage`
- `status == "cancelled"` -> stopped by the user

**Polling interval:** 5-15 seconds. Do not poll more often than once per second.

### Option B: SSE Stream (for live UI)

```http
GET /projects/{projectId}/status
Accept: text/event-stream
```

Events:

- `progress` - `{"percent": 48, "step": "speech_synthesis"}`
- `log` - log line
- `complete` - `{"status": "dubbed"}` or `{"status": "error", "error": "..."}`

SSE is bound to the **active project job**, not to a specific `jobId`. After a one-shot translate request, the project has one main job, so the stream is suitable.

---

## 4. Getting the Result

### Video

```http
GET /projects/{projectId}/download/video
```

Available when the job has finished with `status: completed` and the project is in the `dubbed` state.

Range streaming is also supported:

```http
GET /projects/{projectId}/stream/video
```

### Subtitles (optional)

```http
GET /projects/{projectId}/download/subtitles?format=srt&lang=target
GET /projects/{projectId}/download/subtitles?format=srt&lang=source
```

### Text Segments (optional)

```http
GET /projects/{projectId}/segments
```

---

## 5. Errors and Limits

| HTTP | Reason |
|------|---------|
| `400` | Invalid file format, `targetLang` not specified |
| `404` | Project / job / result not found |
| `413` | File exceeds the upload limit (5 GB by default) |
| `500` | File save error or internal error |

For a `failed` job, details are available in `errorMessage` and `logs`.

---

## 6. Health check

```http
GET /health
```

```json
{"status": "healthy", "version": "1.0.0"}
```

---

## 7. Relationship to the Full API

The external endpoint is a wrapper over the same backend used by the UI:

| External | Full API (equivalent) |
|----------|-------------------------|
| `POST /translate` | `POST /projects` + `PATCH .../config` + `POST .../upload` + `POST .../process/transcribe` (with `autoProcess: true`) |
| `pollUrl` | `GET /projects/{id}/jobs/{jobId}/status` |
| `downloadUrl` | `GET /projects/{id}/download/video` |

Frontend and manual integrations continue to use the project-centric API; external clients can work only through `/translate` + polling + download.
