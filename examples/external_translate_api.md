# DubbLM External Translate API

Краткая документация для внешних клиентов: one-shot перевод видео без многошагового project workflow.

**Base URL:** `http://<host>:<port>/api/v1`

**OpenAPI (Swagger JSON):** `GET /api/v1/openapi.json`  
**Swagger UI:** `GET /api/v1/docs`

---

## Быстрый старт

1. Узнать поддерживаемые языки → `GET /resources/languages`
2. Отправить видео → `POST /translate`
3. Опрашивать прогресс → `GET /projects/{projectId}/jobs/{jobId}/status`
4. Скачать результат → `GET /projects/{projectId}/download/video`

---

## 1. Запуск перевода

### `POST /translate`

Загружает видео и сразу запускает полный пайплайн: транскрипция → перевод → озвучка → сборка MP4.

**Content-Type:** `multipart/form-data`

| Поле | Обязательное | По умолчанию | Описание |
|------|--------------|--------------|----------|
| `file` | да | — | Видео: `.mp4`, `.avi`, `.mkv`, `.mov`, `.webm`, `.m4v` |
| `targetLang` | да | — | Целевой язык (BCP-47 код, напр. `ru`) |
| `preset` | нет | `hq` | Качество: `fast`, `hq`, `ultra` |
| `sourceLang` | нет | `auto` | Язык исходника; `auto` = автоопределение |
| `minimalDiarizationMerge` | нет | `false` | Минимальный merge блоков после диаризации (для screen recording) |
| `keepBackground` | нет | `false` | Сохранить фоновую дорожку из исходника (музыка, ambient) |
| `enableLlmEditor` | нет | `false` | LLM editor pass после перевода (в preset `ultra` включён, но external API переопределяет) |
| `speakerCount` | нет | `1` | Ожидаемое число спикеров для diarization (AssemblyAI) |
| `personaId` | нет | из preset | Персона refinement: `normal`, `informal`, `tractorman`, `none`, … — полный список: `GET /resources/personas` |
| `name` | нет | auto | Имя проекта в системе |

По умолчанию **`keepBackground=false`**: в итоговом видео только озвученная речь. Чтобы смешать с фоном из исходника, передайте `keepBackground=true`.

По умолчанию **`enableLlmEditor=false`**: без дополнительного LLM editor pass (быстрее и дешевле). Для включения: `enableLlmEditor=true`.

**Ответ `202`:**

```json
{
  "projectId": "proj_a1b2c3d4e5f6",
  "jobId": "job_transcribe_abc12345",
  "status": "processing",
  "pollUrl": "/api/v1/projects/proj_a1b2c3d4e5f6/jobs/job_transcribe_abc12345/status",
  "downloadUrl": "/api/v1/projects/proj_a1b2c3d4e5f6/download/video"
}
```

Сохраните `projectId` и `jobId`. `pollUrl` и `downloadUrl` — относительные пути от base URL.

### Пример (curl)

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

### Пример (Python)

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

Если `true`, в конфиг проекта записывается:

- `postDiarizationMergeGap = 0` — почти не объединять соседние блоки одного спикера после диаризации
- `repairSpeakerFragmentation = false` — не «склеивать» короткие фрагменты меток спикеров

Полезно для записей экрана, где важна синхронность речи с действиями на экране.

---

## 2. Доступные языки

### `GET /resources/languages`

Возвращает список языков, которые UI и API принимают в `targetLang` / `sourceLang`:

```json
[
  {"code": "en", "name": "English"},
  {"code": "ru", "name": "Russian"},
  {"code": "es", "name": "Spanish"}
]
```

Полный список — в ответе endpoint. Используйте поле **`code`** в запросах.

**Исходный язык `auto`:** если язык видео неизвестен, передайте `sourceLang=auto` (значение по умолчанию). Транскрибер попытается определить язык автоматически.

**Целевой язык:** указывается явно (`targetLang=ru`, `de`, `en`, …). Код должен быть из списка `/resources/languages` или совместим с BCP-47, который поддерживает backend.

### Пресеты качества

```http
GET /settings/presets
GET /settings/presets/hq
```

| preset | Назначение |
|--------|------------|
| `fast` | Быстрее, проще TTS, 720p |
| `hq` | Баланс качества и скорости (по умолчанию) |
| `ultra` | Максимальное качество, LLM editor, original video quality |

---

## 3. Прогресс перевода

One-shot запрос создаёт один job с `autoProcess=true`: тот же `jobId` проходит транскрипцию **и** озвучку до конца.

### Вариант A: polling (рекомендуется для простых клиентов)

```http
GET /projects/{projectId}/jobs/{jobId}/status
```

**Пример ответа:**

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

**Поля:**

| Поле | Значения | Смысл |
|------|----------|-------|
| `status` | `pending`, `processing`, `completed`, `failed`, `cancelled` | Состояние job |
| `progress` | `0`–`100` | Общий прогресс |
| `currentStep` | см. ниже | Текущая фаза |
| `errorMessage` | string / null | Текст ошибки при `failed` |
| `logs` | массив | Последние сообщения пайплайна |

**Типичные `currentStep`:**

| step | Этап |
|------|------|
| `initialization` | Старт |
| `audio_extraction` | Извлечение аудио |
| `diarization` | Диаризация |
| `speaker_analysis` | Анализ спикеров |
| `translation` | Перевод |
| `handoff` | Переход к озвучке |
| `speech_synthesis` | TTS |
| `background_audio` | Фоновое аудио |
| `video_combine` | Сборка видео |
| `complete` | Готово |

**Когда считать job завершённым:**

- `status == "completed"` → можно скачивать видео
- `status == "failed"` → смотреть `errorMessage`
- `status == "cancelled"` → остановлен пользователем

**Интервал polling:** 5–15 секунд. Не чаще 1 раза в секунду.

### Вариант B: SSE stream (для live UI)

```http
GET /projects/{projectId}/status
Accept: text/event-stream
```

События:

- `progress` — `{"percent": 48, "step": "speech_synthesis"}`
- `log` — строка лога
- `complete` — `{"status": "dubbed"}` или `{"status": "error", "error": "..."}`

SSE привязан к **активному job проекта**, не к конкретному `jobId`. После one-shot translate у проекта один основной job — stream подходит.

---

## 4. Получение результата

### Видео

```http
GET /projects/{projectId}/download/video
```

Доступно когда job завершился со `status: completed` и проект в состоянии `dubbed`.

Также поддерживается range streaming:

```http
GET /projects/{projectId}/stream/video
```

### Субтитры (опционально)

```http
GET /projects/{projectId}/download/subtitles?format=srt&lang=target
GET /projects/{projectId}/download/subtitles?format=srt&lang=source
```

### Сегменты с текстом (опционально)

```http
GET /projects/{projectId}/segments
```

---

## 5. Ошибки и лимиты

| HTTP | Причина |
|------|---------|
| `400` | Неверный формат файла, не указан `targetLang` |
| `404` | Проект / job / результат не найден |
| `413` | Файл больше лимита upload (по умолчанию 5 GB) |
| `500` | Ошибка сохранения файла или внутренняя ошибка |

При `failed` job детали — в `errorMessage` и в `logs`.

---

## 6. Health check

```http
GET /health
```

```json
{"status": "healthy", "version": "1.0.0"}
```

---

## 7. Связь с полным API

External endpoint — обёртка над тем же backend, что и UI:

| External | Полный API (эквивалент) |
|----------|-------------------------|
| `POST /translate` | `POST /projects` + `PATCH .../config` + `POST .../upload` + `POST .../process/transcribe` (с `autoProcess: true`) |
| `pollUrl` | `GET /projects/{id}/jobs/{jobId}/status` |
| `downloadUrl` | `GET /projects/{id}/download/video` |

Frontend и ручные интеграции продолжают использовать project-centric API; external clients могут работать только через `/translate` + polling + download.
