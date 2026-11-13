"""Small OpenRouter client used by the dubbing pipeline.

The project already depends on ``requests`` and on the OpenAI-compatible
OpenRouter endpoints, so this module keeps the direct HTTP calls in one place:
API-key lookup, speech-to-text, text-to-speech and chat completions.
"""

from __future__ import annotations

import base64
import logging
import os
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import requests

logger = logging.getLogger(__name__)

OPENROUTER_API_URL = "https://openrouter.ai/api/v1"
OPENROUTER_TRANSCRIPTIONS_URL = f"{OPENROUTER_API_URL}/audio/transcriptions"
OPENROUTER_SPEECH_URL = f"{OPENROUTER_API_URL}/audio/speech"
OPENROUTER_CHAT_URL = f"{OPENROUTER_API_URL}/chat/completions"
OPENROUTER_KEY_URL = f"{OPENROUTER_API_URL}/key"

DEFAULT_TIMEOUT_SECONDS = 180
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_BACKOFF_SECONDS = 2.0
RETRYABLE_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}


class OpenRouterError(RuntimeError):
    """Raised when OpenRouter returns an error or an unusable response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        payload: Any = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.payload = payload


def get_openrouter_api_key(explicit: Optional[str] = None) -> Optional[str]:
    """Return an OpenRouter key from an argument, environment or settings file."""
    if explicit and str(explicit).strip():
        return str(explicit).strip()
    for env_name in ("OPENROUTER_API_KEY", "OPENROUTER_KEY"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    # Settings are stored in settings.yml; importing lazily avoids a circular
    # import when the API settings service imports this module for validation.
    try:
        from src.api.services.settings_service import get_api_key

        value = get_api_key("openrouter")
        if value and str(value).strip():
            return str(value).strip()
    except Exception:
        pass
    return None


def require_openrouter_api_key(explicit: Optional[str] = None) -> str:
    key = get_openrouter_api_key(explicit)
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is required. Configure it through the web setup "
            "wizard or in .env."
        )
    return key


def _headers(api_key: Optional[str] = None) -> Dict[str, str]:
    return {
        "Authorization": f"Bearer {require_openrouter_api_key(api_key)}",
        "Content-Type": "application/json",
        "HTTP-Referer": os.environ.get("OPENROUTER_SITE_URL", "https://github.com/fpv/DubbLM"),
        "X-OpenRouter-Title": os.environ.get("OPENROUTER_APP_NAME", "DubbLM"),
    }


def _error_message(response: requests.Response) -> Tuple[str, Any]:
    payload: Any = None
    try:
        payload = response.json()
    except Exception:
        payload = response.text[:4000] if response.text else None
    if isinstance(payload, dict):
        error = payload.get("error")
        if isinstance(error, dict):
            text = error.get("message") or error.get("code")
            if text:
                return str(text), payload
        text = payload.get("message") or payload.get("detail")
        if text:
            return str(text), payload
    return f"HTTP {response.status_code}: {payload!r}", payload


def openrouter_request(
    method: str,
    url: str,
    *,
    json_body: Optional[Mapping[str, Any]] = None,
    data: Any = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    backoff_seconds: float = DEFAULT_BACKOFF_SECONDS,
) -> requests.Response:
    """Perform an OpenRouter request with retries on transient errors."""
    attempts = max(1, int(max_attempts or 1))
    last_error: Optional[OpenRouterError] = None

    for attempt in range(attempts):
        try:
            response = requests.request(
                method,
                url,
                headers=_headers(api_key),
                json=json_body,
                data=data,
                timeout=timeout,
            )
        except requests.RequestException as exc:
            last_error = OpenRouterError(f"OpenRouter request failed: {exc}")
        else:
            if response.ok:
                return response
            message, payload = _error_message(response)
            last_error = OpenRouterError(
                message,
                status_code=response.status_code,
                payload=payload,
            )
            if response.status_code not in RETRYABLE_STATUS_CODES:
                raise last_error

        if attempt + 1 < attempts:
            delay = max(0.1, backoff_seconds * (2 ** attempt))
            logger.warning(
                "OpenRouter request failed (%s attempt %d/%d): %s; retrying in %.1fs",
                method,
                attempt + 1,
                attempts,
                last_error,
                delay,
            )
            time.sleep(delay)

    assert last_error is not None
    raise last_error


def validate_openrouter_key(key: str, *, timeout: float = 20.0) -> Dict[str, Any]:
    """Validate a key with OpenRouter and return its metadata.

    Raises :class:`OpenRouterError` with ``status_code`` 401/403 when the key is
    invalid or disabled.
    """
    key = str(key or "").strip()
    if not key:
        raise OpenRouterError("OpenRouter key is empty", status_code=400)
    response = openrouter_request(
        "GET",
        OPENROUTER_KEY_URL,
        api_key=key,
        timeout=timeout,
        max_attempts=1,
    )
    try:
        payload = response.json()
    except Exception:
        payload = {}
    return payload if isinstance(payload, dict) else {"data": payload}


def transcribe_audio(
    audio_bytes: bytes,
    audio_format: str,
    *,
    model: str,
    language: Optional[str] = None,
    response_format: str = "verbose_json",
    timestamp_granularities: Sequence[str] = ("segment", "word"),
    provider_options: Optional[Mapping[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """Transcribe audio through OpenRouter's speech-to-text endpoint."""
    if not audio_bytes:
        raise OpenRouterError("Audio payload is empty", status_code=400)
    body: Dict[str, Any] = {
        "model": model,
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": str(audio_format or "mp3").lstrip("."),
        },
        "response_format": response_format,
        "timestamp_granularities": list(timestamp_granularities),
    }
    if language and str(language).strip().lower() not in ("auto", ""):
        body["language"] = str(language).strip().lower()
    if provider_options:
        body["provider"] = {"options": dict(provider_options)}

    response = openrouter_request(
        "POST",
        OPENROUTER_TRANSCRIPTIONS_URL,
        json_body=body,
        api_key=api_key,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    try:
        payload = response.json()
    except Exception as exc:
        raise OpenRouterError(f"OpenRouter transcription returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpenRouterError("OpenRouter transcription returned an unexpected payload")
    return payload


def synthesize_speech(
    text: str,
    *,
    model: str,
    voice: Optional[str] = None,
    instructions: Optional[str] = None,
    response_format: str = "pcm",
    provider_options: Optional[Mapping[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Tuple[bytes, Mapping[str, str]]:
    """Generate speech and return ``(audio_bytes, response_headers)``."""
    if not text or not text.strip():
        return b"", {}
    body: Dict[str, Any] = {
        "model": model,
        "input": text,
        "response_format": response_format,
    }
    if voice:
        body["voice"] = voice
    if instructions:
        body["instructions"] = instructions
    if provider_options:
        body["provider"] = {"options": dict(provider_options)}

    response = openrouter_request(
        "POST",
        OPENROUTER_SPEECH_URL,
        json_body=body,
        api_key=api_key,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    return response.content or b"", dict(response.headers)


def chat_completion(
    messages: Sequence[Mapping[str, Any]],
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: Optional[int] = None,
    response_format: Optional[Mapping[str, Any]] = None,
    reasoning_effort: Optional[str] = None,
    provider_options: Optional[Mapping[str, Any]] = None,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
) -> Dict[str, Any]:
    """Call an OpenRouter chat-completions model and return the JSON response."""
    body: Dict[str, Any] = {
        "model": model,
        "messages": list(messages),
        "temperature": temperature,
    }
    if max_tokens is not None:
        body["max_tokens"] = int(max_tokens)
    if response_format:
        body["response_format"] = dict(response_format)
    if reasoning_effort:
        body["reasoning"] = {"effort": reasoning_effort}
    if provider_options:
        body["provider"] = {"options": dict(provider_options)}

    response = openrouter_request(
        "POST",
        OPENROUTER_CHAT_URL,
        json_body=body,
        api_key=api_key,
        timeout=timeout,
        max_attempts=max_attempts,
    )
    try:
        payload = response.json()
    except Exception as exc:
        raise OpenRouterError(f"OpenRouter chat returned invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise OpenRouterError("OpenRouter chat returned an unexpected payload")
    return payload


def extract_message_text(payload: Mapping[str, Any]) -> str:
    """Extract assistant text from a chat-completions payload."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    message = choices[0].get("message") if isinstance(choices[0], dict) else None
    if not isinstance(message, dict):
        return ""
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return ""


def audio_input_part(audio_bytes: bytes, audio_format: str = "wav") -> Dict[str, Any]:
    """Build an OpenRouter ``input_audio`` content part."""
    return {
        "type": "input_audio",
        "input_audio": {
            "data": base64.b64encode(audio_bytes).decode("ascii"),
            "format": str(audio_format or "wav").lstrip("."),
        },
    }
