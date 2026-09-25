"""Shared helpers for calling LLMs with a hard timeout and retries.

This module centralises the pattern that used to be scattered (and duplicated)
across the codebase: run a blocking LLM call in a thread so we can enforce a
wall-clock timeout, and retry transient failures with exponential backoff.

Only ``TimeoutError`` and ``ConnectionError`` are retried by default, because
these are the only errors where repeating the same request is safe and likely
to help.  Validation errors (bad JSON, empty but successful response, etc.)
must be handled by the caller.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from typing import Any, Callable, Iterable, Optional, Tuple, Type

from tenacity import (
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Global defaults, configurable via environment variables.
# ---------------------------------------------------------------------------
LLM_CALL_TIMEOUT_SECONDS = int(os.getenv("LLM_CALL_TIMEOUT_SECONDS", "180"))
LLM_CALL_MAX_ATTEMPTS = int(os.getenv("LLM_CALL_MAX_ATTEMPTS", "3"))
LLM_CALL_BACKOFF_INITIAL = float(os.getenv("LLM_CALL_BACKOFF_INITIAL", "1.0"))
LLM_CALL_BACKOFF_MAX = float(os.getenv("LLM_CALL_BACKOFF_MAX", "60.0"))
LLM_CALL_BACKOFF_FACTOR = float(os.getenv("LLM_CALL_BACKOFF_FACTOR", "2.0"))

DEFAULT_RETRYABLE_EXCEPTIONS: Tuple[Type[BaseException], ...] = (
    TimeoutError,
    ConnectionError,
)


def _resolve_timeout(timeout: Optional[int]) -> int:
    return timeout if timeout is not None and timeout > 0 else LLM_CALL_TIMEOUT_SECONDS


def call_with_timeout(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[int] = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn(*args, **kwargs)`` with a hard wall-clock timeout.

    llama-index (and several SDK) synchronous methods do not expose a reliable
    request timeout.  Wrapping the call in a thread lets us abort the waiting
    side after ``timeout`` seconds and raise a catchable :class:`TimeoutError`.

    The underlying network thread is not forcefully killed on timeout, but the
    executor is shut down without waiting so the caller can retry/fallback
    immediately.
    """
    timeout = _resolve_timeout(timeout)
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except FutureTimeoutError:
        logger.warning("LLM call timed out after %ss", timeout)
        future.cancel()
        raise TimeoutError(f"LLM call timed out after {timeout}s")
    finally:
        executor.shutdown(wait=False)


def robust_llm_call(
    fn: Callable[..., Any],
    *args: Any,
    timeout: Optional[int] = None,
    max_attempts: Optional[int] = None,
    retryable_exceptions: Optional[Iterable[Type[BaseException]]] = None,
    **kwargs: Any,
) -> Any:
    """Call ``fn`` with a timeout and retry on transient failures.

    Retries only the exception types supplied in ``retryable_exceptions``
    (default: ``TimeoutError``, ``ConnectionError``) using exponential backoff
    without jitter.  All other exceptions are propagated immediately.
    """
    timeout = _resolve_timeout(timeout)
    attempts = (
        max_attempts
        if max_attempts is not None and max_attempts > 0
        else LLM_CALL_MAX_ATTEMPTS
    )
    retryable = tuple(retryable_exceptions) if retryable_exceptions else DEFAULT_RETRYABLE_EXCEPTIONS

    retryer = Retrying(
        stop=stop_after_attempt(attempts),
        wait=wait_exponential(
            multiplier=LLM_CALL_BACKOFF_INITIAL,
            min=LLM_CALL_BACKOFF_INITIAL,
            max=LLM_CALL_BACKOFF_MAX,
            exp_base=LLM_CALL_BACKOFF_FACTOR,
        ),
        retry=retry_if_exception_type(retryable),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,
    )

    for attempt in retryer:
        with attempt:
            return call_with_timeout(fn, *args, timeout=timeout, **kwargs)

    # Retrying with reraise=True should never let us get here, but keep the
    # return type consistent for static analysis.
    raise TimeoutError("LLM call failed after retries")
