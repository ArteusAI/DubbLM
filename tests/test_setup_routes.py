from __future__ import annotations

import asyncio
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.api.services import settings_service

try:
    from fastapi import HTTPException
    from src.api.routes import setup as setup_routes

    SETUP_ROUTES_AVAILABLE = True
except Exception:  # pragma: no cover - depends on the local venv
    HTTPException = None  # type: ignore[assignment]
    setup_routes = None  # type: ignore[assignment]
    SETUP_ROUTES_AVAILABLE = False


@unittest.skipUnless(SETUP_ROUTES_AVAILABLE, "FastAPI runtime is not installed in this venv")
class SetupRouteTests(unittest.TestCase):
    def test_status_when_not_configured(self) -> None:
        with patch("src.api.routes.setup.has_openrouter_key", return_value=False):
            response = asyncio.run(setup_routes.setup_status())
        self.assertEqual(response["required"], True)
        self.assertEqual(response["configured"], False)

    def test_status_when_configured(self) -> None:
        with patch("src.api.routes.setup.has_openrouter_key", return_value=True):
            response = asyncio.run(setup_routes.setup_status())
        self.assertEqual(response["required"], False)
        self.assertEqual(response["configured"], True)

    def test_rejects_invalid_prefix(self) -> None:
        with patch("src.api.routes.setup.has_openrouter_key", return_value=False):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    setup_routes.configure_openrouter_key(
                        setup_routes.OpenRouterKeyPayload(key="not-an-openrouter-key")
                    )
                )
        self.assertEqual(context.exception.status_code, 400)

    def test_rejects_when_already_configured(self) -> None:
        with patch("src.api.routes.setup.has_openrouter_key", return_value=True):
            with self.assertRaises(HTTPException) as context:
                asyncio.run(
                    setup_routes.configure_openrouter_key(
                        setup_routes.OpenRouterKeyPayload(key="sk-or-v1-test")
                    )
                )
        self.assertEqual(context.exception.status_code, 403)

    def test_persists_valid_key(self) -> None:
        with (
            patch("src.api.routes.setup.has_openrouter_key", return_value=False),
            patch("src.api.routes.setup.validate_openrouter_key", return_value={"data": {}}),
            patch(
                "src.api.routes.setup.save_openrouter_key",
                return_value={"configured": True, "masked": "sk-o...test", "source": "env+settings"},
            ) as save_mock,
        ):
            response = asyncio.run(
                setup_routes.configure_openrouter_key(
                    setup_routes.OpenRouterKeyPayload(key="sk-or-v1-test")
                )
            )
        self.assertEqual(response["configured"], True)
        self.assertEqual(response["masked"], "sk-o...test")
        save_mock.assert_called_once_with("sk-or-v1-test")

class EnvWriterTests(unittest.TestCase):
    def test_env_writer_preserves_comments_and_other_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text(
                "# keep me\nOPENAI_API_KEY=old\nOPENROUTER_API_KEY=old\nBORSCH_CHANNEL=test\n",
                encoding="utf-8",
            )
            with patch.object(settings_service, "_env_file_path", return_value=env_path):
                written = settings_service._write_env_value("OPENROUTER_API_KEY", "sk-or-new")
            text = written.read_text(encoding="utf-8")
        self.assertIn("# keep me", text)
        self.assertIn("OPENAI_API_KEY=old", text)
        self.assertIn("OPENROUTER_API_KEY=sk-or-new", text)
        self.assertNotIn("OPENROUTER_API_KEY=old", text)
        self.assertIn("BORSCH_CHANNEL=test", text)

    def test_env_writer_handles_bind_mount_replace_failure(self) -> None:
        import errno

        with tempfile.TemporaryDirectory() as tmp:
            env_path = Path(tmp) / ".env"
            env_path.write_text("OPENROUTER_API_KEY=old\n", encoding="utf-8")
            with patch.object(settings_service, "_env_file_path", return_value=env_path):
                with patch.object(
                    settings_service.os,
                    "replace",
                    side_effect=OSError(errno.EBUSY, "mount point"),
                ):
                    written = settings_service._write_env_value(
                        "OPENROUTER_API_KEY", "sk-or-new"
                    )
            text = written.read_text(encoding="utf-8")
        self.assertIn("OPENROUTER_API_KEY=sk-or-new", text)


if __name__ == "__main__":
    unittest.main()
