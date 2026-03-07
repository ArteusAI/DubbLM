"""Tests for stable database URL handling."""

from pathlib import Path

from sqlalchemy.engine import make_url

from src.api.database.session import normalize_database_url


def test_normalize_database_url_sqlite_relative_is_absolute() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_db_path = (repo_root / "dubblm.db").resolve()

    normalized = normalize_database_url("sqlite:///./dubblm.db")
    url = make_url(normalized)

    assert url.drivername.startswith("sqlite")
    assert url.database == str(expected_db_path)


def test_normalize_database_url_ignores_cwd_changes(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]
    expected_db_path = (repo_root / "dubblm.db").resolve()

    import os

    original_cwd = Path.cwd()
    try:
        os.chdir(tmp_path)
        normalized = normalize_database_url("sqlite:///./dubblm.db")
    finally:
        os.chdir(original_cwd)

    url = make_url(normalized)
    assert url.database == str(expected_db_path)

