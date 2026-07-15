"""Database session management."""

from pathlib import Path
from typing import Generator, List
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.engine import make_url
from sqlalchemy.orm import sessionmaker, Session

from ..config import get_settings
from .models import Base


# Engine and session factory (initialized lazily)
_engine = None
_SessionLocal = None


def normalize_database_url(database_url: str) -> str:
    """Normalize database URLs so relative SQLite paths are stable across chdir().

    Celery tasks may change the current working directory (e.g. into a project
    folder). If the DB URL uses a relative SQLite path like `sqlite:///./dubblm.db`,
    SQLAlchemy will resolve it relative to the *current* working directory at the
    time the engine is created, which can silently create a new empty DB file and
    later fail with "no such table".
    """

    try:
        url = make_url(database_url)
    except Exception:
        return database_url

    if not url.drivername.startswith("sqlite"):
        return database_url

    if not url.database or url.database == ":memory:" or url.database.startswith("file:"):
        return database_url

    db_path = Path(url.database)
    if db_path.is_absolute():
        return str(url)

    try:
        # In this repo, src/api/database/session.py -> repo root is parents[3]
        base_dir = Path(__file__).resolve().parents[3]
    except Exception:
        base_dir = Path.cwd().resolve()

    abs_path = (base_dir / db_path).resolve()
    return str(url.set(database=str(abs_path)))


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        database_url = normalize_database_url(settings.database_url)
        _engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
            echo=settings.debug,
        )
    return _engine


def get_session_factory():
    """Get or create the session factory."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal


def init_db() -> None:
    """Initialize the database and create all tables."""
    engine = get_engine()
    Base.metadata.create_all(bind=engine)
    _ensure_project_columns(engine)


# Columns added to the `projects` table after initial release. Each entry is
# (column_name, SQL type declaration for SQLite ALTER TABLE ADD COLUMN).
_PROJECT_EXTRA_COLUMNS: List[tuple] = [
    ("source_width", "INTEGER"),
    ("source_height", "INTEGER"),
    ("source_duration", "REAL"),
    ("result_size", "INTEGER"),
    ("result_width", "INTEGER"),
    ("result_height", "INTEGER"),
    ("result_duration", "REAL"),
    ("total_size", "INTEGER"),
]


def _ensure_project_columns(engine) -> None:
    """Add new columns to the `projects` table if missing.

    SQLite supports `ALTER TABLE ... ADD COLUMN` for nullable columns without
    defaults, so existing rows are unaffected. This is idempotent and acts as
    a lightweight migration in the absence of a migration framework.
    """
    inspector = inspect(engine)
    if "projects" not in inspector.get_table_names():
        return

    existing = {col["name"] for col in inspector.get_columns("projects")}
    with engine.begin() as conn:
        for col_name, col_type in _PROJECT_EXTRA_COLUMNS:
            if col_name not in existing:
                conn.execute(
                    text(f'ALTER TABLE projects ADD COLUMN "{col_name}" {col_type}')
                )


def get_db() -> Generator[Session, None, None]:
    """Dependency for getting database sessions."""
    SessionLocal = get_session_factory()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session(fresh: bool = False) -> Session:
    """Get a database session directly (for use in Celery tasks).
    
    Args:
        fresh: If True, creates a completely new engine connection to ensure
               we get the latest data (useful for cross-process synchronization).
    """
    global _engine, _SessionLocal
    
    if fresh:
        # Force a completely fresh connection by recreating the engine
        # This is necessary for SQLite when reading data written by another process
        settings = get_settings()
        database_url = normalize_database_url(settings.database_url)
        fresh_engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False} if "sqlite" in database_url else {},
            echo=settings.debug,
        )
        FreshSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
        return FreshSessionLocal()
    
    SessionLocal = get_session_factory()
    return SessionLocal()
