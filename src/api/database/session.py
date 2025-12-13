"""Database session management."""

from typing import Generator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from ..config import get_settings
from .models import Base


# Engine and session factory (initialized lazily)
_engine = None
_SessionLocal = None


def get_engine():
    """Get or create the database engine."""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
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
        fresh_engine = create_engine(
            settings.database_url,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {},
            echo=settings.debug,
        )
        FreshSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=fresh_engine)
        return FreshSessionLocal()
    
    SessionLocal = get_session_factory()
    return SessionLocal()

