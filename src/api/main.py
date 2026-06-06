"""FastAPI application entry point."""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .database.session import init_db
from .routes import projects, upload, process, segments, resources, download, status, frames, translate
from .routes import settings as settings_routes


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler for startup/shutdown events."""
    # Startup
    app_settings = get_settings()
    app_settings.projects_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    yield
    # Shutdown (cleanup if needed)


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app_settings = get_settings()
    
    app = FastAPI(
        title=app_settings.api_title,
        version=app_settings.api_version,
        lifespan=lifespan,
        docs_url=f"{app_settings.api_prefix}/docs",
        redoc_url=f"{app_settings.api_prefix}/redoc",
        openapi_url=f"{app_settings.api_prefix}/openapi.json",
    )
    
    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include routers
    app.include_router(translate.router, prefix=app_settings.api_prefix)
    app.include_router(projects.router, prefix=app_settings.api_prefix)
    app.include_router(upload.router, prefix=app_settings.api_prefix)
    app.include_router(process.router, prefix=app_settings.api_prefix)
    app.include_router(segments.router, prefix=app_settings.api_prefix)
    app.include_router(resources.router, prefix=app_settings.api_prefix)
    app.include_router(download.router, prefix=app_settings.api_prefix)
    app.include_router(status.router, prefix=app_settings.api_prefix)
    app.include_router(settings_routes.router, prefix=app_settings.api_prefix)
    app.include_router(frames.router, prefix=app_settings.api_prefix)
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        return {"status": "healthy", "version": app_settings.api_version}
    
    return app


app = create_app()
