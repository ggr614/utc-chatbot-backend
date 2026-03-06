"""
FastAPI application for the UTC Chatbot Backend.

Provides:
- Admin endpoints for analytics and management
- Chat endpoint for RAG queries (via OpenWebUI filter)
- Admin dashboard UI
"""
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from api.routes import admin, chat, openai_compat
from core.config import get_settings

# Path to admin dashboard
DASHBOARD_DIR = Path(__file__).parent.parent / "admin-dashboard"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    logger.info("Starting UTC Chatbot API...")
    settings = get_settings()
    logger.info(f"API configured on {settings.API_HOST}:{settings.API_PORT}")
    yield
    logger.info("Shutting down UTC Chatbot API...")


app = FastAPI(
    title="UTC Chatbot Backend API",
    description="RAG backend and admin API for UTC Helpdesk Chatbot",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware - allow Open-webui and local dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8080",
        "*",  # For development - restrict in production
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(admin.router)
app.include_router(chat.router)
app.include_router(openai_compat.router)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "UTC Chatbot Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "admin": "/admin",
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/dashboard")
async def dashboard():
    """Serve the admin dashboard."""
    return FileResponse(DASHBOARD_DIR / "index.html")


if __name__ == "__main__":
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level=settings.API_LOG_LEVEL,
    )
