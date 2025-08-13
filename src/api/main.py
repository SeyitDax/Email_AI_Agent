"""
FastAPI Main Application - BOILERPLATE
Main FastAPI application with middleware and route configuration
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.config import settings
from src.core.exceptions import EmailAgentException
from src.core.models import ErrorResponse
from src.api.routes import emails, health


class TimingMiddleware(BaseHTTPMiddleware):
    """Middleware to track request processing time"""
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        response = await call_next(request)
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        return response


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware for request/response logging"""
    
    def __init__(self, app, logger):
        super().__init__(app)
        self.logger = logger
    
    async def dispatch(self, request: Request, call_next):
        # Log request
        self.logger.info(
            f"Request: {request.method} {request.url.path}",
            extra={
                "method": request.method,
                "url": str(request.url),
                "headers": dict(request.headers),
            }
        )
        
        # Process request
        response = await call_next(request)
        
        # Log response
        self.logger.info(
            f"Response: {response.status_code}",
            extra={
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        )
        
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logging.info("🚀 Email AI Agent starting up...")
    
    # TODO: Initialize database connections
    # TODO: Initialize Redis connections
    # TODO: Load ML models/classifiers
    # TODO: Setup background tasks
    
    logging.info("✅ Email AI Agent startup complete")
    
    yield
    
    # Shutdown
    logging.info("🛑 Email AI Agent shutting down...")
    
    # TODO: Close database connections
    # TODO: Close Redis connections
    # TODO: Cleanup resources
    
    logging.info("✅ Email AI Agent shutdown complete")


def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger(__name__)
    
    # Create FastAPI app
    app = FastAPI(
        title=settings.app_name,
        description="AI-powered email processing and response generation system",
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
        docs_url="/docs" if settings.debug else None,
        redoc_url="/redoc" if settings.debug else None,
    )
    
    # Add security middleware
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.allowed_hosts
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Add custom middleware
    app.add_middleware(TimingMiddleware)
    app.add_middleware(LoggingMiddleware, logger=logger)
    
    # Exception handlers
    @app.exception_handler(EmailAgentException)
    async def email_agent_exception_handler(request: Request, exc: EmailAgentException):
        """Handle custom Email Agent exceptions"""
        return JSONResponse(
            status_code=400,
            content=ErrorResponse(
                error=type(exc).__name__,
                message=exc.message,
                details=exc.details,
            ).dict(),
        )
    
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """Handle HTTP exceptions"""
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(
                error="HTTPException",
                message=exc.detail,
                details={"status_code": exc.status_code},
            ).dict(),
        )
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle unexpected exceptions"""
        logger.error(f"Unexpected error: {str(exc)}", exc_info=True)
        
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="InternalServerError",
                message="An unexpected error occurred",
                details={"exception": str(exc)} if settings.debug else {},
            ).dict(),
        )
    
    # Include routers
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(emails.router, prefix="/api/v1/emails", tags=["emails"])
    
    return app


# Create the app instance
app = create_app()


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Email AI Agent API",
        "version": settings.app_version,
        "status": "running",
        "docs_url": "/docs" if settings.debug else "disabled",
    }