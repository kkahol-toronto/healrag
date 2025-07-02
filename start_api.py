#!/usr/bin/env python3
"""
HEALRAG API Startup Script
==========================

Simple script to start the HEALRAG FastAPI application with customizable settings.
"""

import os
import uvicorn
from pathlib import Path

def main():
    """Start the HEALRAG API server."""
    
    # Load environment variables if .env exists
    try:
        from dotenv import load_dotenv
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv()
            print("✅ Loaded environment variables from .env file")
    except ImportError:
        print("⚠️  python-dotenv not available, using system environment variables")
    
    # Server configuration
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "true").lower() == "true"
    workers = int(os.getenv("WORKERS", "1"))
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("🚀 Starting HEALRAG API Server")
    print("=" * 50)
    print(f"📍 Host: {host}")
    print(f"🔌 Port: {port}")
    print(f"🔄 Reload: {reload}")
    print(f"👥 Workers: {workers}")
    print(f"📝 Log Level: {log_level}")
    print(f"📚 API Docs: http://{host}:{port}/docs")
    print(f"📋 Health Check: http://{host}:{port}/health")
    print("=" * 50)
    
    # Start server
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        workers=workers if not reload else 1,  # Workers only work in non-reload mode
        log_level=log_level,
        access_log=True
    )

if __name__ == "__main__":
    main() 