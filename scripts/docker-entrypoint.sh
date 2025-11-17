#!/bin/bash
# Docker entrypoint script that handles PORT environment variable
# and ensures proper signal handling

set -e

# Get port from environment variable or use default
PORT=${PORT:-8000}

# Use exec to replace shell process with uvicorn (for proper signal handling)
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT"

