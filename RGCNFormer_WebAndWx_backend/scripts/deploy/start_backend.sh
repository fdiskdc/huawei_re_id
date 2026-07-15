#!/bin/bash

# ============================================================================
# mRModN Backend Startup Script
# ============================================================================
# This script starts both Gunicorn (Flask) and Celery worker services
# using uv for Python environment management.
# ============================================================================

# Set environment variables to mitigate PyTorch/Celery multiprocessing deadlocks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Ensure dependencies are synced (skip in production if already deployed)
if [ "$UV_NO_SYNC" != "1" ]; then
    echo "Syncing dependencies with uv..."
    uv sync --locked
fi

# Service Configuration
HOST=0.0.0.0
PORT=8000

# Automatically determine CPU core count
if [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
else
    echo "Could not read /proc/cpuinfo. Defaulting to 4 cores."
    CPU_CORES=4
fi
echo "Detected $CPU_CORES CPU cores."

GUNICORN_WORKERS=1
CELERY_CONCURRENCY=1

# Change to project root directory
cd "$(dirname "$0")/../.."

echo "=========================================="
echo "  mRModN Backend Startup"
echo "=========================================="
echo "Working Directory:  $(pwd)"
echo "Gunicorn Workers:   $GUNICORN_WORKERS"
echo "Celery Concurrency: $CELERY_CONCURRENCY"
echo "=========================================="

# Start Celery Worker (Background)
echo "Starting Celery worker with $CELERY_CONCURRENCY concurrent processes..."
uv run celery -A celery_worker.celery_app worker \
    --concurrency=$CELERY_CONCURRENCY \
    --loglevel=info \
    --pidfile=celery.pid \
    --logfile=celery.log &

CELERY_PID=$!
sleep 2

# Check if Celery started successfully
if ps -p $CELERY_PID > /dev/null; then
    echo "Celery worker started successfully (PID: $CELERY_PID)"
else
    echo "Failed to start Celery worker"
    exit 1
fi

# Start Gunicorn Server (Foreground)
echo "Starting Gunicorn server with $GUNICORN_WORKERS workers on $HOST:$PORT..."
uv run gunicorn -w $GUNICORN_WORKERS \
    -b $HOST:$PORT \
    --timeout 120 \
    --access-logfile gunicorn_access.log \
    --error-logfile gunicorn_error.log \
    wsgi:app

# Cleanup (When Gunicorn exits)
echo ""
echo "Gunicorn stopped, cleaning up Celery worker..."
if ps -p $CELERY_PID > /dev/null; then
    kill $CELERY_PID
    echo "Celery worker stopped"
else
    echo "Celery worker not running"
fi

echo "Backend services stopped completely"
