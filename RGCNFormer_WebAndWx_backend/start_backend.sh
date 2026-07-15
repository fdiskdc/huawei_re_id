#!/bin/bash

# ============================================================================
# DCPRES Backend Startup Script
# ============================================================================
# This script starts both Gunicorn (Flask) and Celery worker services
# with proper conda environment activation and auto-configured concurrency.
# ============================================================================

# ============================================================================
# Conda Environment Configuration
# ============================================================================
CONDA_ENV_NAME="learn"        # Change to your conda environment name
CONDA_BASE_PATH="$HOME/miniconda3"  # Conda installation path

# Activate conda environment
source "$CONDA_BASE_PATH/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    echo "❌ Error: Failed to activate conda environment '$CONDA_ENV_NAME'"
    exit 1
fi

# Set environment variables to mitigate PyTorch/Celery multiprocessing deadlocks
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

echo "✓ Conda environment activated: $CONDA_ENV_NAME"

# ============================================================================
# Service Configuration (Auto-detecting CPU cores)
# ============================================================================
HOST=0.0.0.0
PORT=8000

# Automatically determine CPU core count
if [ -f /proc/cpuinfo ]; then
    CPU_CORES=$(grep -c ^processor /proc/cpuinfo)
else
    echo "⚠️ Could not read /proc/cpuinfo. Defaulting to 4 cores."
    CPU_CORES=4
fi
echo "✓ Detected $CPU_CORES CPU cores."

# Recommended worker counts
# Gunicorn: (2 * cores) + 1 for handling I/O bound requests
# Celery: number of cores for CPU-bound tasks
# GUNICORN_WORKERS=$((2 * CPU_CORES + 1))
# CELERY_CONCURRENCY=$CPU_CORES
GUNICORN_WORKERS=1
CELERY_CONCURRENCY=1
# Change to backend directory
cd "$(dirname "$0")"

echo "=========================================="
echo "  DCPRES Backend Startup"
echo "=========================================="
echo "Conda Environment:  $CONDA_ENV_NAME"
echo "Working Directory:  $(pwd)"
echo "Gunicorn Workers:   $GUNICORN_WORKERS"
echo "Celery Concurrency: $CELERY_CONCURRENCY"
echo "=========================================="

# ============================================================================
# Start Celery Worker (Background)
# ============================================================================
echo "🚀 Starting Celery worker with $CELERY_CONCURRENCY concurrent processes..."
celery -A tasks.celery_app worker \
    --concurrency=$CELERY_CONCURRENCY \
    --loglevel=info \
    --pidfile=celery.pid \
    --logfile=celery.log &

CELERY_PID=$!
sleep 2

# Check if Celery started successfully
if ps -p $CELERY_PID > /dev/null; then
    echo "✓ Celery worker started successfully (PID: $CELERY_PID)"
else
    echo "❌ Failed to start Celery worker"
    exit 1
fi

# ============================================================================
# Start Gunicorn Server (Foreground)
# ============================================================================
echo "🚀 Starting Gunicorn server with $GUNICORN_WORKERS workers on $HOST:$PORT..."
gunicorn -w $GUNICORN_WORKERS \
    -b $HOST:$PORT \
    --timeout 120 \
    --access-logfile gunicorn_access.log \
    --error-logfile gunicorn_error.log \
    wsgi:app

# ============================================================================
# Cleanup (When Gunicorn exits)
# ============================================================================
echo ""
echo "Gunicorn stopped, cleaning up Celery worker..."
if ps -p $CELERY_PID > /dev/null; then
    kill $CELERY_PID
    echo "✓ Celery worker stopped"
else
    echo "⚠ Celery worker not running"
fi

echo "Backend services stopped completely"
