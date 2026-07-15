#!/bin/bash
###
 # @Author: Chao Deng && chaodeng987@outlook.com
 # @Date: 2026-01-22 21:21:08
 # @LastEditors: Chao Deng && chaodeng987@outlook.com
 # @LastEditTime: 2026-01-22 21:21:15
 # @FilePath: /rgcnformer_mobile_web/backend/stop_backend.sh
 # @Description: 
 # 那只是一场游戏一场梦
 #  
 # https://orcid.org/0009-0009-8520-1656
 # DOI: 10.3390/app15158626
 # DOI: 10.3390/rs17142354
 # Copyright (c) 2026 by ${Chao Deng}, All Rights Reserved. 
### 

# ============================================================================
# RGCNFormer Backend Stop Script
# ============================================================================
# This script stops both Gunicorn and Celery worker services.
# ============================================================================

# Change to backend directory
cd "$(dirname "$0")"

echo "=========================================="
echo "  Stopping DCPRES Backend"
echo "=========================================="

# ============================================================================
# Stop Celery Worker
# ============================================================================
if [ -f celery.pid ]; then
    PID=$(cat celery.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✓ Celery worker stopped (PID: $PID)"
    else
        echo "⚠ Celery worker process not found"
    fi
    rm -f celery.pid
else
    echo "⚠ Celery PID file not found"
    echo "   Attempting to stop by process name..."
    pkill -f "celery.*worker" && echo "✓ Celery worker stopped by process name" || echo "⚠ No Celery worker found"
fi

# ============================================================================
# Stop Gunicorn
# ============================================================================
echo ""
echo "Stopping Gunicorn server..."
pkill -f "gunicorn.*wsgi:app" && echo "✓ Gunicorn stopped" || echo "⚠ No Gunicorn process found"

# ============================================================================
# Cleanup
# ============================================================================
echo ""
echo "=========================================="
echo "Backend services stopped completely"
echo "=========================================="

# Display remaining processes (if any)
REMAINING=$(ps aux | grep -E "(gunicorn|celery)" | grep -v grep | grep -v "stop_backend")
if [ -n "$REMAINING" ]; then
    echo ""
    echo "⚠ Warning: Some processes may still be running:"
    echo "$REMAINING"
fi
