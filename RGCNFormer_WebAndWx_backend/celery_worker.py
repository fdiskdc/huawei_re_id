"""
Celery worker entry point.

This thin entry point creates and exports the Celery application
for use with the celery command-line tool.

Usage:
    celery -A celery_worker.celery_app worker --concurrency=1 --loglevel=info
"""
from tasks import celery_app

if __name__ == "__main__":
    celery_app.start()
