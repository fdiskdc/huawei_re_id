"""
Compatibility shim - imports from the new package location.
This file exists for backward compatibility during the refactoring transition.
"""
from mrmodn_backend.workers.tasks import celery_app, run_prediction_task, process_sequence_in_batch
