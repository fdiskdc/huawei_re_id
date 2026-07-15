"""
Tests for Celery tasks - Verify task definitions and names.

Tests that task functions exist and have correct Celery task names.
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class TestCeleryTaskDefinitions:
    """Test that Celery tasks are properly defined."""

    def test_run_prediction_task_exists(self):
        import tasks
        assert hasattr(tasks, 'run_prediction_task')

    def test_process_sequence_in_batch_exists(self):
        import tasks
        assert hasattr(tasks, 'process_sequence_in_batch')

    def test_run_prediction_task_is_callable(self):
        import tasks
        assert callable(tasks.run_prediction_task)

    def test_process_sequence_in_batch_is_callable(self):
        import tasks
        assert callable(tasks.process_sequence_in_batch)

    def test_run_prediction_task_name(self):
        import tasks
        task = tasks.run_prediction_task
        assert task.name == 'tasks.run_prediction_task'

    def test_process_sequence_in_batch_name(self):
        import tasks
        task = tasks.process_sequence_in_batch
        assert task.name == 'tasks.process_sequence_in_batch'

    def test_celery_app_exists(self):
        import tasks
        assert hasattr(tasks, 'celery_app')

    def test_celery_app_name(self):
        import tasks
        assert tasks.celery_app.main == 'rna_prediction_tasks'
