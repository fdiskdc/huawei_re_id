#!/usr/bin/env python3
"""
Integration tests for Redis caching and SHA256 hash-based task IDs.
"""
import pytest
import requests
import hashlib

BASE_URL = "http://localhost:8000"
TEST_SEQUENCE = "ACGUACGUACGUACGUACGU"  # 20 nucleotides for quick testing


def generate_sha256_hash(sequence: str) -> str:
    """Generate SHA256 hash of RNA sequence."""
    return hashlib.sha256(sequence.encode('utf-8')).hexdigest()


@pytest.mark.integration
class TestCacheApi:
    """Tests for Redis caching and SHA256-based task IDs."""

    def test_submit_task_returns_sha256_job_id(self):
        """First request should return a jobId matching SHA256 of the sequence."""
        expected_job_id = generate_sha256_hash(TEST_SEQUENCE)

        response = requests.post(
            f"{BASE_URL}/mrmodn/api/v1/submit-task",
            json={"rnaSequence": TEST_SEQUENCE, "userId": "test_user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("jobId") == expected_job_id

    def test_cached_request_returns_same_job_id(self):
        """Second request for the same sequence should return the same jobId."""
        expected_job_id = generate_sha256_hash(TEST_SEQUENCE)

        # First request (warm the cache)
        requests.post(
            f"{BASE_URL}/mrmodn/api/v1/submit-task",
            json={"rnaSequence": TEST_SEQUENCE, "userId": "test_user"},
        )

        # Second request (should hit cache)
        response = requests.post(
            f"{BASE_URL}/mrmodn/api/v1/submit-task",
            json={"rnaSequence": TEST_SEQUENCE, "userId": "test_user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("jobId") == expected_job_id

    def test_different_sequence_gets_different_job_id(self):
        """A different sequence should produce a different jobId."""
        different_sequence = "GCUAGCUAGCUAGCUAGCUA"
        expected_job_id = generate_sha256_hash(different_sequence)

        response = requests.post(
            f"{BASE_URL}/mrmodn/api/v1/submit-task",
            json={"rnaSequence": different_sequence, "userId": "test_user"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data.get("jobId") == expected_job_id
        assert data.get("jobId") != generate_sha256_hash(TEST_SEQUENCE)

    @pytest.fixture(autouse=True)
    def _require_server(self):
        """Skip tests if the backend server is not reachable."""
        try:
            requests.get(f"{BASE_URL}/mrmodn/api/health", timeout=2)
        except requests.exceptions.ConnectionError:
            pytest.skip("Backend server not running on {BASE_URL}")
