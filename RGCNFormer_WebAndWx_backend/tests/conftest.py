"""
Shared pytest fixtures for mRModN_WebAndWx_backend tests.

Provides project root path and common test configuration.
"""
import sys
import os
import pytest

# Add project root to sys.path so we can import project modules
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


@pytest.fixture
def project_root():
    """Return the absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture
def config():
    """Return the global Config instance for testing."""
    from mrmodn_backend.core.config import config
    return config
