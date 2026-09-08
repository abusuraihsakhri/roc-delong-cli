"""
Pytest configuration for roc-delong-cli tests.
Sets up required environment variables for testing.
"""
import os


def pytest_configure(config):
    """Set required environment variables before tests run."""
    os.environ.setdefault("AUDIT_SECRET_KEY", "test-secret-key-for-pytest-only-1234567890")
