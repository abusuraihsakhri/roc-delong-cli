"""
Tests for CLI batch command path traversal protection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from cli import _safe_resolve_path


class TestSafeResolvePath:
    """Test path traversal protection."""

    def test_valid_path_within_cwd(self, tmp_path, monkeypatch):
        """Paths within the working directory should be accepted."""
        monkeypatch.chdir(tmp_path)
        test_file = tmp_path / "test.csv"
        test_file.write_text("data")
        result = _safe_resolve_path("test.csv", must_exist=True)
        assert result == test_file.resolve()

    def test_path_traversal_rejected(self, tmp_path, monkeypatch):
        """Paths outside the working directory should be rejected."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside the working directory"):
            _safe_resolve_path("../etc/passwd")

    def test_absolute_path_outside_cwd_rejected(self, tmp_path, monkeypatch):
        """Absolute paths outside the working directory should be rejected."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="outside the working directory"):
            _safe_resolve_path("/etc/passwd")

    def test_nonexistent_file_raises_when_required(self, tmp_path, monkeypatch):
        """Non-existent files should raise FileNotFoundError when must_exist=True."""
        monkeypatch.chdir(tmp_path)
        with pytest.raises(FileNotFoundError):
            _safe_resolve_path("nonexistent.csv", must_exist=True)

    def test_nonexistent_file_allowed_when_not_required(self, tmp_path, monkeypatch):
        """Non-existent files should be allowed when must_exist=False."""
        monkeypatch.chdir(tmp_path)
        result = _safe_resolve_path("output.csv", must_exist=False)
        assert result == (tmp_path / "output.csv").resolve()
