"""
Security-focused tests for roc-delong-cli.
Tests for path traversal protection, audit key validation, and HMAC integrity.
"""
import sys
import os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from agents.base import AuditTrail, AuditLogger, PHIGuard, SecurityException


class TestAuditKeyValidation:
    """Test that audit trail requires a proper secret key."""

    def test_audit_trail_requires_key(self):
        """AuditTrail should raise RuntimeError when no key is provided."""
        original_key = os.environ.pop("AUDIT_SECRET_KEY", None)
        try:
            with pytest.raises(RuntimeError, match="AUDIT_SECRET_KEY"):
                AuditTrail()
        finally:
            if original_key is not None:
                os.environ["AUDIT_SECRET_KEY"] = original_key

    def test_audit_trail_rejects_short_key(self):
        """AuditTrail should reject keys shorter than 16 characters."""
        with pytest.raises(ValueError, match="at least 16 characters"):
            AuditTrail(secret_key="short")

    def test_audit_trail_accepts_valid_key(self):
        """AuditTrail should accept a key of sufficient length."""
        trail = AuditTrail(secret_key="a" * 16)
        assert trail is not None

    def test_audit_trail_uses_env_var(self):
        """AuditTrail should read key from environment variable."""
        os.environ["AUDIT_SECRET_KEY"] = "b" * 20
        try:
            trail = AuditTrail()
            assert trail is not None
        finally:
            os.environ.pop("AUDIT_SECRET_KEY", None)


class TestHMACIntegrityVerification:
    """Test that HMAC integrity verification catches tampering."""

    def test_valid_trail_passes_verification(self):
        """A valid audit trail should pass integrity verification."""
        os.environ["AUDIT_SECRET_KEY"] = "test-key-for-integrity-32chars"
        try:
            trail = AuditTrail()
            trail.log("test", "tier", "EVENT", {"data": "value"})
            trail.log("test", "tier", "EVENT2", {"data": "value2"})
            assert trail.verify_integrity() is True
        finally:
            os.environ.pop("AUDIT_SECRET_KEY", None)

    def test_tampered_entry_fails_verification(self):
        """A tampered audit entry should fail integrity verification."""
        os.environ["AUDIT_SECRET_KEY"] = "test-key-for-tampering-32chars"
        try:
            trail = AuditTrail()
            trail.log("test", "tier", "EVENT", {"data": "original"})
            # Tamper with the entry
            trail.logs[0]["payload_hash"] = "tampered_hash"
            assert trail.verify_integrity() is False
        finally:
            os.environ.pop("AUDIT_SECRET_KEY", None)

    def test_broken_chain_fails_verification(self):
        """A broken hash chain should fail integrity verification."""
        os.environ["AUDIT_SECRET_KEY"] = "test-key-for-chain-break-32ch"
        try:
            trail = AuditTrail()
            trail.log("test", "tier", "EVENT1", {"data": "value1"})
            trail.log("test", "tier", "EVENT2", {"data": "value2"})
            # Break the chain by modifying prev_hash of second entry
            trail.logs[1]["prev_hash"] = "broken_chain_hash"
            assert trail.verify_integrity() is False
        finally:
            os.environ.pop("AUDIT_SECRET_KEY", None)


class TestPHIGuard:
    """Test PHI detection patterns."""

    def test_mrn_detected(self):
        with pytest.raises(SecurityException):
            PHIGuard.assert_no_phi("Patient MRN-12345678")

    def test_ssn_detected(self):
        with pytest.raises(SecurityException):
            PHIGuard.assert_no_phi("SSN: 123-45-6789")

    def test_email_detected(self):
        with pytest.raises(SecurityException):
            PHIGuard.assert_no_phi("Contact patient@hospital.com")

    def test_clean_text_passes(self):
        PHIGuard.assert_no_phi("Analytical assay specimen KEY-001 optimal")

    def test_redact_phi(self):
        result = PHIGuard.redact_phi("Patient MRN-12345678 is stable")
        assert "MRN" not in result or "REDACTED" in result
