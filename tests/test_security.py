"""Test suite for CloakMap security features."""

import base64
import os
import tempfile
from pathlib import Path

import pytest

from cloakpivot.core.security import (
    SecurityConfig,
    KeyManager,
    EnvironmentKeyManager,
    FileKeyManager,
    CompositeKeyManager,
    CryptoUtils,
    SecurityValidator,
    SecurityMetadata,
    create_default_key_manager,
    DEFAULT_HMAC_ALGORITHM,
    DEFAULT_PBKDF2_ITERATIONS,
    DEFAULT_SALT_LENGTH
)
from cloakpivot.core.cloakmap import CloakMap, validate_cloakmap_integrity
from cloakpivot.core.anchors import AnchorEntry


class TestSecurityConfig:
    """Test security configuration."""
    
    def test_default_config(self):
        """Test default security configuration."""
        config = SecurityConfig()
        
        assert config.hmac_algorithm == DEFAULT_HMAC_ALGORITHM
        assert config.pbkdf2_iterations == DEFAULT_PBKDF2_ITERATIONS
        assert config.salt_length == DEFAULT_SALT_LENGTH
        assert config.key_derivation_enabled is True
        assert config.constant_time_verification is True
    
    def test_custom_config(self):
        """Test custom security configuration."""
        config = SecurityConfig(
            hmac_algorithm='sha512',
            pbkdf2_iterations=200000,
            salt_length=64,
            key_derivation_enabled=False,
            constant_time_verification=False
        )
        
        assert config.hmac_algorithm == 'sha512'
        assert config.pbkdf2_iterations == 200000
        assert config.salt_length == 64
        assert config.key_derivation_enabled is False
        assert config.constant_time_verification is False
    
    def test_invalid_config(self):
        """Test invalid configuration parameters."""
        with pytest.raises(ValueError, match="Unsupported HMAC algorithm"):
            SecurityConfig(hmac_algorithm='md5')
        
        with pytest.raises(ValueError, match="PBKDF2 iterations must be at least"):
            SecurityConfig(pbkdf2_iterations=1000)
        
        with pytest.raises(ValueError, match="Salt length must be at least"):
            SecurityConfig(salt_length=8)


class TestKeyManagers:
    """Test key management implementations."""
    
    def test_environment_key_manager(self):
        """Test environment variable key manager."""
        # Setup environment
        os.environ['CLOAKPIVOT_KEY_TEST'] = 'test-secret-key'
        os.environ['CLOAKPIVOT_KEY_HEX'] = 'hex:48656c6c6f'  # "Hello" in hex
        os.environ['CLOAKPIVOT_KEY_B64'] = 'b64:SGVsbG8='     # "Hello" in base64
        os.environ['CLOAKPIVOT_KEY_VERSIONED_V1'] = 'version-1-key'
        
        try:
            manager = EnvironmentKeyManager()
            
            # Test basic key retrieval
            key = manager.get_key('test')
            assert key == b'test-secret-key'
            
            # Test hex encoding
            key = manager.get_key('hex')
            assert key == b'Hello'
            
            # Test base64 encoding
            key = manager.get_key('b64')
            assert key == b'Hello'
            
            # Test versioned key
            key = manager.get_key('versioned', version='1')
            assert key == b'version-1-key'
            
            # Test key listing
            keys = manager.list_keys()
            assert 'test' in keys
            assert 'hex' in keys
            assert 'b64' in keys
            assert 'versioned' in keys
            
            # Test missing key
            with pytest.raises(KeyError):
                manager.get_key('missing')
                
        finally:
            # Cleanup
            for key in ['CLOAKPIVOT_KEY_TEST', 'CLOAKPIVOT_KEY_HEX', 
                       'CLOAKPIVOT_KEY_B64', 'CLOAKPIVOT_KEY_VERSIONED_V1']:
                os.environ.pop(key, None)
    
    def test_file_key_manager(self):
        """Test file-based key manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            key_dir = Path(temp_dir)
            
            # Create test key files
            (key_dir / "test.key").write_text("file-secret-key")
            (key_dir / "hex.key").write_text("hex:48656c6c6f")
            (key_dir / "b64.key").write_text("b64:SGVsbG8=")
            (key_dir / "versioned.v1.key").write_text("version-1-key")
            
            manager = FileKeyManager(key_dir)
            
            # Test key retrieval
            key = manager.get_key('test')
            assert key == b'file-secret-key'
            
            # Test hex encoding
            key = manager.get_key('hex')
            assert key == b'Hello'
            
            # Test versioned key
            key = manager.get_key('versioned', version='1')
            assert key == b'version-1-key'
            
            # Test key listing
            keys = manager.list_keys()
            assert 'test' in keys
            assert 'hex' in keys
            assert 'versioned' in keys
            
            # Test missing key
            with pytest.raises(KeyError):
                manager.get_key('missing')
    
    def test_composite_key_manager(self):
        """Test composite key manager."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Setup environment manager
            os.environ['CLOAKPIVOT_KEY_ENV_ONLY'] = 'env-key'
            
            # Setup file manager
            key_dir = Path(temp_dir)
            (key_dir / "file_only.key").write_text("file-key")
            (key_dir / "both.key").write_text("file-version")
            os.environ['CLOAKPIVOT_KEY_BOTH'] = 'env-version'
            
            try:
                env_manager = EnvironmentKeyManager()
                file_manager = FileKeyManager(key_dir)
                composite = CompositeKeyManager([env_manager, file_manager])
                
                # Test env-only key
                key = composite.get_key('env_only')
                assert key == b'env-key'
                
                # Test file-only key
                key = composite.get_key('file_only')
                assert key == b'file-key'
                
                # Test priority (env first)
                key = composite.get_key('both')
                assert key == b'env-version'
                
                # Test key listing from all sources
                keys = composite.list_keys()
                assert 'env_only' in keys
                assert 'file_only' in keys
                assert 'both' in keys
                
            finally:
                os.environ.pop('CLOAKPIVOT_KEY_ENV_ONLY', None)
                os.environ.pop('CLOAKPIVOT_KEY_BOTH', None)


class TestCryptoUtils:
    """Test cryptographic utilities."""
    
    def test_salt_generation(self):
        """Test salt generation."""
        salt1 = CryptoUtils.generate_salt()
        salt2 = CryptoUtils.generate_salt()
        
        assert len(salt1) == DEFAULT_SALT_LENGTH
        assert len(salt2) == DEFAULT_SALT_LENGTH
        assert salt1 != salt2  # Should be different
    
    def test_salted_checksum(self):
        """Test salted checksum computation and verification."""
        config = SecurityConfig()
        data = "sensitive-data"
        salt = CryptoUtils.generate_salt()
        
        # Compute checksum
        checksum = CryptoUtils.compute_salted_checksum(data, salt, config)
        assert isinstance(checksum, str)
        assert len(checksum) == 64  # SHA-256 hex length
        
        # Verify checksum
        is_valid = CryptoUtils.verify_salted_checksum(data, salt, checksum, config)
        assert is_valid
        
        # Test with wrong data
        is_valid = CryptoUtils.verify_salted_checksum("wrong-data", salt, checksum, config)
        assert not is_valid
        
        # Test with wrong salt
        wrong_salt = CryptoUtils.generate_salt()
        is_valid = CryptoUtils.verify_salted_checksum(data, wrong_salt, checksum, config)
        assert not is_valid
    
    def test_hmac_operations(self):
        """Test HMAC computation and verification."""
        data = b"test-data"
        key = b"test-key"
        
        # Compute HMAC
        mac = CryptoUtils.compute_hmac(data, key)
        assert isinstance(mac, str)
        assert len(mac) == 64  # SHA-256 hex length
        
        # Verify HMAC
        is_valid = CryptoUtils.verify_hmac(data, key, mac)
        assert is_valid
        
        # Test with wrong data
        is_valid = CryptoUtils.verify_hmac(b"wrong-data", key, mac)
        assert not is_valid
        
        # Test with wrong key
        is_valid = CryptoUtils.verify_hmac(data, b"wrong-key", mac)
        assert not is_valid
        
        # Test different algorithms
        mac_512 = CryptoUtils.compute_hmac(data, key, 'sha512')
        assert len(mac_512) == 128  # SHA-512 hex length
        
        is_valid = CryptoUtils.verify_hmac(data, key, mac_512, 'sha512')
        assert is_valid


class TestSecureAnchors:
    """Test secure anchor implementations."""
    
    def test_anchor_with_salted_checksum(self):
        """Test anchor creation with salted checksums."""
        config = SecurityConfig()
        
        anchor = AnchorEntry.create_from_detection(
            node_id="test_node",
            start=0,
            end=10,
            entity_type="TEST_TYPE",
            confidence=0.9,
            original_text="secret-data",
            masked_value="[MASKED]",
            strategy_used="test",
            config=config
        )
        
        # Check that salt and checksum are present
        assert anchor.checksum_salt
        assert anchor.original_checksum
        assert len(anchor.original_checksum) == 64
        
        # Decode salt
        salt = base64.b64decode(anchor.checksum_salt)
        assert len(salt) == config.salt_length
        
        # Verify original text
        assert anchor.verify_original_text("secret-data", config)
        assert not anchor.verify_original_text("wrong-data", config)
    
    def test_anchor_backward_compatibility(self):
        """Test anchor backward compatibility with old format."""
        # Create old-format anchor data (without checksum_salt)
        old_data = {
            "node_id": "test_node",
            "start": 0,
            "end": 10,
            "entity_type": "TEST_TYPE",
            "confidence": 0.9,
            "masked_value": "[MASKED]",
            "replacement_id": "repl_123",
            "original_checksum": "a" * 64,  # Valid hex checksum
            "strategy_used": "test"
        }
        
        # Should create anchor with default salt
        anchor = AnchorEntry.from_dict(old_data)
        assert anchor.checksum_salt  # Should have default salt


class TestCloakMapSecurity:
    """Test CloakMap security features."""
    
    def test_enhanced_signing(self):
        """Test enhanced CloakMap signing with key manager."""
        # Setup key manager
        os.environ['CLOAKPIVOT_KEY_TEST'] = 'test-signing-key'
        
        try:
            key_manager = EnvironmentKeyManager()
            config = SecurityConfig()
            
            # Create test CloakMap
            anchors = [
                AnchorEntry.create_from_detection(
                    node_id="test",
                    start=0,
                    end=5,
                    entity_type="TEST",
                    confidence=0.9,
                    original_text="hello",
                    masked_value="[TEST]",
                    strategy_used="test",
                    config=config
                )
            ]
            
            cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="a" * 64,
                anchors=anchors
            )
            
            # Sign with key manager
            signed_map = cloakmap.with_signature(
                key_manager=key_manager,
                key_id="test",
                config=config
            )
            
            assert signed_map.is_signed
            assert signed_map.signature
            assert signed_map.crypto
            assert signed_map.crypto['key_id'] == 'test'
            assert signed_map.crypto['signature_algorithm'] == config.hmac_algorithm
            
            # Verify signature
            is_valid = signed_map.verify_signature(key_manager, config=config)
            assert is_valid
            
            # Test with wrong key
            os.environ['CLOAKPIVOT_KEY_TEST'] = 'wrong-key'
            is_valid = signed_map.verify_signature(key_manager, config=config)
            assert not is_valid
            
        finally:
            os.environ.pop('CLOAKPIVOT_KEY_TEST', None)
    
    def test_security_validator(self):
        """Test comprehensive security validator."""
        os.environ['CLOAKPIVOT_KEY_DEFAULT'] = 'test-key'
        
        try:
            key_manager = EnvironmentKeyManager()
            config = SecurityConfig()
            
            # Create test CloakMap with security features
            anchors = [
                AnchorEntry.create_from_detection(
                    node_id="test",
                    start=0,
                    end=5,
                    entity_type="TEST",
                    confidence=0.9,
                    original_text="hello",
                    masked_value="[TEST]",
                    strategy_used="test",
                    config=config
                )
            ]
            
            cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="a" * 64,
                anchors=anchors
            ).with_signature(key_manager=key_manager, config=config)
            
            # Validate with security validator
            validator = SecurityValidator(config, key_manager)
            results = validator.validate_cloakmap(cloakmap)
            
            assert results["valid"]
            assert results["security_level"] in ["medium", "high"]
            assert results["checks"]["structure"]
            assert results["checks"]["anchors"]
            assert results["checks"]["signature"]
            assert results["performance"]["validation_time_ms"] > 0
            
        finally:
            os.environ.pop('CLOAKPIVOT_KEY_DEFAULT', None)
    
    def test_integrity_validation_enhanced(self):
        """Test enhanced integrity validation function."""
        os.environ['CLOAKPIVOT_KEY_DEFAULT'] = 'test-key'
        
        try:
            key_manager = EnvironmentKeyManager()
            config = SecurityConfig()
            
            anchors = [
                AnchorEntry.create_from_detection(
                    node_id="test",
                    start=0,
                    end=5,
                    entity_type="TEST",
                    confidence=0.9,
                    original_text="hello",
                    masked_value="[TEST]",
                    strategy_used="test",
                    config=config
                )
            ]
            
            cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="a" * 64,
                anchors=anchors
            ).with_signature(key_manager=key_manager, config=config)
            
            # Test enhanced validation
            results = validate_cloakmap_integrity(
                cloakmap, key_manager=key_manager, config=config
            )
            
            assert results["valid"]
            assert "security_level" in results
            assert "performance" in results
            
        finally:
            os.environ.pop('CLOAKPIVOT_KEY_DEFAULT', None)


class TestSecurityMetadata:
    """Test security metadata handling."""
    
    def test_security_metadata_serialization(self):
        """Test security metadata to/from dict conversion."""
        salt = CryptoUtils.generate_salt()
        
        metadata = SecurityMetadata(
            algorithm="sha256",
            key_id="test-key",
            key_version="v1",
            salt=salt,
            iterations=100000
        )
        
        # Test to_dict
        data = metadata.to_dict()
        assert data["algorithm"] == "sha256"
        assert data["key_id"] == "test-key"
        assert data["key_version"] == "v1"
        assert data["iterations"] == 100000
        assert "salt" in data
        
        # Test from_dict
        restored = SecurityMetadata.from_dict(data)
        assert restored.algorithm == metadata.algorithm
        assert restored.key_id == metadata.key_id
        assert restored.key_version == metadata.key_version
        assert restored.salt == metadata.salt
        assert restored.iterations == metadata.iterations


class TestDefaultKeyManager:
    """Test default key manager creation."""
    
    def test_default_key_manager_creation(self):
        """Test creation of default key manager."""
        manager = create_default_key_manager()
        assert isinstance(manager, CompositeKeyManager)
        assert len(manager.managers) >= 1  # At least environment manager
        
        # Test that it can list keys (even if empty)
        keys = manager.list_keys()
        assert isinstance(keys, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])