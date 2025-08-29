"""Test suite for CloakMap security features."""

import base64
import os
import tempfile
from pathlib import Path

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap, validate_cloakmap_integrity
from cloakpivot.core.security import (
    DEFAULT_HMAC_ALGORITHM,
    DEFAULT_PBKDF2_ITERATIONS,
    DEFAULT_SALT_LENGTH,
    CompositeKeyManager,
    CryptoUtils,
    EnvironmentKeyManager,
    FileKeyManager,
    SecurityConfig,
    SecurityMetadata,
    SecurityValidator,
    create_default_key_manager,
)


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


class TestEncryption:
    """Test AES-GCM encryption functionality."""

    def test_encryption_utils(self):
        """Test basic encryption utilities."""
        from cloakpivot.core.security import CryptoUtils

        # Test nonce generation
        nonce1 = CryptoUtils.generate_nonce()
        nonce2 = CryptoUtils.generate_nonce()
        assert len(nonce1) == 12  # Default nonce length
        assert len(nonce2) == 12
        assert nonce1 != nonce2  # Should be different

        # Test key derivation
        base_key = b"test-base-key"
        salt = CryptoUtils.generate_salt()
        config = SecurityConfig()

        derived_key = CryptoUtils.derive_encryption_key(base_key, salt, 32, config)
        assert len(derived_key) == 32

        # Same inputs should produce same key
        derived_key2 = CryptoUtils.derive_encryption_key(base_key, salt, 32, config)
        assert derived_key == derived_key2

        # Different salt should produce different key
        salt2 = CryptoUtils.generate_salt()
        derived_key3 = CryptoUtils.derive_encryption_key(base_key, salt2, 32, config)
        assert derived_key != derived_key3

    def test_aes_gcm_encryption(self):
        """Test AES-GCM encryption and decryption."""
        from cloakpivot.core.security import CryptoUtils

        # Test data
        data = b"sensitive test data to encrypt"
        key = CryptoUtils.generate_salt(32)  # 256-bit key
        associated_data = b"document-id"

        # Encrypt
        ciphertext, nonce = CryptoUtils.encrypt_data(data, key, None, associated_data)

        assert len(nonce) == 12
        assert len(ciphertext) > len(data)  # Includes auth tag
        assert ciphertext != data

        # Decrypt
        decrypted = CryptoUtils.decrypt_data(ciphertext, key, nonce, associated_data)
        assert decrypted == data

        # Test with wrong key
        wrong_key = CryptoUtils.generate_salt(32)
        with pytest.raises(Exception):  # Should raise InvalidTag or similar
            CryptoUtils.decrypt_data(ciphertext, wrong_key, nonce, associated_data)

        # Test with wrong associated data
        with pytest.raises(Exception):
            CryptoUtils.decrypt_data(ciphertext, key, nonce, b"wrong-data")

    def test_encrypted_cloakmap_serialization(self):
        """Test EncryptedCloakMap serialization."""
        from cloakpivot.core.security import EncryptedCloakMap

        encrypted_map = EncryptedCloakMap(
            version="1.0",
            doc_id="test_doc",
            doc_hash="a" * 64,
            algorithm="AES-GCM-256",
            key_id="test_key",
            key_version="v1",
            nonce="dGVzdF9ub25jZQ==",  # base64 encoded
            encrypted_anchors="ZW5jcnlwdGVkX2FuY2hvcnM=",  # base64 encoded
            encrypted_policy="ZW5jcnlwdGVkX3BvbGljeQ==",  # base64 encoded
            encrypted_metadata="ZW5jcnlwdGVkX21ldGE=",  # base64 encoded
        )

        # Test to_dict
        data = encrypted_map.to_dict()
        assert data["version"] == "1.0"
        assert data["doc_id"] == "test_doc"
        assert data["crypto"]["algorithm"] == "AES-GCM-256"
        assert data["crypto"]["key_id"] == "test_key"
        assert "encrypted_content" in data

        # Test to_json
        json_str = encrypted_map.to_json()
        assert "test_doc" in json_str
        assert "encrypted_content" in json_str

        # Test from_dict
        restored = EncryptedCloakMap.from_dict(data)
        assert restored.doc_id == encrypted_map.doc_id
        assert restored.algorithm == encrypted_map.algorithm
        assert restored.key_id == encrypted_map.key_id

        # Test from_json
        restored2 = EncryptedCloakMap.from_json(json_str)
        assert restored2.doc_id == encrypted_map.doc_id


class TestCloakMapEncryption:
    """Test CloakMap encryption integration."""

    def test_cloakmap_encryption_round_trip(self):
        """Test full CloakMap encryption and decryption round trip."""
        # Setup environment
        os.environ['CLOAKPIVOT_KEY_TEST_ENC'] = 'hex:' + '0' * 64  # 32-byte key in hex

        try:
            from cloakpivot.core.security import (
                CloakMapEncryption,
                EnvironmentKeyManager,
                SecurityConfig,
            )

            key_manager = EnvironmentKeyManager()
            config = SecurityConfig()

            # Create test CloakMap
            anchors = [
                AnchorEntry.create_from_detection(
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
            ]

            cloakmap = CloakMap.create(
                doc_id="test_document",
                doc_hash="a" * 64,
                anchors=anchors,
                metadata={"test": "metadata"}
            )

            # Encrypt
            encryption = CloakMapEncryption(key_manager, config)
            encrypted_map = encryption.encrypt_cloakmap(cloakmap, "test_enc")

            assert encrypted_map.doc_id == cloakmap.doc_id
            assert encrypted_map.doc_hash == cloakmap.doc_hash
            assert encrypted_map.key_id == "test_enc"
            assert encrypted_map.nonce  # Should have nonce
            assert encrypted_map.encrypted_anchors  # Should have encrypted data

            # Decrypt
            decrypted_map = encryption.decrypt_cloakmap(encrypted_map)

            assert decrypted_map.doc_id == cloakmap.doc_id
            assert decrypted_map.doc_hash == cloakmap.doc_hash
            assert len(decrypted_map.anchors) == len(cloakmap.anchors)
            assert decrypted_map.metadata == cloakmap.metadata

            # Verify anchor data
            original_anchor = cloakmap.anchors[0]
            decrypted_anchor = decrypted_map.anchors[0]
            assert decrypted_anchor.node_id == original_anchor.node_id
            assert decrypted_anchor.entity_type == original_anchor.entity_type
            assert decrypted_anchor.masked_value == original_anchor.masked_value

        finally:
            os.environ.pop('CLOAKPIVOT_KEY_TEST_ENC', None)

    def test_cloakmap_encryption_methods(self):
        """Test CloakMap encryption methods."""
        # Setup environment
        os.environ['CLOAKPIVOT_KEY_DEFAULT'] = 'hex:' + 'a' * 64  # 32-byte key

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
                doc_hash="b" * 64,
                anchors=anchors
            )

            # Test encrypt method
            encrypted_map = cloakmap.encrypt(key_manager)
            assert encrypted_map.doc_id == cloakmap.doc_id
            assert encrypted_map.key_id == "default"

            # Test save_encrypted and load methods
            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = Path(temp_dir) / "encrypted.json"

                # Save encrypted
                cloakmap.save_encrypted(file_path, key_manager)
                assert file_path.exists()

                # Load encrypted
                loaded_map = CloakMap.load_encrypted(file_path, key_manager)
                assert loaded_map.doc_id == cloakmap.doc_id
                assert len(loaded_map.anchors) == len(cloakmap.anchors)

                # Test auto-detection load
                auto_loaded = CloakMap.load_from_file(file_path, key_manager)
                assert auto_loaded.doc_id == cloakmap.doc_id

        finally:
            os.environ.pop('CLOAKPIVOT_KEY_DEFAULT', None)

    def test_key_rotation(self):
        """Test encryption key rotation."""
        # Setup multiple keys
        os.environ['CLOAKPIVOT_KEY_OLD'] = 'hex:' + '1' * 64
        os.environ['CLOAKPIVOT_KEY_NEW'] = 'hex:' + '2' * 64

        try:
            from cloakpivot.core.security import CloakMapEncryption, KeyRotationManager

            key_manager = EnvironmentKeyManager()
            config = SecurityConfig()
            encryption = CloakMapEncryption(key_manager, config)

            # Create test CloakMap
            anchors = [
                AnchorEntry.create_from_detection(
                    node_id="test",
                    start=0,
                    end=5,
                    entity_type="TEST",
                    confidence=0.9,
                    original_text="data",
                    masked_value="[TEST]",
                    strategy_used="test",
                    config=config
                )
            ]

            cloakmap = CloakMap.create(
                doc_id="test_doc",
                doc_hash="c" * 64,
                anchors=anchors
            )

            # Encrypt with old key
            encrypted_old = encryption.encrypt_cloakmap(cloakmap, "old")
            assert encrypted_old.key_id == "old"

            # Rotate to new key
            encrypted_new = encryption.rotate_encryption_key(encrypted_old, "new")
            assert encrypted_new.key_id == "new"
            assert encrypted_new.doc_id == encrypted_old.doc_id

            # Verify can decrypt with new key
            decrypted = encryption.decrypt_cloakmap(encrypted_new)
            assert decrypted.doc_id == cloakmap.doc_id
            assert len(decrypted.anchors) == len(cloakmap.anchors)

            # Test rotation manager
            rotation_manager = KeyRotationManager(key_manager, config)

            # Test key availability check
            availability = rotation_manager.verify_key_availability(["old", "new", "missing"])
            assert availability["old"] is True
            assert availability["new"] is True
            assert availability["missing"] is False

        finally:
            os.environ.pop('CLOAKPIVOT_KEY_OLD', None)
            os.environ.pop('CLOAKPIVOT_KEY_NEW', None)

    def test_encryption_errors(self):
        """Test encryption error handling."""
        from cloakpivot.core.security import CloakMapEncryption, EncryptedCloakMap

        key_manager = EnvironmentKeyManager()
        config = SecurityConfig()
        encryption = CloakMapEncryption(key_manager, config)

        # Create test CloakMap
        cloakmap = CloakMap.create(
            doc_id="test",
            doc_hash="d" * 64,
            anchors=[]
        )

        # Test missing key (should raise ValueError wrapping KeyError)
        with pytest.raises(ValueError, match="Failed to encrypt CloakMap"):
            encryption.encrypt_cloakmap(cloakmap, "missing_key")

        # Test decryption with wrong key
        os.environ['CLOAKPIVOT_KEY_WRONG'] = 'hex:' + '9' * 64
        try:
            encrypted_map = EncryptedCloakMap(
                version="1.0",
                doc_id="test",
                doc_hash="d" * 64,
                key_id="wrong",
                nonce="dGVzdF9ub25jZQ==",
                encrypted_anchors="aW52YWxpZA==",  # Invalid encrypted data
                encrypted_policy="aW52YWxpZA==",
                encrypted_metadata="aW52YWxpZA=="
            )

            with pytest.raises(ValueError):
                encryption.decrypt_cloakmap(encrypted_map)

        finally:
            os.environ.pop('CLOAKPIVOT_KEY_WRONG', None)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
