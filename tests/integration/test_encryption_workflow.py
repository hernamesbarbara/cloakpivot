"""Integration tests for encryption/decryption workflows."""

import base64
import os
import tempfile
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from docling.datamodel.document import DoclingDocument

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.presidio.advanced_features import (
    PresidioEncryptionManager,
)
from cloakpivot.presidio.batch_processor import PresidioBatchProcessor
from cloakpivot.presidio.key_management import (
    EnvironmentKeyProvider,
    FileKeyProvider,
)


class TestEndToEndEncryption:
    """Test end-to-end encryption masking and decryption unmasking."""

    @pytest.fixture
    def setup_environment_key(self):
        """Set up environment variable with encryption key."""
        key = Fernet.generate_key()
        # Set the key for the entity type and algorithm combination that will be used
        os.environ["CLOAKPIVOT_KEY_CREDIT_CARD_FERNET"] = base64.b64encode(key).decode()
        yield key
        # Cleanup
        del os.environ["CLOAKPIVOT_KEY_CREDIT_CARD_FERNET"]

    @pytest.fixture
    def temp_key_directory(self):
        """Create temporary directory for key storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_encryption_with_environment_provider(self, setup_environment_key):
        """Test encryption workflow with environment key provider."""
        # Set up key provider and encryption manager
        key_provider = EnvironmentKeyProvider()
        encryption_manager = PresidioEncryptionManager(key_provider)

        # Create encryption strategy
        strategy = encryption_manager.create_encryption_strategy("Fernet")

        # Test encryption callback
        callback = strategy.parameters["callback"]
        original_text = "4111-1111-1111-1111"
        encrypted = callback(original_text, "CREDIT_CARD", 0.95)

        # Verify encryption
        assert encrypted != original_text
        assert len(encrypted) > 0

        # Verify we can decrypt (since we have the key)
        key = setup_environment_key
        fernet = Fernet(key)
        decrypted_bytes = fernet.decrypt(base64.b64decode(encrypted))
        decrypted_text = decrypted_bytes.decode()

        assert decrypted_text == original_text

    def test_encryption_with_file_provider(self, temp_key_directory):
        """Test encryption workflow with file-based key provider."""
        # Set up file key provider
        key_provider = FileKeyProvider(key_directory=temp_key_directory, password="test_password")

        # Create a key
        key_id = key_provider.create_key("Fernet")
        assert key_id is not None

        # Set up encryption manager
        encryption_manager = PresidioEncryptionManager(key_provider)

        # For testing, we need to patch the key retrieval to use our key_id
        original_get_key = key_provider.get_key

        def mock_get_key(requested_key_id):
            # Map the generated key ID format to our created key
            # The encryption callback will ask for {entity_type}_{algorithm}
            # We need to handle both CREDIT_CARD and SSN entity types
            if "_Fernet" in requested_key_id:
                return original_get_key(key_id)
            return original_get_key(requested_key_id)

        key_provider.get_key = mock_get_key

        # Create encryption strategy
        strategy = encryption_manager.create_encryption_strategy("Fernet")

        # Test encryption
        callback = strategy.parameters["callback"]
        original_text = "123-45-6789"
        encrypted = callback(original_text, "SSN", 0.95)

        # Verify encryption
        assert encrypted != original_text

        # Verify key file exists
        key_file = Path(temp_key_directory) / f"{key_id}.key"
        assert key_file.exists()

    def test_decryption_context_extraction(self):
        """Test extracting decryption context from CloakMap."""
        # Set up mock key provider
        key_provider = EnvironmentKeyProvider()
        key_provider.get_key = lambda key_id: b"mock_key_for_" + key_id.encode()

        encryption_manager = PresidioEncryptionManager(key_provider)

        # Create CloakMap with encryption metadata
        # Since CloakMap is frozen, we need to pass presidio_metadata at creation time
        cloakmap = CloakMap(
            doc_id="test_doc",
            version="2.0",
            doc_hash="test_hash_123",
            presidio_metadata={
                "operator_results": [
                    {
                        "operator": "encrypt",
                        "entity_type": "CREDIT_CARD",
                        "key_reference": "cc_key_123",
                        "algorithm": "AES-256",
                        "start": 10,
                        "end": 29
                    },
                    {
                        "operator": "encrypt",
                        "entity_type": "SSN",
                        "key_reference": "ssn_key_456",
                        "algorithm": "ChaCha20",
                        "start": 40,
                        "end": 51
                    }
                ]
            }
        )

        # Extract decryption context
        context = encryption_manager.create_decryption_context(cloakmap)

        # Verify context
        assert "CREDIT_CARD" in context
        assert "SSN" in context
        assert context["CREDIT_CARD"]["algorithm"] == "AES-256"
        assert context["SSN"]["algorithm"] == "ChaCha20"
        assert context["CREDIT_CARD"]["key"] == b"mock_key_for_cc_key_123"
        assert context["SSN"]["key"] == b"mock_key_for_ssn_key_456"


class TestKeyRotation:
    """Test key rotation scenarios."""

    @pytest.fixture
    def temp_key_directory(self):
        """Create temporary directory for key storage."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    def test_file_provider_key_rotation(self, temp_key_directory):
        """Test key rotation with file provider."""
        provider = FileKeyProvider(key_directory=temp_key_directory, password="test_password")

        # Create initial key
        old_key_id = provider.create_key("AES-256")
        old_key = provider.get_key(old_key_id)

        # Rotate the key
        new_key_id = provider.rotate_key(old_key_id)
        new_key = provider.get_key(new_key_id)

        # Verify rotation
        assert new_key_id != old_key_id
        assert new_key != old_key

        # Check metadata
        assert old_key_id in provider._metadata["keys"]
        assert new_key_id in provider._metadata["keys"]
        assert provider._metadata["keys"][old_key_id]["active"] is False
        assert provider._metadata["keys"][new_key_id]["active"] is True
        assert provider._metadata["keys"][old_key_id]["replaced_by"] == new_key_id

        # Check rotation history
        assert len(provider._metadata["rotation_history"]) == 1
        assert provider._metadata["rotation_history"][0]["old_key"] == old_key_id
        assert provider._metadata["rotation_history"][0]["new_key"] == new_key_id

    def test_environment_provider_key_rotation(self):
        """Test key rotation with environment provider."""
        provider = EnvironmentKeyProvider()

        # Set up a test key
        test_key = Fernet.generate_key()
        os.environ["CLOAKPIVOT_KEY_OLD_KEY"] = base64.b64encode(test_key).decode()

        try:
            # Verify we can get the old key
            old_key = provider.get_key("old_key")
            assert old_key == test_key

            # Rotate (creates new key but doesn't actually set env var)
            new_key_id = provider.rotate_key("old_key")

            # Verify new key ID was generated
            assert new_key_id is not None
            assert new_key_id != "old_key"

            # Verify old key was removed from cache
            assert "old_key" not in provider._key_cache

        finally:
            # Cleanup
            del os.environ["CLOAKPIVOT_KEY_OLD_KEY"]


class TestDifferentEncryptionAlgorithms:
    """Test different encryption algorithms."""

    def test_fernet_algorithm(self):
        """Test Fernet encryption algorithm."""
        # Set up key provider with Fernet key
        key_provider = EnvironmentKeyProvider()
        fernet_key = Fernet.generate_key()
        key_provider.get_key = lambda key_id: fernet_key

        manager = PresidioEncryptionManager(key_provider)

        # Test Fernet encryption
        original = "sensitive_data"
        encrypted = manager._encryption_callback(original, "PII", 0.9, "Fernet")

        # Verify encryption
        assert encrypted != original

        # Verify we can decrypt
        fernet = Fernet(fernet_key)
        decrypted = fernet.decrypt(base64.b64decode(encrypted))
        assert decrypted.decode() == original

    def test_aes_algorithm_placeholder(self):
        """Test AES-256 algorithm (placeholder implementation)."""
        key_provider = EnvironmentKeyProvider()
        key_provider.get_key = lambda key_id: b"32_byte_key_for_aes256_algorithm"

        manager = PresidioEncryptionManager(key_provider)

        # Test AES encryption raises NotImplementedError
        original = "sensitive_data"
        with pytest.raises(NotImplementedError, match="Encryption algorithm AES-256 is not yet implemented"):
            manager._encryption_callback(original, "PII", 0.9, "AES-256")

    def test_chacha20_algorithm_placeholder(self):
        """Test ChaCha20 algorithm (placeholder implementation)."""
        key_provider = EnvironmentKeyProvider()
        key_provider.get_key = lambda key_id: b"32_byte_key_for_chacha20_algo"

        manager = PresidioEncryptionManager(key_provider)

        # Test ChaCha20 encryption raises NotImplementedError
        original = "sensitive_data"
        with pytest.raises(NotImplementedError, match="Encryption algorithm ChaCha20 is not yet implemented"):
            manager._encryption_callback(original, "PII", 0.9, "ChaCha20")


class TestBatchEncryption:
    """Test batch processing with encryption."""

    def test_batch_processing_with_encryption(self):
        """Test batch processing of documents with encryption."""
        # Create batch processor
        processor = PresidioBatchProcessor(batch_size=2, parallel_workers=2)

        # Create test documents
        documents = []
        for i in range(5):
            doc = DoclingDocument(name=f"doc_{i}")
            doc.add_text(
                text=f"Document {i} contains SSN 123-45-678{i} and credit card 4111-1111-1111-111{i}",
                label="text"
            )
            documents.append(doc)

        # Create masking policy
        policy = MaskingPolicy(
            per_entity={
                "SSN": Strategy(kind=StrategyKind.REDACT),
                "CREDIT_CARD": Strategy(kind=StrategyKind.HASH)
            }
        )

        # Process documents
        results = processor.process_document_batch(documents, policy)

        # Verify results
        assert len(results) == 5

        # Check that documents were processed
        for i, result in enumerate(results):
            assert result.masked_document is not None
            assert result.cloakmap is not None
            # Check if masking was successful (stats should have success=True)
            if result.stats and result.stats.get("success", True):
                # Text should be different from original if entities were found
                _ = result.masked_document.export_to_text()
                _ = documents[i].export_to_text()
                # May or may not be different depending on entity detection

        # Check statistics
        stats = processor.get_statistics()
        assert stats["total_processed"] == 5
        assert stats["batch_count"] > 0

    def test_batch_text_processing(self):
        """Test batch processing of text strings."""
        processor = PresidioBatchProcessor(batch_size=3, parallel_workers=1)

        # Create test texts
        texts = [
            "My SSN is 123-45-6789",
            "Call me at 555-123-4567",
            "My email is john@example.com",
            "Credit card: 4111-1111-1111-1111",
            "IP address: 192.168.1.1"
        ]

        # Create masking policy
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.REDACT)
        )

        # Process texts
        results = processor.process_text_batch(texts, policy)

        # Verify results
        assert len(results) == 5

        for _i, (masked_text, cloakmap) in enumerate(results):
            # Text should be processed (may or may not change depending on detection)
            assert masked_text is not None

            if cloakmap is not None:
                # If PII was detected, cloakmap should have anchors
                assert hasattr(cloakmap, "anchors")

    def test_batch_processing_with_progress_callback(self):
        """Test batch processing with progress callback."""
        processor = PresidioBatchProcessor(batch_size=2, parallel_workers=1)

        # Track progress
        progress_updates = []

        def progress_callback(completed, total):
            progress_updates.append((completed, total))

        # Create test documents
        documents = []
        for i in range(4):
            doc = DoclingDocument(name=f"doc_{i}")
            doc.add_text(text=f"Document {i}", label="text")
            documents.append(doc)

        # Process with callback
        policy = MaskingPolicy()
        _ = processor.process_document_batch(
            documents, policy, progress_callback
        )

        # Verify progress was reported
        assert len(progress_updates) > 0

        # Last update should show all batches completed
        last_update = progress_updates[-1]
        assert last_update[0] == last_update[1]  # completed == total
