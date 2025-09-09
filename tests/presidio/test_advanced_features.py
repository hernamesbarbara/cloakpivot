"""Tests for advanced Presidio features."""

from unittest.mock import MagicMock, patch

import pytest
from presidio_analyzer import RecognizerResult

from cloakpivot.core.strategies import StrategyKind
from cloakpivot.presidio.advanced_features import (
    PresidioAdHocRecognizers,
    PresidioConnectionPool,
    PresidioEncryptionManager,
    PresidioOperatorChain,
)


class TestPresidioEncryptionManager:
    """Test encryption manager functionality."""

    def test_encryption_manager_initialization(self):
        """Test that encryption manager initializes correctly."""
        key_provider = MagicMock()
        manager = PresidioEncryptionManager(key_provider)

        assert manager.key_provider == key_provider
        assert manager.encryption_operators is not None
        assert "AES-256" in manager.encryption_operators
        assert "ChaCha20" in manager.encryption_operators
        assert "Fernet" in manager.encryption_operators

    def test_create_encryption_strategy(self):
        """Test creating an encryption strategy."""
        key_provider = MagicMock()
        key_provider.get_key.return_value = b"test_key_32_bytes_long_for_aes256"

        manager = PresidioEncryptionManager(key_provider)
        strategy = manager.create_encryption_strategy("AES-256")

        assert strategy.kind == StrategyKind.CUSTOM
        assert "callback" in strategy.parameters
        assert strategy.parameters["algorithm"] == "AES-256"
        assert strategy.parameters["key_rotation"] is True

    def test_encryption_callback(self):
        """Test the encryption callback function."""
        key_provider = MagicMock()
        # Fernet key must be 32 url-safe base64-encoded bytes
        from cryptography.fernet import Fernet
        test_key = Fernet.generate_key()
        key_provider.get_key.return_value = test_key

        manager = PresidioEncryptionManager(key_provider)

        # Test Fernet encryption
        encrypted = manager._encryption_callback(
            "sensitive_data", "CREDIT_CARD", 0.95, "Fernet"
        )

        assert encrypted != "sensitive_data"
        assert len(encrypted) > 0

        # Test other algorithm raises NotImplementedError
        with pytest.raises(NotImplementedError, match="Encryption algorithm AES-256 is not yet implemented"):
            manager._encryption_callback(
                "sensitive_data", "SSN", 0.95, "AES-256"
            )

    def test_create_decryption_context(self):
        """Test extracting decryption context from CloakMap."""
        key_provider = MagicMock()
        key_provider.get_key.return_value = b"test_key"

        manager = PresidioEncryptionManager(key_provider)

        # Create a mock CloakMap with presidio metadata
        cloakmap = MagicMock()
        cloakmap.presidio_metadata = {
            "operator_results": [
                {
                    "operator": "encrypt",
                    "entity_type": "CREDIT_CARD",
                    "key_reference": "key_123",
                    "algorithm": "AES-256"
                },
                {
                    "operator": "replace",
                    "entity_type": "NAME",
                    "new_value": "[NAME]"
                }
            ]
        }

        context = manager.create_decryption_context(cloakmap)

        assert "CREDIT_CARD" in context
        assert context["CREDIT_CARD"]["key"] == b"test_key"
        assert context["CREDIT_CARD"]["algorithm"] == "AES-256"
        assert "NAME" not in context  # Only encryption operators


class TestPresidioOperatorChain:
    """Test operator chaining functionality."""

    def test_operator_chain_initialization(self):
        """Test that operator chain initializes correctly."""
        chain = PresidioOperatorChain()

        assert chain.supported_operators is not None
        assert "redact" in chain.supported_operators
        assert "encrypt" in chain.supported_operators

    def test_apply_single_operator_redact(self):
        """Test applying a single redact operator."""
        chain = PresidioOperatorChain()

        result = chain._apply_single_operator(
            "sensitive",
            {"type": "redact", "params": {"char": "*"}}
        )

        assert result == "*********"

    def test_apply_single_operator_replace(self):
        """Test applying a single replace operator."""
        chain = PresidioOperatorChain()

        result = chain._apply_single_operator(
            "john.doe@example.com",
            {"type": "replace", "params": {"new_value": "[EMAIL]"}}
        )

        assert result == "[EMAIL]"

    def test_apply_single_operator_mask(self):
        """Test applying a single mask operator."""
        chain = PresidioOperatorChain()

        result = chain._apply_single_operator(
            "1234567890",
            {"type": "mask", "params": {"char": "*", "visible_chars": 4}}
        )

        assert result == "1234******"

    def test_apply_single_operator_hash(self):
        """Test applying a single hash operator."""
        chain = PresidioOperatorChain()

        result = chain._apply_single_operator(
            "sensitive",
            {"type": "hash", "params": {"algorithm": "sha256"}}
        )

        assert result != "sensitive"
        assert len(result) == 16  # Truncated hash

    def test_is_chain_reversible(self):
        """Test checking if operator chain is reversible."""
        chain = PresidioOperatorChain()

        # Reversible chain (replace and mask can be reversible with metadata)
        reversible_sequence = [
            {"type": "replace", "params": {"new_value": "[TEMP]"}},
            {"type": "mask", "params": {"char": "*", "visible_chars": 4}}
        ]
        assert chain._is_chain_reversible(reversible_sequence) is True

        # Non-reversible chain (contains hash)
        non_reversible_sequence = [
            {"type": "replace", "params": {"new_value": "[TEMP]"}},
            {"type": "hash", "params": {"algorithm": "sha256"}}
        ]
        assert chain._is_chain_reversible(non_reversible_sequence) is False

    def test_create_chained_strategy(self):
        """Test creating a chained strategy."""
        chain = PresidioOperatorChain()

        operator_sequence = [
            {"type": "mask", "params": {"char": "*", "visible_chars": 4}},
            {"type": "replace", "params": {"new_value": "[MASKED]"}}
        ]

        strategy = chain.create_chained_strategy(operator_sequence)

        assert strategy.kind == StrategyKind.CUSTOM
        assert "callback" in strategy.parameters
        assert strategy.parameters["operator_chain"] == operator_sequence
        assert strategy.parameters["reversible"] is True

    def test_reverse_operator_chain(self):
        """Test reversing an operator chain."""
        chain = PresidioOperatorChain()

        chain_metadata = {
            "operator_chain": [
                {"type": "replace", "original": "john.doe@example.com"},
                {"type": "mask", "original": "[EMAIL]"}
            ]
        }

        result = chain.reverse_operator_chain("[MASKED]", chain_metadata)

        # Since we have original values in metadata, it should reverse
        assert result == "john.doe@example.com"


class TestPresidioAdHocRecognizers:
    """Test ad-hoc recognizer functionality."""

    def test_adhoc_recognizer_initialization(self):
        """Test that ad-hoc recognizer manager initializes correctly."""
        recognizers = PresidioAdHocRecognizers()

        assert recognizers.recognizer_cache == {}

    def test_create_pattern_recognizer(self):
        """Test creating a pattern-based recognizer."""
        recognizers = PresidioAdHocRecognizers()

        config = recognizers.create_pattern_recognizer(
            entity_type="CUSTOM_ID",
            patterns=[r"ID-\d{6}", r"REF-[A-Z]{3}\d{3}"],
            confidence=0.9
        )

        assert config["entity_type"] == "CUSTOM_ID"
        assert len(config["patterns"]) == 2
        assert config["patterns"][0]["pattern"] == r"ID-\d{6}"
        assert config["patterns"][0]["score"] == 0.9
        assert config["supported_language"] == "en"

    def test_create_context_recognizer(self):
        """Test creating a context-aware recognizer."""
        recognizers = PresidioAdHocRecognizers()

        config = recognizers.create_context_recognizer(
            entity_type="MEDICAL_TERM",
            context_words=["patient", "diagnosis", "treatment"],
            window_size=10
        )

        assert config["entity_type"] == "MEDICAL_TERM"
        assert config["context_words"] == ["patient", "diagnosis", "treatment"]
        assert config["window_size"] == 10
        assert "patterns" in config
        assert len(config["patterns"]) == 3  # One pattern per context word

    def test_build_recognizer(self):
        """Test building a recognizer from configuration."""
        recognizers = PresidioAdHocRecognizers()

        config = {
            "entity_type": "TEST_ENTITY",
            "patterns": [
                {"pattern": r"\d{3}-\d{3}-\d{4}", "score": 0.8}
            ],
            "supported_language": "en"
        }

        recognizer = recognizers._build_recognizer(config)

        assert recognizer.supported_entities == ["TEST_ENTITY"]
        assert len(recognizer.patterns) == 1
        assert recognizer.patterns[0].regex == r"\d{3}-\d{3}-\d{4}"
        assert recognizer.patterns[0].score == 0.8

    @patch('cloakpivot.presidio.advanced_features.AnalyzerEngine')
    def test_apply_adhoc_recognizers(self, mock_analyzer):
        """Test applying ad-hoc recognizers to text."""
        recognizers = PresidioAdHocRecognizers()

        # Mock analyzer results
        mock_engine = MagicMock()
        mock_engine.analyze.return_value = [
            RecognizerResult(
                entity_type="CUSTOM_ID",
                start=10,
                end=18,
                score=0.9
            )
        ]
        mock_analyzer.return_value = mock_engine

        recognizer_configs = [
            {
                "entity_type": "CUSTOM_ID",
                "patterns": [{"pattern": r"ID-\d{6}", "score": 0.9}],
                "supported_language": "en"
            }
        ]

        results = recognizers.apply_adhoc_recognizers(
            "This is ID-123456 in the text",
            recognizer_configs
        )

        assert len(results) == 1
        assert results[0].entity_type == "CUSTOM_ID"
        assert results[0].score == 0.9


class TestPresidioConnectionPool:
    """Test connection pool functionality."""

    def test_connection_pool_initialization(self):
        """Test that connection pool initializes correctly."""
        pool = PresidioConnectionPool(pool_size=3)

        assert pool.pool_size == 3
        assert pool._analyzer_pool.qsize() == 3
        assert pool._anonymizer_pool.qsize() == 3

    def test_get_analyzer_context_manager(self):
        """Test getting analyzer from pool with context manager."""
        pool = PresidioConnectionPool(pool_size=2)

        # Check initial pool size
        assert pool._analyzer_pool.qsize() == 2

        # Get analyzer from pool
        with pool.get_analyzer() as analyzer:
            # Pool should have one less engine
            assert pool._analyzer_pool.qsize() == 1
            assert analyzer is not None

        # After context exit, engine should be returned
        assert pool._analyzer_pool.qsize() == 2

    def test_get_anonymizer_context_manager(self):
        """Test getting anonymizer from pool with context manager."""
        pool = PresidioConnectionPool(pool_size=2)

        # Check initial pool size
        assert pool._anonymizer_pool.qsize() == 2

        # Get anonymizer from pool
        with pool.get_anonymizer() as anonymizer:
            # Pool should have one less engine
            assert pool._anonymizer_pool.qsize() == 1
            assert anonymizer is not None

        # After context exit, engine should be returned
        assert pool._anonymizer_pool.qsize() == 2

    def test_concurrent_access(self):
        """Test concurrent access to pool."""
        pool = PresidioConnectionPool(pool_size=2)

        # Get multiple engines concurrently
        with pool.get_analyzer() as analyzer1:
            with pool.get_analyzer() as analyzer2:
                # Both engines should be different instances
                assert analyzer1 is not analyzer2
                # Pool should be empty
                assert pool._analyzer_pool.qsize() == 0

        # After context exit, both engines should be returned
        assert pool._analyzer_pool.qsize() == 2
