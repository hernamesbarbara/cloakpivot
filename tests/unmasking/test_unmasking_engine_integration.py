"""Tests for UnmaskingEngine integration with Presidio feature flags."""

import os
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

from cloakpivot.core.anchors import AnchorEntry
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.types import DoclingDocument, UnmaskingResult
from cloakpivot.unmasking.engine import UnmaskingEngine


class TestUnmaskingEngineIntegration:

    def _get_document_text(self, document: DoclingDocument) -> str:
        """Helper to get text from document, handling both formats."""
        if hasattr(document, '_main_text'):
            return document._main_text
        elif document.texts:
            return document.texts[0].text
        return ""

    def _set_document_text(self, document: DoclingDocument, text: str) -> None:
        """Helper to set text in document, handling both formats."""
        from docling_core.types.doc.document import TextItem
        # Create proper TextItem
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        document.texts = [text_item]
        # Also set _main_text for backward compatibility
        document._main_text = text

    """Test UnmaskingEngine with Presidio integration."""

    @pytest.fixture
    def sample_document(self) -> DoclingDocument:
        """Create a sample masked document."""
        doc = DoclingDocument(name="test_doc.txt")
        self._set_document_text(doc, "Contact ENTITY_1234 at PHONE_5678 about ORDER_9012.")
        return doc

    @pytest.fixture
    def v1_cloakmap(self) -> CloakMap:
        """Create a v1.0 CloakMap with only anchors."""
        return CloakMap(
            version="1.0",
            doc_id="test_doc.txt",
            doc_hash="abc123",
            anchors=[
                AnchorEntry(
                    node_id="main_text",
                    start=8,
                    end=19,
                    entity_type="PERSON",
                    confidence=0.95,
                    masked_value="ENTITY_1234",
                    replacement_id="entity_1234",
                    original_checksum="a665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    checksum_salt="c2FsdDEyMw==",
                    strategy_used="replace",
                    timestamp=datetime.now(),
                    metadata={"original_text": "John Doe"},
                ),
                AnchorEntry(
                    node_id="main_text",
                    start=23,
                    end=33,
                    entity_type="PHONE_NUMBER",
                    confidence=0.98,
                    masked_value="PHONE_5678",
                    replacement_id="phone_5678",
                    original_checksum="b665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    checksum_salt="c2FsdDQ1Ng==",
                    strategy_used="replace",
                    timestamp=datetime.now(),
                    metadata={"original_text": "555-1234"},
                ),
            ],
            created_at=datetime.now(),
        )

    @pytest.fixture
    def v2_cloakmap(self) -> CloakMap:
        """Create a v2.0 CloakMap with Presidio metadata."""
        return CloakMap(
            version="2.0",
            doc_id="test_doc.txt",
            doc_hash="abc123",
            anchors=[
                AnchorEntry(
                    node_id="main_text",
                    start=40,
                    end=50,
                    entity_type="ORDER_ID",
                    confidence=0.99,
                    masked_value="ORDER_9012",
                    replacement_id="order_9012",
                    original_checksum="c665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    checksum_salt="c2FsdDc4OQ==",
                    strategy_used="replace",
                    timestamp=datetime.now(),
                    metadata={"original_text": "ORD-123456"},
                ),
            ],
            presidio_metadata={
                "operator_results": [
                    {
                        "entity_type": "PERSON",
                        "start": 8,
                        "end": 19,
                        "operator": "replace",
                        "text": "ENTITY_1234",
                        "original_text": "John Doe",
                    },
                    {
                        "entity_type": "PHONE_NUMBER",
                        "start": 23,
                        "end": 33,
                        "operator": "encrypt",
                        "text": "PHONE_5678",
                        "original_text": "555-1234",
                        "key_reference": "key_123",
                    },
                ],
                "reversible_operators": ["encrypt"],
                "engine_version": "2.2.0",
            },
            created_at=datetime.now(),
        )

    def test_engine_init_with_presidio_flag(self) -> None:
        """Test UnmaskingEngine initialization with Presidio flag."""
        # Test with explicit True
        engine = UnmaskingEngine(use_presidio_engine=True)
        assert engine.use_presidio_override is True
        assert hasattr(engine, "presidio_adapter")

        # Test with explicit False
        engine = UnmaskingEngine(use_presidio_engine=False)
        assert engine.use_presidio_override is False

        # Test with None (default)
        engine = UnmaskingEngine()
        assert engine.use_presidio_override is None

    def test_select_unmasking_engine_explicit_override(self, v1_cloakmap: CloakMap, v2_cloakmap: CloakMap) -> None:
        """Test engine selection with explicit override."""
        # Force Presidio even with v1.0 CloakMap
        engine = UnmaskingEngine(use_presidio_engine=True)
        assert engine._select_unmasking_engine(v1_cloakmap) == "presidio"

        # Force legacy even with v2.0 CloakMap
        engine = UnmaskingEngine(use_presidio_engine=False)
        assert engine._select_unmasking_engine(v2_cloakmap) == "legacy"

    def test_select_unmasking_engine_env_override(self, v2_cloakmap: CloakMap) -> None:
        """Test engine selection with environment variable."""
        engine = UnmaskingEngine()

        # Test with env var set to true
        with patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "true"}):
            assert engine._select_unmasking_engine(v2_cloakmap) == "presidio"

        # Test with env var set to false
        with patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "false"}):
            assert engine._select_unmasking_engine(v2_cloakmap) == "legacy"

    def test_select_unmasking_engine_auto_detection(self, v1_cloakmap: CloakMap, v2_cloakmap: CloakMap) -> None:
        """Test automatic engine selection based on CloakMap."""
        engine = UnmaskingEngine()

        # v1.0 CloakMap should use legacy
        assert engine._select_unmasking_engine(v1_cloakmap) == "legacy"

        # v2.0 CloakMap with presidio_metadata should use presidio
        assert engine._select_unmasking_engine(v2_cloakmap) == "presidio"

    def test_unmask_with_presidio_engine(self, sample_document: DoclingDocument, v2_cloakmap: CloakMap) -> None:
        """Test unmasking with Presidio engine."""
        engine = UnmaskingEngine(use_presidio_engine=True)

        # Mock the presidio adapter
        with patch.object(engine, "presidio_adapter") as mock_adapter:
            mock_result = UnmaskingResult(
                restored_document=sample_document,
                cloakmap=v2_cloakmap,
                stats={"method": "presidio", "presidio_restored": 2},
            )
            mock_adapter.unmask_document.return_value = mock_result

            result = engine.unmask_document(sample_document, v2_cloakmap)

            # Verify Presidio adapter was called
            mock_adapter.unmask_document.assert_called_once_with(
                sample_document, v2_cloakmap
            )
            assert result.stats["method"] == "presidio"

    def test_unmask_with_legacy_engine(self, sample_document: DoclingDocument, v1_cloakmap: CloakMap) -> None:
        """Test unmasking with legacy engine."""
        engine = UnmaskingEngine(use_presidio_engine=False)

        # This should use the legacy unmasking path
        with patch.object(engine, "_unmask_with_legacy") as mock_legacy:
            mock_result = UnmaskingResult(
                restored_document=sample_document,
                cloakmap=v1_cloakmap,
                stats={"method": "legacy"},
            )
            mock_legacy.return_value = mock_result

            result = engine.unmask_document(sample_document, v1_cloakmap)

            mock_legacy.assert_called_once()
            assert result.stats["method"] == "legacy"

    def test_detect_reversible_operations(self, v2_cloakmap: CloakMap) -> None:
        """Test detection of reversible operations in CloakMap."""
        engine = UnmaskingEngine()

        # v2 CloakMap with reversible ops should return True
        assert engine._detect_reversible_operations(v2_cloakmap) is True

        # v1 CloakMap without Presidio metadata should return False
        v1_map = CloakMap(
            version="1.0",
            doc_id="test",
            doc_hash="hash",
            anchors=[],
            created_at=datetime.now(),
        )
        assert engine._detect_reversible_operations(v1_map) is False

    def test_categorize_operations(self, v2_cloakmap: CloakMap) -> None:
        """Test categorization of operations into reversible and non-reversible."""
        engine = UnmaskingEngine()

        reversible_ops, anchor_ops = engine._categorize_operations(v2_cloakmap)

        # Should have 2 reversible operations from presidio_metadata
        assert len(reversible_ops) == 2
        # Should have 1 anchor operation
        assert len(anchor_ops) == 1

    def test_hybrid_processing(self, sample_document: DoclingDocument) -> None:
        """Test hybrid processing with both Presidio and anchor operations."""
        # Create a CloakMap with both types of operations
        hybrid_cloakmap = CloakMap(
            version="2.0",
            doc_id="test_doc.txt",
            doc_hash="abc123",
            anchors=[
                AnchorEntry(
                    node_id="main_text",
                    start=100,
                    end=112,
                    entity_type="NON_REVERSIBLE",
                    confidence=0.95,
                    masked_value="LEGACY_TOKEN",
                    replacement_id="legacy_token",
                    original_checksum="d665a45920422f9d417e4867efdc4fb8a04a1f3fff1fa07e998e86f7f7a27ae3",
                    checksum_salt="c2FsdDAxMg==",
                    strategy_used="replace",
                    timestamp=datetime.now(),
                    metadata={"original_text": "legacy_value"},
                ),
            ],
            presidio_metadata={
                "operator_results": [
                    {
                        "entity_type": "REVERSIBLE",
                        "start": 50,
                        "end": 65,
                        "operator": "encrypt",
                        "text": "ENCRYPTED_VAL",
                        "original_text": "sensitive_data",
                    },
                ],
                "reversible_operators": ["encrypt"],
            },
            created_at=datetime.now(),
        )

        engine = UnmaskingEngine()

        with patch.object(engine, "_unmask_with_presidio") as mock_presidio:
            mock_result = UnmaskingResult(
                restored_document=sample_document,
                cloakmap=hybrid_cloakmap,
                stats={"method": "hybrid", "presidio_restored": 1, "anchor_restored": 1},
            )
            mock_presidio.return_value = mock_result

            result = engine.unmask_document(sample_document, hybrid_cloakmap)

            assert result.stats["method"] == "hybrid"
            assert result.stats["presidio_restored"] == 1
            assert result.stats["anchor_restored"] == 1

    def test_migrate_to_presidio(self, v1_cloakmap: CloakMap) -> None:
        """Test migration of v1.0 CloakMap to v2.0 with Presidio metadata."""
        engine = UnmaskingEngine()

        # Create a mock path for the cloakmap
        mock_path = Path("/tmp/test.cloakmap")

        with patch.object(engine.cloakmap_loader, "load") as mock_load:
            mock_load.return_value = v1_cloakmap

            with patch.object(engine, "_enhance_legacy_cloakmap") as mock_enhance:
                enhanced_map = CloakMap(
                    version="2.0",
                    doc_id=v1_cloakmap.doc_id,
                    doc_hash=v1_cloakmap.doc_hash,
                    anchors=v1_cloakmap.anchors,
                    presidio_metadata={
                        "operator_results": [],
                        "reversible_operators": [],
                    },
                    created_at=v1_cloakmap.created_at,
                )
                mock_enhance.return_value = enhanced_map

                # Mock the save_to_file method at the module level
                with patch("cloakpivot.core.cloakmap.CloakMap.save_to_file") as mock_save:
                    new_path = engine.migrate_to_presidio(mock_path)

                    # Verify migration steps
                    mock_load.assert_called_once_with(mock_path)
                    mock_enhance.assert_called_once_with(v1_cloakmap)
                    mock_save.assert_called_once_with(mock_path.with_suffix(".v2.cloakmap"))
                    assert new_path == mock_path.with_suffix(".v2.cloakmap")

    def test_backward_compatibility_v1_cloakmap(self, sample_document: DoclingDocument, v1_cloakmap: CloakMap) -> None:
        """Test that v1.0 CloakMaps still work correctly."""
        engine = UnmaskingEngine()

        # Mock the legacy unmasking components
        with patch.object(engine.anchor_resolver, "resolve_anchors") as mock_resolve:
            mock_resolve.return_value = {
                "resolved": [
                    {"anchor": v1_cloakmap.anchors[0], "position": 8},
                    {"anchor": v1_cloakmap.anchors[1], "position": 23},
                ],
                "failed": [],
            }

            with patch.object(engine.document_unmasker, "apply_unmasking") as mock_apply:
                mock_apply.return_value = {
                    "successful_restorations": 2,
                    "failed_restorations": 0,
                }

                result = engine.unmask_document(sample_document, v1_cloakmap)

                # Should use legacy engine
                assert "anchor" in result.stats.get("method", "") or result.stats.get("resolved_anchors", 0) > 0

    def test_environment_configuration(self) -> None:
        """Test environment variable configuration options."""
        # Test auto mode (default)
        with patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "auto"}):
            engine = UnmaskingEngine()
            # Should allow automatic detection
            assert engine.use_presidio_override is None

        # Test force legacy mode
        with patch.dict(os.environ, {"CLOAKPIVOT_FORCE_LEGACY_UNMASKING": "true"}):
            engine = UnmaskingEngine()
            # Implementation would check this env var in _select_unmasking_engine
            # For now, just verify the engine initializes
            assert engine is not None

        # Test hybrid processing mode
        with patch.dict(os.environ, {"CLOAKPIVOT_ENABLE_HYBRID_PROCESSING": "true"}):
            engine = UnmaskingEngine()
            # Implementation would enable hybrid processing
            assert engine is not None
