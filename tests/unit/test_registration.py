"""Tests for the registration system."""

import warnings
from unittest.mock import Mock, patch

import pytest

from cloakpivot.engine import CloakEngine
from cloakpivot.registration import (
    get_registered_engine,
    is_registered,
    register_cloak_methods,
    unregister_cloak_methods,
    update_engine,
)
from cloakpivot.type_imports import DoclingDocument


class TestRegistration:
    """Test suite for method registration on DoclingDocument."""

    def setup_method(self):
        """Clean up before each test."""
        unregister_cloak_methods()

    def teardown_method(self):
        """Clean up after each test."""
        unregister_cloak_methods()

    def test_register_cloak_methods_with_default_engine(self):
        """Test registering methods with default engine."""
        assert not is_registered()

        register_cloak_methods()

        assert is_registered()
        assert hasattr(DoclingDocument, "mask_pii")
        assert hasattr(DoclingDocument, "unmask_pii")
        assert hasattr(DoclingDocument, "_cloak_methods_registered")
        assert hasattr(DoclingDocument, "_cloak_engine")

        engine = get_registered_engine()
        assert engine is not None
        assert isinstance(engine, CloakEngine)

    def test_register_cloak_methods_with_custom_engine(self):
        """Test registering methods with a custom engine."""
        custom_engine = CloakEngine()

        register_cloak_methods(custom_engine)

        assert is_registered()
        assert get_registered_engine() is custom_engine

    def test_register_cloak_methods_warning_on_reregister(self):
        """Test that re-registering shows a warning."""
        register_cloak_methods()

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            register_cloak_methods()

            assert len(w) == 1
            assert "already registered" in str(w[0].message)

    def test_unregister_cloak_methods(self):
        """Test unregistering methods removes them from DoclingDocument."""
        register_cloak_methods()
        assert is_registered()

        unregister_cloak_methods()

        assert not is_registered()
        assert not hasattr(DoclingDocument, "mask_pii")
        assert not hasattr(DoclingDocument, "unmask_pii")
        assert not hasattr(DoclingDocument, "_cloak_methods_registered")
        assert not hasattr(DoclingDocument, "_cloak_engine")
        assert get_registered_engine() is None

    def test_unregister_when_not_registered(self):
        """Test unregistering when nothing is registered works fine."""
        assert not is_registered()

        # Should not raise any errors
        unregister_cloak_methods()

        assert not is_registered()

    def test_get_registered_engine_when_not_registered(self):
        """Test getting engine when nothing is registered."""
        assert get_registered_engine() is None

    def test_update_engine_when_registered(self):
        """Test updating the engine after registration."""
        engine1 = CloakEngine()
        register_cloak_methods(engine1)

        engine2 = CloakEngine()
        update_engine(engine2)

        assert get_registered_engine() is engine2
        assert DoclingDocument._cloak_engine is engine2

    def test_update_engine_when_not_registered_raises(self):
        """Test updating engine before registration raises error."""
        engine = CloakEngine()

        with pytest.raises(RuntimeError, match="not yet registered"):
            update_engine(engine)

    def test_mask_pii_method_integration(self):
        """Test that mask_pii method works when registered."""
        from cloakpivot.wrappers import CloakedDocument

        register_cloak_methods()

        # Create a mock document
        doc = DoclingDocument(name="test.txt")

        # Mock the engine's mask_document method
        with patch.object(get_registered_engine(), "mask_document") as mock_mask:
            mock_result = Mock()
            mock_result.document = DoclingDocument(name="masked.txt")
            mock_result.cloakmap = Mock()
            mock_mask.return_value = mock_result

            # Call mask_pii
            result = doc.mask_pii()

            # Verify it returns a CloakedDocument
            assert isinstance(result, CloakedDocument)
            mock_mask.assert_called_once_with(doc, None, None)

    def test_mask_pii_with_custom_params(self):
        """Test mask_pii with custom entities and policy."""
        from cloakpivot.core.policies import MaskingPolicy
        from cloakpivot.wrappers import CloakedDocument

        register_cloak_methods()

        doc = DoclingDocument(name="test.txt")
        entities = ["PERSON", "EMAIL"]
        policy = MaskingPolicy()

        with patch.object(get_registered_engine(), "mask_document") as mock_mask:
            mock_result = Mock()
            mock_result.document = DoclingDocument(name="masked.txt")
            mock_result.cloakmap = Mock()
            mock_mask.return_value = mock_result

            result = doc.mask_pii(entities=entities, policy=policy)

            assert isinstance(result, CloakedDocument)
            mock_mask.assert_called_once_with(doc, entities, policy)

    def test_unmask_pii_on_cloaked_document(self):
        """Test unmask_pii on a CloakedDocument wrapper."""
        register_cloak_methods()

        # Create a mock CloakedDocument-like object
        doc = Mock()
        doc._cloakmap = Mock()
        doc._doc = DoclingDocument(name="masked.txt")

        unmasked_doc = DoclingDocument(name="original.txt")

        with patch.object(
            get_registered_engine(), "unmask_document", return_value=unmasked_doc
        ) as mock_unmask:
            result = DoclingDocument.unmask_pii(doc)

            assert result is unmasked_doc
            mock_unmask.assert_called_once_with(doc._doc, doc._cloakmap)

    def test_unmask_pii_on_regular_document(self):
        """Test unmask_pii on a regular DoclingDocument shows warning."""
        register_cloak_methods()

        doc = DoclingDocument(name="test.txt")

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = doc.unmask_pii()

            assert result is doc  # Returns same document
            assert len(w) == 1
            assert "non-masked document" in str(w[0].message)
