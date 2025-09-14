"""Method registration system for adding masking/unmasking methods to DoclingDocument."""

from typing import Optional, List
import warnings

from docling_core.types import DoclingDocument

from cloakpivot.engine import CloakEngine
from cloakpivot.core.policies import MaskingPolicy


# Global engine instance for method registration
_global_engine: Optional[CloakEngine] = None


def register_cloak_methods(engine: Optional[CloakEngine] = None) -> None:
    """Register masking/unmasking methods on DoclingDocument.

    This allows for natural method chaining:
    doc.mask_pii().export_to_markdown()

    Args:
        engine: Optional CloakEngine instance to use. If not provided,
                a default engine will be created and cached.

    Examples:
        # Register with default engine
        register_cloak_methods()
        doc = converter.convert("file.pdf").document
        masked = doc.mask_pii()
        original = masked.unmask_pii()

        # Register with custom engine
        custom_engine = CloakEngine.builder()
            .with_languages(['en', 'es'])
            .build()
        register_cloak_methods(custom_engine)
    """
    global _global_engine

    # Use provided engine or create/reuse default
    if engine is not None:
        _global_engine = engine
    elif _global_engine is None:
        _global_engine = CloakEngine()

    # Check if methods already registered
    if hasattr(DoclingDocument, '_cloak_methods_registered'):
        warnings.warn(
            "CloakPivot methods already registered on DoclingDocument. "
            "Re-registering with new engine.",
            UserWarning
        )

    def mask_pii(
        self,
        entities: Optional[List[str]] = None,
        policy: Optional[MaskingPolicy] = None
    ):
        """Mask PII in this document and return a CloakedDocument wrapper.

        Args:
            entities: Optional list of entity types to detect
            policy: Optional masking policy to use

        Returns:
            CloakedDocument with masked content and stored CloakMap
        """
        from cloakpivot.wrappers import CloakedDocument

        result = _global_engine.mask_document(self, entities, policy)
        return CloakedDocument(result.document, result.cloakmap)

    def unmask_pii(self):
        """Unmask this document if it's a CloakedDocument.

        Returns:
            Original DoclingDocument with PII restored
        """
        # Check if this is a CloakedDocument wrapper
        if hasattr(self, '_cloakmap') and hasattr(self, '_doc'):
            # This is a CloakedDocument wrapper
            return _global_engine.unmask_document(self._doc, self._cloakmap)
        else:
            # This is a regular DoclingDocument, return as-is
            warnings.warn(
                "unmask_pii() called on non-masked document. Returning document as-is.",
                UserWarning
            )
            return self

    # Register methods on DoclingDocument class
    DoclingDocument.mask_pii = mask_pii
    DoclingDocument.unmask_pii = unmask_pii

    # Mark as registered
    DoclingDocument._cloak_methods_registered = True
    DoclingDocument._cloak_engine = _global_engine


def unregister_cloak_methods() -> None:
    """Remove registered CloakPivot methods from DoclingDocument.

    This is useful for testing or when you want to clean up
    the DoclingDocument class.
    """
    global _global_engine

    # Remove methods if they exist
    if hasattr(DoclingDocument, 'mask_pii'):
        delattr(DoclingDocument, 'mask_pii')
    if hasattr(DoclingDocument, 'unmask_pii'):
        delattr(DoclingDocument, 'unmask_pii')
    if hasattr(DoclingDocument, '_cloak_methods_registered'):
        delattr(DoclingDocument, '_cloak_methods_registered')
    if hasattr(DoclingDocument, '_cloak_engine'):
        delattr(DoclingDocument, '_cloak_engine')

    # Clear global engine
    _global_engine = None


def get_registered_engine() -> Optional[CloakEngine]:
    """Get the currently registered CloakEngine instance.

    Returns:
        The registered CloakEngine or None if not registered
    """
    return _global_engine


def is_registered() -> bool:
    """Check if CloakPivot methods are registered on DoclingDocument.

    Returns:
        True if methods are registered, False otherwise
    """
    return hasattr(DoclingDocument, '_cloak_methods_registered')


def update_engine(engine: CloakEngine) -> None:
    """Update the registered engine without re-registering methods.

    Args:
        engine: New CloakEngine instance to use

    Raises:
        RuntimeError: If methods are not yet registered
    """
    global _global_engine

    if not is_registered():
        raise RuntimeError(
            "CloakPivot methods not yet registered. "
            "Call register_cloak_methods() first."
        )

    _global_engine = engine
    DoclingDocument._cloak_engine = engine