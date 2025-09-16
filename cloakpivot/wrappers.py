"""Wrapper classes for enhanced functionality."""

from typing import Any

from docling_core.types import DoclingDocument  # type: ignore[attr-defined]

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.engine import CloakEngine


class CloakedDocument:
    """Lightweight wrapper that preserves CloakMap while maintaining DoclingDocument interface.

    This wrapper allows a masked DoclingDocument to carry its CloakMap,
    enabling easy unmasking later. It transparently delegates all
    DoclingDocument methods to the underlying document.

    Attributes:
        document: The masked DoclingDocument
        cloakmap: The CloakMap containing original PII values

    Examples:
        # Create through CloakEngine
        engine = CloakEngine()
        result = engine.mask_document(doc)
        cloaked = CloakedDocument(result.document, result.cloakmap)

        # Access document methods transparently
        markdown = cloaked.export_to_markdown()
        text = cloaked.export_to_text()

        # Unmask when needed
        original = cloaked.unmask_pii()

        # Access the CloakMap for persistence
        cloakmap = cloaked.cloakmap
        save_cloakmap(cloakmap)
    """

    def __init__(
        self, document: DoclingDocument, cloakmap: CloakMap, engine: CloakEngine | None = None
    ):
        """Initialize CloakedDocument wrapper.

        Args:
            document: The masked DoclingDocument
            cloakmap: The CloakMap for unmasking
            engine: Optional CloakEngine instance (creates default if not provided)
        """
        self._doc = document
        self._cloakmap = cloakmap
        self._engine = engine or CloakEngine()

    def __getattr__(self, name: str) -> Any:
        """Delegate all DoclingDocument methods transparently.

        Args:
            name: Attribute or method name to access

        Returns:
            The requested attribute or method from the underlying document
        """
        return getattr(self._doc, name)

    def __repr__(self) -> str:
        """String representation of CloakedDocument."""
        num_entities = len(self._cloakmap.anchors)
        return f"CloakedDocument(entities_masked={num_entities})"

    def __str__(self) -> str:
        """String conversion delegates to underlying document."""
        return str(self._doc)

    def unmask_pii(self) -> DoclingDocument:
        """Unmask using stored CloakMap.

        Returns:
            Original DoclingDocument with PII restored
        """
        return self._engine.unmask_document(self._doc, self._cloakmap)

    @property
    def cloakmap(self) -> CloakMap:
        """Access the CloakMap for persistence.

        Returns:
            The CloakMap containing original PII values and positions
        """
        return self._cloakmap

    @property
    def document(self) -> DoclingDocument:
        """Access the underlying masked document.

        Returns:
            The masked DoclingDocument
        """
        return self._doc

    @property
    def is_masked(self) -> bool:
        """Check if this document contains masked entities.

        Returns:
            True if document has masked entities, False if empty CloakMap
        """
        return len(self._cloakmap.anchors) > 0

    @property
    def entities_masked(self) -> int:
        """Get count of masked entities.

        Returns:
            Number of entities in the CloakMap
        """
        return len(self._cloakmap.anchors)

    def save_cloakmap(self, path: str, format: str = "json") -> None:
        """Save the CloakMap to a file.

        Args:
            path: File path to save the CloakMap
            format: Format to use ('json' or 'yaml')

        Raises:
            ValueError: If format is not supported
        """
        from pathlib import Path

        if format == "json":
            from cloakpivot.formats.json import JSONSerializer

            serializer = JSONSerializer()
        elif format == "yaml":
            from cloakpivot.formats.yaml import YAMLSerializer

            serializer = YAMLSerializer()
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'json' or 'yaml'.")

        serialized = serializer.serialize(self._cloakmap)
        Path(path).write_text(serialized)

    @classmethod
    def load_with_cloakmap(
        cls, document_path: str, cloakmap_path: str, engine: CloakEngine | None = None
    ) -> "CloakedDocument":
        """Load a masked document with its CloakMap.

        Args:
            document_path: Path to the masked document
            cloakmap_path: Path to the CloakMap file
            engine: Optional CloakEngine instance

        Returns:
            CloakedDocument instance ready for unmasking
        """
        from pathlib import Path

        from docling.document_converter import DocumentConverter

        # Load document
        converter = DocumentConverter()
        result = converter.convert(document_path)
        doc = result.document

        # Load CloakMap
        cloakmap_path_obj = Path(cloakmap_path)
        cloakmap_data = cloakmap_path_obj.read_text()

        # Detect format and deserialize
        if cloakmap_path_obj.suffix in [".yaml", ".yml"]:
            from cloakpivot.formats.yaml import YAMLSerializer

            serializer = YAMLSerializer()
        else:
            from cloakpivot.formats.json import JSONSerializer

            serializer = JSONSerializer()

        cloakmap = serializer.deserialize(cloakmap_data, CloakMap)

        return cls(doc, cloakmap, engine)
