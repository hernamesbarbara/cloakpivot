"""Document processor for loading and managing DoclingDocument objects."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

from cloakpivot.core.types import DoclingDocument

from ..core.chunking import ChunkedDocumentProcessor

logger = logging.getLogger(__name__)


@dataclass
class DocumentProcessingStats:
    """Statistics for document processing operations."""

    files_processed: int = 0
    errors_encountered: int = 0

    def reset(self) -> None:
        """Reset all statistics."""
        self.files_processed = 0
        self.errors_encountered = 0


class DocumentProcessor:
    """
    Document processor that integrates with DocPivot to load and manage documents.

    This class provides high-level document loading functionality using DocPivot's
    workflow system, with error handling and statistics tracking for CloakPivot's
    document processing pipeline.

    Examples:
        >>> processor = DocumentProcessor()
        >>> doc = processor.load_document("sample.docling.json")
        >>> print(f"Loaded document: {doc.name}")

        >>> # With validation
        >>> doc = processor.load_document("sample.docling.json", validate=True)
    """

    def __init__(self, enable_chunked_processing: bool = True) -> None:
        """Initialize the document processor."""
        self._stats = DocumentProcessingStats()
        self._enable_chunked_processing = enable_chunked_processing
        self._chunked_processor: Optional[ChunkedDocumentProcessor]

        if enable_chunked_processing:
            self._chunked_processor = ChunkedDocumentProcessor()
        else:
            self._chunked_processor = None

        logger.debug(
            f"DocumentProcessor initialized (chunked_processing={enable_chunked_processing})"
        )

    def load_document(
        self, file_path: Union[str, Path], validate: bool = True, **kwargs: Any
    ) -> DoclingDocument:
        """
        Load a document using DocPivot's workflow system.

        This method wraps DocPivot's load_document function with additional
        validation and error handling specific to CloakPivot's needs.

        Args:
            file_path: Path to the document file to load
            validate: Whether to validate the document structure after loading
            **kwargs: Additional parameters to pass to the DocPivot loader

        Returns:
            DoclingDocument: The loaded document

        Raises:
            FileNotFoundError: If the file cannot be found
            ValueError: If the JSON is invalid or document validation fails
            RuntimeError: If document loading fails

        Examples:
            >>> processor = DocumentProcessor()
            >>> doc = processor.load_document("sample.docling.json")
            >>> print(f"Document has {len(doc.texts)} text items")
        """
        file_path_obj = Path(file_path)
        logger.info(f"Loading document from {file_path_obj}")

        try:
            # Load Docling JSON directly (no DocPivot needed)
            with open(file_path, 'r') as f:
                doc_dict = json.load(f)
            document = DoclingDocument.model_validate(doc_dict)
            self._stats.files_processed += 1

            if validate:
                self._validate_document_structure(document)

            # Log document version for v1.7.0 migration awareness
            doc_version = getattr(document, 'version', '1.2.0')
            logger.info(f"Successfully loaded document: {document.name} (version: {doc_version})")
            logger.debug(
                f"Document contains: {len(document.texts)} text items, "
                f"{len(document.tables)} tables, "
                f"{len(document.pictures)} pictures"
            )

            # Note about v1.7.0 changes
            from packaging import version
            if version.parse(str(doc_version)) >= version.parse('1.7.0'):
                logger.debug(
                    "Note: DoclingDocument v1.7.0+ uses segment-local charspans. "
                    "CloakPivot handles this transparently."
                )

            return document

        except FileNotFoundError as e:
            self._stats.errors_encountered += 1
            logger.error(f"File not found: {file_path}")
            raise FileNotFoundError(f"File not found: {file_path}") from e
        except json.JSONDecodeError as e:
            self._stats.errors_encountered += 1
            logger.error(f"Invalid JSON in file {file_path}: {e}")
            raise ValueError(f"Invalid JSON: {e}") from e
        except Exception as e:
            # Wrap unexpected errors
            self._stats.errors_encountered += 1
            logger.error(f"Unexpected error loading document {file_path}: {e}")
            raise RuntimeError(
                f"Unexpected error loading document from '{file_path}': {e}"
            ) from e

    def _validate_document_structure(self, document: DoclingDocument) -> None:
        """
        Validate that the document has the expected structure for CloakPivot processing.

        Args:
            document: The DoclingDocument to validate

        Raises:
            ValueError: If the document structure is invalid
        """
        if not isinstance(document, DoclingDocument):
            raise ValueError(
                f"Expected DoclingDocument, got {type(document).__name__}"
            )

        # Check that the document has a name
        if not document.name:
            logger.warning(
                "Document has no name - this may cause issues with anchor mapping"
            )

        # Check for text-bearing content
        has_text_content = (
            len(document.texts) > 0
            or len(document.tables) > 0
            or len(document.key_value_items) > 0
        )

        if not has_text_content:
            logger.warning(
                "Document appears to have no text content - nothing to process"
            )

        # Validate node structure
        self._validate_node_references(document)

        logger.debug("Document structure validation passed")

    def _validate_node_references(self, document: DoclingDocument) -> None:
        """
        Validate that document nodes have proper self-references for anchor mapping.

        Args:
            document: The DoclingDocument to validate

        Raises:
            ValidationError: If node references are invalid
        """
        issues = []

        # Check text items
        for i, text_item in enumerate(document.texts):
            if not hasattr(text_item, "self_ref") or not text_item.self_ref:
                issues.append(f"Text item {i} missing self_ref")

        # Check table items
        for i, table_item in enumerate(document.tables):
            if not hasattr(table_item, "self_ref") or not table_item.self_ref:
                issues.append(f"Table item {i} missing self_ref")

        if issues:
            logger.warning(f"Node reference issues found: {issues}")
            # Don't raise error for missing self_ref as we can generate them

    def get_processing_stats(self) -> DocumentProcessingStats:
        """
        Get processing statistics for this processor instance.

        Returns:
            DocumentProcessingStats: Current processing statistics
        """
        return self._stats

    def reset_stats(self) -> None:
        """Reset processing statistics."""
        self._stats.reset()
        logger.debug("Processing statistics reset")

    def supports_format(self, file_path: Union[str, Path]) -> bool:
        """
        Check if the given file format is supported by DocPivot.

        Args:
            file_path: Path to check

        Returns:
            bool: True if the format is likely supported
        """
        file_path_obj = Path(file_path)

        # Check common supported extensions
        supported_extensions = {".json"}  # docling.json, lexical.json

        if file_path_obj.suffix.lower() in supported_extensions:
            return True

        # Check for specific format indicators in filename
        if any(
            pattern in file_path_obj.name.lower() for pattern in ["docling", "lexical"]
        ):
            return True

        return False
