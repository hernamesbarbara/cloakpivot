"""Document processor for loading and managing DoclingDocument objects."""

import logging
from pathlib import Path
from typing import Union, Any, cast

from docling_core.types import DoclingDocument
from docpivot import load_document
from docpivot.io.readers.exceptions import (
    FileAccessError,
    UnsupportedFormatError,
    ValidationError,
    TransformationError,
)

from dataclasses import dataclass

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

    def __init__(self) -> None:
        """Initialize the document processor."""
        self._stats = DocumentProcessingStats()
        logger.debug("DocumentProcessor initialized")

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
            FileAccessError: If the file cannot be accessed or read
            UnsupportedFormatError: If no reader can handle the file format
            ValidationError: If the file format is invalid or document validation fails
            TransformationError: If document loading fails

        Examples:
            >>> processor = DocumentProcessor()
            >>> doc = processor.load_document("sample.docling.json")
            >>> print(f"Document has {len(doc.texts)} text items")
        """
        file_path_obj = Path(file_path)
        logger.info(f"Loading document from {file_path_obj}")

        try:
            # Use DocPivot's load_document workflow
            document = cast(DoclingDocument, load_document(file_path, **kwargs))
            self._stats.files_processed += 1

            if validate:
                self._validate_document_structure(document)

            logger.info(f"Successfully loaded document: {document.name}")
            logger.debug(
                f"Document contains: {len(document.texts)} text items, "
                f"{len(document.tables)} tables, "
                f"{len(document.pictures)} pictures"
            )

            return document

        except (
            FileAccessError,
            UnsupportedFormatError,
            ValidationError,
            TransformationError,
        ):
            # Re-raise DocPivot exceptions as-is
            self._stats.errors_encountered += 1
            raise
        except Exception as e:
            # Wrap unexpected errors
            self._stats.errors_encountered += 1
            logger.error(f"Unexpected error loading document {file_path}: {e}")
            raise TransformationError(
                f"Unexpected error loading document from '{file_path}': {e}",
                transformation_type="document_loading",
                context={"file_path": str(file_path), "kwargs": kwargs},
                cause=e,
            ) from e

    def _validate_document_structure(self, document: DoclingDocument) -> None:
        """
        Validate that the document has the expected structure for CloakPivot processing.

        Args:
            document: The DoclingDocument to validate

        Raises:
            ValidationError: If the document structure is invalid
        """
        if not isinstance(document, DoclingDocument):
            raise ValidationError(
                f"Expected DoclingDocument, got {type(document).__name__}",
                validation_errors=["invalid_document_type"],
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
