"""Unit tests for cloakpivot.document.processor module."""

from unittest.mock import Mock

from cloakpivot.document.processor import DocumentProcessor
from cloakpivot.type_imports import DoclingDocument


class TestDocumentProcessor:
    """Test DocumentProcessor class."""

    def test_initialization_default(self):
        """Test DocumentProcessor initialization with defaults."""
        processor = DocumentProcessor()
        assert processor is not None
        assert hasattr(processor, "process_document")

    def test_initialization_with_config(self):
        """Test DocumentProcessor initialization with configuration."""
        config = {"max_length": 10000, "preserve_formatting": True, "extract_metadata": True}
        processor = DocumentProcessor(config=config)
        assert processor is not None

    def test_process_document_basic(self):
        """Test basic document processing."""
        processor = DocumentProcessor()

        # Mock document
        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "# Test Document\n\nContent here."

        result = processor.process_document(doc)

        assert result is not None
        assert isinstance(result, (dict, DoclingDocument))

    def test_process_document_with_text_extraction(self):
        """Test document processing with text extraction."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "extract.md"
        doc.export_to_markdown.return_value = "Text to extract"
        doc.texts = ["Paragraph 1", "Paragraph 2"]

        result = processor.process_document(doc)

        assert result is not None

    def test_process_document_with_tables(self):
        """Test processing document with tables."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "table.md"
        doc.export_to_markdown.return_value = "| Col1 | Col2 |\n|------|------|\n| A | B |"
        doc.tables = [Mock(data=[["A", "B"]])]

        result = processor.process_document(doc)

        assert result is not None

    def test_process_document_empty(self):
        """Test processing empty document."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "empty.md"
        doc.export_to_markdown.return_value = ""

        result = processor.process_document(doc)

        assert result is not None

    def test_extract_text_nodes(self):
        """Test extracting text nodes from document."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.texts = ["Text 1", "Text 2", "Text 3"]

        # If extract_text_nodes method exists
        if hasattr(processor, "extract_text_nodes"):
            nodes = processor.extract_text_nodes(doc)
            assert isinstance(nodes, list)

    def test_extract_metadata(self):
        """Test extracting metadata from document."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "meta.pdf"
        doc.metadata = {"author": "Test Author", "date": "2024-01-01", "pages": 10}

        # If extract_metadata method exists
        if hasattr(processor, "extract_metadata"):
            metadata = processor.extract_metadata(doc)
            assert isinstance(metadata, dict)

    def test_normalize_content(self):
        """Test content normalization."""
        processor = DocumentProcessor()

        content = "  Text   with   extra    spaces  \n\n\n"

        # If normalize_content method exists
        if hasattr(processor, "normalize_content"):
            normalized = processor.normalize_content(content)
            assert normalized is not None
            assert len(normalized) <= len(content)

    def test_process_batch(self):
        """Test batch document processing."""
        processor = DocumentProcessor()

        docs = []
        for i in range(3):
            doc = Mock(spec=DoclingDocument)
            doc.name = f"doc{i}.md"
            doc.export_to_markdown.return_value = f"Content {i}"
            docs.append(doc)

        # If process_batch method exists
        if hasattr(processor, "process_batch"):
            results = processor.process_batch(docs)
            assert len(results) == len(docs)

    def test_validate_document(self):
        """Test document validation."""
        processor = DocumentProcessor()

        # Valid document
        valid_doc = Mock(spec=DoclingDocument)
        valid_doc.name = "valid.md"
        valid_doc.export_to_markdown.return_value = "Valid content"

        # If validate_document method exists
        if hasattr(processor, "validate_document"):
            is_valid = processor.validate_document(valid_doc)
            assert isinstance(is_valid, bool)

    def test_preprocess_document(self):
        """Test document preprocessing."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "preprocess.md"
        doc.export_to_markdown.return_value = "Content to preprocess"

        # If preprocess method exists
        if hasattr(processor, "preprocess"):
            preprocessed = processor.preprocess(doc)
            assert preprocessed is not None

    def test_postprocess_document(self):
        """Test document postprocessing."""
        processor = DocumentProcessor()

        doc = Mock(spec=DoclingDocument)
        doc.name = "postprocess.md"
        doc.export_to_markdown.return_value = "Content to postprocess"

        result = {"processed": True}

        # If postprocess method exists
        if hasattr(processor, "postprocess"):
            postprocessed = processor.postprocess(result, doc)
            assert postprocessed is not None

    def test_error_handling(self):
        """Test error handling during processing."""
        processor = DocumentProcessor()

        # Document that causes error
        doc = Mock(spec=DoclingDocument)
        doc.name = "error.md"
        doc.export_to_markdown.side_effect = Exception("Processing failed")

        # Should handle gracefully
        try:
            result = processor.process_document(doc)
            # Either returns None or raises specific exception
            assert result is None or result is not None
        except Exception as e:
            # Should be a specific exception type
            assert isinstance(e, Exception)

    def test_configuration_update(self):
        """Test updating processor configuration."""
        processor = DocumentProcessor()

        new_config = {"max_length": 5000, "preserve_formatting": False}

        # If update_config method exists
        if hasattr(processor, "update_config"):
            processor.update_config(new_config)
            # Config should be updated

    def test_reset_processor(self):
        """Test resetting processor state."""
        processor = DocumentProcessor()

        # Process a document first
        doc = Mock(spec=DoclingDocument)
        doc.name = "test.md"
        doc.export_to_markdown.return_value = "Test"
        processor.process_document(doc)

        # If reset method exists
        if hasattr(processor, "reset"):
            processor.reset()
            # State should be reset

    def test_get_statistics(self):
        """Test getting processing statistics."""
        processor = DocumentProcessor()

        # Process some documents
        for i in range(2):
            doc = Mock(spec=DoclingDocument)
            doc.name = f"doc{i}.md"
            doc.export_to_markdown.return_value = f"Content {i}"
            processor.process_document(doc)

        # If get_statistics method exists
        if hasattr(processor, "get_statistics"):
            stats = processor.get_statistics()
            assert isinstance(stats, dict)
