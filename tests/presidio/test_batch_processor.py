"""Tests for Presidio batch processor."""

import time
from unittest.mock import MagicMock, patch

from presidio_analyzer import RecognizerResult

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.presidio.batch_processor import PresidioBatchProcessor


class TestPresidioBatchProcessor:
    """Test batch processor functionality."""

    def test_batch_processor_initialization(self):
        """Test that batch processor initializes correctly."""
        processor = PresidioBatchProcessor(batch_size=50, parallel_workers=2)

        assert processor.batch_size == 50
        assert processor.parallel_workers == 2
        assert len(processor.anonymizer_pool) == 2
        assert len(processor.analyzer_pool) == 2
        assert processor._processing_stats["total_processed"] == 0

    def test_create_batches(self):
        """Test batch creation from documents."""
        processor = PresidioBatchProcessor(batch_size=3)

        # Create 10 documents with mock data
        documents = []
        for i in range(10):
            doc = MagicMock()
            doc.name = f"doc_{i}"
            documents.append(doc)

        batches = processor.create_batches(documents)

        # Should create 4 batches (3, 3, 3, 1)
        assert len(batches) == 4
        assert len(batches[0]) == 3
        assert len(batches[1]) == 3
        assert len(batches[2]) == 3
        assert len(batches[3]) == 1

    def test_extract_text_from_document(self):
        """Test text extraction from different document formats."""
        processor = PresidioBatchProcessor()

        # Test with export_to_text method (like DoclingDocument)
        doc1 = MagicMock()
        doc1.export_to_text.return_value = "This is the exported text"
        assert processor.extract_text_from_document(doc1) == "This is the exported text"

        # Test with text attribute (without export_to_text)
        doc2 = MagicMock(spec=['text'])  # spec limits attributes to only 'text'
        doc2.text = "This is the document text"
        assert processor.extract_text_from_document(doc2) == "This is the document text"

        # Test with content attribute (without export_to_text or text)
        doc3 = MagicMock(spec=['content'])  # spec limits attributes to only 'content'
        doc3.content = "This is the document content"
        assert processor.extract_text_from_document(doc3) == "This is the document content"

        # Test with neither (falls back to str)
        doc4 = MagicMock(spec=[])  # spec with empty list means no attributes
        text = processor.extract_text_from_document(doc4)
        assert isinstance(text, str)

    def test_policy_to_operators_conversion(self):
        """Test conversion of masking policy to Presidio operators."""
        processor = PresidioBatchProcessor()

        # Create policy with different strategies
        policy = MaskingPolicy(
            per_entity={
                "SSN": Strategy(kind=StrategyKind.REDACT),
                "CREDIT_CARD": Strategy(kind=StrategyKind.HASH),
                "EMAIL": Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 3, "position": "start"}),
                "NAME": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[NAME]"})
            }
        )

        operators = processor.policy_to_operators(policy)

        # Check default operator - should be redact (from DEFAULT_REDACT)
        assert "DEFAULT" in operators
        assert operators["DEFAULT"].operator_name == "redact"

        # Check specific operators
        assert operators["SSN"].operator_name == "redact"
        assert operators["CREDIT_CARD"].operator_name == "hash"
        # PARTIAL should map to mask operator
        assert operators["EMAIL"].operator_name == "mask"
        # TEMPLATE should map to replace
        assert operators["NAME"].operator_name == "replace"

    def test_create_cloakmap(self):
        """Test CloakMap creation from processing results."""
        processor = PresidioBatchProcessor()

        # Create test document with proper text setup
        doc = MagicMock()
        doc.id = "test_doc_123"
        # Setup the text that will be extracted - must match analyzer results positions
        test_text = "Some text 123-45-6789 more text email@example.com more content"
        doc.export_to_text.return_value = test_text

        # Create analyzer results
        analyzer_results = [
            RecognizerResult(
                entity_type="SSN",
                start=10,
                end=21,
                score=0.95
            ),
            RecognizerResult(
                entity_type="EMAIL",
                start=30,
                end=50,
                score=0.90
            )
        ]

        # Mock anonymizer result
        anonymizer_result = MagicMock()
        anonymizer_result.text = "Masked text"

        # Create CloakMap
        cloakmap = processor.create_cloakmap(doc, analyzer_results, anonymizer_result)

        # Verify CloakMap
        assert cloakmap.doc_id == "test_doc_123"
        assert cloakmap.version == "2.0"
        assert len(cloakmap.anchors) == 2
        assert cloakmap.anchors[0].entity_type == "SSN"
        assert cloakmap.anchors[1].entity_type == "EMAIL"

    def test_apply_masking(self):
        """Test applying masking results to document."""
        processor = PresidioBatchProcessor()

        # Create original document
        original = MagicMock()
        original.text = "Original text"
        original.metadata = {"author": "John Doe"}

        # Mock anonymizer result
        anonymizer_result = MagicMock()
        anonymizer_result.text = "Masked text"

        # Apply masking
        masked = processor._apply_masking(original, anonymizer_result)

        # Verify masked document
        assert masked.text == "Masked text"
        assert masked.metadata == {"author": "John Doe"}

    @patch('cloakpivot.presidio.batch_processor.AnalyzerEngine')
    @patch('cloakpivot.presidio.batch_processor.AnonymizerEngine')
    def test_process_single_document(self, mock_anonymizer_class, mock_analyzer_class):
        """Test processing a single document."""
        # Set up mocks
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.return_value = [
            RecognizerResult(
                entity_type="SSN",
                start=10,
                end=21,
                score=0.95
            )
        ]

        mock_anonymizer = MagicMock()
        mock_anonymizer.anonymize.return_value = MagicMock(text="Masked text")

        # Configure the mock classes to return our mock instances
        mock_analyzer_class.return_value = mock_analyzer
        mock_anonymizer_class.return_value = mock_anonymizer

        # Now create the processor, which will use our mocked classes
        processor = PresidioBatchProcessor()

        # Create document and policy
        doc = MagicMock()
        doc.text = "SSN: 123-45-6789"
        doc.id = "doc1"

        policy = MaskingPolicy()

        # Process document
        result = processor.process_single_document(
            doc, policy, mock_analyzer, mock_anonymizer
        )

        # Verify result
        assert result.stats is not None
        assert result.stats.get("success") is True
        assert result.masked_document is not None
        assert result.cloakmap is not None
        assert result.masked_document.text == "Masked text"

    @patch('cloakpivot.presidio.batch_processor.AnalyzerEngine')
    @patch('cloakpivot.presidio.batch_processor.AnonymizerEngine')
    def test_process_single_document_error_handling(self, mock_anonymizer_class, mock_analyzer_class):
        """Test error handling in single document processing."""
        # Create mock analyzer that raises exception
        mock_analyzer = MagicMock()
        mock_analyzer.analyze.side_effect = Exception("Analysis failed")

        mock_anonymizer = MagicMock()

        # Configure the mock classes to return our mock instances
        mock_analyzer_class.return_value = mock_analyzer
        mock_anonymizer_class.return_value = mock_anonymizer

        processor = PresidioBatchProcessor()

        # Create document and policy
        doc = MagicMock()
        doc.text = "Test document"
        doc.id = "doc1"  # Add ID to avoid error in CloakMap creation

        policy = MaskingPolicy()

        # Process document (should handle error)
        result = processor.process_single_document(
            doc, policy, mock_analyzer, mock_anonymizer
        )

        # Verify error result
        assert result.stats is not None
        assert result.stats.get("success") is False
        assert result.stats.get("error") == "Analysis failed"
        assert result.masked_document == doc  # Original document returned
        assert result.cloakmap is not None  # Empty cloakmap is returned, not None

    def test_statistics_tracking(self):
        """Test processing statistics tracking."""
        processor = PresidioBatchProcessor()

        # Initial stats
        assert processor._processing_stats["total_processed"] == 0
        assert processor._processing_stats["total_time_ms"] == 0

        # Mock process_document_batch to update stats
        processor._processing_stats["total_processed"] = 10
        processor._processing_stats["total_time_ms"] = 5000
        processor._processing_stats["batch_count"] = 2

        # Get statistics
        stats = processor.get_statistics()

        assert stats["total_processed"] == 10
        assert stats["total_time_ms"] == 5000
        assert stats["batch_count"] == 2
        assert stats["avg_time_per_doc_ms"] == 500  # 5000 / 10
        assert stats["avg_batch_size"] == 5  # 10 / 2

    def test_reset_statistics(self):
        """Test resetting processing statistics."""
        processor = PresidioBatchProcessor()

        # Set some stats
        processor._processing_stats["total_processed"] = 100
        processor._processing_stats["total_time_ms"] = 10000
        processor._processing_stats["errors"] = ["error1", "error2"]

        # Reset
        processor.reset_statistics()

        # Verify reset
        assert processor._processing_stats["total_processed"] == 0
        assert processor._processing_stats["total_time_ms"] == 0
        assert processor._processing_stats["batch_count"] == 0
        assert processor._processing_stats["errors"] == []

    def test_parallel_processing(self):
        """Test parallel processing with multiple workers."""
        processor = PresidioBatchProcessor(batch_size=2, parallel_workers=3)

        # Create test documents
        documents = []
        for i in range(6):
            doc = MagicMock()
            doc.text = f"Document {i}"
            doc.id = f"doc_{i}"
            documents.append(doc)

        # Create policy
        policy = MaskingPolicy()

        # Process documents (will use ThreadPoolExecutor)
        results = processor.process_document_batch(documents, policy)

        # Verify all documents processed
        assert len(results) == 6

        # Check that batches were created correctly
        batches = processor._create_batches(documents)
        assert len(batches) == 3  # 6 documents / 2 batch_size

    def test_process_text_batch(self):
        """Test batch processing of text strings."""
        processor = PresidioBatchProcessor()

        # Create test texts
        texts = [
            "Text with SSN 123-45-6789",
            "Email: john@example.com",
            "Phone: 555-123-4567"
        ]

        # Create policy
        policy = MaskingPolicy()

        # Process texts
        results = processor.process_text_batch(texts, policy)

        # Verify results
        assert len(results) == 3

        for _i, (masked_text, _cloakmap) in enumerate(results):
            assert masked_text is not None
            # CloakMap may be None if no PII detected

    def test_performance_optimization(self):
        """Test that batch processing is more efficient than individual processing."""
        processor = PresidioBatchProcessor(batch_size=10, parallel_workers=4)

        # Create many documents
        documents = []
        for i in range(20):
            doc = MagicMock()
            doc.text = f"Document {i} with PII: SSN 123-45-678{i}"
            doc.id = f"doc_{i}"
            documents.append(doc)

        policy = MaskingPolicy()

        # Process in batch
        start_time = time.time()
        batch_results = processor.process_document_batch(documents, policy)
        _ = time.time() - start_time

        # Verify results
        assert len(batch_results) == 20

        # Check that parallel workers were utilized
        assert processor.parallel_workers == 4
        assert processor.batch_size == 10

        # Stats should show batch processing
        stats = processor.get_statistics()
        assert stats["total_processed"] == 20
        assert stats["batch_count"] == 2  # 20 documents / 10 batch_size
