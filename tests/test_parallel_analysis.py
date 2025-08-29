"""Tests for parallel analysis engine."""

import os
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor

import pytest
from presidio_analyzer import RecognizerResult

from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.core.parallel_analysis import (
    ChunkAnalysisResult,
    ParallelAnalysisEngine,
    ParallelAnalysisResult,
)
from cloakpivot.core.chunking import ChunkBoundary
from cloakpivot.core.policies import MaskingPolicy


class TestChunkAnalysisResult:
    """Test ChunkAnalysisResult dataclass."""

    def test_chunk_analysis_result_creation(self):
        """Test basic ChunkAnalysisResult creation."""
        result = ChunkAnalysisResult(
            chunk_id="chunk_1",
            entities=[],
            processing_time_ms=100.5,
            chunk_size=1024
        )

        assert result.chunk_id == "chunk_1"
        assert result.entities == []
        assert result.processing_time_ms == 100.5
        assert result.chunk_size == 1024
        assert result.error is None

    def test_chunk_analysis_result_with_error(self):
        """Test ChunkAnalysisResult with error."""
        result = ChunkAnalysisResult(
            chunk_id="chunk_1",
            entities=[],
            processing_time_ms=50.0,
            chunk_size=512,
            error="Analysis failed"
        )

        assert result.error == "Analysis failed"


class TestParallelAnalysisResult:
    """Test ParallelAnalysisResult dataclass."""

    def test_parallel_analysis_result_creation(self):
        """Test basic ParallelAnalysisResult creation."""
        chunk_result = ChunkAnalysisResult(
            chunk_id="chunk_1",
            entities=[],
            processing_time_ms=100.0,
            chunk_size=1024
        )

        result = ParallelAnalysisResult(
            entities=[],
            chunk_results=[chunk_result],
            total_processing_time_ms=150.0,
            total_chunks=1,
            total_entities=0,
            threads_used=2,
            performance_stats={}
        )

        assert result.entities == []
        assert len(result.chunk_results) == 1
        assert result.total_processing_time_ms == 150.0
        assert result.total_chunks == 1
        assert result.total_entities == 0
        assert result.threads_used == 2
        assert result.performance_stats == {}


class TestParallelAnalysisEngine:
    """Test ParallelAnalysisEngine functionality."""

    def test_default_initialization(self):
        """Test engine initializes with default configuration."""
        engine = ParallelAnalysisEngine()

        assert engine.analyzer_config is not None
        assert engine.max_workers > 0
        assert engine.enable_performance_monitoring is True
        assert engine._analyzer_cache == {}
        assert engine.chunked_processor is not None

    def test_initialization_with_config(self):
        """Test engine initialization with custom config."""
        config = AnalyzerConfig(language="es", min_confidence=0.8)
        engine = ParallelAnalysisEngine(
            analyzer_config=config,
            max_workers=4,
            enable_performance_monitoring=False
        )

        assert engine.analyzer_config == config
        assert engine.max_workers == 4
        assert engine.enable_performance_monitoring is False

    def test_calculate_optimal_workers_default(self):
        """Test optimal worker calculation with default system values."""
        engine = ParallelAnalysisEngine()
        workers = engine._calculate_optimal_workers()

        # Should be at least 4 workers, at most 32
        assert 4 <= workers <= 32

    @patch.dict(os.environ, {"CLOAKPIVOT_MAX_WORKERS": "8"})
    def test_calculate_optimal_workers_with_env(self):
        """Test optimal worker calculation with environment override."""
        engine = ParallelAnalysisEngine()
        workers = engine._calculate_optimal_workers()

        assert workers == 8

    @patch.dict(os.environ, {"CLOAKPIVOT_MAX_WORKERS": "invalid"})
    def test_calculate_optimal_workers_invalid_env(self):
        """Test optimal worker calculation with invalid environment value."""
        engine = ParallelAnalysisEngine()
        workers = engine._calculate_optimal_workers()

        # Should fall back to calculated value
        assert 4 <= workers <= 32

    @patch.dict(os.environ, {"CLOAKPIVOT_MAX_WORKERS": "-1"})
    def test_calculate_optimal_workers_negative_env(self):
        """Test optimal worker calculation with negative environment value."""
        engine = ParallelAnalysisEngine()
        workers = engine._calculate_optimal_workers()

        # Should fall back to calculated value
        assert 4 <= workers <= 32

    def test_get_thread_analyzer_creates_instance(self):
        """Test thread analyzer instance creation."""
        engine = ParallelAnalysisEngine()

        with patch('cloakpivot.core.parallel_analysis.AnalyzerEngineWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_wrapper.return_value = mock_instance

            analyzer = engine._get_thread_analyzer()

            assert analyzer == mock_instance
            mock_wrapper.assert_called_once_with(engine.analyzer_config)

    def test_get_thread_analyzer_reuses_instance(self):
        """Test thread analyzer instance reuse."""
        engine = ParallelAnalysisEngine()

        with patch('cloakpivot.core.parallel_analysis.AnalyzerEngineWrapper') as mock_wrapper:
            mock_instance = Mock()
            mock_wrapper.return_value = mock_instance

            # First call creates instance
            analyzer1 = engine._get_thread_analyzer()
            # Second call reuses instance
            analyzer2 = engine._get_thread_analyzer()

            assert analyzer1 == analyzer2 == mock_instance
            mock_wrapper.assert_called_once()

    def test_analyze_document_parallel_empty_document(self, simple_document):
        """Test parallel analysis with empty document (no chunks)."""
        engine = ParallelAnalysisEngine()
        policy = MaskingPolicy()

        with patch.object(engine.chunked_processor, 'chunk_document', return_value=[]):
            result = engine.analyze_document_parallel(simple_document, policy)

            assert result.entities == []
            assert result.chunk_results == []
            assert result.total_chunks == 0
            assert result.total_entities == 0
            assert result.threads_used == 0
            assert result.performance_stats == {}

    def test_analyze_document_parallel_with_chunks(self, simple_document):
        """Test parallel analysis with document chunks."""
        engine = ParallelAnalysisEngine()
        policy = MaskingPolicy()

        # Mock chunks
        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=0,
            end_offset=100,
            segments=[],
            cross_chunk_segments=[]
        )

        # Mock chunk analysis result
        mock_entities = [
            RecognizerResult(entity_type="PERSON", start=0, end=8, score=0.9)
        ]
        mock_chunk_result = ChunkAnalysisResult(
            chunk_id="chunk_1",
            entities=mock_entities,
            processing_time_ms=50.0,
            chunk_size=100
        )

        with patch.object(engine.chunked_processor, 'chunk_document', return_value=[mock_chunk]):
            with patch.object(engine, '_analyze_chunks_parallel', return_value=[mock_chunk_result]):
                result = engine.analyze_document_parallel(simple_document, policy)

                assert len(result.entities) == 1
                assert result.entities[0].entity_type == "PERSON"
                assert len(result.chunk_results) == 1
                assert result.total_chunks == 1
                assert result.total_entities == 1
                assert result.threads_used == 1

    def test_analyze_document_parallel_with_custom_chunk_size(self, simple_document):
        """Test parallel analysis with custom chunk size."""
        engine = ParallelAnalysisEngine()
        policy = MaskingPolicy()

        with patch('cloakpivot.core.parallel_analysis.ChunkedDocumentProcessor') as mock_processor_class:
            mock_processor = Mock()
            mock_processor.chunk_document.return_value = []
            mock_processor_class.return_value = mock_processor

            engine.analyze_document_parallel(simple_document, policy, chunk_size=2048)

            # Should create new processor with custom chunk size
            mock_processor_class.assert_called_once_with(chunk_size=2048)

    def test_analyze_chunks_parallel_success(self):
        """Test successful parallel chunk analysis."""
        engine = ParallelAnalysisEngine()
        policy = MaskingPolicy()

        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=0,
            end_offset=100,
            segments=[],
            cross_chunk_segments=[]
        )

        mock_result = ChunkAnalysisResult(
            chunk_id="chunk_1",
            entities=[],
            processing_time_ms=50.0,
            chunk_size=100
        )

        with patch.object(engine, '_analyze_single_chunk', return_value=mock_result):
            results = engine._analyze_chunks_parallel([mock_chunk], policy)

            assert len(results) == 1
            assert results[0] == mock_result

    def test_analyze_chunks_parallel_with_error(self):
        """Test parallel chunk analysis with error handling."""
        engine = ParallelAnalysisEngine()
        policy = MaskingPolicy()

        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=0,
            end_offset=100,
            segments=[],
            cross_chunk_segments=[]
        )

        with patch.object(engine, '_analyze_single_chunk', side_effect=Exception("Test error")):
            results = engine._analyze_chunks_parallel([mock_chunk], policy)

            assert len(results) == 1
            assert results[0].chunk_id == "chunk_1"
            assert results[0].error == "Test error"
            assert results[0].entities == []

    def test_analyze_single_chunk_success(self):
        """Test successful single chunk analysis."""
        engine = ParallelAnalysisEngine()

        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=10,
            end_offset=110,
            segments=[],
            cross_chunk_segments=[]
        )

        config = AnalyzerConfig()

        # Mock analyzer and detection results
        mock_detection = Mock()
        mock_detection.entity_type = "PERSON"
        mock_detection.start = 5
        mock_detection.end = 13
        mock_detection.confidence = 0.9

        mock_analyzer = Mock()
        mock_analyzer.analyze_text.return_value = [mock_detection]

        with patch.object(engine, '_get_thread_analyzer', return_value=mock_analyzer):
            with patch.object(engine.chunked_processor, 'extract_chunk_text', return_value="John Doe"):
                result = engine._analyze_single_chunk(mock_chunk, config)

                assert result.chunk_id == "chunk_1"
                assert len(result.entities) == 1
                # Check global offset adjustment
                assert result.entities[0].start == 15  # 10 + 5
                assert result.entities[0].end == 23    # 10 + 13
                assert result.entities[0].entity_type == "PERSON"
                assert result.entities[0].score == 0.9
                assert result.error is None

    def test_analyze_single_chunk_empty_text(self):
        """Test single chunk analysis with empty text."""
        engine = ParallelAnalysisEngine()

        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=0,
            end_offset=100,
            segments=[],
            cross_chunk_segments=[]
        )

        config = AnalyzerConfig()

        with patch.object(engine.chunked_processor, 'extract_chunk_text', return_value="   "):
            result = engine._analyze_single_chunk(mock_chunk, config)

            assert result.chunk_id == "chunk_1"
            assert result.entities == []
            assert result.processing_time_ms == 0.0
            assert result.error is None

    def test_analyze_single_chunk_with_error(self):
        """Test single chunk analysis with error."""
        engine = ParallelAnalysisEngine()

        mock_chunk = ChunkBoundary(
            chunk_id="chunk_1",
            start_offset=0,
            end_offset=100,
            segments=[],
            cross_chunk_segments=[]
        )

        config = AnalyzerConfig()

        with patch.object(engine, '_get_thread_analyzer', side_effect=Exception("Analysis error")):
            result = engine._analyze_single_chunk(mock_chunk, config)

            assert result.chunk_id == "chunk_1"
            assert result.entities == []
            assert result.error == "Analysis error"

    def test_calculate_performance_stats_disabled(self):
        """Test performance stats calculation when disabled."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=False)

        result = engine._calculate_performance_stats([], [])
        assert result == {}

    def test_calculate_performance_stats_successful_chunks(self):
        """Test performance stats calculation with successful chunks."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        chunk_results = [
            ChunkAnalysisResult("chunk_1", [], 100.0, 1000),
            ChunkAnalysisResult("chunk_2", [Mock()], 200.0, 1500),
        ]

        chunks = [Mock(), Mock()]

        stats = engine._calculate_performance_stats(chunk_results, chunks)

        assert stats["successful_chunks"] == 2
        assert stats["failed_chunks"] == 0
        assert stats["average_chunk_processing_time_ms"] == 150.0
        assert stats["min_chunk_processing_time_ms"] == 100.0
        assert stats["max_chunk_processing_time_ms"] == 200.0
        assert stats["average_chunk_size"] == 1250.0
        assert stats["average_entities_per_chunk"] == 0.5
        assert stats["total_text_processed"] == 2500

    def test_calculate_performance_stats_with_errors(self):
        """Test performance stats calculation with error chunks."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        chunk_results = [
            ChunkAnalysisResult("chunk_1", [], 100.0, 1000),
            ChunkAnalysisResult("chunk_2", [], 0.0, 500, error="Failed"),
        ]

        chunks = [Mock(), Mock()]

        stats = engine._calculate_performance_stats(chunk_results, chunks)

        assert stats["successful_chunks"] == 1
        assert stats["failed_chunks"] == 1
        assert stats["errors"] == ["Failed"]

    def test_calculate_performance_stats_all_failed(self):
        """Test performance stats calculation with all failed chunks."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        chunk_results = [
            ChunkAnalysisResult("chunk_1", [], 0.0, 500, error="Error 1"),
            ChunkAnalysisResult("chunk_2", [], 0.0, 500, error="Error 2"),
        ]

        chunks = [Mock(), Mock()]

        stats = engine._calculate_performance_stats(chunk_results, chunks)

        assert stats["successful_chunks"] == 0
        assert stats["failed_chunks"] == 2
        assert stats["errors"] == ["Error 1", "Error 2"]

    def test_cleanup_analyzer_cache(self):
        """Test analyzer cache cleanup."""
        engine = ParallelAnalysisEngine()

        # Add some mock entries to cache
        engine._analyzer_cache[123] = Mock()
        engine._analyzer_cache[456] = Mock()

        assert len(engine._analyzer_cache) == 2

        engine.cleanup_analyzer_cache()

        assert len(engine._analyzer_cache) == 0

    def test_get_performance_recommendations_disabled(self):
        """Test performance recommendations when monitoring disabled."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=False)

        result = ParallelAnalysisResult(
            entities=[], chunk_results=[], total_processing_time_ms=0.0,
            total_chunks=0, total_entities=0, threads_used=0, performance_stats={}
        )

        recommendations = engine.get_performance_recommendations(result)

        assert len(recommendations) == 1
        assert "Enable performance monitoring" in recommendations[0]

    def test_get_performance_recommendations_low_throughput(self):
        """Test performance recommendations for low throughput."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        result = ParallelAnalysisResult(
            entities=[], chunk_results=[], total_processing_time_ms=100.0,
            total_chunks=1, total_entities=0, threads_used=1,
            performance_stats={"processing_rate_chars_per_second": 5000}
        )

        recommendations = engine.get_performance_recommendations(result)

        assert any("throughput" in r for r in recommendations)

    def test_get_performance_recommendations_failed_chunks(self):
        """Test performance recommendations for failed chunks."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        result = ParallelAnalysisResult(
            entities=[], chunk_results=[], total_processing_time_ms=100.0,
            total_chunks=2, total_entities=0, threads_used=2,
            performance_stats={"failed_chunks": 1}
        )

        recommendations = engine.get_performance_recommendations(result)

        assert any("failed chunks" in r for r in recommendations)

    def test_get_performance_recommendations_underutilized_threads(self):
        """Test performance recommendations for underutilized threads."""
        engine = ParallelAnalysisEngine(max_workers=4, enable_performance_monitoring=True)

        result = ParallelAnalysisResult(
            entities=[], chunk_results=[], total_processing_time_ms=100.0,
            total_chunks=2, total_entities=0, threads_used=2,
            performance_stats={}
        )

        recommendations = engine.get_performance_recommendations(result)

        assert any("utilize available threads" in r for r in recommendations)

    def test_get_performance_recommendations_high_variance(self):
        """Test performance recommendations for high processing time variance."""
        engine = ParallelAnalysisEngine(enable_performance_monitoring=True)

        result = ParallelAnalysisResult(
            entities=[], chunk_results=[], total_processing_time_ms=100.0,
            total_chunks=2, total_entities=0, threads_used=2,
            performance_stats={
                "min_chunk_processing_time_ms": 10.0,
                "max_chunk_processing_time_ms": 100.0
            }
        )

        recommendations = engine.get_performance_recommendations(result)

        assert any("variance" in r for r in recommendations)