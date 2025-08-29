"""Parallel analysis engine for efficient PII detection across document chunks."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Any, Optional

from presidio_analyzer import RecognizerResult

from .analyzer import AnalyzerEngineWrapper, AnalyzerConfig, EntityDetectionResult
from .chunking import ChunkBoundary, ChunkedDocumentProcessor
from .policies import MaskingPolicy

logger = logging.getLogger(__name__)


@dataclass
class ChunkAnalysisResult:
    """Result of analyzing a single document chunk."""
    chunk_id: str
    entities: list[RecognizerResult]
    processing_time_ms: float
    chunk_size: int
    error: Optional[str] = None


@dataclass
class ParallelAnalysisResult:
    """Result of parallel analysis across all chunks."""
    entities: list[RecognizerResult]
    chunk_results: list[ChunkAnalysisResult]
    total_processing_time_ms: float
    total_chunks: int
    total_entities: int
    threads_used: int
    performance_stats: dict[str, Any]


class ParallelAnalysisEngine:
    """
    Parallel analysis engine that processes document chunks concurrently.
    
    This engine manages thread pools, coordinates parallel PII analysis,
    and aggregates results while maintaining proper entity ordering.
    """

    def __init__(
        self,
        analyzer_config: Optional[AnalyzerConfig] = None,
        max_workers: Optional[int] = None,
        enable_performance_monitoring: bool = True,
    ) -> None:
        """
        Initialize parallel analysis engine.
        
        Args:
            analyzer_config: Configuration for underlying analyzer
            max_workers: Maximum number of worker threads (None for auto-detect)
            enable_performance_monitoring: Whether to collect detailed performance metrics
        """
        self.analyzer_config = analyzer_config or AnalyzerConfig()
        self.max_workers = max_workers or self._calculate_optimal_workers()
        self.enable_performance_monitoring = enable_performance_monitoring
        
        # Thread-local analyzer instances to avoid contention
        self._analyzer_cache: dict[int, AnalyzerEngineWrapper] = {}
        self._cache_lock = Lock()
        
        self.chunked_processor = ChunkedDocumentProcessor()
        
        logger.info(
            f"ParallelAnalysisEngine initialized with {self.max_workers} workers, "
            f"performance_monitoring={self.enable_performance_monitoring}"
        )

    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of worker threads based on system resources."""
        try:
            cpu_count = os.cpu_count() or 4
        except (AttributeError, OSError):
            cpu_count = 4
        
        # Use heuristic: min(32, cpu_count + 4) for I/O vs CPU bound balance
        # This accounts for I/O wait time during Presidio analysis
        optimal_workers = min(32, cpu_count + 4)
        
        # Allow environment override
        env_workers = os.environ.get("CLOAKPIVOT_MAX_WORKERS")
        if env_workers:
            try:
                env_workers_int = int(env_workers)
                if env_workers_int > 0:
                    optimal_workers = env_workers_int
                    logger.info(f"Using environment-specified max workers: {optimal_workers}")
                else:
                    logger.warning(f"Invalid CLOAKPIVOT_MAX_WORKERS: {env_workers}, using calculated value")
            except ValueError:
                logger.warning(f"Invalid CLOAKPIVOT_MAX_WORKERS format: {env_workers}, using calculated value")
        
        return optimal_workers

    def _get_thread_analyzer(self) -> AnalyzerEngineWrapper:
        """Get thread-local analyzer instance."""
        import threading
        thread_id = threading.get_ident()
        
        with self._cache_lock:
            if thread_id not in self._analyzer_cache:
                # Create new analyzer instance for this thread
                self._analyzer_cache[thread_id] = AnalyzerEngineWrapper(self.analyzer_config)
                logger.debug(f"Created analyzer instance for thread {thread_id}")
            
            return self._analyzer_cache[thread_id]

    def analyze_document_parallel(
        self,
        document,  # DoclingDocument - avoiding import for type hint
        policy: MaskingPolicy,
        chunk_size: Optional[int] = None,
    ) -> ParallelAnalysisResult:
        """
        Analyze document for PII entities using parallel processing.
        
        Args:
            document: DoclingDocument to analyze
            policy: MaskingPolicy defining detection parameters
            chunk_size: Override default chunk size for this analysis
            
        Returns:
            ParallelAnalysisResult with aggregated entities and performance metrics
        """
        import time
        start_time = time.perf_counter()
        
        logger.info(f"Starting parallel analysis of document {document.name}")
        
        # Configure chunked processor if custom chunk size provided
        if chunk_size:
            processor = ChunkedDocumentProcessor(chunk_size=chunk_size)
        else:
            processor = self.chunked_processor
        
        # Create chunks
        chunks = processor.chunk_document(document)
        
        if not chunks:
            logger.warning(f"No chunks created for document {document.name}")
            return ParallelAnalysisResult(
                entities=[],
                chunk_results=[],
                total_processing_time_ms=0.0,
                total_chunks=0,
                total_entities=0,
                threads_used=0,
                performance_stats={},
            )
        
        logger.info(f"Created {len(chunks)} chunks for parallel analysis")
        
        # Analyze chunks in parallel
        chunk_results = self._analyze_chunks_parallel(chunks, policy)
        
        # Aggregate results
        all_entities = []
        for result in chunk_results:
            if result.entities:
                all_entities.extend(result.entities)
        
        # Sort entities by position for deterministic ordering
        all_entities.sort(key=lambda e: (e.start, e.end, e.entity_type))
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        performance_stats = self._calculate_performance_stats(chunk_results, chunks)
        
        logger.info(
            f"Parallel analysis completed: {len(all_entities)} entities found "
            f"in {total_time:.1f}ms using {self.max_workers} threads"
        )
        
        return ParallelAnalysisResult(
            entities=all_entities,
            chunk_results=chunk_results,
            total_processing_time_ms=total_time,
            total_chunks=len(chunks),
            total_entities=len(all_entities),
            threads_used=min(len(chunks), self.max_workers),
            performance_stats=performance_stats,
        )

    def _analyze_chunks_parallel(
        self, chunks: list[ChunkBoundary], policy: MaskingPolicy
    ) -> list[ChunkAnalysisResult]:
        """Analyze chunks in parallel using ThreadPoolExecutor."""
        chunk_results = []
        
        # Prepare analyzer configuration from policy
        analyzer_config = AnalyzerConfig.from_policy(policy)
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk analysis tasks
            future_to_chunk = {}
            
            for chunk in chunks:
                future = executor.submit(
                    self._analyze_single_chunk, chunk, analyzer_config
                )
                future_to_chunk[future] = chunk
            
            # Collect results as they complete
            for future in as_completed(future_to_chunk):
                chunk = future_to_chunk[future]
                try:
                    result = future.result()
                    chunk_results.append(result)
                    
                    if self.enable_performance_monitoring:
                        logger.debug(
                            f"Chunk {result.chunk_id} analyzed: {len(result.entities)} entities "
                            f"in {result.processing_time_ms:.1f}ms"
                        )
                        
                except Exception as e:
                    logger.error(f"Error analyzing chunk {chunk.chunk_id}: {e}")
                    # Create error result
                    error_result = ChunkAnalysisResult(
                        chunk_id=chunk.chunk_id,
                        entities=[],
                        processing_time_ms=0.0,
                        chunk_size=chunk.size,
                        error=str(e),
                    )
                    chunk_results.append(error_result)
        
        # Sort results by chunk_id to maintain order
        chunk_results.sort(key=lambda r: r.chunk_id)
        
        return chunk_results

    def _analyze_single_chunk(
        self, chunk: ChunkBoundary, analyzer_config: AnalyzerConfig
    ) -> ChunkAnalysisResult:
        """Analyze a single chunk for PII entities."""
        import time
        start_time = time.perf_counter()
        
        try:
            # Get thread-local analyzer
            analyzer = self._get_thread_analyzer()
            
            # Extract text from chunk
            chunk_text = self.chunked_processor.extract_chunk_text(chunk)
            
            if not chunk_text.strip():
                return ChunkAnalysisResult(
                    chunk_id=chunk.chunk_id,
                    entities=[],
                    processing_time_ms=0.0,
                    chunk_size=chunk.size,
                )
            
            # Analyze text for entities
            detection_results = analyzer.analyze_text(chunk_text)
            
            # Convert EntityDetectionResult back to RecognizerResult
            # and adjust offsets to document-global coordinates
            recognizer_results = []
            
            for detection in detection_results:
                # Adjust positions to global document coordinates
                global_start = chunk.start_offset + detection.start
                global_end = chunk.start_offset + detection.end
                
                recognizer_result = RecognizerResult(
                    entity_type=detection.entity_type,
                    start=global_start,
                    end=global_end,
                    score=detection.confidence,
                )
                recognizer_results.append(recognizer_result)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return ChunkAnalysisResult(
                chunk_id=chunk.chunk_id,
                entities=recognizer_results,
                processing_time_ms=processing_time,
                chunk_size=chunk.size,
            )
            
        except Exception as e:
            processing_time = (time.perf_counter() - start_time) * 1000
            logger.error(f"Error in chunk analysis {chunk.chunk_id}: {e}")
            
            return ChunkAnalysisResult(
                chunk_id=chunk.chunk_id,
                entities=[],
                processing_time_ms=processing_time,
                chunk_size=chunk.size,
                error=str(e),
            )

    def _calculate_performance_stats(
        self, chunk_results: list[ChunkAnalysisResult], chunks: list[ChunkBoundary]
    ) -> dict[str, Any]:
        """Calculate detailed performance statistics."""
        if not self.enable_performance_monitoring:
            return {}
        
        successful_results = [r for r in chunk_results if r.error is None]
        error_results = [r for r in chunk_results if r.error is not None]
        
        if successful_results:
            processing_times = [r.processing_time_ms for r in successful_results]
            chunk_sizes = [r.chunk_size for r in successful_results]
            entity_counts = [len(r.entities) for r in successful_results]
            
            stats = {
                "successful_chunks": len(successful_results),
                "failed_chunks": len(error_results),
                "average_chunk_processing_time_ms": sum(processing_times) / len(processing_times),
                "min_chunk_processing_time_ms": min(processing_times),
                "max_chunk_processing_time_ms": max(processing_times),
                "average_chunk_size": sum(chunk_sizes) / len(chunk_sizes),
                "average_entities_per_chunk": sum(entity_counts) / len(entity_counts),
                "total_text_processed": sum(chunk_sizes),
                "processing_rate_chars_per_second": (
                    sum(chunk_sizes) / (sum(processing_times) / 1000)
                    if sum(processing_times) > 0
                    else 0
                ),
                "errors": [r.error for r in error_results if r.error],
            }
        else:
            stats = {
                "successful_chunks": 0,
                "failed_chunks": len(error_results),
                "errors": [r.error for r in error_results if r.error],
            }
        
        return stats

    def cleanup_analyzer_cache(self) -> None:
        """Clean up thread-local analyzer cache."""
        with self._cache_lock:
            logger.debug(f"Cleaning up {len(self._analyzer_cache)} cached analyzers")
            self._analyzer_cache.clear()

    def get_performance_recommendations(
        self, analysis_result: ParallelAnalysisResult
    ) -> list[str]:
        """Generate performance optimization recommendations based on analysis results."""
        recommendations = []
        
        if not self.enable_performance_monitoring:
            recommendations.append("Enable performance monitoring for detailed recommendations")
            return recommendations
        
        stats = analysis_result.performance_stats
        
        # Check processing rate
        if stats.get("processing_rate_chars_per_second", 0) < 10000:
            recommendations.append("Consider increasing chunk size or thread count for better throughput")
        
        # Check for failed chunks
        if stats.get("failed_chunks", 0) > 0:
            recommendations.append(
                f"Investigate {stats['failed_chunks']} failed chunks to improve reliability"
            )
        
        # Check thread utilization
        if analysis_result.threads_used < self.max_workers:
            recommendations.append(
                "Consider smaller chunk sizes to better utilize available threads"
            )
        
        # Check processing time variance
        min_time = stats.get("min_chunk_processing_time_ms", 0)
        max_time = stats.get("max_chunk_processing_time_ms", 0)
        if min_time > 0 and (max_time / min_time) > 5:
            recommendations.append(
                "High variance in chunk processing times - consider more uniform chunk sizes"
            )
        
        return recommendations