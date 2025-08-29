"""Memory optimization utilities for processing large documents efficiently."""

import gc
import logging
import mmap
import os
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import psutil
from docling_core.types import DoclingDocument

logger = logging.getLogger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""

    total_mb: float
    available_mb: float
    used_mb: float
    percent_used: float
    process_memory_mb: float
    process_memory_percent: float


class MemoryMonitor:
    """Monitor system and process memory usage."""

    def __init__(self) -> None:
        """Initialize memory monitor."""
        self.process = psutil.Process()
        self._baseline_memory: Optional[float] = None

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # System memory
        system_memory = psutil.virtual_memory()

        # Process memory
        process_memory = self.process.memory_info()

        return MemoryStats(
            total_mb=system_memory.total / (1024 * 1024),
            available_mb=system_memory.available / (1024 * 1024),
            used_mb=(system_memory.total - system_memory.available) / (1024 * 1024),
            percent_used=system_memory.percent,
            process_memory_mb=process_memory.rss / (1024 * 1024),
            process_memory_percent=self.process.memory_percent(),
        )

    def set_baseline(self) -> None:
        """Set baseline memory usage for comparison."""
        stats = self.get_memory_stats()
        self._baseline_memory = stats.process_memory_mb
        logger.debug(f"Memory baseline set: {self._baseline_memory:.1f} MB")

    def get_memory_delta(self) -> Optional[float]:
        """Get memory usage change from baseline in MB."""
        if self._baseline_memory is None:
            return None

        current_stats = self.get_memory_stats()
        return current_stats.process_memory_mb - self._baseline_memory

    def log_memory_status(self, operation: str = "memory_check") -> None:
        """Log current memory status."""
        stats = self.get_memory_stats()
        delta = self.get_memory_delta()

        delta_str = f" (Δ{delta:+.1f} MB)" if delta is not None else ""

        logger.info(
            f"{operation}: Process using {stats.process_memory_mb:.1f} MB "
            f"({stats.process_memory_percent:.1f}%){delta_str}, "
            f"System: {stats.used_mb:.1f}/{stats.total_mb:.1f} MB "
            f"({stats.percent_used:.1f}%)"
        )


class MemoryOptimizedDocumentProcessor:
    """Document processor with memory optimization strategies."""

    LARGE_DOCUMENT_THRESHOLD = 50 * 1024 * 1024  # 50MB

    def __init__(
        self,
        use_memory_mapping: bool = True,
        enable_gc_tuning: bool = True,
        chunk_processing: bool = True,
    ) -> None:
        """
        Initialize memory-optimized document processor.

        Args:
            use_memory_mapping: Use memory-mapped files for large documents
            enable_gc_tuning: Enable garbage collection tuning for better performance
            chunk_processing: Enable chunked processing to reduce memory usage
        """
        self.use_memory_mapping = use_memory_mapping
        self.enable_gc_tuning = enable_gc_tuning
        self.chunk_processing = chunk_processing
        self.memory_monitor = MemoryMonitor()

        if self.enable_gc_tuning:
            self._configure_garbage_collection()

        logger.info(
            f"MemoryOptimizedDocumentProcessor initialized: "
            f"memory_mapping={use_memory_mapping}, gc_tuning={enable_gc_tuning}, "
            f"chunk_processing={chunk_processing}"
        )

    def _configure_garbage_collection(self) -> None:
        """Configure garbage collection for better performance with large documents."""
        # Increase garbage collection thresholds for better performance
        # This reduces GC overhead when processing large amounts of data
        gc.set_threshold(2000, 15, 15)  # Default is (700, 10, 10)

        logger.debug(
            "Configured garbage collection thresholds for large document processing"
        )

    @contextmanager
    def memory_optimized_processing(
        self, operation_name: str = "document_processing"
    ) -> Generator[None, None, None]:
        """
        Context manager for memory-optimized processing.

        This context manager:
        - Sets memory baseline
        - Monitors memory usage
        - Triggers garbage collection when needed
        - Logs memory statistics
        """
        self.memory_monitor.set_baseline()
        self.memory_monitor.log_memory_status(f"{operation_name}_start")

        try:
            yield
        finally:
            # Force garbage collection to clean up temporary objects
            if self.enable_gc_tuning:
                collected = gc.collect()
                logger.debug(f"Garbage collection freed {collected} objects")

            self.memory_monitor.log_memory_status(f"{operation_name}_end")

    def should_use_memory_mapping(self, file_path: Union[str, Path]) -> bool:
        """Determine if memory mapping should be used for a file."""
        if not self.use_memory_mapping:
            return False

        try:
            file_size = os.path.getsize(file_path)
            return file_size >= self.LARGE_DOCUMENT_THRESHOLD
        except (OSError, TypeError):
            return False

    @contextmanager
    def memory_mapped_file(
        self, file_path: Union[str, Path]
    ) -> Generator[Optional[mmap.mmap], None, None]:
        """
        Context manager for memory-mapped file access.

        Args:
            file_path: Path to file to memory-map

        Yields:
            Memory-mapped file object or None if mapping fails
        """
        if not self.should_use_memory_mapping(file_path):
            yield None
            return

        try:
            with open(file_path, "rb") as f:
                with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as mm:
                    logger.debug(f"Memory-mapped file: {file_path} ({len(mm)} bytes)")
                    yield mm
        except (OSError, ValueError) as e:
            logger.warning(f"Failed to memory-map {file_path}: {e}")
            yield None

    def optimize_document_copy(self, document: DoclingDocument) -> DoclingDocument:
        """
        Create an optimized copy of a document, minimizing memory usage.

        This method implements selective deep copying, only copying the parts
        that need to be modified during processing.
        """
        logger.debug(f"Creating optimized copy of document {document.name}")

        with self.memory_optimized_processing("document_copy"):
            # Instead of deep copying the entire document, create a new document
            # with references to immutable parts and copies of mutable parts
            import copy

            # Create shallow copy first
            optimized_doc = copy.copy(document)

            # Deep copy only the text-bearing content that might be modified
            if hasattr(document, "texts") and document.texts:
                optimized_doc.texts = copy.deepcopy(document.texts)

            if hasattr(document, "tables") and document.tables:
                optimized_doc.tables = copy.deepcopy(document.tables)

            if hasattr(document, "key_value_items") and document.key_value_items:
                optimized_doc.key_value_items = copy.deepcopy(document.key_value_items)

            # Keep references to immutable content
            # Pictures, forms, and other non-text content don't need deep copying

        return optimized_doc

    def get_memory_recommendations(self) -> list[str]:
        """Get memory optimization recommendations based on current system state."""
        stats = self.get_memory_stats()
        recommendations = []

        # Check system memory pressure
        if stats.percent_used > 80:
            recommendations.append(
                "System memory usage is high (>80%) - consider reducing chunk sizes "
                "or processing fewer documents concurrently"
            )

        # Check process memory usage
        if stats.process_memory_percent > 25:
            recommendations.append(
                "Process memory usage is high (>25%) - consider enabling memory mapping "
                "or garbage collection tuning"
            )

        # Check available memory
        if stats.available_mb < 1024:  # Less than 1GB available
            recommendations.append(
                "Low available system memory (<1GB) - enable chunk processing "
                "and reduce concurrent operations"
            )

        return recommendations

    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        return self.memory_monitor.get_memory_stats()


class StreamingTextExtractor:
    """Text extractor that processes documents in streaming fashion to reduce memory usage."""

    def __init__(
        self,
        buffer_size: int = 1024 * 1024,  # 1MB buffer
        yield_threshold: int = 10240,  # Yield every 10KB
    ) -> None:
        """
        Initialize streaming text extractor.

        Args:
            buffer_size: Size of internal buffer for text processing
            yield_threshold: Minimum text size before yielding a segment
        """
        self.buffer_size = buffer_size
        self.yield_threshold = yield_threshold

        logger.debug(
            f"StreamingTextExtractor initialized: buffer_size={buffer_size}, "
            f"yield_threshold={yield_threshold}"
        )

    def extract_text_streaming(
        self, document: DoclingDocument
    ) -> Generator[str, None, None]:
        """
        Extract text from document in streaming fashion.

        Args:
            document: DoclingDocument to extract text from

        Yields:
            Text segments as they are processed
        """
        current_text = ""

        # Process text items
        for text_item in document.texts:
            if hasattr(text_item, "text") and text_item.text:
                current_text += text_item.text + "\n\n"

                # Yield when we have enough text
                if len(current_text) >= self.yield_threshold:
                    yield current_text
                    current_text = ""

        # Process tables
        for table_item in document.tables:
            if hasattr(table_item, "data") and table_item.data:
                table_text = self._extract_table_text_streaming(table_item)
                current_text += table_text + "\n\n"

                if len(current_text) >= self.yield_threshold:
                    yield current_text
                    current_text = ""

        # Process key-value items
        for kv_item in document.key_value_items:
            kv_text = self._extract_kv_text_streaming(kv_item)
            if kv_text:
                current_text += kv_text + "\n\n"

                if len(current_text) >= self.yield_threshold:
                    yield current_text
                    current_text = ""

        # Yield any remaining text
        if current_text.strip():
            yield current_text

    def _extract_table_text_streaming(self, table_item: Any) -> str:
        """Extract text from table in memory-efficient way."""
        table_text = ""

        if hasattr(table_item, "data") and hasattr(table_item.data, "table_cells"):
            for row in table_item.data.table_cells:
                for cell in row:
                    if hasattr(cell, "text") and cell.text:
                        table_text += cell.text + " "
                table_text += "\n"

        return table_text

    def _extract_kv_text_streaming(self, kv_item: Any) -> str:
        """Extract text from key-value item in memory-efficient way."""
        kv_text = ""

        if hasattr(kv_item, "key") and kv_item.key and hasattr(kv_item.key, "text"):
            kv_text += kv_item.key.text

        if (
            hasattr(kv_item, "value")
            and kv_item.value
            and hasattr(kv_item.value, "text")
        ):
            kv_text += ": " + kv_item.value.text

        return kv_text
