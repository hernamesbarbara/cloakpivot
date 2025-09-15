"""Batch processing engine for multi-document operations."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from threading import Lock
from typing import Any, Optional, Protocol

from .error_handling import ErrorCollector
# Performance profiling removed - simplified implementation
from .policies import MaskingPolicy

logger = logging.getLogger(__name__)


class BatchOperationType(Enum):
    """Type of batch operation to perform."""

    MASK = "mask"
    UNMASK = "unmask"
    ANALYZE = "analyze"


class BatchStatus(Enum):
    """Status of batch operation."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class BatchFileItem:
    """Represents a file in a batch operation."""

    file_path: Path
    output_path: Optional[Path] = None
    cloakmap_path: Optional[Path] = None
    status: BatchStatus = BatchStatus.PENDING
    error: Optional[str] = None
    processing_time_ms: float = 0.0
    file_size_bytes: int = 0
    entities_processed: int = 0


@dataclass
class BatchConfig:
    """Configuration for batch operations."""

    # Core settings
    operation_type: BatchOperationType
    input_patterns: list[str]
    output_directory: Optional[Path] = None
    cloakmap_directory: Optional[Path] = None

    # Processing settings
    max_workers: int = 4
    chunk_size: Optional[int] = None
    max_retries: int = 2
    retry_delay_seconds: float = 1.0

    # Resource management
    max_memory_mb: Optional[float] = None
    max_files_per_batch: Optional[int] = None
    throttle_delay_ms: float = 0.0

    # Output settings
    output_format: str = "lexical"
    preserve_directory_structure: bool = True
    overwrite_existing: bool = False

    # Policy and validation
    masking_policy: Optional[MaskingPolicy] = None
    validate_outputs: bool = True

    # Progress and logging
    progress_reporting: bool = True
    verbose_logging: bool = False


@dataclass
class BatchResult:
    """Result of a batch operation."""

    config: BatchConfig
    status: BatchStatus
    start_time: float
    end_time: float
    total_files: int
    successful_files: int
    failed_files: int
    skipped_files: int
    total_processing_time_ms: float
    total_entities_processed: int
    file_results: list[BatchFileItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    performance_stats: dict[str, Any] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Total batch operation duration in milliseconds."""
        return (self.end_time - self.start_time) * 1000

    @property
    def success_rate(self) -> float:
        """Success rate as a percentage."""
        if self.total_files == 0:
            return 0.0
        return (self.successful_files / self.total_files) * 100.0

    @property
    def throughput_files_per_second(self) -> float:
        """Processing throughput in files per second."""
        if self.duration_ms == 0:
            return 0.0
        return self.successful_files / (self.duration_ms / 1000)


class BatchProgressCallback(Protocol):
    """Protocol for batch progress callbacks."""

    def on_batch_start(self, total_files: int) -> None:
        """Called when batch processing starts."""
        ...

    def on_file_start(self, file_item: BatchFileItem, current_index: int) -> None:
        """Called when file processing starts."""
        ...

    def on_file_complete(self, file_item: BatchFileItem, current_index: int) -> None:
        """Called when file processing completes."""
        ...

    def on_file_error(
        self, file_item: BatchFileItem, error: str, current_index: int
    ) -> None:
        """Called when file processing encounters an error."""
        ...

    def on_batch_complete(self, result: BatchResult) -> None:
        """Called when batch processing completes."""
        ...


class DefaultProgressCallback:
    """Default console-based progress callback."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self._start_time = 0.0
        self._last_update_time = 0.0

    def on_batch_start(self, total_files: int) -> None:
        """Called when batch processing starts."""
        self._start_time = time.time()
        logger.info(f"Starting batch processing of {total_files} files")
        if self.verbose:
            print(f"🚀 Starting batch processing of {total_files} files")

    def on_file_start(self, file_item: BatchFileItem, current_index: int) -> None:
        """Called when file processing starts."""
        if self.verbose:
            print(f"📄 Processing {current_index + 1}: {file_item.file_path.name}")

    def on_file_complete(self, file_item: BatchFileItem, current_index: int) -> None:
        """Called when file processing completes."""
        # Throttle console updates to avoid spam
        current_time = time.time()
        if current_time - self._last_update_time >= 1.0:  # Update every second
            elapsed = current_time - self._start_time
            if elapsed > 0:
                rate = (current_index + 1) / elapsed
                print(f"✅ Completed {current_index + 1} files ({rate:.1f} files/sec)")
            self._last_update_time = current_time

    def on_file_error(
        self, file_item: BatchFileItem, error: str, current_index: int
    ) -> None:
        """Called when file processing encounters an error."""
        logger.error(f"Error processing {file_item.file_path}: {error}")
        if self.verbose:
            print(f"❌ Error processing {file_item.file_path.name}: {error}")

    def on_batch_complete(self, result: BatchResult) -> None:
        """Called when batch processing completes."""
        success_rate = result.success_rate
        logger.info(
            f"Batch processing completed: {result.successful_files}/{result.total_files} "
            f"files successful ({success_rate:.1f}%)"
        )
        print("🏁 Batch processing completed!")
        print(f"   Total files: {result.total_files}")
        print(f"   Successful: {result.successful_files}")
        print(f"   Failed: {result.failed_files}")
        print(f"   Success rate: {success_rate:.1f}%")
        print(f"   Duration: {result.duration_ms / 1000:.1f} seconds")
        print(f"   Throughput: {result.throughput_files_per_second:.1f} files/sec")


class BatchProcessor:
    """
    Main batch processing engine for multi-document operations.

    This processor coordinates file discovery, parallel processing, progress reporting,
    error handling, and resource management for batch operations.
    """

    def __init__(
        self,
        config: BatchConfig,
        progress_callback: Optional[BatchProgressCallback] = None,
        # profiler parameter removed - simplified implementation
        error_collector: Optional[ErrorCollector] = None,
    ):
        """
        Initialize batch processor.

        Args:
            config: Batch processing configuration
            progress_callback: Callback for progress reporting (None for default)
            profiler: Performance profiler instance (None to create default)
            error_collector: Error collector instance (None to create default)
        """
        self.config = config
        self.progress_callback = progress_callback or DefaultProgressCallback(
            verbose=config.verbose_logging
        )
        # Performance profiling removed
        self.profiler = None
        self.error_collector = error_collector or ErrorCollector()

        # Legacy compatibility - some tests expect error_handler attribute
        self.error_handler = self.error_collector

        # Processing state
        self._is_cancelled = False
        self._processing_lock = Lock()

        # Resource monitoring
        self._memory_monitor = None
        if config.max_memory_mb:
            try:
                from .memory_optimization import MemoryMonitor

                self._memory_monitor = MemoryMonitor()
            except ImportError:
                logger.warning(
                    "Memory monitoring not available - disabling memory limits"
                )

        logger.info(
            f"BatchProcessor initialized for {config.operation_type.value} operation "
            f"with max_workers={config.max_workers}"
        )

    def discover_files(self) -> list[BatchFileItem]:
        """
        Discover files matching the input patterns.

        Returns:
            List of BatchFileItem objects for discovered files
        """
        import glob

        discovered_files = []

        for pattern in self.config.input_patterns:
            # Handle both absolute and relative patterns
            if Path(pattern).is_absolute():
                matches = glob.glob(str(pattern), recursive=True)
            else:
                matches = glob.glob(pattern, recursive=True)

            for match in matches:
                file_path = Path(match)

                # Skip directories and non-existent files
                if not file_path.is_file():
                    continue

                # Calculate output paths
                output_path = self._calculate_output_path(file_path)
                cloakmap_path = self._calculate_cloakmap_path(file_path)

                # Get file size
                try:
                    file_size = file_path.stat().st_size
                except OSError:
                    logger.warning(f"Cannot access file {file_path} - skipping")
                    continue

                file_item = BatchFileItem(
                    file_path=file_path,
                    output_path=output_path,
                    cloakmap_path=cloakmap_path,
                    file_size_bytes=file_size,
                )

                discovered_files.append(file_item)

        # Apply file limits if configured
        if (
            self.config.max_files_per_batch
            and len(discovered_files) > self.config.max_files_per_batch
        ):
            logger.warning(
                f"Limiting batch to {self.config.max_files_per_batch} files "
                f"(found {len(discovered_files)})"
            )
            discovered_files = discovered_files[: self.config.max_files_per_batch]

        logger.info(f"Discovered {len(discovered_files)} files for batch processing")
        return discovered_files

    def _calculate_output_path(self, input_path: Path) -> Optional[Path]:
        """Calculate output path for a given input file."""
        if not self.config.output_directory:
            return None

        if self.config.preserve_directory_structure:
            # Maintain relative directory structure
            relative_path = input_path.relative_to(input_path.anchor)
            output_path = self.config.output_directory / relative_path
        else:
            # Flat output directory
            output_path = self.config.output_directory / input_path.name

        # Add format suffix if needed
        if self.config.operation_type in [
            BatchOperationType.MASK,
            BatchOperationType.UNMASK,
        ]:
            suffix = f".{self.config.output_format}.json"
            output_path = output_path.with_suffix(suffix)

        return output_path

    def _calculate_cloakmap_path(self, input_path: Path) -> Optional[Path]:
        """Calculate CloakMap path for a given input file."""
        if self.config.operation_type == BatchOperationType.UNMASK:
            # For unmask operations, CloakMap should be in cloakmap_directory
            if self.config.cloakmap_directory:
                if self.config.preserve_directory_structure:
                    relative_path = input_path.relative_to(input_path.anchor)
                    cloakmap_path = self.config.cloakmap_directory / relative_path
                else:
                    cloakmap_path = self.config.cloakmap_directory / input_path.name
                return cloakmap_path.with_suffix(".cloakmap.json")
        elif self.config.operation_type == BatchOperationType.MASK:
            # For mask operations, generate CloakMap alongside output
            if self.config.output_directory:
                output_path = self._calculate_output_path(input_path)
                if output_path:
                    return output_path.with_suffix(".cloakmap.json")

        return None

    def process_batch(self) -> BatchResult:
        """
        Process the entire batch of files.

        Returns:
            BatchResult with comprehensive results and statistics
        """
        start_time = time.time()

        # Discover files
        with self.profiler.measure_operation("batch.file_discovery") as metric:
            files = self.discover_files()
            metric.metadata["files_discovered"] = len(files)

        if not files:
            logger.warning("No files found matching the specified patterns")
            return BatchResult(
                config=self.config,
                status=BatchStatus.COMPLETED,
                start_time=start_time,
                end_time=time.time(),
                total_files=0,
                successful_files=0,
                failed_files=0,
                skipped_files=0,
                total_processing_time_ms=0.0,
                total_entities_processed=0,
            )

        # Create output directories
        self._create_output_directories(files)

        # Check for conflicts with existing files
        if not self.config.overwrite_existing:
            files = self._filter_existing_files(files)

        # Start batch processing
        self.progress_callback.on_batch_start(len(files))

        # Process files in parallel
        with self.profiler.measure_operation("batch.parallel_processing") as metric:
            processed_files = self._process_files_parallel(files)
            metric.metadata["files_processed"] = len(processed_files)

        # Calculate final statistics
        end_time = time.time()
        successful_files = [
            f for f in processed_files if f.status == BatchStatus.COMPLETED
        ]
        failed_files = [f for f in processed_files if f.status == BatchStatus.FAILED]

        total_processing_time = sum(f.processing_time_ms for f in processed_files)
        total_entities = sum(f.entities_processed for f in processed_files)

        result = BatchResult(
            config=self.config,
            status=(
                BatchStatus.CANCELLED if self._is_cancelled else BatchStatus.COMPLETED
            ),
            start_time=start_time,
            end_time=end_time,
            total_files=len(processed_files),
            successful_files=len(successful_files),
            failed_files=len(failed_files),
            skipped_files=len(files) - len(processed_files),
            total_processing_time_ms=total_processing_time,
            total_entities_processed=total_entities,
            file_results=processed_files,
            errors=[f.error for f in failed_files if f.error],
            performance_stats=self.profiler.generate_performance_report(),
        )

        self.progress_callback.on_batch_complete(result)

        logger.info(
            f"Batch processing completed: {result.successful_files}/{result.total_files} "
            f"successful ({result.success_rate:.1f}%)"
        )

        return result

    def _create_output_directories(self, files: list[BatchFileItem]) -> None:
        """Create necessary output directories."""
        directories_to_create = set()

        for file_item in files:
            if file_item.output_path:
                directories_to_create.add(file_item.output_path.parent)
            if file_item.cloakmap_path:
                directories_to_create.add(file_item.cloakmap_path.parent)

        for directory in directories_to_create:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Created directory: {directory}")
            except OSError as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise

    def _filter_existing_files(self, files: list[BatchFileItem]) -> list[BatchFileItem]:
        """Filter out files that would overwrite existing outputs."""
        filtered_files = []

        for file_item in files:
            should_skip = False

            if file_item.output_path and file_item.output_path.exists():
                logger.warning(f"Output file exists, skipping: {file_item.output_path}")
                file_item.status = BatchStatus.FAILED
                file_item.error = "Output file already exists"
                should_skip = True

            if file_item.cloakmap_path and file_item.cloakmap_path.exists():
                logger.warning(
                    f"CloakMap file exists, skipping: {file_item.cloakmap_path}"
                )
                file_item.status = BatchStatus.FAILED
                file_item.error = "CloakMap file already exists"
                should_skip = True

            if not should_skip:
                filtered_files.append(file_item)

        return filtered_files

    def _process_files_parallel(
        self, files: list[BatchFileItem]
    ) -> list[BatchFileItem]:
        """Process files in parallel using ThreadPoolExecutor."""
        processed_files = []

        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all file processing tasks
            future_to_file = {}

            for i, file_item in enumerate(files):
                if self._is_cancelled:
                    break

                future = executor.submit(self._process_single_file, file_item, i)
                future_to_file[future] = (file_item, i)

            # Collect results as they complete
            for future in as_completed(future_to_file):
                file_item, index = future_to_file[future]

                try:
                    result = future.result()
                    processed_files.append(result)

                    if result.status == BatchStatus.COMPLETED:
                        self.progress_callback.on_file_complete(result, index)
                    else:
                        self.progress_callback.on_file_error(
                            result, result.error or "Unknown error", index
                        )

                except Exception as e:
                    logger.error(
                        f"Unexpected error processing {file_item.file_path}: {e}"
                    )
                    file_item.status = BatchStatus.FAILED
                    file_item.error = f"Unexpected error: {e}"
                    processed_files.append(file_item)
                    self.progress_callback.on_file_error(file_item, str(e), index)

                # Check memory limits
                if self._memory_monitor and self.config.max_memory_mb:
                    current_memory = (
                        self._memory_monitor.get_memory_stats().process_memory_mb
                    )
                    if current_memory > self.config.max_memory_mb:
                        logger.warning(
                            f"Memory limit exceeded: {current_memory}MB > {self.config.max_memory_mb}MB"
                        )
                        self.cancel()

                # Apply throttling if configured
                if self.config.throttle_delay_ms > 0:
                    time.sleep(self.config.throttle_delay_ms / 1000.0)

        return processed_files

    def _process_single_file(
        self, file_item: BatchFileItem, index: int
    ) -> BatchFileItem:
        """Process a single file with error handling and retry logic."""
        start_time = time.time()

        self.progress_callback.on_file_start(file_item, index)

        for attempt in range(self.config.max_retries + 1):
            if self._is_cancelled:
                file_item.status = BatchStatus.CANCELLED
                return file_item

            try:
                with self.profiler.measure_operation(
                    f"batch.{self.config.operation_type.value}_file"
                ):
                    # Delegate to specific operation handler
                    if self.config.operation_type == BatchOperationType.MASK:
                        self._process_mask_operation(file_item)
                    elif self.config.operation_type == BatchOperationType.UNMASK:
                        self._process_unmask_operation(file_item)
                    elif self.config.operation_type == BatchOperationType.ANALYZE:
                        self._process_analyze_operation(file_item)
                    else:
                        raise ValueError(
                            f"Unsupported operation type: {self.config.operation_type}"
                        )

                    file_item.status = BatchStatus.COMPLETED
                    break  # Success, exit retry loop

            except Exception as e:
                error_msg = str(e)
                logger.warning(
                    f"Attempt {attempt + 1} failed for {file_item.file_path}: {error_msg}"
                )

                if attempt < self.config.max_retries:
                    # Retry with delay
                    time.sleep(
                        self.config.retry_delay_seconds * (2**attempt)
                    )  # Exponential backoff
                    continue
                else:
                    # Final failure
                    file_item.status = BatchStatus.FAILED
                    file_item.error = error_msg

        file_item.processing_time_ms = (time.time() - start_time) * 1000
        return file_item

    def _process_mask_operation(self, file_item: BatchFileItem) -> None:
        """Process a single file for masking operation."""
        import json

        from ..core.detection import EntityDetectionPipeline
        from ..document.extractor import TextExtractor
        from ..document.processor import DocumentProcessor
        from ..masking.engine import MaskingEngine

        # Load document
        processor = DocumentProcessor()
        document = processor.load_document(file_item.file_path, validate=True)

        # Detect entities
        detection_pipeline = EntityDetectionPipeline()
        detection_result = detection_pipeline.analyze_document(
            document, self.config.masking_policy or MaskingPolicy()
        )

        # Convert to RecognizerResult format
        from presidio_analyzer import RecognizerResult

        entities = []
        for segment_result in detection_result.segment_results:
            for entity in segment_result.entities:
                recognizer_result = RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start + segment_result.segment.start_offset,
                    end=entity.end + segment_result.segment.start_offset,
                    score=entity.confidence,
                )
                entities.append(recognizer_result)

        # Mask document
        masking_engine = MaskingEngine()
        text_extractor = TextExtractor()
        text_segments = text_extractor.extract_text_segments(document)

        masking_result = masking_engine.mask_document(
            document=document,
            entities=entities,
            policy=self.config.masking_policy or MaskingPolicy(),
            text_segments=text_segments,
        )

        # Save masked document
        if file_item.output_path:
            from docpivot import DocPivotEngine

            engine = DocPivotEngine()
            result = engine.convert_to_lexical(masking_result.masked_document)

            with open(file_item.output_path, "w", encoding="utf-8") as f:
                f.write(result.content)

        # Save CloakMap
        if file_item.cloakmap_path:
            with open(file_item.cloakmap_path, "w", encoding="utf-8") as f:
                json.dump(masking_result.cloakmap.to_dict(), f, indent=2, default=str)

        file_item.entities_processed = len(entities)

    def _process_unmask_operation(self, file_item: BatchFileItem) -> None:
        """Process a single file for unmasking operation."""
        import json

        from ..core.cloakmap import CloakMap
        from ..document.processor import DocumentProcessor
        from ..unmasking.engine import UnmaskingEngine

        if not file_item.cloakmap_path or not file_item.cloakmap_path.exists():
            raise FileNotFoundError(f"CloakMap not found: {file_item.cloakmap_path}")

        # Load CloakMap
        with open(file_item.cloakmap_path, encoding="utf-8") as f:
            cloakmap_data = json.load(f)
        cloakmap = CloakMap.from_dict(cloakmap_data)

        # Load masked document
        processor = DocumentProcessor()
        masked_document = processor.load_document(file_item.file_path, validate=True)

        # Unmask document
        unmasking_engine = UnmaskingEngine()
        unmasking_result = unmasking_engine.unmask_document(
            masked_document=masked_document,
            cloakmap=cloakmap,
            verify_integrity=self.config.validate_outputs,
        )

        # Save restored document
        if file_item.output_path:
            from docpivot import DocPivotEngine

            engine = DocPivotEngine()
            result = engine.convert_to_lexical(unmasking_result.restored_document)

            with open(file_item.output_path, "w", encoding="utf-8") as f:
                f.write(result.content)

        file_item.entities_processed = len(cloakmap.anchors)

    def _process_analyze_operation(self, file_item: BatchFileItem) -> None:
        """Process a single file for analysis operation."""
        import json

        from ..core.detection import EntityDetectionPipeline
        from ..document.processor import DocumentProcessor

        # Load document
        processor = DocumentProcessor()
        document = processor.load_document(file_item.file_path, validate=True)

        # Detect entities
        detection_pipeline = EntityDetectionPipeline()
        detection_result = detection_pipeline.analyze_document(
            document, self.config.masking_policy or MaskingPolicy()
        )

        file_item.entities_processed = detection_result.total_entities

        # Save analysis results if output path specified
        if file_item.output_path:
            analysis_data = {
                "file_path": str(file_item.file_path),
                "total_entities": detection_result.total_entities,
                "entity_breakdown": detection_result.entity_breakdown,
                "segment_results": [
                    {
                        "segment_id": segment.segment.node_id,
                        "start_offset": segment.segment.start_offset,
                        "end_offset": segment.segment.end_offset,
                        "entities": [
                            {
                                "entity_type": entity.entity_type,
                                "start": entity.start,
                                "end": entity.end,
                                "confidence": entity.confidence,
                                "text": entity.text,
                            }
                            for entity in segment.entities
                        ],
                    }
                    for segment in detection_result.segment_results
                ],
                "analysis_timestamp": time.time(),
            }

            with open(file_item.output_path, "w", encoding="utf-8") as f:
                json.dump(analysis_data, f, indent=2, default=str)

    def cancel(self) -> None:
        """Cancel the batch processing operation."""
        with self._processing_lock:
            self._is_cancelled = True
            logger.info("Batch processing cancelled by user request")

    @property
    def is_cancelled(self) -> bool:
        """Check if batch processing has been cancelled."""
        return self._is_cancelled
