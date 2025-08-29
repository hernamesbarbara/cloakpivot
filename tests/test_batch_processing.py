"""Tests for batch processing functionality."""

import json
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from cloakpivot.core.batch import (
    BatchConfig,
    BatchFileItem,
    BatchOperationType,
    BatchProcessor,
    BatchResult,
    BatchStatus,
    DefaultProgressCallback,
)
from cloakpivot.core.policies import MaskingPolicy


class TestBatchConfig:
    """Tests for BatchConfig configuration class."""
    
    def test_batch_config_creation(self):
        """Test creating a basic batch configuration."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("./output")
        )
        
        assert config.operation_type == BatchOperationType.MASK
        assert config.input_patterns == ["*.pdf"]
        assert config.output_directory == Path("./output")
        assert config.max_workers == 4  # Default value
        assert config.max_retries == 2  # Default value
        assert config.overwrite_existing is False  # Default value
    
    def test_batch_config_with_all_options(self):
        """Test creating a configuration with all options specified."""
        masking_policy = MaskingPolicy()
        
        config = BatchConfig(
            operation_type=BatchOperationType.UNMASK,
            input_patterns=["data/*.json", "docs/**/*.pdf"],
            output_directory=Path("./output"),
            cloakmap_directory=Path("./cloakmaps"),
            max_workers=8,
            chunk_size=1000,
            max_retries=3,
            retry_delay_seconds=2.0,
            max_memory_mb=2048.0,
            max_files_per_batch=100,
            throttle_delay_ms=50.0,
            output_format="docling",
            preserve_directory_structure=False,
            overwrite_existing=True,
            masking_policy=masking_policy,
            validate_outputs=False,
            progress_reporting=False,
            verbose_logging=True,
        )
        
        assert config.operation_type == BatchOperationType.UNMASK
        assert len(config.input_patterns) == 2
        assert config.output_directory == Path("./output")
        assert config.cloakmap_directory == Path("./cloakmaps")
        assert config.max_workers == 8
        assert config.chunk_size == 1000
        assert config.max_retries == 3
        assert config.retry_delay_seconds == 2.0
        assert config.max_memory_mb == 2048.0
        assert config.max_files_per_batch == 100
        assert config.throttle_delay_ms == 50.0
        assert config.output_format == "docling"
        assert config.preserve_directory_structure is False
        assert config.overwrite_existing is True
        assert config.masking_policy == masking_policy
        assert config.validate_outputs is False
        assert config.progress_reporting is False
        assert config.verbose_logging is True


class TestBatchFileItem:
    """Tests for BatchFileItem data class."""
    
    def test_batch_file_item_creation(self):
        """Test creating a BatchFileItem."""
        file_path = Path("test.pdf")
        output_path = Path("output/test.masked.json")
        cloakmap_path = Path("output/test.cloakmap.json")
        
        item = BatchFileItem(
            file_path=file_path,
            output_path=output_path,
            cloakmap_path=cloakmap_path,
            file_size_bytes=1024
        )
        
        assert item.file_path == file_path
        assert item.output_path == output_path
        assert item.cloakmap_path == cloakmap_path
        assert item.status == BatchStatus.PENDING  # Default
        assert item.error is None
        assert item.processing_time_ms == 0.0
        assert item.file_size_bytes == 1024
        assert item.entities_processed == 0
    
    def test_batch_file_item_with_error(self):
        """Test BatchFileItem with error status."""
        item = BatchFileItem(
            file_path=Path("test.pdf"),
            status=BatchStatus.FAILED,
            error="File not found",
            processing_time_ms=250.5,
            entities_processed=0
        )
        
        assert item.status == BatchStatus.FAILED
        assert item.error == "File not found"
        assert item.processing_time_ms == 250.5
        assert item.entities_processed == 0


class TestBatchResult:
    """Tests for BatchResult data class."""
    
    def test_batch_result_properties(self):
        """Test BatchResult calculated properties."""
        start_time = time.time()
        end_time = start_time + 10.5  # 10.5 seconds later
        
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("./output")
        )
        
        result = BatchResult(
            config=config,
            status=BatchStatus.COMPLETED,
            start_time=start_time,
            end_time=end_time,
            total_files=10,
            successful_files=8,
            failed_files=2,
            skipped_files=0,
            total_processing_time_ms=8000.0,
            total_entities_processed=150
        )
        
        # Test calculated properties
        assert result.duration_ms == pytest.approx(10500.0, rel=1e-3)
        assert result.success_rate == 80.0  # 8/10 * 100
        assert result.throughput_files_per_second == pytest.approx(0.762, rel=1e-2)  # 8 successful / 10.5 seconds
    
    def test_batch_result_zero_files(self):
        """Test BatchResult with zero files."""
        config = BatchConfig(
            operation_type=BatchOperationType.ANALYZE,
            input_patterns=["*.pdf"],
        )
        
        result = BatchResult(
            config=config,
            status=BatchStatus.COMPLETED,
            start_time=time.time(),
            end_time=time.time(),
            total_files=0,
            successful_files=0,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=0.0,
            total_entities_processed=0
        )
        
        assert result.success_rate == 0.0
        assert result.throughput_files_per_second == 0.0


class TestDefaultProgressCallback:
    """Tests for DefaultProgressCallback."""
    
    def test_progress_callback_creation(self):
        """Test creating a progress callback."""
        callback = DefaultProgressCallback(verbose=True)
        assert callback.verbose is True
    
    def test_batch_start_callback(self, capsys):
        """Test batch start callback output."""
        callback = DefaultProgressCallback(verbose=True)
        callback.on_batch_start(5)
        
        captured = capsys.readouterr()
        assert "Starting batch processing of 5 files" in captured.out
    
    def test_file_complete_callback(self, capsys):
        """Test file complete callback with throttling."""
        callback = DefaultProgressCallback(verbose=True)
        callback._start_time = time.time() - 2.0  # 2 seconds ago
        callback._last_update_time = 0.0  # Force update
        
        file_item = BatchFileItem(file_path=Path("test1.pdf"))
        callback.on_file_complete(file_item, 0)
        
        captured = capsys.readouterr()
        assert "Completed 1 files" in captured.out
    
    def test_file_error_callback(self, capsys):
        """Test file error callback output."""
        callback = DefaultProgressCallback(verbose=True)
        
        file_item = BatchFileItem(file_path=Path("test.pdf"))
        callback.on_file_error(file_item, "Test error", 0)
        
        captured = capsys.readouterr()
        assert "Error processing test.pdf: Test error" in captured.out
    
    def test_batch_complete_callback(self, capsys):
        """Test batch complete callback output."""
        callback = DefaultProgressCallback(verbose=True)
        
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
        )
        
        result = BatchResult(
            config=config,
            status=BatchStatus.COMPLETED,
            start_time=time.time() - 5.0,
            end_time=time.time(),
            total_files=10,
            successful_files=8,
            failed_files=2,
            skipped_files=0,
            total_processing_time_ms=4000.0,
            total_entities_processed=100
        )
        
        callback.on_batch_complete(result)
        
        captured = capsys.readouterr()
        assert "Batch processing completed!" in captured.out
        assert "Total files: 10" in captured.out
        assert "Successful: 8" in captured.out
        assert "Failed: 2" in captured.out
        assert "Success rate: 80.0%" in captured.out


class TestBatchProcessor:
    """Tests for BatchProcessor main class."""
    
    def test_batch_processor_creation(self):
        """Test creating a BatchProcessor."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("./output")
        )
        
        processor = BatchProcessor(config)
        
        assert processor.config == config
        assert processor.progress_callback is not None
        assert processor.profiler is not None
        assert processor.error_handler is not None
        assert processor._is_cancelled is False
    
    @patch('glob.glob')
    def test_discover_files(self, mock_glob):
        """Test file discovery functionality."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf", "data/*.json"],
            output_directory=Path("./output")
        )
        
        # Mock glob results
        mock_glob.side_effect = [
            ["file1.pdf", "file2.pdf"],  # First pattern
            ["data/doc1.json", "data/doc2.json"],  # Second pattern
        ]
        
        # Mock Path.is_file() and Path.stat()
        with patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat') as mock_stat:
            
            mock_stat.return_value.st_size = 1024
            
            processor = BatchProcessor(config)
            files = processor.discover_files()
            
            assert len(files) == 4
            assert all(isinstance(f, BatchFileItem) for f in files)
            assert all(f.file_size_bytes == 1024 for f in files)
            
            # Check that output paths are calculated
            assert all(f.output_path is not None for f in files)
            assert all(f.cloakmap_path is not None for f in files)
    
    @patch('glob.glob')
    def test_discover_files_with_limit(self, mock_glob):
        """Test file discovery with file limit."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("./output"),
            max_files_per_batch=2
        )
        
        mock_glob.return_value = ["file1.pdf", "file2.pdf", "file3.pdf", "file4.pdf"]
        
        with patch.object(Path, 'is_file', return_value=True), \
             patch.object(Path, 'stat') as mock_stat:
            
            mock_stat.return_value.st_size = 1024
            
            processor = BatchProcessor(config)
            files = processor.discover_files()
            
            assert len(files) == 2  # Limited by max_files_per_batch
    
    def test_calculate_output_path(self):
        """Test output path calculation."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("/output"),
            preserve_directory_structure=True,
            output_format="lexical"
        )
        
        processor = BatchProcessor(config)
        
        input_path = Path("/data/subdir/document.pdf")
        output_path = processor._calculate_output_path(input_path)
        
        # Should preserve structure and add format suffix
        expected = Path("/output/data/subdir/document.lexical.json")
        assert output_path == expected
    
    def test_calculate_output_path_flat(self):
        """Test output path calculation with flat structure."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("/output"),
            preserve_directory_structure=False,
            output_format="docling"
        )
        
        processor = BatchProcessor(config)
        
        input_path = Path("/data/subdir/document.pdf")
        output_path = processor._calculate_output_path(input_path)
        
        # Should flatten structure
        expected = Path("/output/document.docling.json")
        assert output_path == expected
    
    def test_calculate_cloakmap_path_mask(self):
        """Test CloakMap path calculation for mask operations."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("/output"),
            preserve_directory_structure=False,
        )
        
        processor = BatchProcessor(config)
        
        input_path = Path("/data/document.pdf")
        cloakmap_path = processor._calculate_cloakmap_path(input_path)
        
        expected = Path("/output/document.lexical.cloakmap.json")
        assert cloakmap_path == expected
    
    def test_calculate_cloakmap_path_unmask(self):
        """Test CloakMap path calculation for unmask operations."""
        config = BatchConfig(
            operation_type=BatchOperationType.UNMASK,
            input_patterns=["*.json"],
            cloakmap_directory=Path("/cloakmaps"),
        )
        
        processor = BatchProcessor(config)
        
        input_path = Path("/masked/document.json")
        cloakmap_path = processor._calculate_cloakmap_path(input_path)
        
        expected = Path("/cloakmaps/masked/document.cloakmap.json")
        assert cloakmap_path == expected
    
    def test_filter_existing_files(self):
        """Test filtering out existing files when overwrite is disabled."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
            output_directory=Path("/output"),
            overwrite_existing=False
        )
        
        processor = BatchProcessor(config)
        
        # Create test file items
        file1 = BatchFileItem(
            file_path=Path("/data/file1.pdf"),
            output_path=Path("/output/file1.json")
        )
        file2 = BatchFileItem(
            file_path=Path("/data/file2.pdf"),
            output_path=Path("/output/file2.json")
        )
        
        files = [file1, file2]
        
        # Mock file1 output exists, file2 doesn't
        def mock_exists(path_instance):
            return "file1" in str(path_instance)
        
        with patch.object(Path, 'exists', mock_exists):
            filtered_files = processor._filter_existing_files(files)
            
            assert len(filtered_files) == 1
            assert filtered_files[0] == file2
            assert file1.status == BatchStatus.FAILED
            assert "already exists" in file1.error
    
    def test_cancellation(self):
        """Test batch processing cancellation."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=["*.pdf"],
        )
        
        processor = BatchProcessor(config)
        
        assert processor.is_cancelled is False
        
        processor.cancel()
        
        assert processor.is_cancelled is True


class TestBatchProcessorIntegration:
    """Integration tests for BatchProcessor."""
    
    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create input directory with test files
            input_dir = workspace / "input"
            input_dir.mkdir()
            
            # Create sample input files
            (input_dir / "doc1.json").write_text('{"text": "John Doe lives at 123 Main St."}')
            (input_dir / "doc2.json").write_text('{"text": "Contact: jane@example.com or (555) 123-4567"}')
            (input_dir / "subdir").mkdir()
            (input_dir / "subdir" / "doc3.json").write_text('{"text": "SSN: 123-45-6789"}')
            
            # Create output directories
            output_dir = workspace / "output"
            output_dir.mkdir()
            cloakmap_dir = workspace / "cloakmaps"
            cloakmap_dir.mkdir()
            
            yield {
                'workspace': workspace,
                'input_dir': input_dir,
                'output_dir': output_dir,
                'cloakmap_dir': cloakmap_dir,
            }
    
    def test_process_batch_empty_patterns(self, temp_workspace):
        """Test batch processing with patterns that match no files."""
        config = BatchConfig(
            operation_type=BatchOperationType.ANALYZE,
            input_patterns=[str(temp_workspace['input_dir'] / "*.nonexistent")],
        )
        
        processor = BatchProcessor(config)
        result = processor.process_batch()
        
        assert result.status == BatchStatus.COMPLETED
        assert result.total_files == 0
        assert result.successful_files == 0
        assert result.failed_files == 0
    
    @patch('cloakpivot.core.batch.BatchProcessor._process_mask_operation')
    @patch('cloakpivot.core.batch.BatchProcessor._create_output_directories')
    def test_process_batch_mock_operations(self, mock_create_dirs, mock_mask_op, temp_workspace):
        """Test batch processing with mocked operations."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=[str(temp_workspace['input_dir'] / "**/*.json")],
            output_directory=temp_workspace['output_dir'],
            cloakmap_directory=temp_workspace['cloakmap_dir'],
            max_workers=1,  # Single worker for predictable testing
        )
        
        # Mock the mask operation to succeed
        def mock_mask_operation(file_item):
            file_item.entities_processed = 5
            time.sleep(0.01)  # Simulate processing time
        
        mock_mask_op.side_effect = mock_mask_operation
        
        processor = BatchProcessor(config)
        result = processor.process_batch()
        
        assert result.status == BatchStatus.COMPLETED
        assert result.total_files == 3  # doc1.json, doc2.json, subdir/doc3.json
        assert result.successful_files == 3
        assert result.failed_files == 0
        assert result.total_entities_processed == 15  # 5 per file * 3 files
        assert result.success_rate == 100.0
        
        # Check that create directories was called
        mock_create_dirs.assert_called_once()
        
        # Check that mask operation was called for each file
        assert mock_mask_op.call_count == 3
    
    @patch('cloakpivot.core.batch.BatchProcessor._process_mask_operation')
    def test_process_batch_with_failures(self, mock_mask_op, temp_workspace):
        """Test batch processing with some file failures."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=[str(temp_workspace['input_dir'] / "**/*.json")],
            output_directory=temp_workspace['output_dir'],
            max_workers=1,
            max_retries=1,
        )
        
        # Mock the mask operation to fail on specific files
        def mock_mask_operation(file_item):
            if "doc2" in str(file_item.file_path):
                raise ValueError("Simulated processing error")
            file_item.entities_processed = 3
        
        mock_mask_op.side_effect = mock_mask_operation
        
        processor = BatchProcessor(config)
        result = processor.process_batch()
        
        assert result.status == BatchStatus.COMPLETED  # Batch completes despite individual failures
        assert result.total_files == 3
        assert result.successful_files == 2  # doc1 and doc3 succeed
        assert result.failed_files == 1   # doc2 fails
        assert result.total_entities_processed == 6  # 3 per successful file * 2 files
        assert result.success_rate == pytest.approx(66.67, rel=1e-2)
        
        # Check that failed file has error information
        failed_files = [f for f in result.file_results if f.status == BatchStatus.FAILED]
        assert len(failed_files) == 1
        assert "Simulated processing error" in failed_files[0].error
    
    @patch('cloakpivot.core.batch.BatchProcessor._process_mask_operation')
    def test_process_batch_with_retries(self, mock_mask_op, temp_workspace):
        """Test batch processing retry mechanism."""
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=[str(temp_workspace['input_dir'] / "doc1.json")],
            output_directory=temp_workspace['output_dir'],
            max_workers=1,
            max_retries=2,
            retry_delay_seconds=0.01,  # Very short delay for testing
        )
        
        # Mock the operation to fail first time, succeed second time
        call_count = 0
        def mock_mask_operation(file_item):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ValueError("Transient error")
            file_item.entities_processed = 4
        
        mock_mask_op.side_effect = mock_mask_operation
        
        processor = BatchProcessor(config)
        result = processor.process_batch()
        
        assert result.status == BatchStatus.COMPLETED
        assert result.total_files == 1
        assert result.successful_files == 1  # Eventually succeeds
        assert result.failed_files == 0
        assert mock_mask_op.call_count == 2  # Called twice (first failure, then success)


if __name__ == "__main__":
    pytest.main([__file__])