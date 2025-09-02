"""Tests for batch CLI commands."""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from click.testing import CliRunner

from cloakpivot.cli.batch import batch
from cloakpivot.core.batch import BatchOperationType, BatchResult, BatchStatus
from cloakpivot.core.policies import MaskingPolicy


class TestBatchCLI:
    """Tests for batch CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Create a CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_workspace(self):
        """Create a temporary workspace for CLI testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)

            # Create input files
            input_dir = workspace / "input"
            input_dir.mkdir()

            (input_dir / "doc1.pdf").write_text("Sample document 1")
            (input_dir / "doc2.pdf").write_text("Sample document 2")
            (input_dir / "subdir").mkdir()
            (input_dir / "subdir" / "doc3.pdf").write_text("Sample document 3")

            # Create output directories
            output_dir = workspace / "output"
            output_dir.mkdir()
            cloakmap_dir = workspace / "cloakmaps"
            cloakmap_dir.mkdir()

            # Create a sample policy file
            policy_file = workspace / "policy.yaml"
            policy_file.write_text("""
locale: "en"
default_strategy:
  kind: "redact"
  parameters:
    redact_char: "*"
per_entity:
  PERSON:
    kind: "hash"
    threshold: 0.8
""")

            yield {
                "workspace": workspace,
                "input_dir": input_dir,
                "output_dir": output_dir,
                "cloakmap_dir": cloakmap_dir,
                "policy_file": policy_file,
            }

    def test_batch_mask_help(self, cli_runner):
        """Test batch mask command help."""
        result = cli_runner.invoke(batch, ["mask", "--help"])

        assert result.exit_code == 0
        assert "Batch mask PII in multiple documents" in result.output
        assert "--out-dir" in result.output
        assert "--policy" in result.output
        assert "--max-workers" in result.output

    def test_batch_unmask_help(self, cli_runner):
        """Test batch unmask command help."""
        result = cli_runner.invoke(batch, ["unmask", "--help"])

        assert result.exit_code == 0
        assert "Batch unmask previously masked documents" in result.output
        assert "--cloakmap-dir" in result.output
        assert "--out-dir" in result.output
        assert "--verify-integrity" in result.output

    def test_batch_analyze_help(self, cli_runner):
        """Test batch analyze command help."""
        result = cli_runner.invoke(batch, ["analyze", "--help"])

        assert result.exit_code == 0
        assert "Batch analyze documents for PII" in result.output
        assert "--summary-only" in result.output
        assert "--out-dir" in result.output

    def test_batch_config_sample(self, cli_runner):
        """Test batch config sample command."""
        result = cli_runner.invoke(batch, ["config-sample"])

        assert result.exit_code == 0
        assert "CloakPivot Batch Processing Configuration Options" in result.output
        assert "operation_type" in result.output
        assert "max_workers" in result.output

    def test_batch_config_sample_yaml(self, cli_runner):
        """Test batch config sample with YAML format."""
        result = cli_runner.invoke(batch, ["config-sample", "--format", "yaml"])

        assert result.exit_code == 0
        assert "# CloakPivot Batch Processing Configuration Sample" in result.output
        assert 'operation_type: "mask"' in result.output
        assert "input_patterns:" in result.output

    def test_batch_config_sample_json(self, cli_runner):
        """Test batch config sample with JSON format."""
        result = cli_runner.invoke(batch, ["config-sample", "--format", "json"])

        assert result.exit_code == 0

        # Should be valid JSON
        try:
            json.loads(result.output)
        except json.JSONDecodeError:
            pytest.fail("Output is not valid JSON")

        assert '"operation_type": "mask"' in result.output
        assert '"input_patterns"' in result.output

    def test_batch_mask_missing_arguments(self, cli_runner):
        """Test batch mask with missing required arguments."""
        result = cli_runner.invoke(batch, ["mask"])

        assert result.exit_code != 0
        assert "Missing argument" in result.output or "Usage:" in result.output

    def test_batch_mask_missing_output_dir(self, cli_runner):
        """Test batch mask with missing output directory."""
        result = cli_runner.invoke(batch, ["mask", "*.pdf"])

        assert result.exit_code != 0
        assert "Missing option" in result.output or "--out-dir" in result.output

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_mask_basic(self, mock_processor_class, cli_runner, temp_workspace):
        """Test basic batch mask command."""
        # Mock successful batch processing
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1005.0,
            total_files=2,
            successful_files=2,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=4000.0,
            total_entities_processed=10,
        )
        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "*.pdf")
        output_dir = str(temp_workspace["output_dir"])

        result = cli_runner.invoke(
            batch,
            [
                "mask",
                input_pattern,
                "--out-dir",
                output_dir,
            ],
        )

        assert result.exit_code == 0
        mock_processor_class.assert_called_once()
        mock_processor.process_batch.assert_called_once()

        # Check that the configuration was created correctly
        call_args = mock_processor_class.call_args[0][
            0
        ]  # First positional argument (config)
        assert call_args.operation_type == BatchOperationType.MASK
        assert call_args.output_directory == Path(output_dir)

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_mask_with_all_options(
        self, mock_processor_class, cli_runner, temp_workspace
    ):
        """Test batch mask command with all options specified."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1010.0,
            total_files=3,
            successful_files=3,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=8000.0,
            total_entities_processed=25,
        )
        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "**/*.pdf")
        output_dir = str(temp_workspace["output_dir"])
        cloakmap_dir = str(temp_workspace["cloakmap_dir"])
        policy_file = str(temp_workspace["policy_file"])

        with patch("cloakpivot.cli.batch._load_masking_policy") as mock_load_policy:
            mock_load_policy.return_value = MaskingPolicy()

            result = cli_runner.invoke(
                batch,
                [
                    "mask",
                    input_pattern,
                    "--out-dir",
                    output_dir,
                    "--cloakmap-dir",
                    cloakmap_dir,
                    "--policy",
                    policy_file,
                    "--format",
                    "docling",
                    "--max-workers",
                    "8",
                    "--max-files",
                    "100",
                    "--max-retries",
                    "3",
                    "--overwrite",
                    "--throttle-delay",
                    "0.1",
                    "--max-memory",
                    "2048",
                    "--verbose",
                ],
            )

            assert result.exit_code == 0

            # Verify configuration
            call_args = mock_processor_class.call_args[0][0]
            assert call_args.operation_type == BatchOperationType.MASK
            assert call_args.output_directory == Path(output_dir)
            assert call_args.cloakmap_directory == Path(cloakmap_dir)
            assert call_args.output_format == "docling"
            assert call_args.max_workers == 8
            assert call_args.max_files_per_batch == 100
            assert call_args.max_retries == 3
            assert call_args.overwrite_existing is True
            assert call_args.throttle_delay_ms == 100.0  # 0.1 * 1000
            assert call_args.max_memory_mb == 2048.0
            assert call_args.verbose_logging is True

            # Check that policy loader was called
            mock_load_policy.assert_called_once_with(Path(policy_file), True)

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_mask_with_failures(
        self, mock_processor_class, cli_runner, temp_workspace
    ):
        """Test batch mask command with some failures."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1005.0,
            total_files=3,
            successful_files=2,
            failed_files=1,
            skipped_files=0,
            total_processing_time_ms=4000.0,
            total_entities_processed=15,
        )

        # Add file results with errors
        from cloakpivot.core.batch import BatchFileItem

        mock_result.file_results = [
            BatchFileItem(file_path=Path("doc1.pdf"), status=BatchStatus.COMPLETED),
            BatchFileItem(file_path=Path("doc2.pdf"), status=BatchStatus.COMPLETED),
            BatchFileItem(
                file_path=Path("doc3.pdf"),
                status=BatchStatus.FAILED,
                error="Processing failed",
            ),
        ]

        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "*.pdf")
        output_dir = str(temp_workspace["output_dir"])

        result = cli_runner.invoke(
            batch,
            [
                "mask",
                input_pattern,
                "--out-dir",
                output_dir,
            ],
        )

        assert (
            result.exit_code == 1
        )  # Should exit with failure code due to failed files
        assert "files failed processing" in result.output
        assert "Processing failed" in result.output

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_unmask_basic(self, mock_processor_class, cli_runner, temp_workspace):
        """Test basic batch unmask command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1003.0,
            total_files=2,
            successful_files=2,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=2500.0,
            total_entities_processed=12,
        )
        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "*.json")
        output_dir = str(temp_workspace["output_dir"])
        cloakmap_dir = str(temp_workspace["cloakmap_dir"])

        result = cli_runner.invoke(
            batch,
            [
                "unmask",
                input_pattern,
                "--cloakmap-dir",
                cloakmap_dir,
                "--out-dir",
                output_dir,
            ],
        )

        assert result.exit_code == 0
        mock_processor_class.assert_called_once()
        mock_processor.process_batch.assert_called_once()

        # Check configuration
        call_args = mock_processor_class.call_args[0][0]
        assert call_args.operation_type == BatchOperationType.UNMASK
        assert call_args.cloakmap_directory == Path(cloakmap_dir)
        assert call_args.output_directory == Path(output_dir)

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_analyze_basic(
        self, mock_processor_class, cli_runner, temp_workspace
    ):
        """Test basic batch analyze command."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1002.0,
            total_files=3,
            successful_files=3,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=1800.0,
            total_entities_processed=25,
        )
        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "**/*.pdf")

        result = cli_runner.invoke(
            batch,
            [
                "analyze",
                input_pattern,
                "--summary-only",
            ],
        )

        assert result.exit_code == 0
        assert "Batch Analysis Summary" in result.output
        assert "Total files processed: 3" in result.output
        assert "Successful analyses: 3" in result.output
        assert "Total entities found: 25" in result.output
        assert "Success rate: 100.0%" in result.output

        # Check configuration
        call_args = mock_processor_class.call_args[0][0]
        assert call_args.operation_type == BatchOperationType.ANALYZE
        assert call_args.output_directory is None  # summary-only mode

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_analyze_with_output(
        self, mock_processor_class, cli_runner, temp_workspace
    ):
        """Test batch analyze command with output directory."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        mock_result = BatchResult(
            config=Mock(),
            status=BatchStatus.COMPLETED,
            start_time=1000.0,
            end_time=1001.0,
            total_files=2,
            successful_files=2,
            failed_files=0,
            skipped_files=0,
            total_processing_time_ms=800.0,
            total_entities_processed=18,
        )
        mock_processor.process_batch.return_value = mock_result

        input_pattern = str(temp_workspace["input_dir"] / "*.pdf")
        output_dir = str(temp_workspace["output_dir"])

        result = cli_runner.invoke(
            batch,
            [
                "analyze",
                input_pattern,
                "--out-dir",
                output_dir,
                "--verbose",
            ],
        )

        assert result.exit_code == 0
        assert "Starting batch analysis operation" in result.output

        # Check configuration
        call_args = mock_processor_class.call_args[0][0]
        assert call_args.operation_type == BatchOperationType.ANALYZE
        assert call_args.output_directory == Path(output_dir)
        assert call_args.verbose_logging is True

    @patch("cloakpivot.cli.batch.BatchProcessor")
    def test_batch_keyboard_interrupt(
        self, mock_processor_class, cli_runner, temp_workspace
    ):
        """Test handling of keyboard interrupt during batch processing."""
        mock_processor = Mock()
        mock_processor_class.return_value = mock_processor

        # Simulate KeyboardInterrupt
        mock_processor.process_batch.side_effect = KeyboardInterrupt()

        input_pattern = str(temp_workspace["input_dir"] / "*.pdf")
        output_dir = str(temp_workspace["output_dir"])

        result = cli_runner.invoke(
            batch,
            [
                "mask",
                input_pattern,
                "--out-dir",
                output_dir,
            ],
        )

        assert result.exit_code == 130  # SIGINT exit code
        assert "cancelled by user" in result.output
        mock_processor.cancel.assert_called_once()

    def test_validate_patterns(self):
        """Test pattern validation function."""
        from cloakpivot.cli.batch import _validate_patterns

        # Test with valid patterns
        patterns = ["*.pdf", "data/**/*.json"]
        result = _validate_patterns(patterns)

        assert len(result) == 2
        # Should convert to absolute paths
        assert all(Path(p).is_absolute() for p in result)

    def test_validate_patterns_empty(self):
        """Test pattern validation with empty patterns."""
        from click.exceptions import ClickException

        from cloakpivot.cli.batch import _validate_patterns

        with pytest.raises(ClickException) as exc_info:
            _validate_patterns([])

        assert "At least one input pattern must be specified" in str(exc_info.value)

    @patch("cloakpivot.core.policy_loader.PolicyLoader")
    @patch("builtins.open")
    @patch("yaml.safe_load")
    def test_load_masking_policy_yaml(
        self, mock_yaml_load, mock_open, mock_loader_class
    ):
        """Test loading masking policy from YAML file."""
        from cloakpivot.cli.batch import _load_masking_policy

        # Make the enhanced loader fail so it falls back to basic YAML
        mock_loader_class.side_effect = ImportError("PolicyLoader not available")

        policy_data = {
            "locale": "en",
            "default_strategy": {"kind": "redact"},
            "per_entity": {"PERSON": {"kind": "hash"}},
        }
        mock_yaml_load.return_value = policy_data

        with patch(
            "cloakpivot.core.policies.MaskingPolicy.from_dict"
        ) as mock_from_dict:
            mock_policy = MaskingPolicy()
            mock_from_dict.return_value = mock_policy

            result = _load_masking_policy(Path("policy.yaml"), verbose=False)

            assert result == mock_policy
            mock_from_dict.assert_called_once_with(policy_data)

    def test_load_masking_policy_none(self):
        """Test loading masking policy when no file specified."""
        from cloakpivot.cli.batch import _load_masking_policy

        result = _load_masking_policy(None, verbose=False)
        assert result is None

    @patch("cloakpivot.core.policy_loader.PolicyLoader")
    def test_load_masking_policy_enhanced_loader(self, mock_loader_class):
        """Test loading masking policy with enhanced loader."""
        from cloakpivot.cli.batch import _load_masking_policy

        mock_loader = Mock()
        mock_loader_class.return_value = mock_loader
        mock_policy = MaskingPolicy()
        mock_loader.load_policy.return_value = mock_policy

        result = _load_masking_policy(Path("policy.yaml"), verbose=True)

        assert result == mock_policy
        mock_loader.load_policy.assert_called_once_with(Path("policy.yaml"))


if __name__ == "__main__":
    pytest.main([__file__])
