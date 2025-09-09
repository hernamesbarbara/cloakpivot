"""Tests for Presidio CLI integration."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from click.testing import CliRunner

from cloakpivot.cli.main import cli


@pytest.fixture
def runner():
    """Create a Click test runner."""
    return CliRunner()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_document(temp_dir):
    """Create a sample document for testing."""
    doc_path = temp_dir / "sample.txt"
    doc_path.write_text(
        "John Doe's email is john.doe@example.com and phone is 555-123-4567."
    )
    return doc_path


@pytest.fixture
def sample_policy(temp_dir):
    """Create a sample policy file."""
    policy_path = temp_dir / "policy.yaml"
    policy_data = {
        "version": "1.0",
        "default_strategy": {
            "kind": "redact",
            "parameters": {"redact_char": "*", "preserve_length": True},
        },
    }
    with open(policy_path, "w") as f:
        yaml.dump(policy_data, f)
    return policy_path


@pytest.fixture
def presidio_config(temp_dir):
    """Create a Presidio configuration file."""
    config_path = temp_dir / "presidio_config.yml"
    config_data = {
        "presidio": {
            "engine": "presidio",
            "fallback_on_error": True,
            "confidence_threshold": 0.8,
            "operators": {
                "default_redact": {"redact_char": "#", "preserve_length": True}
            },
        }
    }
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)
    return config_path


class TestEngineSelection:
    """Test engine selection via CLI flags."""

    def test_global_engine_selection_auto(self, runner):
        """Test auto engine selection at global level."""
        result = runner.invoke(cli, ["--engine", "auto", "--help"])
        assert result.exit_code == 0

    def test_global_engine_selection_presidio(self, runner):
        """Test Presidio engine selection at global level."""
        result = runner.invoke(cli, ["--engine", "presidio", "--help"])
        assert result.exit_code == 0

    def test_global_engine_selection_legacy(self, runner):
        """Test legacy engine selection at global level."""
        result = runner.invoke(cli, ["--engine", "legacy", "--help"])
        assert result.exit_code == 0

    def test_invalid_engine_selection(self, runner):
        """Test invalid engine selection."""
        result = runner.invoke(cli, ["--engine", "invalid"])
        assert result.exit_code != 0
        assert "Invalid value" in result.output or "Error" in result.output

    def test_presidio_fallback_flag(self, runner):
        """Test Presidio fallback flag."""
        result = runner.invoke(cli, ["--no-presidio-fallback", "--help"])
        assert result.exit_code == 0


class TestMaskCommandWithPresidio:
    """Test mask command with Presidio integration."""

    @patch("cloakpivot.masking.engine.MaskingEngine")
    @patch("cloakpivot.document.processor.DocumentProcessor")
    @patch("cloakpivot.core.detection.EntityDetectionPipeline")
    def test_mask_with_engine_override(
        self,
        mock_pipeline,
        mock_processor,
        mock_engine,
        runner,
        sample_document,
        sample_policy,
        temp_dir,
    ):
        """Test mask command with engine override."""
        # Setup mocks
        mock_doc = MagicMock()
        mock_doc.name = "test_doc"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_processor.return_value.load_document.return_value = mock_doc

        mock_detection = MagicMock()
        mock_detection.total_entities = 2
        mock_detection.entity_breakdown = {"PERSON": 1, "EMAIL": 1}
        mock_detection.segment_results = []
        mock_pipeline.return_value.analyze_document.return_value = mock_detection

        mock_result = MagicMock()
        mock_result.masked_document = mock_doc
        mock_result.cloakmap = MagicMock()
        mock_result.cloakmap.anchors = []
        mock_result.cloakmap.to_dict.return_value = {}
        mock_result.stats = {"total_entities_masked": 2}
        mock_engine.return_value.mask_document.return_value = mock_result

        output_path = temp_dir / "masked.json"
        cloakmap_path = temp_dir / "map.json"

        runner.invoke(
            cli,
            [
                "mask",
                str(sample_document),
                "--out",
                str(output_path),
                "--cloakmap",
                str(cloakmap_path),
                "--policy",
                str(sample_policy),
                "--engine",
                "presidio",
            ],
        )

        # Check that MaskingEngine was called with presidio engine
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args.kwargs
        assert call_kwargs["use_presidio_engine"] is True

    @patch("cloakpivot.cli.config.load_presidio_config")
    @patch("cloakpivot.masking.engine.MaskingEngine")
    @patch("cloakpivot.document.processor.DocumentProcessor")
    @patch("cloakpivot.core.detection.EntityDetectionPipeline")
    def test_mask_with_presidio_config(
        self,
        mock_pipeline,
        mock_processor,
        mock_engine,
        mock_load_config,
        runner,
        sample_document,
        sample_policy,
        presidio_config,
        temp_dir,
    ):
        """Test mask command with Presidio configuration file."""
        # Setup mocks
        mock_doc = MagicMock()
        mock_doc.name = "test_doc"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_processor.return_value.load_document.return_value = mock_doc

        mock_detection = MagicMock()
        mock_detection.total_entities = 2
        mock_detection.entity_breakdown = {"PERSON": 1, "EMAIL": 1}
        mock_detection.segment_results = []
        mock_pipeline.return_value.analyze_document.return_value = mock_detection

        mock_result = MagicMock()
        mock_result.masked_document = mock_doc
        mock_result.cloakmap = MagicMock()
        mock_result.cloakmap.anchors = []
        mock_result.cloakmap.to_dict.return_value = {}
        mock_result.stats = {"total_entities_masked": 2}
        mock_engine.return_value.mask_document.return_value = mock_result

        mock_load_config.return_value = {
            "engine": "presidio",
            "confidence_threshold": 0.8,
        }

        output_path = temp_dir / "masked.json"
        cloakmap_path = temp_dir / "map.json"

        runner.invoke(
            cli,
            [
                "mask",
                str(sample_document),
                "--out",
                str(output_path),
                "--cloakmap",
                str(cloakmap_path),
                "--policy",
                str(sample_policy),
                "--presidio-config",
                str(presidio_config),
            ],
        )

        # Check that config was loaded (convert to string for comparison)
        mock_load_config.assert_called_once_with(str(presidio_config))


class TestMigrationCommands:
    """Test CloakMap migration commands."""

    def test_upgrade_cloakmap_v1_to_v2(self, runner, temp_dir):
        """Test upgrading v1.0 CloakMap to v2.0."""
        # Create v1.0 CloakMap
        cloakmap_v1 = {
            "version": "1.0",
            "doc_id": "test-doc",
            "doc_hash": "abc123",
            "anchors": [],
            "created_at": "2024-01-01T00:00:00",
        }

        cloakmap_path = temp_dir / "map_v1.json"
        with open(cloakmap_path, "w") as f:
            json.dump(cloakmap_v1, f)

        result = runner.invoke(
            cli, ["migrate", "upgrade-cloakmap", str(cloakmap_path)]
        )

        assert result.exit_code == 0
        assert "upgraded successfully" in result.output

        # Check upgraded file
        with open(cloakmap_path) as f:
            upgraded = json.load(f)

        assert upgraded["version"] == "2.0"
        assert upgraded["engine_used"] == "legacy"
        assert upgraded["metadata"]["upgraded_from_v1"] is True

    def test_upgrade_cloakmap_already_v2(self, runner, temp_dir):
        """Test upgrading already v2.0 CloakMap."""
        # Create v2.0 CloakMap
        cloakmap_v2 = {
            "version": "2.0",
            "doc_id": "test-doc",
            "doc_hash": "abc123",
            "anchors": [],
            "engine_used": "presidio",
        }

        cloakmap_path = temp_dir / "map_v2.json"
        with open(cloakmap_path, "w") as f:
            json.dump(cloakmap_v2, f)

        result = runner.invoke(
            cli, ["migrate", "upgrade-cloakmap", str(cloakmap_path)]
        )

        assert result.exit_code == 0
        assert "already v2.0 format" in result.output

    def test_upgrade_cloakmap_with_backup(self, runner, temp_dir):
        """Test upgrading with backup creation."""
        # Create v1.0 CloakMap
        cloakmap_v1 = {"version": "1.0", "doc_id": "test-doc", "doc_hash": "abc123", "anchors": []}

        cloakmap_path = temp_dir / "map.json"
        with open(cloakmap_path, "w") as f:
            json.dump(cloakmap_v1, f)

        result = runner.invoke(
            cli,
            ["-v", "migrate", "upgrade-cloakmap", str(cloakmap_path), "--backup"],
        )

        assert result.exit_code == 0
        assert "Backup created" in result.output

        # Check that backup exists
        backup_files = list(temp_dir.glob("*.backup.*.json"))
        assert len(backup_files) == 1


class TestPresidioCommands:
    """Test Presidio-specific commands."""

    def test_list_operators(self, runner):
        """Test listing available Presidio operators."""
        result = runner.invoke(cli, ["presidio", "list-operators"])

        assert result.exit_code == 0
        assert "Available Presidio Operators" in result.output
        assert "redact" in result.output
        assert "hash" in result.output
        assert "encrypt" in result.output
        assert "mask" in result.output

    def test_list_operators_verbose(self, runner):
        """Test listing operators with verbose output."""
        result = runner.invoke(cli, ["-v", "presidio", "list-operators"])

        assert result.exit_code == 0
        assert "Parameters:" in result.output
        assert "redact_char" in result.output
        assert "algorithm" in result.output

    @patch("cloakpivot.masking.presidio_adapter.PresidioMaskingAdapter")
    def test_apply_operator(self, mock_adapter, runner, temp_dir):
        """Test applying a Presidio operator."""
        # Create input file
        input_file = temp_dir / "input.txt"
        input_file.write_text("Test content with PII")

        result = runner.invoke(
            cli,
            [
                "presidio",
                "apply-operator",
                str(input_file),
                "--operator",
                "redact",
                "--config",
                '{"redact_char": "#"}',
            ],
        )

        # Note: Current implementation shows info message about needing entity detection
        assert "Direct operator application requires entity detection" in result.output

    def test_apply_operator_invalid_config(self, runner, temp_dir):
        """Test applying operator with invalid configuration."""
        input_file = temp_dir / "input.txt"
        input_file.write_text("Test content")

        result = runner.invoke(
            cli,
            [
                "presidio",
                "apply-operator",
                str(input_file),
                "--operator",
                "redact",
                "--config",
                "invalid json",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid JSON configuration" in result.output


class TestBackwardCompatibility:
    """Test backward compatibility with existing CLI usage."""

    @patch("cloakpivot.masking.engine.MaskingEngine")
    @patch("cloakpivot.document.processor.DocumentProcessor")
    @patch("cloakpivot.core.detection.EntityDetectionPipeline")
    def test_mask_without_engine_flags(
        self,
        mock_pipeline,
        mock_processor,
        mock_engine,
        runner,
        sample_document,
        sample_policy,
        temp_dir,
    ):
        """Test that mask command works without engine flags (backward compatibility)."""
        # Setup mocks
        mock_doc = MagicMock()
        mock_doc.name = "test_doc"
        mock_doc.texts = []
        mock_doc.tables = []
        mock_processor.return_value.load_document.return_value = mock_doc

        mock_detection = MagicMock()
        mock_detection.total_entities = 2
        mock_detection.entity_breakdown = {"PERSON": 1, "EMAIL": 1}
        mock_detection.segment_results = []
        mock_pipeline.return_value.analyze_document.return_value = mock_detection

        mock_result = MagicMock()
        mock_result.masked_document = mock_doc
        mock_result.cloakmap = MagicMock()
        mock_result.cloakmap.anchors = []
        mock_result.cloakmap.to_dict.return_value = {}
        mock_result.stats = {"total_entities_masked": 2}
        mock_engine.return_value.mask_document.return_value = mock_result

        output_path = temp_dir / "masked.json"
        cloakmap_path = temp_dir / "map.json"

        # Run without any engine-related flags
        runner.invoke(
            cli,
            [
                "mask",
                str(sample_document),
                "--out",
                str(output_path),
                "--cloakmap",
                str(cloakmap_path),
                "--policy",
                str(sample_policy),
            ],
        )

        # Should use default (auto) engine selection
        mock_engine.assert_called_once()
        call_kwargs = mock_engine.call_args.kwargs
        # For auto mode, use_presidio_engine should be None
        assert call_kwargs.get("use_presidio_engine") is None


class TestErrorHandling:
    """Test error handling and help display."""

    def test_help_display(self, runner):
        """Test that help includes new engine options."""
        result = runner.invoke(cli, ["--help"])

        assert result.exit_code == 0
        assert "--engine" in result.output
        assert "auto" in result.output
        assert "presidio" in result.output
        assert "legacy" in result.output
        assert "--presidio-fallback" in result.output

    def test_mask_help_display(self, runner):
        """Test that mask command help includes new options."""
        result = runner.invoke(cli, ["mask", "--help"])

        assert result.exit_code == 0
        assert "--engine" in result.output
        assert "--presidio-config" in result.output

    def test_migrate_help_display(self, runner):
        """Test migrate command group help."""
        result = runner.invoke(cli, ["migrate", "--help"])

        assert result.exit_code == 0
        assert "CloakMap migration utilities" in result.output
        assert "upgrade-cloakmap" in result.output

    def test_presidio_help_display(self, runner):
        """Test presidio command group help."""
        result = runner.invoke(cli, ["presidio", "--help"])

        assert result.exit_code == 0
        assert "Presidio-specific features" in result.output
        assert "apply-operator" in result.output
        assert "list-operators" in result.output
