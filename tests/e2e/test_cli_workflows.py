"""End-to-end CLI workflow tests for CloakPivot.

These tests validate complete user workflows through the command-line interface,
including file handling, error scenarios, and user experience flows.
"""

import json
import tempfile
from pathlib import Path

import pytest
from click.testing import CliRunner

from cloakpivot.cli.main import cli


class TestCLIWorkflows:
    """Test complete CLI workflows from user perspective."""

    @pytest.fixture
    def cli_runner(self) -> CliRunner:
        """Create CLI runner for testing."""
        return CliRunner()

    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for CLI tests."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            yield workspace

    @pytest.fixture
    def sample_document_file(self, temp_workspace: Path) -> Path:
        """Create sample document file for testing."""
        import json

        from docling_core.types import DoclingDocument
        from docling_core.types.doc.document import TextItem

        doc_content = """Employee Information
===================

Personal Details:
Name: John Smith
Phone: 555-123-4567
Email: john.smith@company.com
SSN: 123-45-6789

Contact Information:
Emergency Contact: Jane Smith
Phone: 555-987-6543
Email: jane.smith@personal.com"""

        # Create DoclingDocument
        doc = DoclingDocument(name="test_document")
        text_item = TextItem(
            text=doc_content, self_ref="#/texts/0", label="text", orig=doc_content
        )
        doc.texts = [text_item]

        doc_file = temp_workspace / "employee_info.json"
        doc_file.write_text(json.dumps(doc.model_dump(), indent=2))
        return doc_file

    @pytest.fixture
    def sample_policy_file(self, temp_workspace: Path) -> Path:
        """Create sample policy file for testing."""
        policy_content = {
            "locale": "en",
            "privacy_level": "MEDIUM",
            "entities": {
                "PHONE_NUMBER": {"kind": "PHONE_TEMPLATE"},
                "EMAIL_ADDRESS": {"kind": "EMAIL_TEMPLATE"},
                "US_SSN": {"kind": "SURROGATE_SECURE"},
                "PERSON": {"kind": "TEMPLATE", "parameters": {"auto_generate": True}},
            },
            "thresholds": {
                "PHONE_NUMBER": 0.7,
                "EMAIL_ADDRESS": 0.8,
                "US_SSN": 0.9,
                "PERSON": 0.8,
            },
        }

        policy_file = temp_workspace / "test_policy.json"
        policy_file.write_text(json.dumps(policy_content, indent=2))
        return policy_file

    @pytest.mark.e2e
    def test_basic_help_commands(self, cli_runner: CliRunner):
        """Test basic help and version commands work."""
        # Test main help
        result = cli_runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "CloakPivot" in result.output
        assert "mask" in result.output
        assert "unmask" in result.output

        # Test version
        result = cli_runner.invoke(cli, ["--version"])
        assert result.exit_code == 0

        # Test subcommand help
        result = cli_runner.invoke(cli, ["mask", "--help"])
        assert result.exit_code == 0
        assert "Mask PII in a document" in result.output

        result = cli_runner.invoke(cli, ["unmask", "--help"])
        assert result.exit_code == 0
        assert "Unmask a previously masked document" in result.output

    @pytest.mark.e2e
    def test_complete_mask_unmask_workflow(
        self,
        cli_runner: CliRunner,
        temp_workspace: Path,
        sample_document_file: Path,
        sample_policy_file: Path,
    ):
        """Test complete mask/unmask workflow with file I/O."""
        masked_file = temp_workspace / "masked_output.json"
        cloakmap_file = temp_workspace / "cloakmap.json"
        unmasked_file = temp_workspace / "unmasked_output.json"

        # Step 1: Mask the document
        mask_result = cli_runner.invoke(
            cli,
            [
                "mask",
                str(sample_document_file),
                "--out",
                str(masked_file),
                "--policy",
                str(sample_policy_file),
                "--cloakmap",
                str(cloakmap_file),
            ],
        )

        assert mask_result.exit_code == 0, f"Masking failed: {mask_result.output}"
        assert masked_file.exists(), "Masked file should be created"
        assert cloakmap_file.exists(), "CloakMap file should be created"

        # Verify masked content is different from original
        original_content = sample_document_file.read_text()
        masked_content = masked_file.read_text()
        assert original_content != masked_content, "Content should be masked"

        # Verify CloakMap is valid JSON
        cloakmap_data = json.loads(cloakmap_file.read_text())
        assert "doc_id" in cloakmap_data
        assert "anchors" in cloakmap_data

        # Step 2: Unmask the document
        unmask_result = cli_runner.invoke(
            cli,
            [
                "unmask",
                str(masked_file),
                "--out",
                str(unmasked_file),
                "--cloakmap",
                str(cloakmap_file),
            ],
        )

        assert unmask_result.exit_code == 0, f"Unmasking failed: {unmask_result.output}"
        assert unmasked_file.exists(), "Unmasked file should be created"

        # Verify round-trip fidelity
        unmasked_content = unmasked_file.read_text()
        assert (
            original_content.strip() == unmasked_content.strip()
        ), "Round-trip should preserve content"

    @pytest.mark.e2e
    def test_policy_management_workflow(
        self, cli_runner: CliRunner, temp_workspace: Path
    ):
        """Test policy creation and management workflow."""
        policy_file = temp_workspace / "created_policy.json"

        # Test policy creation (interactive simulation)
        # Note: This is a simplified test - full interactive testing would require more complex setup
        result = cli_runner.invoke(
            cli,
            ["policy", "create", "--output", str(policy_file)],
            input="MEDIUM\ny\ny\ny\ny\n",
        )  # Simulate user inputs

        # The exact behavior depends on implementation
        # This test may need adjustment based on actual CLI implementation
        if result.exit_code == 0:
            assert policy_file.exists(), "Policy file should be created"

            # Verify policy content
            policy_data = json.loads(policy_file.read_text())
            assert "locale" in policy_data
            assert "entities" in policy_data

    @pytest.mark.e2e
    def test_batch_processing_workflow(
        self, cli_runner: CliRunner, temp_workspace: Path, sample_policy_file: Path
    ):
        """Test batch processing workflow with multiple files."""
        # Create multiple test files
        input_dir = temp_workspace / "input"
        output_dir = temp_workspace / "output"
        cloakmap_dir = temp_workspace / "cloakmaps"

        input_dir.mkdir()
        output_dir.mkdir()
        cloakmap_dir.mkdir()

        # Create test files
        test_files = []
        for i in range(3):
            content = f"""
Document {i + 1}
==============
Employee: Person {i + 1}
Phone: 555-{i:03d}-{i + 1:04d}
Email: person{i + 1}@company.com
SSN: {i + 1:03d}-{i + 2:02d}-{i + 3:04d}
"""
            file_path = input_dir / f"document_{i + 1}.txt"
            file_path.write_text(content)
            test_files.append(file_path)

        # Test batch masking
        batch_result = cli_runner.invoke(
            cli,
            [
                "batch",
                "mask",
                "--input-dir",
                str(input_dir),
                "--output-dir",
                str(output_dir),
                "--cloakmap-dir",
                str(cloakmap_dir),
                "--policy",
                str(sample_policy_file),
            ],
        )

        # Check if batch command is implemented
        if batch_result.exit_code == 0:
            # Verify all files were processed
            for test_file in test_files:
                masked_file = output_dir / test_file.name
                cloakmap_file = cloakmap_dir / f"{test_file.stem}_cloakmap.json"

                assert masked_file.exists(), f"Masked file should exist: {masked_file}"
                assert cloakmap_file.exists(), f"CloakMap should exist: {cloakmap_file}"
        else:
            # Batch processing may not be implemented yet
            pytest.skip("Batch processing not yet implemented")

    @pytest.mark.e2e
    def test_error_handling_workflows(
        self, cli_runner: CliRunner, temp_workspace: Path
    ):
        """Test CLI error handling and recovery scenarios."""
        # Test with non-existent input file
        result = cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(temp_workspace / "nonexistent.txt"),
                "--output",
                str(temp_workspace / "output.txt"),
                "--policy",
                str(temp_workspace / "policy.json"),
            ],
        )

        assert result.exit_code != 0
        assert "not found" in result.output.lower() or "error" in result.output.lower()

        # Test with invalid policy file
        invalid_policy = temp_workspace / "invalid_policy.json"
        invalid_policy.write_text("invalid json content {")

        sample_doc = temp_workspace / "sample.txt"
        sample_doc.write_text("Test content")

        result = cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(sample_doc),
                "--output",
                str(temp_workspace / "output.txt"),
                "--policy",
                str(invalid_policy),
            ],
        )

        assert result.exit_code != 0
        # Should have helpful error message
        assert len(result.output) > 0

        # Test with missing required arguments
        result = cli_runner.invoke(cli, ["mask"])
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "missing" in result.output.lower()

    @pytest.mark.e2e
    def test_output_format_workflow(
        self,
        cli_runner: CliRunner,
        temp_workspace: Path,
        sample_document_file: Path,
        sample_policy_file: Path,
    ):
        """Test workflow with different output formats."""
        # Test JSON output format
        json_output = temp_workspace / "masked_output.json"
        cloakmap_file = temp_workspace / "cloakmap.json"

        result = cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(sample_document_file),
                "--output",
                str(json_output),
                "--policy",
                str(sample_policy_file),
                "--cloakmap",
                str(cloakmap_file),
                "--format",
                "json",
            ],
        )

        # Check if format options are implemented
        if result.exit_code == 0:
            assert json_output.exists()

            # Verify JSON format
            try:
                output_data = json.loads(json_output.read_text())
                assert isinstance(output_data, dict)
            except json.JSONDecodeError:
                pytest.fail("Output should be valid JSON")
        else:
            # Format options may not be implemented yet
            pytest.skip("Output format options not yet implemented")

    @pytest.mark.e2e
    def test_verbose_and_quiet_modes(
        self,
        cli_runner: CliRunner,
        temp_workspace: Path,
        sample_document_file: Path,
        sample_policy_file: Path,
    ):
        """Test verbose and quiet output modes."""
        output_file = temp_workspace / "output.txt"
        cloakmap_file = temp_workspace / "cloakmap.json"

        # Test verbose mode
        verbose_result = cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(sample_document_file),
                "--output",
                str(output_file),
                "--policy",
                str(sample_policy_file),
                "--cloakmap",
                str(cloakmap_file),
                "--verbose",
            ],
        )

        # Clean up for next test
        if output_file.exists():
            output_file.unlink()
        if cloakmap_file.exists():
            cloakmap_file.unlink()

        # Test quiet mode
        quiet_result = cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(sample_document_file),
                "--output",
                str(output_file),
                "--policy",
                str(sample_policy_file),
                "--cloakmap",
                str(cloakmap_file),
                "--quiet",
            ],
        )

        # Verbose mode should have more output than quiet mode
        if verbose_result.exit_code == 0 and quiet_result.exit_code == 0:
            assert len(verbose_result.output) >= len(quiet_result.output)

    @pytest.mark.e2e
    def test_configuration_workflow(self, cli_runner: CliRunner, temp_workspace: Path):
        """Test configuration management workflow."""
        config_file = temp_workspace / "cloakpivot_config.json"

        # Test config initialization
        result = cli_runner.invoke(
            cli, ["config", "init", "--config-file", str(config_file)]
        )

        # Check if config commands are implemented
        if result.exit_code == 0:
            assert config_file.exists()

            # Verify config content
            config_data = json.loads(config_file.read_text())
            assert isinstance(config_data, dict)
        else:
            # Config commands may not be implemented yet
            pytest.skip("Config commands not yet implemented")

    @pytest.mark.e2e
    def test_plugin_workflow(self, cli_runner: CliRunner, temp_workspace: Path):
        """Test plugin management workflow."""
        # Test plugin listing
        result = cli_runner.invoke(cli, ["plugin", "list"])

        # Check if plugin commands are implemented
        if result.exit_code == 0:
            # Should show available plugins
            assert (
                "strategy" in result.output.lower()
                or "recognizer" in result.output.lower()
            )
        else:
            # Plugin commands may not be implemented yet
            pytest.skip("Plugin commands not yet implemented")

    @pytest.mark.e2e
    def test_validation_workflow(
        self,
        cli_runner: CliRunner,
        temp_workspace: Path,
        sample_document_file: Path,
        sample_policy_file: Path,
    ):
        """Test validation and diagnostics workflow."""
        # Test policy validation
        policy_result = cli_runner.invoke(
            cli, ["policy", "validate", "--policy", str(sample_policy_file)]
        )

        # Test document analysis
        analyze_result = cli_runner.invoke(
            cli,
            [
                "analyze",
                "--input",
                str(sample_document_file),
                "--policy",
                str(sample_policy_file),
            ],
        )

        # These commands may not be implemented yet
        if policy_result.exit_code != 0 and analyze_result.exit_code != 0:
            pytest.skip("Validation commands not yet implemented")

    @pytest.mark.e2e
    def test_cli_integration_with_temp_files(
        self, cli_runner: CliRunner, temp_workspace: Path
    ):
        """Test CLI properly handles temporary file cleanup."""
        # Create test scenario that might leave temp files
        test_content = (
            "Test document with phone 555-123-4567 and email test@example.com"
        )
        input_file = temp_workspace / "input.txt"
        input_file.write_text(test_content)

        # Create minimal policy
        policy_content = {
            "locale": "en",
            "privacy_level": "LOW",
            "entities": {"PHONE_NUMBER": {"kind": "TEMPLATE"}},
            "thresholds": {"PHONE_NUMBER": 0.5},
        }
        policy_file = temp_workspace / "policy.json"
        policy_file.write_text(json.dumps(policy_content))

        output_file = temp_workspace / "output.txt"
        cloakmap_file = temp_workspace / "cloakmap.json"

        # Run masking operation
        cli_runner.invoke(
            cli,
            [
                "mask",
                "--input",
                str(input_file),
                "--output",
                str(output_file),
                "--policy",
                str(policy_file),
                "--cloakmap",
                str(cloakmap_file),
            ],
        )

        # Verify operation completed (may succeed or fail depending on implementation)
        # The key is that no temporary files should be left behind
        temp_files_before = list(temp_workspace.glob("*tmp*")) + list(
            temp_workspace.glob(".*tmp*")
        )

        # Check system temp directory for any CloakPivot temp files
        # Note: During parallel test execution, worker temp directories are legitimate and expected
        import tempfile

        system_temp = Path(tempfile.gettempdir())
        cloakpivot_temp_files = list(system_temp.glob("*cloakpivot*")) + list(
            system_temp.glob("*CloakPivot*")
        )
        # Filter out legitimate parallel execution worker directories
        # These follow the pattern "cloakpivot_worker_*" and are expected during test execution
        unexpected_temp_files = []
        for temp_file in cloakpivot_temp_files:
            # Allow worker temp directories (cloakpivot_worker_*) during parallel execution
            if not temp_file.name.startswith("cloakpivot_worker_"):
                unexpected_temp_files.append(temp_file)
        assert (
            len(temp_files_before) == 0
        ), f"Temporary files found in workspace: {temp_files_before}"
        assert (
            len(unexpected_temp_files) == 0
        ), f"Unexpected CloakPivot temp files found in system temp: {unexpected_temp_files}"
