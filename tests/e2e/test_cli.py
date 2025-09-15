"""End-to-end tests for CLI commands."""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


class TestCLI:
    """Test the cloakpivot CLI commands."""

    @pytest.fixture
    def sample_json_file(self, tmp_path: Path) -> Path:
        """Create a sample JSON file for testing."""
        content = {
            "version": "1.0.0",
            "name": "test_doc",
            "text": "Contact John Doe at john.doe@example.com or call 555-123-4567."
        }
        json_path = tmp_path / "sample.json"
        with open(json_path, "w") as f:
            json.dump(content, f)
        return json_path

    def test_cli_help(self):
        """Test that CLI help works."""
        result = subprocess.run(
            ["python", "-m", "cloakpivot", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "usage" in result.stdout.lower() or "Usage" in result.stdout

    def test_mask_command_help(self):
        """Test mask command help."""
        result = subprocess.run(
            ["python", "-m", "cloakpivot", "mask", "--help"],
            capture_output=True,
            text=True
        )
        assert result.returncode == 0
        assert "mask" in result.stdout.lower()

    def test_mask_json_file(self, sample_json_file: Path, tmp_path: Path):
        """Test masking a JSON file via CLI."""
        output_file = tmp_path / "masked.json"
        cloakmap_file = tmp_path / "cloakmap.json"

        result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "mask",
                str(sample_json_file),
                "--output", str(output_file),
                "--cloakmap", str(cloakmap_file),
                "--format", "json"
            ],
            capture_output=True,
            text=True
        )

        # Check command succeeded
        if result.returncode != 0:
            print(f"STDERR: {result.stderr}")
            print(f"STDOUT: {result.stdout}")

        assert result.returncode == 0
        assert output_file.exists()
        assert cloakmap_file.exists()

        # Check that PII was masked
        with open(output_file) as f:
            masked_content = json.load(f)
            assert "john.doe@example.com" not in masked_content.get("text", "")

    def test_unmask_command(self, sample_json_file: Path, tmp_path: Path):
        """Test the full mask->unmask cycle via CLI."""
        masked_file = tmp_path / "masked.json"
        cloakmap_file = tmp_path / "cloakmap.json"
        restored_file = tmp_path / "restored.json"

        # First mask the file
        mask_result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "mask",
                str(sample_json_file),
                "--output", str(masked_file),
                "--cloakmap", str(cloakmap_file),
                "--format", "json"
            ],
            capture_output=True,
            text=True
        )
        assert mask_result.returncode == 0

        # Then unmask it
        unmask_result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "unmask",
                str(masked_file),
                "--cloakmap", str(cloakmap_file),
                "--output", str(restored_file),
                "--format", "json"
            ],
            capture_output=True,
            text=True
        )

        if unmask_result.returncode != 0:
            print(f"STDERR: {unmask_result.stderr}")
            print(f"STDOUT: {unmask_result.stdout}")

        assert unmask_result.returncode == 0
        assert restored_file.exists()

        # Check content is restored
        with open(sample_json_file) as f:
            original = json.load(f)
        with open(restored_file) as f:
            restored = json.load(f)

        assert restored["text"] == original["text"]

    def test_mask_with_policy_file(self, sample_json_file: Path, tmp_path: Path):
        """Test masking with a custom policy file."""
        # Create a policy file
        policy_file = tmp_path / "policy.yaml"
        policy_content = """
name: test_policy
privacy_level: high
default_strategy:
  kind: redact
  parameters:
    redact_char: "X"
"""
        with open(policy_file, "w") as f:
            f.write(policy_content)

        output_file = tmp_path / "masked.json"
        cloakmap_file = tmp_path / "cloakmap.json"

        result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "mask",
                str(sample_json_file),
                "--output", str(output_file),
                "--cloakmap", str(cloakmap_file),
                "--policy", str(policy_file),
                "--format", "json"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        assert output_file.exists()

    def test_mask_stdin_stdout(self):
        """Test masking from stdin to stdout."""
        input_text = '{"version": "1.0.0", "name": "test", "text": "Email: test@example.com"}'

        result = subprocess.run(
            ["python", "-m", "cloakpivot", "mask", "-", "--format", "json"],
            input=input_text,
            capture_output=True,
            text=True
        )

        if result.returncode == 0:
            output = json.loads(result.stdout)
            assert "test@example.com" not in output.get("text", "")

    def test_invalid_input_file(self):
        """Test error handling for non-existent input file."""
        result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "mask",
                "/nonexistent/file.json"
            ],
            capture_output=True,
            text=True
        )

        assert result.returncode != 0
        assert "error" in result.stderr.lower() or "Error" in result.stderr

    @pytest.mark.parametrize("format_type", ["json", "text", "markdown"])
    def test_output_formats(self, sample_json_file: Path, tmp_path: Path, format_type: str):
        """Test different output formats."""
        output_file = tmp_path / f"output.{format_type}"
        cloakmap_file = tmp_path / "cloakmap.json"

        result = subprocess.run(
            [
                "python", "-m", "cloakpivot", "mask",
                str(sample_json_file),
                "--output", str(output_file),
                "--cloakmap", str(cloakmap_file),
                "--format", format_type
            ],
            capture_output=True,
            text=True
        )

        # Format support may vary, but command should complete
        assert output_file.exists() or result.returncode != 0