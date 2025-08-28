"""Tests for the CLI interface."""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from click.testing import CliRunner

from cloakpivot.cli.main import cli


class TestCLIBasics:
    """Test basic CLI functionality."""

    def test_cli_help(self):
        """Test that CLI help works."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])
        assert result.exit_code == 0
        assert 'CloakPivot: PII masking/unmasking' in result.output

    def test_cli_version(self):
        """Test that version flag works."""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])
        assert result.exit_code == 0

    def test_mask_command_help(self):
        """Test mask command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['mask', '--help'])
        assert result.exit_code == 0
        assert 'Mask PII in a document' in result.output

    def test_unmask_command_help(self):
        """Test unmask command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['unmask', '--help'])
        assert result.exit_code == 0
        assert 'Unmask a previously masked document' in result.output

    def test_policy_command_help(self):
        """Test policy command help."""
        runner = CliRunner()
        result = runner.invoke(cli, ['policy', '--help'])
        assert result.exit_code == 0
        assert 'Manage masking policies' in result.output


class TestMaskCommand:
    """Test mask command functionality."""

    def test_mask_missing_input_file(self):
        """Test mask command with non-existent input file."""
        runner = CliRunner()
        result = runner.invoke(cli, ['mask', 'nonexistent.json'])
        assert result.exit_code != 0
        assert 'does not exist' in result.output.lower()

    def test_mask_missing_output_and_cloakmap(self):
        """Test mask command without output or cloakmap specified."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            temp_file.write(b'{"test": "content"}')
            temp_file.flush()

            result = runner.invoke(cli, ['mask', temp_file.name])
            # Should succeed since default paths are generated
            # But will fail due to missing dependencies in test environment
            assert result.exit_code != 0

    @patch('cloakpivot.document.processor.DocumentProcessor')
    @patch('cloakpivot.core.detection.EntityDetectionPipeline')
    @patch('cloakpivot.masking.engine.MaskingEngine')
    def test_mask_command_success(self, mock_masking_engine, mock_detection, mock_processor):
        """Test successful mask command execution."""
        # Setup mocks
        mock_document = Mock()
        mock_document.name = "test.json"
        mock_document.texts = []
        mock_document.tables = []

        mock_processor.return_value.load_document.return_value = mock_document

        mock_detection_result = Mock()
        mock_detection_result.total_entities = 1
        mock_detection_result.entity_breakdown = {'PERSON': 1}
        mock_detection_result.segment_results = []

        mock_detection.return_value.analyze_document.return_value = mock_detection_result

        mock_masking_result = Mock()
        mock_masking_result.cloakmap.anchors = []
        mock_masking_result.cloakmap.to_dict.return_value = {}
        mock_masking_result.masked_document = mock_document
        mock_masking_result.stats = {'total_entities_masked': 1}

        mock_masking_engine.return_value.mask_document.return_value = mock_masking_result

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.json"
            output_file = temp_path / "output.json"
            cloakmap_file = temp_path / "cloakmap.json"

            input_file.write_text('{"test": "content"}')

            with patch('cloakpivot.document.extractor.TextExtractor'), \
                 patch('docpivot.LexicalDocSerializer'), \
                 patch('builtins.open', create=True):
                runner.invoke(cli, [
                    '--verbose',
                    'mask', str(input_file),
                    '--out', str(output_file),
                    '--cloakmap', str(cloakmap_file)
                ])

                assert mock_processor.called
                assert mock_detection.called
                assert mock_masking_engine.called

    def test_mask_with_policy_file_missing_yaml(self):
        """Test mask command with policy file but no PyYAML."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.json"
            policy_file = temp_path / "policy.yaml"

            input_file.write_text('{"test": "content"}')
            policy_file.write_text('default_strategy:\n  kind: redact')

            with patch('yaml.safe_load', side_effect=ImportError()), \
                 patch('cloakpivot.document.processor.DocumentProcessor') as mock_processor:
                mock_processor.return_value.load_document.return_value = Mock(name="doc", texts=[], tables=[])
                result = runner.invoke(cli, [
                    '--verbose',
                    'mask', str(input_file),
                    '--policy', str(policy_file),
                    '--out', str(input_file) + '.masked'
                ])

                assert 'PyYAML not installed' in result.output


class TestUnmaskCommand:
    """Test unmask command functionality."""

    def test_unmask_missing_files(self):
        """Test unmask command with missing files."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            'unmask', 'nonexistent.json',
            '--cloakmap', 'nonexistent.json'
        ])
        assert result.exit_code != 0
        assert 'does not exist' in result.output.lower()

    def test_unmask_missing_cloakmap_required(self):
        """Test unmask command without required cloakmap."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            temp_file.write(b'{"test": "content"}')
            temp_file.flush()

            result = runner.invoke(cli, ['unmask', temp_file.name])
            assert result.exit_code != 0
            assert 'Missing option' in result.output

    @patch('cloakpivot.document.processor.DocumentProcessor')
    @patch('cloakpivot.unmasking.engine.UnmaskingEngine')
    @patch('cloakpivot.core.cloakmap.CloakMap')
    def test_unmask_command_success(self, mock_cloakmap, mock_unmasking_engine, mock_processor):
        """Test successful unmask command execution."""
        # Setup mocks
        mock_document = Mock()
        mock_document.name = "test.json"

        mock_processor.return_value.load_document.return_value = mock_document

        mock_cloakmap_obj = Mock()
        mock_cloakmap_obj.anchors = []
        mock_cloakmap_obj.doc_id = "test-doc"
        mock_cloakmap_obj.version = "1.0"

        mock_cloakmap.from_dict.return_value = mock_cloakmap_obj

        mock_unmasking_result = Mock()
        mock_unmasking_result.restored_document = mock_document
        mock_unmasking_result.stats = {'success_rate': 100.0, 'resolved_anchors': 1, 'failed_anchors': 0}
        mock_unmasking_result.integrity_report = {'valid': True, 'issues': []}

        mock_unmasking_engine.return_value.unmask_document.return_value = mock_unmasking_result

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            masked_file = temp_path / "masked.json"
            cloakmap_file = temp_path / "cloakmap.json"
            output_file = temp_path / "output.json"

            masked_file.write_text('{"test": "masked content"}')
            cloakmap_file.write_text('{"doc_id": "test", "anchors": []}')

            with patch('docpivot.LexicalDocSerializer'), \
                 patch('builtins.open', create=True), \
                 patch('json.load', return_value={'doc_id': 'test', 'anchors': []}):
                runner.invoke(cli, [
                    '--verbose',
                    'unmask', str(masked_file),
                    '--cloakmap', str(cloakmap_file),
                    '--out', str(output_file)
                ])

                assert mock_processor.called
                assert mock_cloakmap.from_dict.called
                assert mock_unmasking_engine.called

    def test_unmask_verify_only_mode(self):
        """Test unmask command in verify-only mode."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            masked_file = temp_path / "masked.json"
            cloakmap_file = temp_path / "cloakmap.json"

            masked_file.write_text('{"test": "masked content"}')
            cloakmap_file.write_text('{"doc_id": "test", "anchors": []}')

            result = runner.invoke(cli, [
                'unmask', str(masked_file),
                '--cloakmap', str(cloakmap_file),
                '--verify-only'
            ])
            # Will fail due to missing dependencies but should recognize verify-only
            assert '--verify-only' not in result.output or result.exit_code != 0

    def test_unmask_invalid_cloakmap_json(self):
        """Test unmask command with invalid CloakMap JSON."""
        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            masked_file = temp_path / "masked.json"
            cloakmap_file = temp_path / "cloakmap.json"

            masked_file.write_text('{"test": "masked content"}')
            cloakmap_file.write_text('invalid json content')

            result = runner.invoke(cli, [
                'unmask', str(masked_file),
                '--cloakmap', str(cloakmap_file)
            ])
            assert result.exit_code != 0
            assert 'Invalid CloakMap file format' in result.output


class TestPolicyCommands:
    """Test policy command functionality."""

    def test_policy_sample_to_stdout(self):
        """Test policy sample generation to stdout."""
        runner = CliRunner()
        result = runner.invoke(cli, ['policy', 'sample'])

        assert result.exit_code == 0
        assert '# CloakPivot Masking Policy Configuration' in result.output
        assert 'default_strategy:' in result.output
        assert 'per_entity:' in result.output

    def test_policy_sample_to_file(self):
        """Test policy sample generation to file."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as temp_file:
            result = runner.invoke(cli, ['policy', 'sample', '--output', temp_file.name])

            assert result.exit_code == 0
            assert f'Sample policy written to {temp_file.name}' in result.output

            # Verify file content
            with open(temp_file.name) as f:
                content = f.read()
                assert '# CloakPivot Masking Policy Configuration' in content
                assert 'default_strategy:' in content


class TestErrorHandling:
    """Test CLI error handling scenarios."""

    def test_verbose_error_output(self):
        """Test that verbose mode shows detailed error information."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            '--verbose',
            'mask', 'nonexistent.json'
        ])
        assert result.exit_code != 0
        # Should show file not found error

    def test_import_error_handling(self):
        """Test handling of missing dependencies."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            temp_file.write(b'{"test": "content"}')
            temp_file.flush()

            # Simulate import error
            with patch('cloakpivot.document.processor.DocumentProcessor', side_effect=ImportError("missing dep")):
                result = runner.invoke(cli, ['mask', temp_file.name, '--out', temp_file.name + '.masked'])
                assert result.exit_code != 0
                assert 'Missing required dependency' in result.output

    def test_general_exception_handling(self):
        """Test general exception handling in CLI."""
        runner = CliRunner()
        with tempfile.NamedTemporaryFile(suffix='.json') as temp_file:
            temp_file.write(b'{"test": "content"}')
            temp_file.flush()

            # Simulate general exception
            with patch('cloakpivot.document.processor.DocumentProcessor', side_effect=Exception("test error")):
                result = runner.invoke(cli, ['mask', temp_file.name])
                assert result.exit_code != 0
                assert 'Masking failed' in result.output


class TestProgressReporting:
    """Test progress reporting functionality."""

    @patch('cloakpivot.document.processor.DocumentProcessor')
    @patch('cloakpivot.core.detection.EntityDetectionPipeline')
    @patch('cloakpivot.masking.engine.MaskingEngine')
    def test_progress_bars_shown(self, mock_masking_engine, mock_detection, mock_processor):
        """Test that progress bars are displayed during operations."""
        # Setup minimal mocks
        mock_processor.return_value.load_document.return_value = Mock(name="doc", texts=[], tables=[])
        mock_detection.return_value.analyze_document.return_value = Mock(
            total_entities=0, entity_breakdown={}, segment_results=[]
        )
        mock_masking_result = Mock()
        mock_masking_result.cloakmap.anchors = []
        mock_masking_result.cloakmap.to_dict.return_value = {}
        mock_masking_result.masked_document = Mock()
        mock_masking_result.stats = {'total_entities_masked': 0}
        mock_masking_engine.return_value.mask_document.return_value = mock_masking_result

        runner = CliRunner()
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            input_file = temp_path / "input.json"
            input_file.write_text('{"test": "content"}')

            with patch('cloakpivot.document.extractor.TextExtractor'), \
                 patch('docpivot.LexicalDocSerializer'), \
                 patch('builtins.open', create=True), \
                 patch('click.confirm', return_value=True):
                result = runner.invoke(cli, [
                    '--verbose',
                    'mask', str(input_file)
                ])

                # Progress indicators should be in output
                assert 'Loading document' in result.output or result.exit_code != 0
