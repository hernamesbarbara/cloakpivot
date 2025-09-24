"""Unit tests for cloakpivot.compat module."""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, MagicMock, patch

import pytest

# Mock docpivot module before importing cloakpivot.compat
mock_docpivot = MagicMock()
mock_engine_class = MagicMock()
mock_docpivot.DocPivotEngine = mock_engine_class
sys.modules['docpivot'] = mock_docpivot

from cloakpivot.compat import load_document, to_lexical
from cloakpivot.type_imports import DoclingDocument


class TestLoadDocument:
    """Test load_document function."""

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch("cloakpivot.compat.DoclingDocument.model_validate")
    def test_load_document_success(self, mock_validate, mock_json_load, mock_open_file):
        """Test successful document loading."""
        # Setup
        test_data = {"test": "data", "version": "1.0"}
        mock_json_load.return_value = test_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc

        # Execute
        result = load_document("test.json")

        # Verify
        assert result == mock_doc
        mock_json_load.assert_called_once()
        mock_validate.assert_called_once_with(test_data)

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch("cloakpivot.compat.DoclingDocument.model_validate")
    def test_load_document_with_path_object(self, mock_validate, mock_json_load, mock_open_file):
        """Test loading document with Path object."""
        # Setup
        test_data = {"test": "data"}
        mock_json_load.return_value = test_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc
        test_path = Path("test.json")

        # Execute
        result = load_document(test_path)

        # Verify
        assert result == mock_doc
        mock_validate.assert_called_once_with(test_data)

    @patch("cloakpivot.compat.Path.open", side_effect=FileNotFoundError("File not found"))
    def test_load_document_file_not_found(self, mock_open_file):
        """Test load_document with non-existent file."""
        with pytest.raises(FileNotFoundError, match="File not found"):
            load_document("nonexistent.json")

    @patch("cloakpivot.compat.Path.open")
    @patch(
        "cloakpivot.compat.json.load", side_effect=json.JSONDecodeError("Invalid JSON", "doc", 0)
    )
    def test_load_document_invalid_json(self, mock_json_load, mock_open_file):
        """Test load_document with invalid JSON."""
        with pytest.raises(json.JSONDecodeError, match="Invalid JSON"):
            load_document("invalid.json")

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch(
        "cloakpivot.compat.DoclingDocument.model_validate",
        side_effect=ValueError("Invalid document"),
    )
    def test_load_document_invalid_docling_format(
        self, mock_validate, mock_json_load, mock_open_file
    ):
        """Test load_document with invalid DoclingDocument format."""
        mock_json_load.return_value = {"invalid": "data"}

        with pytest.raises(ValueError, match="Invalid document"):
            load_document("test.json")

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch("cloakpivot.compat.DoclingDocument.model_validate")
    def test_load_document_with_complex_data(self, mock_validate, mock_json_load, mock_open_file):
        """Test loading document with complex nested data."""
        # Setup
        complex_data = {
            "version": "2.0",
            "metadata": {
                "author": "Test Author",
                "date": "2024-01-01",
                "tags": ["test", "document", "complex"],
            },
            "content": [
                {"type": "paragraph", "text": "Test content"},
                {"type": "table", "rows": [["A", "B"], ["1", "2"]]},
                {"type": "list", "items": ["item1", "item2", "item3"]},
            ],
        }
        mock_json_load.return_value = complex_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc

        # Execute
        result = load_document("complex.json")

        # Verify
        assert result == mock_doc
        mock_validate.assert_called_once_with(complex_data)


class TestToLexical:
    """Test to_lexical function."""

    def test_to_lexical_default(self):
        """Test to_lexical with default parameters."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            lexical_content = {"root": {"type": "root", "children": []}}
            mock_result = Mock()
            mock_result.content = json.dumps(lexical_content)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            result = to_lexical(mock_document)

            # Verify
            assert result == lexical_content
            mock_engine.convert_to_lexical.assert_called_once_with(mock_document, pretty=False)

    def test_to_lexical_with_pretty_true(self):
        """Test to_lexical with pretty=True."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            lexical_content = {
                "root": {"type": "root", "children": [{"type": "paragraph", "text": "Test"}]}
            }
            mock_result = Mock()
            mock_result.content = json.dumps(lexical_content, indent=2)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            result = to_lexical(mock_document, pretty=True)

            # Verify
            assert result == lexical_content
            mock_engine.convert_to_lexical.assert_called_once_with(mock_document, pretty=True)

    def test_to_lexical_with_complex_lexical_structure(self):
        """Test to_lexical with complex Lexical structure."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            complex_lexical = {
                "root": {
                    "type": "root",
                    "children": [
                        {
                            "type": "paragraph",
                            "children": [
                                {"type": "text", "text": "Bold text", "format": ["bold"]},
                                {"type": "text", "text": " normal text"},
                                {"type": "text", "text": " italic", "format": ["italic"]},
                            ],
                        },
                        {
                            "type": "heading",
                            "tag": "h2",
                            "children": [{"type": "text", "text": "Section Title"}],
                        },
                        {
                            "type": "list",
                            "listType": "bullet",
                            "children": [
                                {
                                    "type": "listitem",
                                    "value": 1,
                                    "children": [{"type": "text", "text": "Item 1"}],
                                },
                                {
                                    "type": "listitem",
                                    "value": 2,
                                    "children": [{"type": "text", "text": "Item 2"}],
                                },
                            ],
                        },
                    ],
                }
            }
            mock_result = Mock()
            mock_result.content = json.dumps(complex_lexical)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            result = to_lexical(mock_document)

            # Verify
            assert result == complex_lexical
            assert "root" in result
            assert result["root"]["type"] == "root"
            assert len(result["root"]["children"]) == 3

    def test_to_lexical_empty_document(self):
        """Test to_lexical with empty document."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            empty_lexical = {"root": {"type": "root", "children": []}}
            mock_result = Mock()
            mock_result.content = json.dumps(empty_lexical)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            result = to_lexical(mock_document)

            # Verify
            assert result == empty_lexical
            assert result["root"]["children"] == []

    def test_to_lexical_conversion_error(self):
        """Test to_lexical when conversion fails."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            mock_engine.convert_to_lexical.side_effect = Exception("Conversion failed")

            # Execute & Verify
            with pytest.raises(Exception, match="Conversion failed"):
                to_lexical(mock_document)

    def test_to_lexical_invalid_json_response(self):
        """Test to_lexical when engine returns invalid JSON."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            mock_result = Mock()
            mock_result.content = "invalid json {"  # Invalid JSON
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute & Verify
            with pytest.raises(json.JSONDecodeError):
                to_lexical(mock_document)

    def test_to_lexical_with_special_characters(self):
        """Test to_lexical with special characters in content."""
        # Setup
        mock_document = Mock(spec=DoclingDocument)

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine

            lexical_with_special = {
                "root": {
                    "type": "root",
                    "children": [{"type": "text", "text": "Special chars: €£¥ • — " "''"}],
                }
            }
            mock_result = Mock()
            mock_result.content = json.dumps(lexical_with_special)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            result = to_lexical(mock_document)

            # Verify
            assert result == lexical_with_special
            assert "€£¥" in result["root"]["children"][0]["text"]


class TestCompatibilityIntegration:
    """Test integration between load_document and to_lexical."""

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch("cloakpivot.compat.DoclingDocument.model_validate")
    def test_load_and_convert_workflow(
        self, mock_validate, mock_json_load, mock_open_file
    ):
        """Test complete workflow of loading and converting a document."""
        # Setup - Load document
        doc_data = {"content": "test document"}
        mock_json_load.return_value = doc_data
        mock_doc = Mock(spec=DoclingDocument)
        mock_validate.return_value = mock_doc

        with patch("cloakpivot.compat.DocPivotEngine") as mock_engine_class:
            # Setup - Convert to lexical
            mock_engine = Mock()
            mock_engine_class.return_value = mock_engine
            lexical_content = {"root": {"type": "root", "children": []}}
            mock_result = Mock()
            mock_result.content = json.dumps(lexical_content)
            mock_engine.convert_to_lexical.return_value = mock_result

            # Execute
            loaded_doc = load_document("test.json")
            lexical_output = to_lexical(loaded_doc)

            # Verify
            assert loaded_doc == mock_doc
            assert lexical_output == lexical_content
            mock_validate.assert_called_once_with(doc_data)
            mock_engine.convert_to_lexical.assert_called_once_with(mock_doc, pretty=False)

    @patch("cloakpivot.compat.Path.open")
    @patch("cloakpivot.compat.json.load")
    @patch("cloakpivot.compat.DoclingDocument.model_validate")
    def test_load_document_multiple_files(self, mock_validate, mock_json_load, mock_open_file):
        """Test loading multiple documents sequentially."""
        # Setup
        doc1_data = {"id": 1, "content": "doc1"}
        doc2_data = {"id": 2, "content": "doc2"}
        mock_json_load.side_effect = [doc1_data, doc2_data]

        mock_doc1 = Mock(spec=DoclingDocument, id=1)
        mock_doc2 = Mock(spec=DoclingDocument, id=2)
        mock_validate.side_effect = [mock_doc1, mock_doc2]

        # Execute
        result1 = load_document("doc1.json")
        result2 = load_document("doc2.json")

        # Verify
        assert result1 == mock_doc1
        assert result2 == mock_doc2
        assert mock_validate.call_count == 2