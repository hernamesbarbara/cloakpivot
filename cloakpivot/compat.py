"""Compatibility module for docpivot v2.0.1 migration."""

import json
from pathlib import Path
from typing import Union
from docling_core.types import DoclingDocument
from docpivot import DocPivotEngine


def load_document(file_path: Union[str, Path]) -> DoclingDocument:
    """Load a Docling JSON document directly.

    This function replaces the old docpivot.load_document function
    for loading Docling JSON files.

    Args:
        file_path: Path to the Docling JSON file

    Returns:
        DoclingDocument: The loaded document

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the JSON is invalid
    """
    with open(file_path, 'r') as f:
        doc_dict = json.load(f)
    return DoclingDocument.model_validate(doc_dict)


def to_lexical(document: DoclingDocument, pretty: bool = False) -> dict:
    """Convert a DoclingDocument to Lexical format.

    This function replaces the old docpivot.to_lexical function.

    Args:
        document: DoclingDocument to convert
        pretty: Whether to format the output with indentation

    Returns:
        dict: The Lexical JSON as a dictionary
    """
    engine = DocPivotEngine()
    result = engine.convert_to_lexical(document, pretty=pretty)
    return json.loads(result.content)