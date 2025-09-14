# Test Data Directory

This directory contains sample data files for running the examples.

## Structure

- `pdf/` - Sample PDF files for testing
- `docling/` - Sample DoclingDocument JSON files

## Creating Test Data

### Option 1: Use the provided sample generator

```bash
python examples/generate_test_data.py
```

### Option 2: Add your own files

Place your test files in the appropriate subdirectories:
- PDF files → `data/pdf/`
- DoclingDocument JSON files → `data/docling/`

## Privacy Note

⚠️ **Do not commit real documents containing actual PII to this repository!**

The sample files should only contain synthetic/fake PII for testing purposes.