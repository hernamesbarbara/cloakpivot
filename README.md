# CloakPivot

[![PyPI version](https://img.shields.io/pypi/v/cloakpivot.svg)](https://pypi.python.org/pypi/cloakpivot)
[![Python versions](https://img.shields.io/pypi/pyversions/cloakpivot.svg)](https://pypi.python.org/pypi/cloakpivot)
[![CI status](https://github.com/your-org/cloakpivot/workflows/CI/badge.svg)](https://github.com/your-org/cloakpivot/actions)

CloakPivot is a Python package that enables **reversible document masking** while preserving structure and formatting. It leverages DocPivot for robust document processing and Presidio for PII detection and anonymization.

## 🔑 Key Features

- **🔄 Reversible Masking**: Mask PII while maintaining the ability to restore original content
- **📋 Structure Preservation**: Maintain document layout, formatting, and hierarchy during masking
- **⚙️ Policy-Driven**: Configurable masking strategies per entity type with comprehensive policy system
- **📄 Format Support**: Works with multiple document formats through DocPivot integration
- **🔒 Security**: Optional encryption and integrity verification for CloakMaps
- **🖥️ CLI & API**: Both command-line interface and programmatic Python API

## 🚀 Quick Start

### Installation

```bash
pip install cloakpivot
```

### Basic Usage

#### CLI Example
```bash
# Mask a document
cloakpivot mask document.pdf --out masked.json --cloakmap map.json

# Unmask later
cloakpivot unmask masked.json --cloakmap map.json --out restored.json
```

#### Python API Example
```python
from cloakpivot import mask_document, unmask_document

# Mask a document
result = mask_document("document.pdf", policy="my-policy.yaml")

# Unmask later
restored = unmask_document(result.masked_path, result.cloakmap_path)
```

## 🎯 How It Works

CloakPivot creates a **CloakMap** - a secure mapping between original and masked content that enables perfect restoration:

1. **Document Processing**: Load documents using DocPivot integration
2. **PII Detection**: Identify sensitive information using Presidio
3. **Strategic Masking**: Apply configurable masking strategies per entity type
4. **CloakMap Generation**: Create secure anchors for reversible transformations
5. **Perfect Restoration**: Unmask documents with 100% accuracy

### Masking Strategies

- **Redaction**: Replace with characters (e.g., `████████`)
- **Template**: Use templates (e.g., `[PERSON]`, `[EMAIL]`)
- **Partial**: Show partial content (e.g., `joh***@company.com`)
- **Hash**: Consistent hashing (e.g., `a7b2c8d1`)

## 📖 Documentation

Comprehensive documentation is available in the [`docs/`](docs/) directory:

- **[Getting Started Guide](docs/notebooks/01_getting_started.ipynb)** - Interactive tutorial
- **[CLI Reference](docs/cli/overview.rst)** - Complete command-line documentation
- **[API Documentation](docs/api/)** - Python API reference
- **[Policy Development](docs/notebooks/02_policy_development.ipynb)** - Creating custom policies
- **[Examples](examples/)** - Integration examples and use cases

### Quick Links

- 📘 [Full Documentation](docs/index.rst)
- 🔧 [CLI Commands](docs/cli/overview.rst)
- 📝 [Policy Configuration](docs/policies/)
- 🧪 [Jupyter Notebooks](docs/notebooks/)
- 💼 [Industry Examples](examples/)

## 🏗️ Repository Structure

```
cloakpivot/
├── cloakpivot/           # Main package
│   ├── cli/              # Command-line interface
│   ├── core/             # Core masking/unmasking logic
│   ├── document/         # DocPivot integration
│   ├── masking/          # Masking engines and strategies
│   ├── unmasking/        # Unmasking and restoration
│   ├── policies/         # Policy management and examples
│   ├── plugins/          # Extension system
│   ├── storage/          # CloakMap storage backends
│   └── observability/    # Monitoring and diagnostics
├── docs/                 # Comprehensive documentation
├── examples/             # Integration examples
├── tests/                # Test suite (142+ tests)
├── policies/             # Policy templates and examples
└── specification/        # Technical specifications
```

## 🔧 Development

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=cloakpivot

# Run specific test categories
python -m pytest -m unit        # Unit tests
python -m pytest -m integration # Integration tests
python -m pytest -m e2e         # End-to-end tests
```

### Code Quality

```bash
# Format code
black cloakpivot/ tests/

# Lint code
ruff check cloakpivot/ tests/

# Type checking
mypy cloakpivot/
```

## 🏥 Use Cases

### Healthcare & HIPAA Compliance
```python
# HIPAA-compliant masking for medical records
result = mask_document(
    "patient_record.pdf", 
    policy="policies/industries/healthcare/hipaa-compliant.yaml"
)
```

### Document Processing Pipelines
```python
# Batch processing with custom policies
from cloakpivot.core.batch import BatchProcessor

processor = BatchProcessor(policy="balanced.yaml")
results = processor.process_directory("./sensitive_docs/")
```

### Development & Testing
```python
# Mask production data for development environments
result = mask_document(
    "production_db_export.json",
    policy="policies/templates/permissive.yaml"
)
```

## 📊 Performance

- **Fast Processing**: Optimized for large documents and batch operations
- **Memory Efficient**: Streaming processing for large files
- **Parallel Analysis**: Multi-core entity detection and processing
- **Comprehensive Testing**: 142+ tests including performance benchmarks

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/development/contributing.rst) for details on:

- Setting up the development environment
- Running tests and quality checks
- Submitting pull requests
- Plugin development
- Documentation improvements

## 📄 License

CloakPivot is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🔗 Related Projects

- **[DocPivot](https://github.com/example/docpivot)** - Document processing and format conversion
- **[Presidio](https://github.com/microsoft/presidio)** - PII detection and anonymization
- **[Policy Templates](policies/)** - Industry-specific masking policies

## 💡 Examples

### Healthcare Document Masking
```python
from cloakpivot import mask_document

# Mask medical record with HIPAA compliance
result = mask_document(
    "medical_record.pdf",
    policy_path="policies/industries/healthcare/hipaa-compliant.yaml",
    output_format="docling"
)

print(f"Masked {result.stats.total_entities_found} PII entities")
print(f"Document: {result.masked_path}")
print(f"CloakMap: {result.cloakmap_path}")
```

### Custom Policy Development
```python
from cloakpivot import MaskingPolicy, Strategy, StrategyKind

# Create custom policy
policy = MaskingPolicy(
    locale="en",
    default_strategy=Strategy(
        kind=StrategyKind.TEMPLATE,
        parameters={"template": "[REDACTED]"}
    ),
    per_entity={
        "EMAIL_ADDRESS": Strategy(
            kind=StrategyKind.PARTIAL,
            parameters={"visible_chars": 3, "position": "start"}
        ),
        "PHONE_NUMBER": Strategy(
            kind=StrategyKind.REDACT,
            parameters={"redact_char": "X"}
        )
    }
)

result = mask_document("document.json", policy=policy)
```

For more examples, see the [`examples/`](examples/) directory and [documentation notebooks](docs/notebooks/).