# CloakPivot

[![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)](https://github.com/hernamesbarbara/cloakpivot)
[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Simple, reversible PII masking for documents.** One-line masking and unmasking while preserving document structure.

CloakPivot provides a Presidio-like simple API for detecting and masking PII in documents, with the unique ability to perfectly restore the original content later using a secure mapping file (CloakMap).

## ✨ Key Features

- **🎯 One-line masking**: `engine.mask_document(doc)` - that's it!
- **🔄 Perfect reversal**: Restore original content exactly with CloakMap
- **📄 Document-aware**: Works with Docling documents, preserving structure
- **🛡️ Smart defaults**: Detects common PII types automatically (emails, phones, SSNs, etc.)
- **⚙️ Flexible policies**: Customize masking strategies per entity type
- **🚀 Builder pattern**: Advanced configuration when you need it

## 🚀 Quick Start

### Installation

```bash
# From GitHub (until PyPI release)
pip install git+https://github.com/hernamesbarbara/cloakpivot.git

# Or clone and install locally
git clone https://github.com/hernamesbarbara/cloakpivot.git
cd cloakpivot
pip install -e .
```

### Basic Usage

```python
from cloakpivot import CloakEngine
from docling.document_converter import DocumentConverter

# Convert your document
converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# One-line PII masking!
engine = CloakEngine()
result = engine.mask_document(doc)

print(f"Masked {result.entities_masked} PII entities")
# Save the masked document and CloakMap...

# Later, restore the original
original = engine.unmask_document(result.document, result.cloakmap)
```

### CLI Example

```bash
# Mask a document
cloakpivot mask document.pdf -o masked.md -c document.cloakmap.json

# Unmask later
cloakpivot unmask masked.md document.cloakmap.json -o restored.md
```

## 📖 More Examples

### Using Different Policies

```python
from cloakpivot import CloakEngine, get_conservative_policy, get_permissive_policy

# Maximum privacy - redact everything
engine = CloakEngine(default_policy=get_conservative_policy())
result = engine.mask_document(doc)

# Minimal masking - only critical PII
engine = CloakEngine(default_policy=get_permissive_policy())
result = engine.mask_document(doc)
```

### Advanced Configuration with Builder

```python
# Fine-tune detection and masking
engine = CloakEngine.builder() \
    .with_confidence_threshold(0.9) \
    .with_languages(['en', 'es']) \
    .with_custom_policy(my_policy) \
    .build()

result = engine.mask_document(doc)
```

### Detect Specific Entity Types

```python
# Only mask emails and credit cards
result = engine.mask_document(doc, entities=['EMAIL_ADDRESS', 'CREDIT_CARD'])
```

## 🎯 How It Works

CloakPivot creates a **CloakMap** - a secure mapping between original and masked content that enables perfect restoration:

1. **📄 Document Loading**: Use Docling to convert any document format
2. **🔍 PII Detection**: Presidio identifies sensitive information
3. **🎭 Smart Masking**: Apply configurable strategies per entity type
4. **🗺️ CloakMap Creation**: Store original values and positions securely
5. **♻️ Perfect Restoration**: Unmask with 100% accuracy

### Masking Strategies

| Strategy | Example Input | Example Output | Use Case |
|----------|--------------|----------------|----------|
| **REDACT** | `john.doe@email.com` | `████████████████` | Maximum privacy |
| **TEMPLATE** | `John Smith` | `[PERSON]` | Clear entity types |
| **PARTIAL** | `555-123-4567` | `555-XXX-XXXX` | Preserve format |
| **HASH** | `123-45-6789` | `a7b2c8d1` | Consistent replacement |

## 📖 Documentation

- **[Quick Start](examples/simple_usage.py)** - Basic usage with test data
- **[Advanced Configuration](examples/advanced_usage.py)** - Builder pattern and policies
- **[PDF Workflow](examples/pdf_workflow.py)** - Complete PDF processing example
- **[Pipeline Integration](examples/docling_integration.py)** - Working with DoclingDocument files
- **[API Reference](docs/API.md)** - Complete API documentation
- **[Migration Guide](docs/MIGRATION.md)** - Upgrading from v1.x

## 🏗️ Project Structure

```
cloakpivot/
├── cloakpivot/           # Main package
│   ├── engine.py         # CloakEngine - main API
│   ├── engine_builder.py # Builder pattern configuration
│   ├── defaults.py       # Default policies and settings
│   ├── cli/              # Command-line interface
│   ├── core/             # Core functionality (anchors, policies, etc.)
│   ├── masking/          # Masking engine
│   └── unmasking/        # Unmasking engine
├── examples/             # Usage examples
│   ├── simple_usage.py   # Quick start with test data
│   ├── advanced_usage.py # Builder pattern and policies
│   ├── pdf_workflow.py   # Complete PDF processing
│   └── docling_integration.py # Pipeline integration
├── tests/                # Test suite (60+ tests)
└── config/policies/      # Policy templates
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

## 🏥 Common Use Cases

- **🏥 Healthcare**: De-identify patient records while preserving document structure
- **💳 Financial**: Mask credit cards, SSNs, and account numbers in reports
- **👥 HR**: Redact employee PII in documents for compliance
- **🧪 Development**: Create safe test data from production documents
- **📝 Legal**: Redact sensitive information in legal documents
- **📧 Customer Support**: Remove PII from support tickets and logs

## 🎯 Supported Entity Types

Default detection includes:
- **Personal**: Names, phone numbers, addresses
- **Financial**: Credit cards, bank accounts, SSNs
- **Digital**: Email addresses, URLs, IP addresses
- **Healthcare**: Medical license numbers, patient IDs
- **Dates & Times**: Birthdays, appointments
- **Custom**: Add your own entity recognizers

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone the repository
git clone https://github.com/hernamesbarbara/cloakpivot.git
cd cloakpivot

# Install in development mode
pip install -e ".[dev]"

# Run tests
python -m pytest
```

## 📄 License

CloakPivot is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🔗 Dependencies

- **[Docling](https://github.com/DS4SD/docling)** - Document parsing and conversion
- **[Presidio](https://github.com/microsoft/presidio)** - PII detection engine
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation

## ℹ️ Version 2.0 Breaking Changes

Version 2.0 introduces the simplified CloakEngine API. If upgrading from v1.x:
- `MaskingEngine` and `UnmaskingEngine` are deprecated
- Use `CloakEngine` for all operations
- See [Migration Guide](docs/MIGRATION.md) for details

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/hernamesbarbara/cloakpivot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hernamesbarbara/cloakpivot/discussions)
- **Examples**: [examples/](examples/)

---

<p align="center">
  Made with ❤️ for document privacy
</p>