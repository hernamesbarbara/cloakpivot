# CloakPivot

[![CI](https://github.com/austinogilvie/cloakpivot/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/austinogilvie/cloakpivot/actions/workflows/ci.yml)

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
pip install git+https://github.com/austinogilvie/cloakpivot.git

# Or clone and install locally
git clone https://github.com/austinogilvie/cloakpivot.git
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
print(f"Generated CloakMap with {result.cloakmap.anchor_count} anchors")

# Save the results
result.cloakmap.save_to_file("document.cloakmap.json")

# Later, restore the original
original = engine.unmask_document(result.document, result.cloakmap)
```

### CLI Example

```bash
# Mask a document with default policy
cloakpivot mask document.pdf -o masked.md -c document.cloakmap.json

# Mask with specific policy
cloakpivot mask document.pdf -o masked.md -c document.cloakmap.json --policy conservative

# Unmask later
cloakpivot unmask masked.md document.cloakmap.json -o restored.md

# View CloakMap statistics
cloakpivot stats document.cloakmap.json
```

## 📖 More Examples

### Using Different Policies

```python
from cloakpivot import CloakEngine, get_conservative_policy, get_permissive_policy

# Maximum privacy - redact everything
engine = CloakEngine(default_policy=get_conservative_policy())
result = engine.mask_document(doc)
print(f"Conservative masking: {result.entities_masked} entities masked")

# Minimal masking - only critical PII
engine = CloakEngine(default_policy=get_permissive_policy())
result = engine.mask_document(doc)
print(f"Permissive masking: {result.entities_masked} entities masked")

# Use specific entity types
result = engine.mask_document(doc, entities=['EMAIL_ADDRESS', 'CREDIT_CARD'])
print(f"Targeted masking: {result.entities_masked} entities masked")
```

### Advanced Configuration with Builder

```python
from cloakpivot import CloakEngine, MaskingPolicy, Strategy, StrategyKind

# Create custom policy
custom_policy = MaskingPolicy(
    per_entity={
        'EMAIL_ADDRESS': Strategy(StrategyKind.PARTIAL, {'visible_chars': 3, 'position': 'start'}),
        'PHONE_NUMBER': Strategy(StrategyKind.TEMPLATE, {'template': '[PHONE-NUMBER]'}),
        'CREDIT_CARD': Strategy(StrategyKind.HASH, {'algorithm': 'sha256', 'truncate': 8})
    }
)

# Fine-tune detection and masking
engine = CloakEngine.builder() \
    .with_confidence_threshold(0.9) \
    .with_languages(['en', 'es']) \
    .with_custom_policy(custom_policy) \
    .build()

result = engine.mask_document(doc)
print(f"Custom policy applied to {result.entities_masked} entities")
```

### Detect Specific Entity Types

```python
# Only mask emails and credit cards
result = engine.mask_document(doc, entities=['EMAIL_ADDRESS', 'CREDIT_CARD'])
```

### Working with DocPivot

```python
from docpivot import DocPivotEngine
from cloakpivot import CloakEngine
import json

# Load Docling JSON files directly  
with open('document.docling.json', 'r') as f:
    docling_data = json.load(f)

# Process with CloakPivot
engine = CloakEngine()
result = engine.mask_document(docling_data)

# Convert to various formats using DocPivot
pivot_engine = DocPivotEngine()
markdown = pivot_engine.to_markdown(result.document)
lexical = pivot_engine.to_lexical(result.document)
html = pivot_engine.to_html(result.document)

print(f"Converted masked document to multiple formats")
```

## 🎯 How It Works

CloakPivot creates a **CloakMap** - a secure mapping between original and masked content that enables perfect restoration:

1. **📄 Document Loading**: Use Docling to convert any document format
2. **🔍 PII Detection**: Presidio identifies sensitive information
3. **🎭 Smart Masking**: Apply configurable strategies per entity type
4. **🗺️ CloakMap Creation**: Store original values and positions securely
5. **♻️ Perfect Restoration**: Unmask with 100% accuracy

### Visual Workflow

For detailed visual representations of the processing pipeline, see our [Workflow Diagrams](docs/WORKFLOW_DIAGRAMS.md) which show:
- PDF → JSON → Masked JSON → Masked Markdown flow
- Entity detection and strategy application
- CloakMap generation and structure  
- Unmasking and restoration process
- Performance optimizations and error handling

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
- **[Docling to Lexical](examples/docling_to_lexical_workflow.py)** - Convert documents to Lexical format

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
├── tests/                # Test suite (32 tests, v2.0 API)
└── config/policies/      # Policy templates
```

## 🔧 Development

### Setup

```bash
# Clone the repository
git clone https://github.com/austinogilvie/cloakpivot.git
cd cloakpivot

# Setup development environment (one command!)
make dev
```

### Development Workflow

```bash
# Show all available commands
make help

# Quick validation before committing
make check  # Runs format + lint + type + test-fast

# Run full CI/CD pipeline locally
make all    # Runs format + lint + type + test

# Individual commands
make format      # Format with Black
make lint        # Lint with Ruff
make type        # Type check with MyPy
make test        # Run tests with coverage
```

### Testing

```bash
# Run all tests with coverage
make test

# Run specific test types
make test-unit        # Unit tests only
make test-integration # Integration tests only
make test-e2e         # End-to-end tests only

# Generate HTML coverage report
make coverage-html    # Open htmlcov/index.html

# Run tests without coverage (faster)
make test-fast
```

### Project Configuration

All project configuration is centralized in `pyproject.toml`:
- **Black**: line-length=100, target-version=py311
- **Ruff**: Comprehensive rules with integrated isort
- **MyPy**: Gradual typing with per-module overrides
- **Pytest**: Coverage integration, test markers
- **Coverage**: Branch coverage, multiple report formats

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
git clone https://github.com/austinogilvie/cloakpivot.git
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
- **[DocPivot](https://github.com/austinogilvie/docpivot)** v2.0.1+ - Document format conversions
- **[Presidio](https://github.com/microsoft/presidio)** - PII detection engine
- **[Pydantic](https://pydantic-docs.helpmanual.io/)** - Data validation

## ℹ️ Recent Updates

### DocPivot v2.0.1 Integration
CloakPivot now uses DocPivot v2.0.1 with improved performance:
- Direct JSON loading for Docling documents
- Single `DocPivotEngine` for all conversions
- Backward compatibility via `cloakpivot.compat` module

### Version 2.0 Features
Version 2.0 introduces the simplified CloakEngine API:
- Single `CloakEngine` for all masking/unmasking operations
- Clean builder pattern for advanced configuration
- Improved performance with direct JSON loading

## 💬 Support

- **Issues**: [GitHub Issues](https://github.com/austinogilvie/cloakpivot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/austinogilvie/cloakpivot/discussions)
- **Examples**: [examples/](examples/)

---

<p align="center">
  Made with ❤️ for document privacy
</p>