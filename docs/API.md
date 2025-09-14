# CloakPivot API Reference

## Core Components

### CloakEngine

The main API for PII masking and unmasking operations.

```python
from cloakpivot import CloakEngine
```

#### Constructor

```python
CloakEngine(
    analyzer_config: Optional[Dict[str, Any]] = None,
    default_policy: Optional[MaskingPolicy] = None,
    confidence_threshold: Optional[float] = None,
    conflict_resolution_config: Optional[Dict[str, Any]] = None
)
```

**Parameters:**
- `analyzer_config`: Configuration for Presidio AnalyzerEngine
  - `languages`: List of languages to support (default: `['en']`)
  - `confidence_threshold`: Minimum confidence for entity detection (0.0-1.0, default: 0.7)
- `default_policy`: Default masking policy (uses sensible defaults if not provided)
- `confidence_threshold`: Shorthand for setting analyzer confidence threshold
- `conflict_resolution_config`: Configuration for handling overlapping entities

#### Methods

##### mask_document

```python
mask_document(
    document: DoclingDocument,
    entities: Optional[List[str]] = None,
    policy: Optional[MaskingPolicy] = None
) -> MaskResult
```

Mask PII entities in a document.

**Parameters:**
- `document`: DoclingDocument to mask
- `entities`: Entity types to detect (default: common PII types)
- `policy`: Masking policy to use (default: uses engine's default policy)

**Returns:**
- `MaskResult` with:
  - `document`: Masked DoclingDocument
  - `cloakmap`: CloakMap for reversal
  - `entities_found`: Number of entities detected
  - `entities_masked`: Number of entities masked

**Example:**
```python
engine = CloakEngine()
result = engine.mask_document(doc)
print(f"Masked {result.entities_masked} entities")
```

##### unmask_document

```python
unmask_document(
    document: DoclingDocument,
    cloakmap: CloakMap
) -> DoclingDocument
```

Restore original PII using a CloakMap.

**Parameters:**
- `document`: Masked DoclingDocument
- `cloakmap`: CloakMap containing original values

**Returns:**
- `DoclingDocument` with original content restored

**Example:**
```python
original = engine.unmask_document(masked_doc, cloakmap)
```

##### builder (class method)

```python
@classmethod
builder() -> CloakEngineBuilder
```

Create a builder for advanced configuration.

**Returns:**
- `CloakEngineBuilder` instance

**Example:**
```python
engine = CloakEngine.builder()
    .with_confidence_threshold(0.9)
    .with_languages(['en', 'es'])
    .build()
```

### CloakEngineBuilder

Fluent builder for advanced CloakEngine configuration.

```python
from cloakpivot import CloakEngineBuilder
```

#### Methods

##### with_confidence_threshold

```python
with_confidence_threshold(threshold: float) -> CloakEngineBuilder
```

Set minimum confidence for entity detection (0.0-1.0).

##### with_languages

```python
with_languages(languages: List[str]) -> CloakEngineBuilder
```

Set supported languages for entity detection.

##### with_custom_policy

```python
with_custom_policy(policy: MaskingPolicy) -> CloakEngineBuilder
```

Set a custom default masking policy.

##### with_analyzer_config

```python
with_analyzer_config(config: dict) -> CloakEngineBuilder
```

Set advanced Presidio analyzer configuration.

##### with_conflict_resolution

```python
with_conflict_resolution(config: ConflictResolutionConfig) -> CloakEngineBuilder
```

Configure handling of overlapping entities.

##### with_presidio_engine

```python
with_presidio_engine(use: bool = True) -> CloakEngineBuilder
```

Enable/disable Presidio engine (default: enabled).

##### build

```python
build() -> CloakEngine
```

Build the configured CloakEngine instance.

### MaskResult

Result of a masking operation.

```python
@dataclass
class MaskResult:
    document: DoclingDocument      # Masked document
    cloakmap: CloakMap             # Mapping for reversal
    entities_found: int            # Number of entities detected
    entities_masked: int           # Number of entities masked
```

### CloakMap

Secure mapping between original and masked content.

#### Methods

##### save_to_file

```python
save_to_file(path: Path) -> None
```

Save CloakMap to a JSON file.

##### load_from_file (class method)

```python
@classmethod
load_from_file(path: Path) -> CloakMap
```

Load CloakMap from a JSON file.

**Example:**
```python
# Save
result.cloakmap.save_to_file("document.cloakmap.json")

# Load
cloakmap = CloakMap.load_from_file("document.cloakmap.json")
```

## Masking Policies

### MaskingPolicy

Defines how different entity types should be masked.

```python
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
```

#### Structure

```python
@dataclass
class MaskingPolicy:
    per_entity: Dict[str, Strategy]  # Entity-specific strategies
    default_strategy: Strategy        # Fallback strategy
```

### Pre-configured Policies

#### get_default_policy

```python
from cloakpivot.defaults import get_default_policy

policy = get_default_policy()
```

Balanced policy with readable templates:
- Emails → `[EMAIL]`
- Names → `[NAME]`
- Phones → `[PHONE]`
- Credit cards → `[CARD]`
- SSNs → Partial masking (last 4 digits visible)

#### get_conservative_policy

```python
from cloakpivot.defaults import get_conservative_policy

policy = get_conservative_policy()
```

Maximum privacy with full redaction:
- All entities → `████████████`

#### get_permissive_policy

```python
from cloakpivot.defaults import get_permissive_policy

policy = get_permissive_policy()
```

Minimal masking for readability:
- Critical PII → Templates
- Dates/locations → Preserved
- Names → Partial masking

### Custom Policies

```python
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind

custom_policy = MaskingPolicy(
    per_entity={
        "EMAIL_ADDRESS": Strategy(
            kind=StrategyKind.HASH,
            params={"length": 8}
        ),
        "PERSON": Strategy(
            kind=StrategyKind.PARTIAL,
            params={"visible_chars": 2, "position": "start"}
        ),
        "CREDIT_CARD": Strategy(
            kind=StrategyKind.REDACT,
            params={}
        ),
    },
    default_strategy=Strategy(
        kind=StrategyKind.TEMPLATE,
        params={"template": "[REDACTED]"}
    )
)
```

## Masking Strategies

### StrategyKind Enum

```python
from cloakpivot.core.strategies import StrategyKind
```

Available masking strategies:

| Strategy | Description | Example |
|----------|-------------|---------|
| `REDACT` | Replace with redaction characters | `john@example.com` → `████████████████` |
| `TEMPLATE` | Replace with entity type template | `John Smith` → `[PERSON]` |
| `PARTIAL` | Show partial content | `555-123-4567` → `555-XXX-XXXX` |
| `HASH` | Replace with deterministic hash | `123-45-6789` → `a7b2c8d1` |
| `KEEP` | Preserve original value | `2024-01-15` → `2024-01-15` |

### Strategy Configuration

```python
from cloakpivot.core.strategies import Strategy, StrategyKind

# Redaction strategy
redact = Strategy(
    kind=StrategyKind.REDACT,
    params={"char": "█"}  # Optional custom character
)

# Template strategy
template = Strategy(
    kind=StrategyKind.TEMPLATE,
    params={"template": "[PRIVATE]"}
)

# Partial masking
partial = Strategy(
    kind=StrategyKind.PARTIAL,
    params={
        "visible_chars": 4,      # Number of chars to show
        "position": "end",       # "start", "end", or "middle"
        "mask_char": "X"         # Character for masking
    }
)

# Hash strategy
hash_strategy = Strategy(
    kind=StrategyKind.HASH,
    params={
        "length": 8,             # Hash output length
        "preserve_format": True  # Maintain original format
    }
)
```

## Entity Types

Default entity types detected:

| Entity Type | Description | Example |
|-------------|-------------|---------|
| `EMAIL_ADDRESS` | Email addresses | `john@example.com` |
| `PERSON` | Person names | `John Smith` |
| `PHONE_NUMBER` | Phone numbers | `555-123-4567` |
| `CREDIT_CARD` | Credit card numbers | `4111-1111-1111-1111` |
| `US_SSN` | US Social Security Numbers | `123-45-6789` |
| `LOCATION` | Addresses and locations | `123 Main St, NYC` |
| `DATE_TIME` | Dates and times | `2024-01-15` |
| `MEDICAL_LICENSE` | Medical license numbers | `MD12345` |
| `URL` | Web URLs | `https://example.com` |
| `IP_ADDRESS` | IP addresses | `192.168.1.1` |

## CLI Usage

### Installation

```bash
pip install cloakpivot
```

### Commands

#### mask

Mask PII in a document:

```bash
cloakpivot mask document.pdf -o masked.md -c document.cloakmap.json
```

**Options:**
- `-o, --output`: Output file path
- `-c, --cloakmap`: CloakMap output file
- `-p, --policy`: Path to policy YAML file
- `-t, --confidence`: Confidence threshold (0.0-1.0)
- `-f, --format`: Output format (markdown/json/text)

#### unmask

Restore original content:

```bash
cloakpivot unmask masked.md document.cloakmap.json -o restored.md
```

**Options:**
- `-o, --output`: Output file path
- `-f, --format`: Output format (markdown/json/text)

#### version

Show version:

```bash
cloakpivot version
```

## Error Handling

### Common Exceptions

```python
from cloakpivot.core.exceptions import (
    CloakPivotError,          # Base exception
    MaskingError,             # Masking operation failed
    UnmaskingError,           # Unmasking operation failed
    PolicyError,              # Invalid policy configuration
    CloakMapError            # CloakMap loading/saving error
)
```

### Error Handling Example

```python
from cloakpivot import CloakEngine
from cloakpivot.core.exceptions import CloakPivotError

try:
    engine = CloakEngine()
    result = engine.mask_document(doc)
except CloakPivotError as e:
    print(f"Masking failed: {e}")
```

## Working with Documents

### Document Conversion

```python
from docling.document_converter import DocumentConverter

# Convert various formats to DoclingDocument
converter = DocumentConverter()
result = converter.convert("document.pdf")  # Also: .docx, .pptx, .html, etc.
doc = result.document
```

### Export Formats

```python
# After masking
masked_doc = result.document

# Export to different formats
markdown = masked_doc.export_to_markdown()
json_dict = masked_doc.export_to_dict()
text = masked_doc.export_to_text()
```

## Advanced Topics

### Custom Entity Recognizers

```python
# Configure custom recognizers via analyzer_config
engine = CloakEngine(
    analyzer_config={
        'custom_recognizers': [my_custom_recognizer],
        'enabled_recognizers': ['EMAIL_ADDRESS', 'MY_CUSTOM_TYPE']
    }
)
```

### Conflict Resolution

When entities overlap, configure resolution strategy:

```python
from cloakpivot.core.normalization import ConflictResolutionConfig

config = ConflictResolutionConfig(
    strategy="PREFER_LONGER",  # or "PREFER_HIGHER_CONFIDENCE"
    min_overlap_ratio=0.5
)

engine = CloakEngine(conflict_resolution_config=config)
```

### Performance Optimization

```python
# For large documents, tune confidence threshold
engine = CloakEngine.builder()
    .with_confidence_threshold(0.8)  # Higher = fewer false positives
    .build()

# Process specific entity types only
result = engine.mask_document(
    doc,
    entities=['EMAIL_ADDRESS', 'CREDIT_CARD']  # Skip other types
)
```

## Examples

### Round-trip Example

```python
from cloakpivot import CloakEngine
from docling.document_converter import DocumentConverter

# Convert document
converter = DocumentConverter()
doc = converter.convert("sensitive.pdf").document

# Mask PII
engine = CloakEngine()
result = engine.mask_document(doc)

# Save masked document and CloakMap
with open("masked.md", "w") as f:
    f.write(result.document.export_to_markdown())
result.cloakmap.save_to_file("sensitive.cloakmap.json")

print(f"Masked {result.entities_masked} PII entities")

# Later: restore original
from cloakpivot.core.cloakmap import CloakMap

cloakmap = CloakMap.load_from_file("sensitive.cloakmap.json")
original = engine.unmask_document(result.document, cloakmap)
```

### Policy Customization Example

```python
from cloakpivot import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind

# Create custom policy
policy = MaskingPolicy(
    per_entity={
        "EMAIL_ADDRESS": Strategy(
            StrategyKind.PARTIAL,
            {"visible_chars": 3, "position": "start"}
        ),
        "CREDIT_CARD": Strategy(
            StrategyKind.REDACT,
            {}
        ),
    },
    default_strategy=Strategy(
        StrategyKind.TEMPLATE,
        {"template": "[CONFIDENTIAL]"}
    )
)

# Use custom policy
engine = CloakEngine(default_policy=policy)
result = engine.mask_document(doc)
```

## Version Compatibility

- **CloakPivot 2.0+**: Uses CloakEngine API
- **CloakPivot 1.x**: Legacy MaskingEngine/UnmaskingEngine (deprecated)
- **CloakMap Format**: Supports v1.0 and v2.0
- **Python**: 3.8+
- **Docling**: 1.7.0+
- **Presidio**: 2.2+

## Migration from v1.x

See [Migration Guide](MIGRATION.md) for upgrading from v1.x to v2.0.

## Support

- **Issues**: [GitHub Issues](https://github.com/hernamesbarbara/cloakpivot/issues)
- **Discussions**: [GitHub Discussions](https://github.com/hernamesbarbara/cloakpivot/discussions)
- **Examples**: [examples/](https://github.com/hernamesbarbara/cloakpivot/tree/main/examples)