# Migration Guide: CloakPivot v1.x to v2.0

## Overview

CloakPivot v2.0 introduces the simplified `CloakEngine` API, replacing the separate `MaskingEngine` and `UnmaskingEngine` classes. This guide helps you migrate existing code to the new API.

## Key Changes

### 1. Single Engine Class

**Before (v1.x):**
```python
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine

masking_engine = MaskingEngine(config)
unmasking_engine = UnmaskingEngine()
```

**After (v2.0):**
```python
from cloakpivot import CloakEngine

engine = CloakEngine()  # Single engine for both operations
```

### 2. Simplified Masking

**Before (v1.x):**
```python
# Complex multi-step process
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.core.analyzer import AnalyzerConfig
from presidio_analyzer import AnalyzerEngine

# Extract text
extractor = TextExtractor()
text_result = extractor.extract_full_text(doc)
segments = extractor.extract_text_segments(doc)

# Configure analyzer
analyzer_config = AnalyzerConfig(language="en", min_confidence=0.7)
analyzer = AnalyzerEngine()

# Detect entities
entities = analyzer.analyze(text_result, language="en")

# Apply masking
masking_engine = MaskingEngine()
result = masking_engine.mask_document(doc, entities, policy, segments)
```

**After (v2.0):**
```python
# One-line masking
engine = CloakEngine()
result = engine.mask_document(doc)
```

### 3. Simplified Unmasking

**Before (v1.x):**
```python
unmasking_engine = UnmaskingEngine()
result = unmasking_engine.unmask_document(masked_doc, cloakmap)
unmasked_doc = result.unmasked_document
```

**After (v2.0):**
```python
unmasked_doc = engine.unmask_document(masked_doc, cloakmap)
```

### 4. Return Types

**Before (v1.x):**
```python
# MaskingEngine returns MaskingResult
result = masking_engine.mask_document(...)
masked_doc = result.masked_document
cloakmap = result.cloakmap

# UnmaskingEngine returns UnmaskingResult
result = unmasking_engine.unmask_document(...)
unmasked_doc = result.unmasked_document
```

**After (v2.0):**
```python
# CloakEngine.mask_document returns MaskResult
result = engine.mask_document(doc)
masked_doc = result.document  # Note: 'document' not 'masked_document'
cloakmap = result.cloakmap
entities_found = result.entities_found
entities_masked = result.entities_masked

# CloakEngine.unmask_document returns DoclingDocument directly
unmasked_doc = engine.unmask_document(masked_doc, cloakmap)
```

## Common Migration Patterns

### Pattern 1: Basic Document Masking

**Before:**
```python
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.core.policies import load_policy
from presidio_analyzer import AnalyzerEngine

# Setup
extractor = TextExtractor()
analyzer = AnalyzerEngine()
masking_engine = MaskingEngine()
policy = load_policy("default")

# Process document
text = extractor.extract_full_text(doc)
segments = extractor.extract_text_segments(doc)
entities = analyzer.analyze(text, language="en")
result = masking_engine.mask_document(doc, entities, policy, segments)
```

**After:**
```python
from cloakpivot import CloakEngine

engine = CloakEngine()
result = engine.mask_document(doc)
```

### Pattern 2: Custom Configuration

**Before:**
```python
from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.masking.engine import MaskingEngine

config = AnalyzerConfig(
    language="es",
    min_confidence=0.8,
    enabled_recognizers=["EMAIL_ADDRESS", "PERSON"]
)
engine = MaskingEngine(analyzer_config=config)
```

**After:**
```python
from cloakpivot import CloakEngine

engine = CloakEngine(
    analyzer_config={
        'languages': ['es'],
        'confidence_threshold': 0.8,
        'enabled_recognizers': ["EMAIL_ADDRESS", "PERSON"]
    }
)
```

### Pattern 3: Custom Policies

**Before:**
```python
from cloakpivot.core.policies import MaskingPolicy, load_policy
from cloakpivot.masking.engine import MaskingEngine

policy = load_policy("path/to/policy.yaml")
engine = MaskingEngine()
result = engine.mask_document(doc, entities, policy, segments)
```

**After:**
```python
from cloakpivot import CloakEngine
from cloakpivot.defaults import get_conservative_policy

# Use preset policies
engine = CloakEngine(default_policy=get_conservative_policy())
result = engine.mask_document(doc)

# Or pass policy per operation
result = engine.mask_document(doc, policy=custom_policy)
```

### Pattern 4: Builder Pattern (New)

The v2.0 API introduces a builder pattern for complex configurations:

```python
from cloakpivot import CloakEngine

engine = CloakEngine.builder()
    .with_confidence_threshold(0.9)
    .with_languages(['en', 'es'])
    .with_custom_policy(policy)
    .with_conflict_resolution(config)
    .build()
```

## CLI Changes

### Before (v1.x):
```bash
# Many complex commands
cloakpivot mask --analyzer-config config.yaml --policy policy.yaml ...
cloakpivot unmask ...
cloakpivot migrate ...
cloakpivot diagnose ...
```

### After (v2.0):
```bash
# Simplified commands
cloakpivot mask document.pdf -o masked.md -c document.cloakmap.json
cloakpivot unmask masked.md document.cloakmap.json -o restored.md
cloakpivot version
```

## Removed Features

The following features were removed in v2.0 for simplicity:

### Removed Modules
- `cloakpivot.migration.*` - CloakMap migration tools
- `cloakpivot.storage.*` - Cloud storage backends (S3, GCS)
- `cloakpivot.plugins.*` - Plugin system
- `cloakpivot.diagnostics.*` - Diagnostic tools
- `cloakpivot.security.*` - Advanced security features
- `cloakpivot.observability.exporters.*` - Metric exporters
- `cloakpivot.core.parallel_analysis` - Parallel processing
- `cloakpivot.core.performance` - Performance optimization

### Removed CLI Commands
- `migrate` - CloakMap migration
- `diagnose` - System diagnostics
- `batch` - Batch processing
- `plugin` - Plugin management

If you need these features, consider:
1. Using v1.x for legacy projects
2. Implementing custom solutions
3. Opening an issue for critical missing features

## Import Changes

### Common Import Updates

```python
# Old imports (remove)
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
from cloakpivot.document.extractor import TextExtractor
from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.storage import S3Storage
from cloakpivot.migration import CloakMapMigrator
from cloakpivot.plugins import PluginManager

# New imports (use)
from cloakpivot import (
    CloakEngine,
    CloakEngineBuilder,
    get_default_policy,
    get_conservative_policy,
    get_permissive_policy
)
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.cloakmap import CloakMap
```

## Deprecation Warnings

v2.0 includes deprecation wrappers for smooth migration:

```python
# This will work but show a deprecation warning
from cloakpivot.deprecated import MaskingEngine, UnmaskingEngine

masking_engine = MaskingEngine()  # DeprecationWarning
# Internally uses CloakEngine
```

To silence warnings during migration:

```python
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
```

## Step-by-Step Migration

### Step 1: Update Imports

Replace old engine imports with CloakEngine:

```python
# Replace this
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine

# With this
from cloakpivot import CloakEngine
```

### Step 2: Simplify Initialization

Remove complex setup code:

```python
# Remove all of this
extractor = TextExtractor()
analyzer = AnalyzerEngine()
masking_engine = MaskingEngine(config)
unmasking_engine = UnmaskingEngine()

# Replace with
engine = CloakEngine()
```

### Step 3: Update Method Calls

Update masking calls:

```python
# Replace
text = extractor.extract_full_text(doc)
segments = extractor.extract_text_segments(doc)
entities = analyzer.analyze(text, language="en")
result = masking_engine.mask_document(doc, entities, policy, segments)

# With
result = engine.mask_document(doc)
```

Update unmasking calls:

```python
# Replace
result = unmasking_engine.unmask_document(masked_doc, cloakmap)
unmasked = result.unmasked_document

# With
unmasked = engine.unmask_document(masked_doc, cloakmap)
```

### Step 4: Update Result Access

```python
# Replace
masked_doc = result.masked_document

# With
masked_doc = result.document
```

### Step 5: Test Thoroughly

Run your test suite to ensure functionality is preserved:

```bash
python -m pytest tests/
```

## Configuration Mapping

### Analyzer Configuration

| v1.x Parameter | v2.0 Parameter | Notes |
|---------------|----------------|-------|
| `language` | `languages[0]` | v2.0 uses list |
| `min_confidence` | `confidence_threshold` | Same range (0.0-1.0) |
| `enabled_recognizers` | `enabled_recognizers` | No change |
| `disabled_recognizers` | `disabled_recognizers` | No change |

### Policy Configuration

| v1.x | v2.0 | Notes |
|------|------|-------|
| Load from YAML | Pass MaskingPolicy object | More explicit |
| Global policy file | Per-engine default policy | More flexible |
| Runtime policy override | `policy` parameter | Same concept |

## Performance Considerations

v2.0 optimizations:
- Simplified initialization (faster startup)
- Removed unnecessary abstractions
- Single engine reduces memory overhead
- Default configurations optimized for common cases

## Getting Help

### Resources
- [API Reference](API.md)
- [Examples](../examples/)
- [GitHub Issues](https://github.com/hernamesbarbara/cloakpivot/issues)

### Common Issues

**Issue:** `ModuleNotFoundError: No module named 'cloakpivot.masking.engine'`
**Solution:** Update imports to use `from cloakpivot import CloakEngine`

**Issue:** `AttributeError: 'MaskResult' object has no attribute 'masked_document'`
**Solution:** Use `result.document` instead of `result.masked_document`

**Issue:** Tests failing after upgrade
**Solution:** Update test fixtures and assertions to match new API

## Rollback Plan

If you need to rollback to v1.x:

```bash
pip install cloakpivot==1.8.3
```

Consider maintaining both versions during migration:

```python
try:
    # Try v2.0 API
    from cloakpivot import CloakEngine
    USE_V2 = True
except ImportError:
    # Fall back to v1.x
    from cloakpivot.masking.engine import MaskingEngine
    USE_V2 = False
```

## Summary

The v2.0 migration simplifies your code significantly:

- **Less code:** ~10 lines → 2 lines for basic masking
- **Fewer imports:** 5-6 imports → 1 import
- **Single engine:** No more separate masking/unmasking engines
- **Smart defaults:** Works out-of-the-box for common cases
- **Builder pattern:** Advanced configuration when needed

Most migrations can be completed in under an hour for typical projects. The simplified API reduces maintenance burden and makes the code more readable.