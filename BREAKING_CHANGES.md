# Breaking Changes Migration Guide

This document outlines the breaking changes introduced during the CloakPivot refactoring process (PRs 001-011) and provides migration examples to help update your code.

## Overview

The refactoring process involved major architectural changes to improve code organization, reduce file sizes, and eliminate code duplication. The primary breaking changes are related to import path reorganization following the core layer restructuring in PR-011.

**⚠️ Important**: While there are breaking changes in import paths, the public API functionality remains the same. Most breaking changes can be resolved by updating import statements.

## Core Layer Reorganization (PR-011)

The most significant breaking changes come from reorganizing the core layer into logical subpackages. Modules were moved from a flat structure to a nested hierarchy.

### Import Path Changes

#### Before (❌ Broken)
```python
# These imports will no longer work
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.anchors import AnchorEntry, AnchorIndex
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.normalization import ConflictResolutionConfig
from cloakpivot.core.config import get_default_config
from cloakpivot.core.exceptions import CloakPivotError, ValidationError
from cloakpivot.core.model_info import MODEL_CHARACTERISTICS
from cloakpivot.core.surrogate import SurrogateGenerator
from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.core.cloakmap_enhancer import CloakMapEnhancer
from cloakpivot.core.presidio_mapper import StrategyToOperatorMapper
```

#### After (✅ New Correct Paths)
```python
# New nested import paths
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.core.types.strategies import Strategy, StrategyKind
from cloakpivot.core.types.anchors import AnchorEntry, AnchorIndex
from cloakpivot.core.policies.policies import MaskingPolicy
from cloakpivot.core.processing.normalization import ConflictResolutionConfig
from cloakpivot.core.utilities.config import get_default_config
from cloakpivot.core.types.exceptions import CloakPivotError, ValidationError
from cloakpivot.core.types.model_info import MODEL_CHARACTERISTICS
from cloakpivot.core.processing.surrogate import SurrogateGenerator
from cloakpivot.core.processing.analyzer import AnalyzerConfig
from cloakpivot.core.processing.cloakmap_enhancer import CloakMapEnhancer
from cloakpivot.core.processing.presidio_mapper import StrategyToOperatorMapper
```

#### Recommended (✅ Backward Compatible)
```python
# Use core module exports for backward compatibility
from cloakpivot.core import (
    CloakMap,
    Strategy,
    StrategyKind,
    AnchorEntry,
    AnchorIndex,
    MaskingPolicy,
    # Note: Some items may need specific imports if not re-exported
)

# Or use main package exports
from cloakpivot import (
    CloakMap,
    Strategy,
    StrategyKind,
    AnchorEntry,
    AnchorIndex,
    MaskingPolicy,
    # Most commonly used items are available here
)
```

### Core Module Structure

The new core module structure is organized as follows:

```
cloakpivot/core/
├── types/           # Data structures and type definitions
│   ├── cloakmap.py         # CloakMap class
│   ├── strategies.py       # Strategy and StrategyKind
│   ├── anchors.py          # AnchorEntry, AnchorIndex
│   ├── results.py          # Result classes
│   ├── exceptions.py       # Custom exceptions
│   └── model_info.py       # Model information
├── policies/        # Policy definitions and loading
│   ├── policies.py         # MaskingPolicy and presets
│   └── policy_loader.py    # Policy loading utilities
├── processing/      # Core processing logic
│   ├── analyzer.py         # Analysis configuration
│   ├── normalization.py    # Conflict resolution
│   ├── surrogate.py        # Surrogate generation
│   ├── cloakmap_enhancer.py # CloakMap enhancement
│   └── presidio_mapper.py  # Strategy mapping
└── utilities/       # Utility functions and helpers
    ├── config.py           # Configuration utilities
    ├── validation.py       # Validation helpers
    ├── cloakmap_validator.py # CloakMap validation
    └── cloakmap_serializer.py # CloakMap serialization
```

## Module Splits (PRs 007-010)

The module splits primarily affected internal implementation and generally preserved public API compatibility. However, some internal classes may have moved.

### PresidioMaskingAdapter Split (PRs 007-008)

The large `PresidioMaskingAdapter` was split into focused modules:

#### Internal Implementation Changes
- **strategy_processors.py**: Strategy-specific processing logic
- **entity_processor.py**: Entity processing and handling
- **text_processor.py**: Text processing utilities
- **document_reconstructor.py**: Document reconstruction logic
- **metadata_manager.py**: Metadata and CloakMap management

**✅ No Breaking Changes**: The main `PresidioMaskingAdapter` class interface remains the same.

### CloakMap Split (PR-009)

The `CloakMap` implementation was split into focused components:

#### Before (❌ If accessing internals)
```python
# These internal methods may have moved
from cloakpivot.core.cloakmap import CloakMap

# Internal validation/serialization was part of CloakMap
cloakmap = CloakMap()
# Some internal methods may have changed
```

#### After (✅ Public API unchanged)
```python
from cloakpivot.core.types.cloakmap import CloakMap

# Public API remains the same
cloakmap = CloakMap()
# All public methods work identically
```

**Note**: Validation and serialization logic moved to separate utility classes but are used internally by CloakMap.

### MaskingApplicator Split (PR-010)

The `MaskingApplicator` was split into helper modules:

#### Internal Implementation Changes
- **conflict_resolver.py**: Conflict resolution logic
- **strategy_executor.py**: Strategy execution
- **template_helpers.py**: Template processing helpers
- **format_helpers.py**: Format processing utilities

**✅ No Breaking Changes**: The main `StrategyApplicator` class interface remains the same.

## Migration Examples

### Example 1: Basic Import Updates

#### Before
```python
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.strategies import Strategy, StrategyKind, DEFAULT_REDACT
from cloakpivot.core.policies import MaskingPolicy, CONSERVATIVE_POLICY

engine = CloakEngine()
policy = CONSERVATIVE_POLICY
```

#### After (Option 1: Use new paths)
```python
from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.core.types.strategies import Strategy, StrategyKind, DEFAULT_REDACT
from cloakpivot.core.policies.policies import MaskingPolicy, CONSERVATIVE_POLICY

engine = CloakEngine()
policy = CONSERVATIVE_POLICY
```

#### After (Option 2: Use package exports - Recommended)
```python
from cloakpivot import (
    CloakEngine,
    Strategy,
    StrategyKind,
    DEFAULT_REDACT,
    MaskingPolicy,
    CONSERVATIVE_POLICY,
)

engine = CloakEngine()
policy = CONSERVATIVE_POLICY
```

### Example 2: Advanced Usage with Internal Components

#### Before
```python
from cloakpivot.core.normalization import ConflictResolutionConfig, ConflictResolutionStrategy
from cloakpivot.core.analyzer import AnalyzerConfig

config = ConflictResolutionConfig(
    strategy=ConflictResolutionStrategy.PRIORITIZE_LONGER
)
analyzer_config = AnalyzerConfig()
```

#### After
```python
from cloakpivot.core.processing.normalization import ConflictResolutionConfig, ConflictResolutionStrategy
from cloakpivot.core.processing.analyzer import AnalyzerConfig

config = ConflictResolutionConfig(
    strategy=ConflictResolutionStrategy.PRIORITIZE_LONGER
)
analyzer_config = AnalyzerConfig()
```

### Example 3: Custom Policy Creation

#### Before
```python
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind

custom_policy = MaskingPolicy(
    default_strategy=Strategy(
        kind=StrategyKind.REDACT,
        char="*"
    )
)
```

#### After (Recommended)
```python
from cloakpivot import MaskingPolicy, Strategy, StrategyKind

custom_policy = MaskingPolicy(
    default_strategy=Strategy(
        kind=StrategyKind.REDACT,
        char="*"
    )
)
```

## Automated Migration

### Find and Replace Patterns

You can use these patterns to automatically update most import statements:

```bash
# Core module imports - use with caution and test thoroughly
find . -name "*.py" -type f -exec sed -i 's/from cloakpivot\.core\.cloakmap import/from cloakpivot.core.types.cloakmap import/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/from cloakpivot\.core\.strategies import/from cloakpivot.core.types.strategies import/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/from cloakpivot\.core\.anchors import/from cloakpivot.core.types.anchors import/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/from cloakpivot\.core\.policies import/from cloakpivot.core.policies.policies import/g' {} \;
find . -name "*.py" -type f -exec sed -i 's/from cloakpivot\.core\.normalization import/from cloakpivot.core.processing.normalization import/g' {} \;
```

**⚠️ Warning**: Always review and test automated changes. Consider using the recommended package-level imports instead.

## Compatibility Notes

### Backward Compatibility

- The main package exports (`from cloakpivot import ...`) provide the most stable interface
- Core module exports (`from cloakpivot.core import ...`) provide access to essential classes
- Public API functionality is preserved - only import paths changed
- All existing method signatures remain the same

### What's NOT Breaking

- ✅ Public method signatures
- ✅ Class constructors and initialization
- ✅ Configuration options and parameters
- ✅ Serialization formats
- ✅ CLI interface
- ✅ Core functionality and behavior

### What IS Breaking

- ❌ Direct imports from old flat core structure
- ❌ Some internal class locations (if you were accessing internals)
- ❌ Module-level imports of utilities now in nested packages

## Testing Your Migration

After updating your imports, run these checks:

```bash
# 1. Check for any remaining old imports
grep -r "from cloakpivot\.core\.[^.]* import" --include="*.py" .

# 2. Run your test suite
python -m pytest

# 3. Check import compatibility
python -c "from cloakpivot import CloakEngine, MaskingPolicy, Strategy; print('✅ Main imports work')"
python -c "from cloakpivot.core import CloakMap, StrategyKind; print('✅ Core imports work')"
```

## Getting Help

If you encounter issues during migration:

1. Check that you're using the latest version of CloakPivot
2. Prefer package-level imports (`from cloakpivot import ...`) over internal module imports
3. Consult the API documentation for current import paths
4. File an issue if you discover missing exports or compatibility problems

## Summary

The refactoring improved code organization without changing core functionality. Most breaking changes can be resolved by updating import statements to use the new nested module structure or preferably using the stable package-level imports.

**Recommended Migration Strategy:**
1. Replace direct core imports with package-level imports where possible
2. Update remaining imports to use new nested paths
3. Test thoroughly with your existing workflows
4. Consider this an opportunity to simplify your import statements