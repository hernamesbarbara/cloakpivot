# CloakPivot Baseline & Inventory Report
*Generated: 2025-09-24*

## Executive Summary
- **Total Python Files**: 50
- **Total Source Lines**: 17,975 LOC (cloakpivot package)
- **Total Test Lines**: 10,227 LOC
- **Test Coverage Ratio**: 0.57:1 (test:source)
- **Module Count**: 7 main modules (cli, core, document, formats, masking, unmasking, utils)
- **Function/Class Definitions**: ~169 total symbols

## Module Organization

### Directory Structure
```
cloakpivot/
├── cli/          # Command-line interface
├── core/         # Core functionality and types
├── document/     # Document processing
├── formats/      # Format serialization
├── masking/      # PII masking operations
├── unmasking/    # PII unmasking operations
└── utils/        # Utility functions
```

## Top 30 Largest Files (Hotspots)

| Rank | File | LOC | Category |
|------|------|-----|----------|
| 1 | masking/presidio_adapter.py | 1310 | **CRITICAL** - Oversized |
| 2 | core/cloakmap.py | 1005 | **CRITICAL** - Oversized |
| 3 | masking/applicator.py | 861 | **HIGH** - Oversized |
| 4 | unmasking/document_unmasker.py | 772 | **HIGH** - Oversized |
| 5 | core/surrogate.py | 632 | **MEDIUM** - Large |
| 6 | core/normalization.py | 618 | **MEDIUM** - Large |
| 7 | masking/engine.py | 611 | **MEDIUM** - Large |
| 8 | core/analyzer.py | 597 | **MEDIUM** - Large |
| 9 | core/results.py | 563 | **MEDIUM** - Large |
| 10 | unmasking/anchor_resolver.py | 561 | **MEDIUM** - Large |
| 11 | core/policy_loader.py | 554 | **MEDIUM** - Large |
| 12 | loaders.py | 543 | **MEDIUM** - Large |
| 13 | core/policies.py | 535 | **MEDIUM** - Large |
| 14 | core/validation.py | 515 | **MEDIUM** - Large |
| 15 | core/anchors.py | 472 | Normal |
| 16 | core/error_handling.py | 465 | Normal |
| 17 | unmasking/engine.py | 458 | Normal |
| 18 | document/extractor.py | 458 | Normal |
| 19 | unmasking/presidio_adapter.py | 426 | Normal |
| 20 | document/mapper.py | 424 | Normal |
| 21 | core/detection.py | 391 | Normal |
| 22 | unmasking/cloakmap_loader.py | 382 | Normal |
| 23 | formats/serialization.py | 382 | Normal |
| 24 | core/exceptions.py | 381 | Normal |
| 25 | core/cloakmap_enhancer.py | 360 | Normal |
| 26 | cli/config.py | 352 | Normal |
| 27 | core/presidio_mapper.py | 327 | Normal |
| 28 | core/model_info.py | 301 | Normal |
| 29 | core/config.py | 265 | Normal |

## Public API Surface (from tests and examples)

### Core Imports (High Frequency)
```python
# Primary entry points
from cloakpivot import CloakEngine, CloakEngineBuilder

# Policy and strategy types
from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind
from cloakpivot.core.policies import MaskingPolicy, Strategy

# Document types
from cloakpivot.type_imports import DoclingDocument
from cloakpivot.core.types import DoclingDocument

# Specialized features
from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.compat import load_document, to_lexical
```

### Test Coverage Areas
- **Unit tests**: Comprehensive coverage of individual components
- **Integration tests**: Workflow testing (test_masking_workflow.py)
- **CLI tests**: Command-line interface testing
- **Compatibility tests**: Document format compatibility

## Complexity Analysis (Cyclomatic Complexity)

### High Complexity Functions (Sample)
```
registration.py:
  - unregister_cloak_methods: A (5)
  - register_cloak_methods: A (4)

engine_builder.py:
  - CloakEngineBuilder.build: A (5)
  - CloakEngineBuilder.with_analyzer_config: A (4)

loaders.py:
  - get_detection_pipeline: B (7) ⚠️
  - _validate_language: A (5)
  - get_presidio_analyzer_from_config: A (5)

engine.py:
  - CloakEngine.__init__: C (11) ⚠️⚠️
```

## Internal Dependencies Analysis

### Cross-module Import Patterns
- **31 internal imports** found across 15 files
- Heavy coupling between:
  - masking/* ↔ core/*
  - unmasking/* ↔ core/*
  - document/* → core/types

## Initial Findings & Risk Areas

### 🔴 Critical Issues
1. **[OVERSIZED]** `masking/presidio_adapter.py` (1310 LOC) - Far exceeds recommended module size
2. **[OVERSIZED]** `core/cloakmap.py` (1005 LOC) - Complex data structure needing split
3. **[DRY-SUSPECT]** Duplicate adapters: `masking/presidio_adapter.py` vs `unmasking/presidio_adapter.py`

### 🟡 Medium Priority
1. **[COMPLEXITY]** `CloakEngine.__init__` with C(11) complexity
2. **[OVERSIZED]** Multiple 600+ LOC files in core/* suggesting mixed responsibilities
3. **[BOUNDARY-SUSPECT]** Heavy cross-module dependencies need audit

### 🟢 Strengths
1. Good test coverage ratio (0.57:1)
2. Clear module separation at top level
3. Consistent use of type imports

## Hotspot Triage List (Top 20 by Impact)

Priority = LOC × estimated_fan_in × complexity_factor

1. **masking/presidio_adapter.py** - Priority: CRITICAL
2. **core/cloakmap.py** - Priority: CRITICAL
3. **masking/applicator.py** - Priority: HIGH
4. **unmasking/document_unmasker.py** - Priority: HIGH
5. **masking/engine.py** - Priority: HIGH (public API)
6. **core/surrogate.py** - Priority: MEDIUM
7. **core/normalization.py** - Priority: MEDIUM
8. **core/analyzer.py** - Priority: MEDIUM
9. **core/results.py** - Priority: MEDIUM
10. **unmasking/anchor_resolver.py** - Priority: MEDIUM
11. **core/policy_loader.py** - Priority: MEDIUM
12. **loaders.py** - Priority: MEDIUM
13. **core/validation.py** - Priority: MEDIUM
14. **unmasking/engine.py** - Priority: MEDIUM
15. **document/extractor.py** - Priority: LOW
16. **unmasking/presidio_adapter.py** - Priority: LOW (DRY candidate)
17. **document/mapper.py** - Priority: LOW
18. **core/detection.py** - Priority: LOW
19. **formats/serialization.py** - Priority: LOW
20. **core/exceptions.py** - Priority: LOW

## Next Steps (Session 2 Preview)
- Deep dive into DRY violations between masking/unmasking adapters
- Analyze boundary violations between core/masking/unmasking
- Document mixed responsibilities in oversized modules
- Create dependency graph visualization

---
*End of Baseline Report*