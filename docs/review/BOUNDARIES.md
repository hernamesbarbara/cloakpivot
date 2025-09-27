# Boundary Analysis & Architecture Violations Report
*Generated: 2025-09-24*

## Target Architecture

```
┌─────────────────────────────────────────────┐
│                    CLI                       │ ← Entry point, orchestration only
├─────────────────────────────────────────────┤
│                  Engine                      │ ← High-level API & orchestration
├─────────────────┬───────────────────────────┤
│     Masking     │       Unmasking           │ ← Domain-specific operations
├─────────────────┴───────────────────────────┤
│                   Core                       │ ← Shared types, policies, utilities
├─────────────────────────────────────────────┤
│    Document     │      Formats              │ ← I/O and serialization
├─────────────────┴───────────────────────────┤
│                  Utils                       │ ← Low-level utilities
└─────────────────────────────────────────────┘
```

### Expected Dependencies
- ↓ Direction only (higher layers depend on lower)
- No circular dependencies
- No cross-cutting at same level (masking ↔ unmasking)

## Boundary Violations Found

### 1. Cross-Layer Violations [HIGH PRIORITY]

#### [LEAKY-BOUNDARY] FIND-0016: Engine imports from masking/unmasking
**Location**: `engine.py:18-20`
```python
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
```
**Issue**: Top-level engine directly imports domain-specific engines
**Impact**: Tight coupling, difficult to test in isolation
**Fix**: Use dependency injection or factory pattern

#### [LEAKY-BOUNDARY] FIND-0017: Relative imports crossing boundaries
**Locations**: 10 files using `from ..` imports
- `masking/presidio_adapter.py`
- `unmasking/presidio_adapter.py`
- `unmasking/document_unmasker.py`
- etc.

**Issue**: Modules reaching across package boundaries
**Impact**: Fragile architecture, refactoring risk
**Fix**: Use absolute imports through public APIs

### 2. Circular Dependency Risks [MEDIUM PRIORITY]

#### [LEAKY-BOUNDARY] FIND-0018: Core depending on higher layers
**Pattern**: None found (✓ Good)
- Core modules correctly avoid importing from higher layers

#### [LEAKY-BOUNDARY] FIND-0019: Document/Format layer violations
**Pattern**: Clean separation maintained (✓ Good)
- Document modules only import from core/types

### 3. Horizontal Dependencies [LOW PRIORITY]

#### [LEAKY-BOUNDARY] FIND-0020: Masking ↔ Unmasking coupling
**Pattern**: No direct cross-imports found (✓ Good)
- Both domains communicate through core types only

## Layer Responsibility Analysis

### CLI Layer (cli/)
**Current Responsibilities**: ✓ Correct
- Command parsing
- Configuration loading
- Engine orchestration

**Violations**: None found

### Engine Layer (engine.py, engine_builder.py)
**Current Responsibilities**:
- ✓ High-level API
- ✓ Builder pattern
- ✗ Direct instantiation of domain engines

**Violations**:
- Imports specific engine implementations
- Should use factory or registry pattern

### Masking Layer (masking/)
**Current Responsibilities**: ✓ Mostly Correct
- PII detection
- Entity masking
- Surrogate generation
- Presidio adapter

**Violations**:
- Heavy coupling to core (acceptable)
- Large monolithic adapter (1310 LOC)

### Unmasking Layer (unmasking/)
**Current Responsibilities**: ✓ Correct
- Reverse masking operations
- Anchor resolution
- CloakMap processing

**Violations**:
- Some duplication with masking layer
- Could share more utilities

### Core Layer (core/)
**Current Responsibilities**: ⚠️ Overloaded
- Types and interfaces
- Policies and strategies
- Validation and normalization
- Error handling
- Analyzers
- Surrogates
- CloakMap
- Results

**Issues**:
- Too many responsibilities (13+ modules)
- Mix of data types and business logic
- Should be split into core-types and core-logic

### Document Layer (document/)
**Current Responsibilities**: ✓ Correct
- Text extraction
- Document mapping
- Processing utilities

**Violations**: None significant

### Formats Layer (formats/)
**Current Responsibilities**: ✓ Correct
- Serialization/deserialization
- Format conversions

**Violations**: None found

### Utils Layer (utils/)
**Current Responsibilities**: ✓ Correct
- Low-level utilities
- No domain knowledge

**Violations**: None found

## Recommended Architecture Improvements

### 1. Split Core Layer
```
core/
├── types/         # Pure data types, interfaces
├── policies/      # Policy definitions
├── processing/    # Business logic
└── utilities/     # Shared utilities
```

### 2. Introduce Factory Pattern for Engines
```python
# engine_factory.py
class EngineFactory:
    def create_masking_engine(config) -> MaskingEngine
    def create_unmasking_engine(config) -> UnmaskingEngine
```

### 3. Extract Shared Domain Utilities
```
shared/
├── presidio_common.py
├── text_processing.py
└── validation.py
```

### 4. Enforce Import Rules
- Add import linter rules
- Document allowed import patterns
- Use `__all__` to define public APIs

## Risk Assessment

| Violation | Impact | Effort | Risk | Priority |
|-----------|--------|--------|------|----------|
| Engine direct imports | HIGH | LOW | LOW | P1 |
| Core layer overload | MEDIUM | HIGH | MEDIUM | P2 |
| Relative imports | LOW | MEDIUM | LOW | P3 |
| Presidio adapter size | MEDIUM | HIGH | MEDIUM | P2 |
| Missing factories | MEDIUM | LOW | LOW | P1 |

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
- [ ] Replace relative imports with absolute
- [ ] Introduce engine factory
- [ ] Document import rules

### Phase 2: Core Restructuring (3-5 days)
- [ ] Split core into subpackages
- [ ] Extract shared utilities
- [ ] Add import linting

### Phase 3: Domain Refactoring (1 week)
- [ ] Split large adapters
- [ ] Extract common presidio utilities
- [ ] Consolidate duplicate patterns

---
*End of Boundary Analysis Report*