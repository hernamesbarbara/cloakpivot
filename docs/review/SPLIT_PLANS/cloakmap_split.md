# Split Plan: core/cloakmap.py
*Current: 1005 LOC → Target: 3 modules, ~300-400 LOC each*

## Rationale
The `CloakMap` class contains data structure definition, validation, serialization, and manipulation logic all in one file. This should be separated into core data structure, validation rules, and serialization concerns.

## New Modules Structure

### 1. `core/cloakmap.py` (350 lines)
**Purpose:** Core CloakMap data structure and essential operations
```python
class CloakMap:
    def __init__(self)
    def add_entry(self, entry: AnchorEntry)
    def get_entry(self, anchor_id: str)
    def remove_entry(self, anchor_id: str)
    def merge(self, other: CloakMap)
    def clear(self)
    def to_dict(self)
    def from_dict(cls, data: dict)
    @property
    def anchors(self)
    @property
    def metadata(self)
```

### 2. `core/cloakmap_validator.py` (300 lines)
**Purpose:** Validation logic for CloakMap operations
```python
class CloakMapValidator:
    def validate_structure(self, cloakmap: CloakMap)
    def validate_anchor_entry(self, entry: AnchorEntry)
    def validate_metadata(self, metadata: dict)
    def validate_merge_compatibility(self, map1: CloakMap, map2: CloakMap)
    def check_integrity(self, cloakmap: CloakMap, document: DoclingDocument)
    def validate_positions(self, anchors: List[AnchorEntry])
```

### 3. `core/cloakmap_serializer.py` (355 lines)
**Purpose:** Serialization, encryption (deprecated), and persistence
```python
class CloakMapSerializer:
    def serialize_to_json(self, cloakmap: CloakMap) -> str
    def deserialize_from_json(self, data: str) -> CloakMap
    def serialize_to_yaml(self, cloakmap: CloakMap) -> str
    def deserialize_from_yaml(self, data: str) -> CloakMap
    def save(self, cloakmap: CloakMap, path: Path, format: str)
    def load(self, path: Path) -> CloakMap
    # Deprecated encryption methods (with NotImplementedError)
    def encrypt(self, cloakmap: CloakMap, **kwargs)
    def save_encrypted(self, cloakmap: CloakMap, path: Path, **kwargs)
    def load_encrypted(self, path: Path, **kwargs)
```

## Move Map

### Keep in `cloakmap.py`:
- Lines 1-100: Core class definition and initialization
- Lines 101-250: Basic CRUD operations
- Lines 251-350: Core properties and dict conversion

### Move to `cloakmap_validator.py`:
- Lines 351-450: Validation helper methods
- Lines 451-550: Integrity checking methods
- Lines 551-650: Position and boundary validation

### Move to `cloakmap_serializer.py`:
- Lines 651-750: JSON serialization methods
- Lines 751-850: YAML serialization methods
- Lines 851-950: File I/O methods
- Lines 951-1005: Deprecated encryption stubs

## Re-exports

```python
# core/__init__.py
from .cloakmap import CloakMap
from .cloakmap_validator import CloakMapValidator
from .cloakmap_serializer import CloakMapSerializer

__all__ = ['CloakMap']  # Only expose main class by default
```

## Implementation Steps

1. Extract validation logic to validator class
2. Extract serialization to serializer class
3. Update CloakMap to delegate to helpers
4. Add proper error handling at boundaries
5. Update tests to cover new structure

## Test Impact

- Split `test_cloakmap.py` into three test files
- No changes to integration tests
- Add specific serialization format tests

## Risks

- **Risk:** Breaking serialization compatibility
- **Mitigation:** Extensive round-trip testing

## Acceptance Criteria

- [ ] All existing tests pass
- [ ] Each module under 400 LOC
- [ ] No public API changes
- [ ] Serialization format unchanged
- [ ] Clear separation of concerns

---
*Estimated effort: 2 days*
*Risk level: LOW*
*Priority: HIGH (second largest file)*