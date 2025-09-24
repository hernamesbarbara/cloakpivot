# Split Plan: masking/applicator.py
*Current: 861 LOC → Target: 3 modules, ~250-350 LOC each*

## Rationale
The `MaskingApplicator` class handles strategy application, conflict resolution, and text processing in a single large file. These are distinct responsibilities that should be separated.

## New Modules Structure

### 1. `masking/applicator.py` (300 lines)
**Purpose:** Core applicator orchestration and public API
```python
class MaskingApplicator:
    def __init__(self, adapter: PresidioMaskingAdapter)
    def apply_masking(self, text: str, entities: List, policy: MaskingPolicy)
    def process_document(self, document: DoclingDocument, entities: List)
    def get_strategy_for_entity(self, entity: Entity, policy: MaskingPolicy)
    # Delegates to specialized processors
```

### 2. `masking/conflict_resolver.py` (280 lines)
**Purpose:** Entity conflict and overlap resolution
```python
class ConflictResolver:
    def __init__(self, strategy: ConflictResolutionStrategy)
    def resolve_overlapping_entities(self, entities: List[Entity])
    def merge_adjacent_entities(self, entities: List[Entity])
    def prioritize_entities(self, entities: List[Entity])
    def apply_resolution_strategy(self, conflicts: List)
    def validate_resolution(self, original: List, resolved: List)
```

### 3. `masking/strategy_executor.py` (281 lines)
**Purpose:** Strategy execution and fallback handling
```python
class StrategyExecutor:
    def __init__(self, default_strategy: Strategy)
    def execute_strategy(self, text: str, entity: Entity, strategy: Strategy)
    def handle_strategy_failure(self, error: Exception, entity: Entity)
    def apply_fallback_strategy(self, text: str, entity: Entity)
    def validate_replacement(self, original: str, replacement: str)
    def batch_execute_strategies(self, text: str, entity_strategy_pairs: List)
```

## Move Map

### Keep in `applicator.py`:
- Lines 1-150: Core class and initialization
- Lines 151-300: Main orchestration methods

### Move to `conflict_resolver.py`:
- Lines 301-450: Overlap detection methods
- Lines 451-580: Conflict resolution strategies

### Move to `strategy_executor.py`:
- Lines 581-730: Strategy execution logic
- Lines 731-861: Fallback and error handling

## Re-exports

```python
# masking/__init__.py
from .applicator import MaskingApplicator
# Internal classes not exposed by default
```

## Implementation Steps

1. Extract conflict resolution logic
2. Extract strategy execution logic
3. Wire dependencies via composition
4. Update error handling boundaries
5. Add integration tests

## Test Impact

- Split unit tests into three files
- Add specific conflict resolution tests
- Mock strategy execution for applicator tests

## Risks

- **Risk:** Performance impact from additional abstraction
- **Mitigation:** Profile hot paths, optimize if needed

## Acceptance Criteria

- [ ] All existing tests pass
- [ ] Each module under 350 LOC
- [ ] Clear separation of concerns
- [ ] No behavior changes
- [ ] Improved testability

---
*Estimated effort: 2 days*
*Risk level: LOW*
*Priority: MEDIUM*