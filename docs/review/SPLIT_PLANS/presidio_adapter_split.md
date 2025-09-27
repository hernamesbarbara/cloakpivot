# Split Plan: masking/presidio_adapter.py
*Current: 1310 LOC → Target: 6 modules, ~200-350 LOC each*

## Rationale
The `PresidioMaskingAdapter` class is a monolithic 1310-line file handling all aspects of Presidio-based masking. This violates single responsibility principle and makes the code difficult to test, maintain, and extend.

## New Modules Structure

### 1. `masking/presidio_adapter.py` (280 lines)
**Purpose:** Core adapter class maintaining public API
```python
# Main orchestration and public interface
class PresidioMaskingAdapter:
    def __init__(self, engine_config)
    def mask_document(self, document, entities, policy, segments, format)
    def apply_strategy(self, text, entity_type, strategy, confidence)
    @property
    def anonymizer(self)

    # Delegates to:
    self._strategy_processor = StrategyProcessor()
    self._entity_processor = EntityProcessor()
    self._text_processor = TextProcessor()
    self._document_reconstructor = DocumentReconstructor()
    self._metadata_manager = MetadataManager()
```

### 2. `masking/strategy_processors.py` (220 lines)
**Purpose:** Individual strategy implementations
```python
class StrategyProcessor:
    def apply_hash_strategy(text, entity_type, params)
    def apply_partial_strategy(text, entity_type, params)
    def apply_custom_strategy(text, entity_type, callback)
    def apply_surrogate_strategy(text, entity_type, generator)
    def fallback_redaction(text, entity_type)
```

### 3. `masking/entity_processor.py` (350 lines)
**Purpose:** Entity validation and batch processing
```python
class EntityProcessor:
    def filter_overlapping_entities(entities)
    def validate_entities(entities, text_length)
    def validate_entities_against_boundaries(entities, boundaries)
    def batch_process_entities(entities, full_text, strategies)
    def prepare_strategies(entities, policy)
```

### 4. `masking/text_processor.py` (200 lines)
**Purpose:** Text manipulation and span operations
```python
class TextProcessor:
    def build_full_text_and_boundaries(segments)
    def apply_spans(text, spans, replacements)
    def apply_masks_to_text(segments, replacements)
    def compute_replacements(operator_results, text)
```

### 5. `masking/document_reconstructor.py` (280 lines)
**Purpose:** Document structure preservation
```python
class DocumentReconstructor:
    def create_masked_document(original_doc, masked_segments)
    def update_table_cells(document, masked_text, boundaries)
    def get_table_node_id(segment, position)
    def find_segment_for_position(segments, position)
```

### 6. `masking/metadata_manager.py` (160 lines)
**Purpose:** CloakMap and metadata operations
```python
class MetadataManager:
    def create_anchor_entries(entities, operator_results)
    def enhance_cloakmap_with_metadata(cloakmap, metadata)
    def operator_result_to_dict(result)
    def get_reversible_operators()
    def create_synthetic_result(entity, replacement)
    def cleanup_large_results(results)
```

## Move Map

### From `presidio_adapter.py` to `strategy_processors.py`:
- Lines 450-550: `_apply_hash_strategy()`
- Lines 551-650: `_apply_partial_strategy()`
- Lines 651-750: `_apply_custom_strategy()`
- Lines 751-850: `_apply_surrogate_strategy()`
- Lines 851-900: `_fallback_redaction()`

### From `presidio_adapter.py` to `entity_processor.py`:
- Lines 200-320: `_filter_overlapping_entities()`
- Lines 321-420: `_validate_entities()`
- Lines 421-520: `_validate_entities_against_boundaries()`
- Lines 901-1000: `_batch_process_entities()`
- Lines 1001-1100: `_prepare_strategies()`

### From `presidio_adapter.py` to `text_processor.py`:
- Lines 100-199: `_build_full_text_and_boundaries()`
- Lines 1101-1150: `_apply_spans()`
- Lines 1151-1200: `_apply_masks_to_text()`
- Lines 1201-1250: `_compute_replacements()`

### From `presidio_adapter.py` to `document_reconstructor.py`:
- Lines 600-700: `_create_masked_document()`
- Lines 701-800: `_update_table_cells()`
- Lines 801-850: Helper methods for tables
- Lines 851-900: Position mapping utilities

### From `presidio_adapter.py` to `metadata_manager.py`:
- Lines 1251-1310: All metadata and CloakMap enhancement methods

## Re-exports for Backward Compatibility

```python
# masking/__init__.py
from .presidio_adapter import PresidioMaskingAdapter

# Optionally expose internal processors for testing
from .entity_processor import EntityProcessor
from .strategy_processors import StrategyProcessor
from .text_processor import TextProcessor

__all__ = ['PresidioMaskingAdapter']  # Maintain public API
```

## Call-site Updates

### Internal Updates Required:
- None - all internal methods remain accessible through composition

### External Updates Required:
- None - public API remains identical

### Test Updates Required:
- Split unit tests to match new module structure
- Add integration tests for component interactions
- Mock individual processors for focused testing

## Test Impact

### Affected Test Files:
- `tests/unit/test_presidio_adapter.py` - Split into 6 test files
- `tests/integration/test_masking_workflow.py` - No changes needed
- New test files needed:
  - `tests/unit/test_strategy_processors.py`
  - `tests/unit/test_entity_processor.py`
  - `tests/unit/test_text_processor.py`
  - `tests/unit/test_document_reconstructor.py`
  - `tests/unit/test_metadata_manager.py`

### Test Strategy:
1. Create comprehensive unit tests for each processor
2. Mock processor dependencies in adapter tests
3. Keep integration tests unchanged to verify behavior

## Risks & Mitigations

### Risk 1: Circular Dependencies
**Mitigation:** Use dependency injection and interfaces
```python
# Use protocols/interfaces
from typing import Protocol

class TextProcessorProtocol(Protocol):
    def apply_spans(self, text: str, spans: List, replacements: Dict) -> str: ...
```

### Risk 2: Performance Overhead
**Mitigation:** Profile before/after to ensure no regression
- Benchmark text processing operations
- Monitor memory usage for large documents
- Consider caching frequently used processors

### Risk 3: Breaking Changes
**Mitigation:** Comprehensive integration testing
- Run all existing tests without modification
- Add regression tests for edge cases
- Maintain exact same return types and exceptions

## Implementation Steps

1. **Phase 1: Extract Pure Functions** (Low Risk)
   - Move utility functions to new modules
   - Keep them as module-level functions initially

2. **Phase 2: Create Processor Classes** (Medium Risk)
   - Wrap functions in processor classes
   - Add proper initialization and configuration

3. **Phase 3: Wire Dependencies** (Medium Risk)
   - Update main adapter to use processors
   - Ensure proper composition and delegation

4. **Phase 4: Test & Validate** (Low Risk)
   - Run full test suite
   - Benchmark performance
   - Validate no behavior changes

5. **Phase 5: Clean Up** (Low Risk)
   - Remove old methods from adapter
   - Update documentation
   - Add deprecation notices if needed

## Acceptance Criteria

- [ ] All existing tests pass without modification
- [ ] No performance regression (±5% benchmark tolerance)
- [ ] Each new module is under 400 LOC
- [ ] 100% backward compatibility for public API
- [ ] New unit tests achieve >90% coverage per module
- [ ] Documentation updated with new structure
- [ ] No circular dependencies introduced

## Rollback Plan

If issues arise:
1. Keep original file as `presidio_adapter_legacy.py`
2. Use feature flag to switch implementations
3. Gradual migration with parallel testing
4. Full revert possible via single import change

---
*Estimated effort: 3-4 days*
*Risk level: MEDIUM*
*Priority: HIGH (largest file in codebase)*