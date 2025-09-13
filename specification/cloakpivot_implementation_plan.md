# CloakPivot Implementation Plan

**Date:** September 13, 2025  
**Priority:** HIGH - Implement First  
**Estimated Effort:** 3-4 days  

## Overview

This memo outlines the specific technical implementation tasks for improving CloakPivot based on the code review. The primary goal is to create a simplified `CloakEngine` API that matches Presidio's simplicity pattern while encapsulating the current complex workflow.

## Implementation Tasks

### Task 1: Create CloakEngine Core Class
**File:** `cloakpivot/engine.py` (new file)

```python
# Core structure to implement
class CloakEngine:
    """High-level API for PII masking/unmasking operations on DoclingDocument instances.
    
    Provides a Presidio-like simple interface while encapsulating:
    - TextExtractor initialization
    - AnalyzerEngine configuration
    - MaskingEngine setup
    - Policy management
    """
    
    def __init__(self, 
                 analyzer_config=None,
                 default_policy=None,
                 conflict_resolution_config=None):
        """Initialize with sensible defaults for all components."""
        pass
    
    def mask_document(self, 
                     document: DoclingDocument,
                     entities: List[str] = None,
                     policy: MaskingPolicy = None) -> MaskResult:
        """One-line masking with auto-detection if entities not specified."""
        pass
    
    def unmask_document(self, 
                       document: DoclingDocument,
                       cloakmap: CloakMap) -> DoclingDocument:
        """Simple unmasking using stored CloakMap."""
        pass
    
    @classmethod
    def builder(cls) -> 'CloakEngineBuilder':
        """Builder pattern for advanced configuration."""
        pass
```

### Task 2: Implement Builder Pattern for Advanced Configuration
**File:** `cloakpivot/engine_builder.py` (new file)

```python
class CloakEngineBuilder:
    """Fluent builder for CloakEngine with advanced configuration."""
    
    def with_custom_policy(self, policy: MaskingPolicy) -> 'CloakEngineBuilder':
        pass
    
    def with_analyzer_config(self, config: dict) -> 'CloakEngineBuilder':
        pass
    
    def with_conflict_resolution(self, config: ConflictResolutionConfig) -> 'CloakEngineBuilder':
        pass
    
    def with_presidio_engine(self, use: bool = True) -> 'CloakEngineBuilder':
        pass
    
    def build(self) -> CloakEngine:
        pass
```

### Task 3: Create Smart Defaults System
**File:** `cloakpivot/defaults.py` (new file)

```python
# Implement smart defaults that cover 90% of use cases
DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER", 
    "CREDIT_CARD", "US_SSN", "LOCATION", "DATE_TIME"
]

def get_default_policy() -> MaskingPolicy:
    """Return a sensible default policy for common PII types."""
    return MaskingPolicy(
        per_entity={
            "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
            "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
            "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
            "CREDIT_CARD": Strategy(StrategyKind.TEMPLATE, {"template": "[CARD]"}),
            "US_SSN": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
            "LOCATION": Strategy(StrategyKind.TEMPLATE, {"template": "[LOCATION]"}),
            "DATE_TIME": Strategy(StrategyKind.KEEP, {}),
        },
        default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
    )

def get_default_analyzer_config() -> dict:
    """Return optimized analyzer configuration."""
    return {
        "languages": ["en"],
        "confidence_threshold": 0.7,
        "return_decision_process": False
    }
```

### Task 4: Add Method Registration System
**File:** `cloakpivot/registration.py` (new file)

```python
def register_cloak_methods():
    """Register masking/unmasking methods on DoclingDocument.
    
    This allows for natural method chaining:
    doc.mask_pii().export_to_markdown()
    """
    from docling_core.types import DoclingDocument
    
    # Store engine as class variable for reuse
    if not hasattr(DoclingDocument, '_cloak_engine'):
        DoclingDocument._cloak_engine = CloakEngine()
    
    def mask_pii(self, entities=None, policy=None):
        """Mask PII and return CloakedDocument wrapper."""
        result = self._cloak_engine.mask_document(self, entities, policy)
        return CloakedDocument(result.document, result.cloakmap)
    
    def unmask_pii(self):
        """Unmask if this is a CloakedDocument."""
        if hasattr(self, '_cloakmap'):
            return self._cloak_engine.unmask_document(self._doc, self._cloakmap)
        return self
    
    DoclingDocument.mask_pii = mask_pii
    DoclingDocument.unmask_pii = unmask_pii
```

### Task 5: Implement CloakedDocument Wrapper
**File:** `cloakpivot/wrappers.py` (new file)

```python
class CloakedDocument:
    """Lightweight wrapper that preserves CloakMap while maintaining DoclingDocument interface."""
    
    def __init__(self, document: DoclingDocument, cloakmap: CloakMap):
        self._doc = document
        self._cloakmap = cloakmap
        self._engine = CloakEngine()  # Reuse singleton
    
    def __getattr__(self, name):
        """Delegate all DoclingDocument methods transparently."""
        return getattr(self._doc, name)
    
    def unmask_pii(self) -> DoclingDocument:
        """Unmask using stored CloakMap."""
        return self._engine.unmask_document(self._doc, self._cloakmap)
    
    @property
    def cloakmap(self) -> CloakMap:
        """Access the CloakMap for persistence."""
        return self._cloakmap
    
    @property
    def document(self) -> DoclingDocument:
        """Access the underlying masked document."""
        return self._doc
```

### Task 6: Update Package Exports
**File:** `cloakpivot/__init__.py` (modify existing)

```python
# Add new exports
from .engine import CloakEngine
from .engine_builder import CloakEngineBuilder
from .registration import register_cloak_methods
from .wrappers import CloakedDocument
from .defaults import DEFAULT_ENTITIES, get_default_policy

__all__ = [
    # ... existing exports ...
    
    # New simplified API
    "CloakEngine",
    "CloakEngineBuilder",
    "register_cloak_methods",
    "CloakedDocument",
    "DEFAULT_ENTITIES",
    "get_default_policy",
]
```

### Task 7: Create Usage Examples
**File:** `examples/simple_masking.py` (new file)

```python
"""Example: Simple PII masking with CloakEngine."""

from pathlib import Path
from docling.document_converter import DocumentConverter
from cloakpivot import CloakEngine

# Convert document
converter = DocumentConverter()
result = converter.convert("document.pdf")
doc = result.document

# Simple one-line masking
engine = CloakEngine()
masked_result = engine.mask_document(doc)

print(f"Masked {len(masked_result.cloakmap.anchors)} PII entities")
print(masked_result.document.export_to_markdown())

# Unmask when needed
original = engine.unmask_document(masked_result.document, masked_result.cloakmap)
```

### Task 8: Create Integration Tests
**File:** `tests/test_cloak_engine.py` (new file)

```python
"""Test suite for simplified CloakEngine API."""

def test_default_initialization():
    """Test CloakEngine initializes with sensible defaults."""
    engine = CloakEngine()
    assert engine is not None
    # Test has default policy, analyzer, etc.

def test_simple_masking():
    """Test one-line masking workflow."""
    engine = CloakEngine()
    doc = load_test_document()
    result = engine.mask_document(doc)
    assert result.document != doc
    assert len(result.cloakmap.anchors) > 0

def test_round_trip():
    """Test mask/unmask round trip preserves document."""
    engine = CloakEngine()
    original = load_test_document()
    masked_result = engine.mask_document(original)
    unmasked = engine.unmask_document(masked_result.document, masked_result.cloakmap)
    assert unmasked.export_to_dict() == original.export_to_dict()

def test_builder_pattern():
    """Test advanced configuration via builder."""
    engine = CloakEngine.builder() \
        .with_custom_policy(custom_policy) \
        .with_analyzer_config({"languages": ["es"]}) \
        .build()
    assert engine is not None

def test_method_registration():
    """Test methods can be registered on DoclingDocument."""
    from cloakpivot import register_cloak_methods
    register_cloak_methods()
    
    doc = load_test_document()
    assert hasattr(doc, 'mask_pii')
    assert hasattr(doc, 'unmask_pii')
    
    masked = doc.mask_pii()
    assert hasattr(masked, 'cloakmap')
    unmasked = masked.unmask_pii()
    assert unmasked.export_to_dict() == doc.export_to_dict()
```

## Implementation Order

1. **Day 1:** Tasks 1-3 (CloakEngine core, Builder, Defaults)
2. **Day 2:** Tasks 4-5 (Registration system, CloakedDocument wrapper)
3. **Day 3:** Tasks 6-7 (Update exports, Create examples)
4. **Day 4:** Task 8 (Integration tests) + Documentation

## Success Criteria

- [ ] Users can mask a document in 1-2 lines of code
- [ ] Default configuration handles 90% of use cases
- [ ] Builder pattern available for advanced users
- [ ] Method chaining works naturally
- [ ] All existing functionality remains accessible
- [ ] 100% backward compatibility maintained
- [ ] Comprehensive test coverage for new API

## Notes for Implementation Session

- Start with Task 1 (CloakEngine core) as it's the foundation
- Keep all existing code intact - this is purely additive
- Focus on developer experience - every decision should reduce friction
- Test with the existing `docling2cloaked.py` script to ensure it can be simplified
- Document every public method with clear examples