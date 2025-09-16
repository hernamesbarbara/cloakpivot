# Feature Request: Enhance Surrogate Strategy to Generate Realistic Fake Data Using Faker

## Problem Statement

The current implementation of CloakPivot's `SURROGATE` strategy defaults to asterisk masking (`***`) instead of generating realistic fake data, despite having a well-designed `SurrogateGenerator` class that can produce quality fake data when called directly.

### Current Behavior
When using `StrategyKind.SURROGATE`:
- The Presidio adapter's surrogate operator defaults to asterisks
- Example: `"John Doe"` → `"********"` instead of `"Sarah Johnson"`
- Example: `"john@example.com"` → `"*****************"` instead of `"sarah.johnson@demo.com"`

### Expected Behavior
The surrogate strategy should generate realistic, format-preserving fake data:
- Names should be replaced with fake names
- Emails should be replaced with fake emails
- Dates should be replaced with fake dates in similar format
- Phone numbers should be replaced with fake phone numbers

## Root Cause Analysis

### 1. Working Components
The `SurrogateGenerator` class at `cloakpivot/core/surrogate.py` works correctly when called directly:

```python
from cloakpivot.core.surrogate import SurrogateGenerator

gen = SurrogateGenerator(seed='test')
print(gen.generate_surrogate('test@example.com', 'EMAIL_ADDRESS'))  # Returns: hoby@demo.com
print(gen.generate_surrogate('John Doe', 'PERSON'))                 # Returns: Alex Garcia
print(gen.generate_surrogate('2025-01-01', 'DATE_TIME'))           # Returns: 8303-97-15
```

### 2. Integration Issue
The problem occurs in the Presidio adapter integration. When examining the flow:

1. `CloakEngine` with `MaskingPolicy(default_strategy=Strategy(kind=StrategyKind.SURROGATE))`
2. → `PresidioAdapter._apply_surrogate_strategy()` at line 647
3. → Should call `self._surrogate_generator.generate_surrogate()`
4. → But Presidio's anonymizer appears to override with asterisks

### 3. Presidio Anonymizer Behavior
The issue appears to be in how Presidio's `AnonymizerEngine` handles the surrogate operator. The cloakmap shows `"presidio_operator": "surrogate"` but the output is asterisks, suggesting Presidio's default surrogate implementation is being used instead of CloakPivot's enhanced generator.

## Implementation Requirements

### 1. Fix Presidio Integration
Modify `cloakpivot/masking/presidio_adapter.py` to ensure the `SurrogateGenerator` is properly invoked:

```python
def _apply_surrogate_strategy(self, text: str, entity_type: str, strategy: Strategy) -> str:
    """Apply surrogate strategy with high-quality fake data generation."""
    try:
        # This should work but currently returns asterisks
        return self._surrogate_generator.generate_surrogate(text, entity_type)
    except Exception as e:
        logger.warning(f"Surrogate generation failed: {e}")
        return f"[{entity_type}]"
```

### 2. Custom Operator for Presidio
Consider implementing a custom Presidio operator that properly uses faker:

```python
from presidio_anonymizer.operators import Operator
from faker import Faker

class FakerOperator(Operator):
    def operate(self, text: str, params: dict = None) -> str:
        entity_type = params.get("entity_type")
        seed = params.get("seed")

        faker = Faker()
        if seed:
            faker.seed_instance(seed)

        # Map entity types to faker methods
        faker_map = {
            "PERSON": faker.name,
            "EMAIL_ADDRESS": faker.email,
            "PHONE_NUMBER": faker.phone_number,
            "DATE_TIME": lambda: faker.date_time().strftime("%Y-%m-%d"),
            "LOCATION": faker.city,
            "CREDIT_CARD": faker.credit_card_number,
        }

        generator = faker_map.get(entity_type, lambda: f"[{entity_type}]")
        return generator()
```

### 3. Ensure Deterministic Output
For consistency across runs (important for testing and reproducibility):

```python
def generate_consistent_fake(original_text: str, entity_type: str, seed: str = None):
    """Generate consistent fake data based on original text hash"""
    import hashlib

    # Use hash of original text + seed for consistency
    combined = f"{original_text}:{seed or ''}"
    hash_val = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)

    faker = Faker()
    faker.seed_instance(hash_val)

    # Generate appropriate fake data...
```

## Testing Requirements

### Test Case 1: Basic Surrogate Generation
```python
policy = MaskingPolicy(
    default_strategy=Strategy(
        kind=StrategyKind.SURROGATE,
        parameters={}
    ),
    seed="test-seed"
)
engine = CloakEngine(default_policy=policy)
result = engine.mask_document(doc)

# Should produce fake data, not asterisks
assert "*" not in result.document.texts[0].text  # Should fail currently
assert any(name in result.document.texts[0].text
          for name in ["Sarah", "John", "Michael", "Jennifer"])  # Fake names
```

### Test Case 2: Consistency
```python
# Same input with same seed should produce same fake data
result1 = engine.mask_document(doc)
result2 = engine.mask_document(doc)
assert result1.document.texts == result2.document.texts
```

### Test Case 3: Format Preservation
```python
# Email format should be preserved
original = "john.doe@company.com"
masked = mask_email(original)
assert "@" in masked
assert "." in masked
assert masked != original
assert "*" not in masked
```

## Files to Modify

1. **`cloakpivot/masking/presidio_adapter.py`**
   - Fix `_apply_surrogate_strategy()` method
   - Ensure `SurrogateGenerator` is properly initialized and used
   - Add proper error handling and logging

2. **`cloakpivot/masking/applicator.py`**
   - Review `_apply_surrogate_strategy()` at line 716
   - Ensure it's not falling back to `_apply_legacy_surrogate_strategy()`
   - Check the surrogate generator initialization

3. **`cloakpivot/core/surrogate.py`**
   - Already works correctly, but may need minor adjustments for Presidio integration
   - Consider adding a method specifically for Presidio operator compatibility

4. **`cloakpivot/engine.py`**
   - Verify that the policy is correctly passed through to the masking engine
   - Check if any configuration is being overridden

## Workaround (Current)

Until this is fixed, users can use the CUSTOM strategy with faker callbacks:

```python
from faker import Faker
import hashlib

def create_faker_callback(entity_type):
    def callback(text):
        faker = Faker()
        hash_val = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        faker.seed_instance(hash_val)

        if entity_type == "PERSON":
            return faker.name()
        elif entity_type == "EMAIL_ADDRESS":
            return faker.email()
        # ... etc
    return callback

policy = MaskingPolicy(
    default_strategy=Strategy(
        kind=StrategyKind.CUSTOM,
        parameters={"callback": create_faker_callback("PERSON")}
    )
)
```

However, this workaround also currently produces asterisks, suggesting the issue is deeper in the Presidio integration layer.

## Success Criteria

1. ✅ `StrategyKind.SURROGATE` produces realistic fake data, not asterisks
2. ✅ Fake data is deterministic when seed is provided
3. ✅ Format is preserved (emails look like emails, phones like phones)
4. ✅ Different entity types use appropriate faker methods
5. ✅ Backwards compatibility maintained (existing code still works)
6. ✅ Tests pass showing fake data generation works correctly

## Additional Context

This issue was discovered while implementing multiple masking strategies in the testcloak repository. The SurrogateGenerator class is well-designed and functional, but the integration with Presidio's anonymizer appears to bypass it, defaulting to asterisk replacement instead.

The fix should maintain the existing architecture while ensuring the SurrogateGenerator is actually invoked during the masking process. This will enable CloakPivot to provide high-quality, realistic data masking that's useful for testing and demonstration purposes while maintaining privacy.