# Enhanced Surrogate Generation

CloakPivot's enhanced surrogate generation system provides format-preserving, realistic-looking replacement values that maintain original data characteristics while being completely artificial.

## Overview

The surrogate generation system consists of three main components:

1. **FormatPattern Analysis** - Analyzes original text to extract format patterns
2. **SurrogateGenerator** - Generates format-preserving surrogates with collision detection  
3. **Quality Metrics** - Tracks generation quality and validation success

## Key Features

### Format-Preserving Generation

Surrogates maintain the exact format characteristics of the original data:

- **Character Classes**: Preserves digits, letters, separators, and special characters in exact positions
- **Length Preservation**: Generated surrogates match original text length exactly  
- **Structure Preservation**: Maintains formatting like dashes, dots, spaces, and parentheses

### Deterministic Generation

- **Consistent Results**: Identical inputs with the same seed produce identical surrogates
- **Per-Entity Seeding**: Different entity types use isolated seed spaces to avoid cross-contamination
- **Collision Detection**: Automatically resolves collisions with retry logic and suffix fallback

### Entity-Specific Rules

Built-in generators for common PII types:

- **Phone Numbers**: Generates valid US phone numbers with realistic area codes
- **SSNs**: Creates surrogates avoiding invalid SSN ranges (000-xx-xxxx, 666-xx-xxxx, 9xx-xx-xxxx)
- **Email Addresses**: Uses safe domains (example.com, test.org, sample.net)
- **Credit Cards**: Generates test card numbers with known test prefixes
- **Names**: Uses common placeholder names (Jordan Smith, Alex Johnson, etc.)

## Usage Examples

### Basic Surrogate Generation

```python
from cloakpivot.core.surrogate import SurrogateGenerator

# Create generator with deterministic seed
generator = SurrogateGenerator(seed="my_document_seed")

# Generate format-preserving surrogates
phone_surrogate = generator.generate_surrogate("555-123-4567", "PHONE_NUMBER")
print(phone_surrogate)  # "208-456-7890" (preserves XXX-XXX-XXXX format)

ssn_surrogate = generator.generate_surrogate("123-45-6789", "US_SSN")  
print(ssn_surrogate)    # "456-78-9012" (preserves XXX-XX-XXXX format)

email_surrogate = generator.generate_surrogate("user@company.com", "EMAIL_ADDRESS")
print(email_surrogate)  # "alex@example.com" (preserves user@domain.tld format)
```

### Using with StrategyApplicator

```python
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.masking.applicator import StrategyApplicator

# Create applicator with seed
applicator = StrategyApplicator(seed="document_seed")

# Use surrogate strategy  
strategy = Strategy(StrategyKind.SURROGATE)

# Apply to different entity types
masked_phone = applicator.apply_strategy("555-123-4567", "PHONE_NUMBER", strategy, 0.95)
masked_email = applicator.apply_strategy("john@acme.com", "EMAIL_ADDRESS", strategy, 0.95)
masked_ssn = applicator.apply_strategy("123-45-6789", "US_SSN", strategy, 0.95)

# Check quality metrics
metrics = applicator.get_surrogate_quality_metrics()
print(f"Format preservation rate: {metrics.format_preservation_rate:.2f}")
print(f"Uniqueness rate: {metrics.uniqueness_rate:.2f}")
```

### Custom Pattern Generation

```python
# Generate from explicit patterns
# X = uppercase letter, x = lowercase letter, 9 = digit, ? = alphanumeric
custom_surrogate = generator.generate_from_pattern("XXX-XX-9999")
print(custom_surrogate)  # "ABC-DE-1234"

# Use custom patterns in strategies
custom_strategy = Strategy(StrategyKind.SURROGATE, {"pattern": "XX9-9X9"})
result = applicator.apply_strategy("AB1-2C3", "CUSTOM", custom_strategy, 0.95)
```

## Format Pattern Analysis

The `FormatPattern` class analyzes original text to extract structural information:

```python
from cloakpivot.core.surrogate import FormatPattern

pattern = FormatPattern.analyze("555-123-4567")
print(f"Length: {pattern.original_length}")           # 12
print(f"Character classes: {pattern.character_classes}")  # "DDDSDDDSDDDD"  
print(f"Digit positions: {pattern.digit_positions}")      # [0,1,2,4,5,6,8,9,10,11]
print(f"Separators: {pattern.separator_positions}")       # {3: '-', 7: '-'}
print(f"Detected format: {pattern.detected_format}")      # "phone"
```

## Quality Metrics and Validation

The system tracks comprehensive quality metrics:

```python
metrics = generator.get_quality_metrics()

print(f"Total generated: {metrics.total_generated}")
print(f"Format preservation rate: {metrics.format_preservation_rate:.2f}")
print(f"Uniqueness rate: {metrics.uniqueness_rate:.2f}")  
print(f"Validation success rate: {metrics.validation_success_rate:.2f}")
print(f"Collision count: {metrics.collision_count}")
```

### Validation Checks

The system validates generated surrogates:

1. **Format Preservation**: Ensures character classes match original positions
2. **Uniqueness**: Tracks uniqueness within document scope  
3. **Quality**: Validates surrogates don't match original data
4. **Safety**: Basic checks to avoid accidentally generating real PII patterns

## Document Scope Management

For processing multiple documents, reset scope between documents:

```python
# Process first document
surrogate1 = generator.generate_surrogate("555-123-4567", "PHONE_NUMBER")

# Reset for new document (clears uniqueness tracking)
generator.reset_document_scope()

# Process second document  
surrogate2 = generator.generate_surrogate("555-123-4567", "PHONE_NUMBER")

# Results will be identical due to deterministic seeding
assert surrogate1 == surrogate2
```

## Advanced Configuration

### Custom Entity Generators

The system can be extended with custom entity-specific generators by subclassing `SurrogateGenerator` and overriding the `_initialize_entity_generators()` method.

### Collision Handling

The system handles collisions automatically:

1. **Retry Logic**: Up to 10 attempts with different seeds
2. **Suffix Fallback**: Adds numeric suffix if retries fail  
3. **Ultimate Fallback**: Hash-based suffix for extreme edge cases

### Error Handling

Robust error handling with graceful degradation:

1. **Enhanced Generation**: Primary format-preserving generation
2. **Legacy Fallback**: Falls back to simpler generation if enhanced fails
3. **Simple Fallback**: Character-by-character replacement as last resort

## Performance Considerations

- **Caching**: Results cached within document scope for consistency
- **Seed Optimization**: SHA256-based deterministic seeding  
- **Memory Management**: Bounded collision tracking and metrics storage
- **Efficiency**: Pattern analysis performed once per unique format

## Integration Notes

The enhanced surrogate generation is fully integrated with the existing CloakPivot masking system:

- **Backward Compatibility**: Existing surrogate strategies continue to work
- **Transparent Upgrade**: Enhanced generation used automatically when available
- **Fallback Support**: Graceful degradation to legacy methods when needed
- **Quality Monitoring**: Comprehensive metrics for monitoring generation quality

For more implementation details, see the source code in `cloakpivot/core/surrogate.py` and integration tests in `tests/test_strategy_surrogate_integration.py`.