# Documentation Recreation Plan

**Date:** September 13, 2025
**Priority:** MEDIUM - After API stabilization
**Estimated Effort:** 2-3 days

## Background

The entire `docs/` directory was removed on September 13, 2025, because it was completely outdated after the CloakEngine refactoring. The old documentation referenced removed modules (storage, plugins, diagnostics, migration) and used the old multi-step API instead of the simplified CloakEngine API.

## Current State

### What Was Removed
- Sphinx-based documentation in `docs/` directory
- RST files covering old API patterns
- References to 7+ removed modules
- CLI documentation for removed commands
- API documentation for MaskingEngine/UnmaskingEngine (replaced by CloakEngine)

### What Exists Now
- `README.md` - Basic project overview (needs update)
- `examples/simple_usage.py` - Shows basic CloakEngine usage
- `examples/advanced_usage.py` - Shows builder pattern and advanced features
- `specification/*.md` - Implementation documentation (internal)
- Docstrings in code (especially in `engine.py`, `engine_builder.py`, `defaults.py`)

## Documentation Recreation Strategy

### Phase 1: Core Documentation (Priority: HIGH)

#### 1. Update README.md
```markdown
# CloakPivot

Simple, reversible PII masking for documents.

## Quick Start

```python
from cloakpivot import CloakEngine
from docling.document_converter import DocumentConverter

# Convert document
converter = DocumentConverter()
doc = converter.convert("document.pdf").document

# Mask PII with one line
engine = CloakEngine()
result = engine.mask_document(doc)

print(f"Masked {result.entities_masked} PII entities")

# Unmask when needed
original = engine.unmask_document(result.document, result.cloakmap)
```
```

#### 2. Create docs/quickstart.md
- Installation instructions
- Basic usage example
- Common use cases
- Link to examples

#### 3. Create docs/api_reference.md
- CloakEngine class documentation
- CloakEngineBuilder documentation
- Default policies documentation
- MaskResult and CloakMap structures

### Phase 2: User Guides (Priority: MEDIUM)

#### 1. docs/user_guide/masking_policies.md
- How policies work
- Available strategies (REDACT, TEMPLATE, PARTIAL, etc.)
- Creating custom policies
- Using policy presets

#### 2. docs/user_guide/cli_usage.md
- `cloakpivot mask` command
- `cloakpivot unmask` command
- Configuration options
- Examples

#### 3. docs/user_guide/builder_pattern.md
- Using CloakEngineBuilder
- Configuration options
- Language support
- Confidence thresholds

### Phase 3: Developer Documentation (Priority: LOW)

#### 1. docs/development/architecture.md
- System architecture overview
- How CloakEngine wraps internal components
- Document processing flow
- Anchor system explanation

#### 2. docs/development/extending.md
- Adding new masking strategies
- Custom entity recognizers
- Integration with other systems

#### 3. docs/development/testing.md
- Running tests
- Test structure
- Adding new tests

## Documentation Format Recommendations

### Use Markdown Instead of RST
- Simpler to write and maintain
- Better GitHub integration
- Can still use MkDocs or similar for site generation

### Focus on Examples
- Every feature should have a code example
- Use the actual CloakEngine API
- Show both simple and advanced usage

### Keep It Minimal
- Don't document removed features
- Focus on the 80% use case
- Link to code for advanced details

## Documentation Tools

### Option 1: MkDocs (Recommended)
```yaml
# mkdocs.yml
site_name: CloakPivot
theme:
  name: material
  features:
    - navigation.sections
    - navigation.expand
    - content.code.copy

nav:
  - Home: index.md
  - Quick Start: quickstart.md
  - User Guide:
    - Masking Policies: user_guide/masking_policies.md
    - CLI Usage: user_guide/cli_usage.md
    - Builder Pattern: user_guide/builder_pattern.md
  - API Reference: api_reference.md
  - Examples:
    - Simple Usage: examples/simple.md
    - Advanced Usage: examples/advanced.md
```

### Option 2: GitHub Wiki
- No build process needed
- Easy to edit
- Good for small projects

### Option 3: Just Markdown in repo
- Simplest approach
- No tooling needed
- Good enough for most users

## Key Documentation Principles

### 1. Start with Why
- Explain what problem CloakPivot solves
- Show the value proposition upfront
- Make it clear why someone would use this

### 2. Show, Don't Tell
- Lead with code examples
- Minimize explanatory text
- Let the API speak for itself

### 3. Document the Current State
- Don't mention removed features
- Don't show old API patterns
- Focus on CloakEngine only

### 4. Progressive Disclosure
- Start with simplest usage
- Add complexity gradually
- Link to advanced topics

## Example Documentation Structure

```
docs/
├── README.md                 # Overview and quick start
├── installation.md           # Installation instructions
├── quickstart.md            # Getting started guide
├── user_guide/
│   ├── basic_usage.md       # Simple masking/unmasking
│   ├── policies.md          # Policy configuration
│   ├── builder.md           # Advanced configuration
│   └── cli.md               # CLI reference
├── api/
│   ├── cloakengine.md       # CloakEngine API
│   ├── builder.md           # Builder API
│   ├── policies.md          # Policy API
│   └── cloakmap.md          # CloakMap structure
├── examples/
│   ├── simple.md            # Basic examples
│   ├── advanced.md          # Advanced examples
│   └── real_world.md        # Real-world scenarios
└── development/
    ├── contributing.md      # How to contribute
    ├── architecture.md      # System design
    └── testing.md           # Testing guide
```

## Priority Documentation Items

### Must Have (Week 1)
1. Updated README.md with CloakEngine examples
2. Basic API reference for CloakEngine
3. Installation instructions
4. Migration guide from old API

### Should Have (Week 2)
1. Policy configuration guide
2. CLI documentation
3. Builder pattern guide
4. Troubleshooting guide

### Nice to Have (Week 3)
1. Architecture documentation
2. Performance tuning guide
3. Security best practices
4. Integration examples

## Documentation Testing

### Code Examples
- All code examples should be executable
- Consider using doctest or similar
- Test examples in CI/CD

### Link Checking
- Ensure all internal links work
- Check external links periodically
- Use tools like linkchecker

### User Testing
- Have someone unfamiliar with the project try to use it
- Note where they get stuck
- Improve those sections

## Conclusion

The documentation needs a complete rewrite to match the new CloakEngine API. Start with a minimal set of essential documentation (README, quickstart, API reference) and expand based on user feedback. Focus on showing how simple the new API is compared to the old approach.