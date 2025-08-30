# CloakPivot Policy Library

This directory contains a comprehensive collection of reference policies for common use cases and industry scenarios.

## Directory Structure

```
policies/
├── README.md                    # This file
├── templates/                   # Base templates for different security levels
│   ├── conservative.yaml        # High security, strict thresholds
│   ├── balanced.yaml           # Balanced security and usability
│   └── permissive.yaml         # Development-friendly settings
├── industries/                  # Industry-specific policies
│   ├── healthcare/             # HIPAA-compliant policies
│   ├── finance/               # PCI DSS and financial regulations
│   ├── legal/                 # Legal document protection
│   ├── education/             # FERPA and student data
│   └── government/            # Government and public sector
├── examples/                   # Example policies for learning
│   ├── basic_examples.yaml    # Simple policy configurations
│   ├── advanced_features.yaml # Complex policy patterns
│   └── context_rules.yaml    # Context-aware masking examples
└── custom/                    # Space for your custom policies
    └── .gitkeep
```

## Quick Start

### Choose a Template

Start with one of the base templates:

```bash
# Copy a template as your starting point
cp policies/templates/balanced.yaml my-policy.yaml

# Apply to a document
cloakpivot mask document.pdf --policy my-policy.yaml
```

### Industry-Specific Policies

Use pre-configured policies for your industry:

```bash
# Healthcare
cloakpivot mask patient-records.pdf --policy policies/industries/healthcare/hipaa-compliant.yaml

# Finance
cloakpivot mask financial-statement.pdf --policy policies/industries/finance/pci-dss.yaml
```

## Policy Development Workflow

1. **Start with a template** that matches your security requirements
2. **Customize** entity strategies and thresholds for your use case
3. **Test** the policy with sample documents
4. **Validate** using the policy validation tools
5. **Deploy** with proper version control and governance

```bash
# Development workflow
cloakpivot policy template balanced > my-policy.yaml
cloakpivot policy validate my-policy.yaml
cloakpivot policy test my-policy.yaml --text "Test data with John Doe"
cloakpivot mask test-document.pdf --policy my-policy.yaml --verbose
```

## Policy Categories

### Security Levels

| Template | Use Case | Characteristics |
|----------|----------|-----------------|
| **Conservative** | High-security environments | High thresholds, hash/template strategies, minimal visibility |
| **Balanced** | Production systems | Moderate thresholds, mix of strategies, good usability |
| **Permissive** | Development/testing | Lower thresholds, partial masking, development-friendly |

### Industry Compliance

| Industry | Regulations | Key Features |
|----------|-------------|--------------|
| **Healthcare** | HIPAA, HITECH | Patient privacy, PHI protection, audit trails |
| **Finance** | PCI DSS, SOX, GDPR | Payment card security, financial privacy |
| **Legal** | Attorney-client privilege | Document confidentiality, case sensitivity |
| **Education** | FERPA, COPPA | Student privacy, educational records |
| **Government** | FIPS, FedRAMP | Government data classification |

## Common Policy Patterns

### Entity Strategies

```yaml
# Template replacement - Good for consistent formatting
PERSON:
  kind: "template"
  parameters:
    template: "[PATIENT]"

# Partial masking - Preserves format while hiding sensitive data
CREDIT_CARD:
  kind: "partial"
  parameters:
    visible_chars: 4
    position: "end"

# Hashing - For deterministic anonymization
EMAIL_ADDRESS:
  kind: "hash"
  parameters:
    algorithm: "sha256"
    truncate: 8
```

### Context Rules

```yaml
# Context-aware masking
context_rules:
  heading:
    enabled: false  # Don't mask titles/headings
  table:
    enabled: true
    threshold_overrides:
      PERSON: 0.9  # Higher confidence in tables
  footer:
    enabled: false  # Preserve contact info in footers
```

### Allow/Deny Lists

```yaml
# Explicit control over masking
allow_list:
  - "Customer Service"
  - "support@company.com"
  
deny_list:
  - "confidential"
  - "internal use only"
```

## Policy Validation

Always validate policies before deployment:

```bash
# Validate syntax and logic
cloakpivot policy validate my-policy.yaml

# Test with sample data
cloakpivot policy test my-policy.yaml --verbose

# Generate policy information
cloakpivot policy info my-policy.yaml
```

## Custom Policy Development

### Extending Templates

```yaml
# Start with a base template
version: "1.0"
name: "my-custom-policy"
extends: "templates/balanced.yaml"  # Inherit from template

# Add your customizations
per_entity:
  CUSTOM_ID:
    kind: "hash"
    parameters:
      algorithm: "sha256"
      prefix: "ID_"
```

### Organization Standards

Create organization-wide standards:

```yaml
# Base organizational policy
organization:
  name: "ACME Corp"
  compliance: ["SOC2", "GDPR"]
  
default_settings:
  locale: "en"
  seed: "acme-corp-2023"
  
standard_allow_list:
  - "ACME Corporation"
  - "support@acme.com"
```

## Testing and Validation

### Test Data Sets

Use representative test data for each policy:

```bash
# Healthcare test data
echo "Patient John Doe, DOB: 1980-01-01, SSN: 123-45-6789" > test-health.txt
cloakpivot policy test policies/industries/healthcare/hipaa-compliant.yaml \
  --text "$(cat test-health.txt)"

# Financial test data
echo "Account: Robert Smith, Card: 4532-1234-5678-9012" > test-finance.txt
cloakpivot policy test policies/industries/finance/pci-dss.yaml \
  --text "$(cat test-finance.txt)"
```

### Validation Checklist

- [ ] YAML syntax is valid
- [ ] All required fields are present
- [ ] Thresholds are in valid range (0.0-1.0)
- [ ] Strategy parameters are correct
- [ ] Entity types are recognized or properly defined
- [ ] Context rules are logically consistent
- [ ] Allow/deny lists don't conflict
- [ ] Policy has been tested with representative data

## Best Practices

### Security

1. **Use appropriate seeds** for deterministic operations
2. **Rotate salts** regularly for hash strategies
3. **Version control** all policy changes
4. **Restrict access** to policy files in production
5. **Audit policy usage** and modifications

### Performance

1. **Use higher thresholds** to reduce false positives
2. **Choose efficient strategies** (redact > template > partial > hash)
3. **Limit entity types** to what's actually needed
4. **Use context rules** to skip unnecessary processing
5. **Test with realistic document sizes**

### Maintenance

1. **Document policy decisions** and rationale
2. **Maintain test suites** for policy validation
3. **Review policies regularly** for effectiveness
4. **Update for new regulations** and requirements
5. **Share learnings** across teams and organizations

## Contributing

To contribute new policies or improvements:

1. Follow the existing naming conventions
2. Include comprehensive documentation
3. Provide test cases and validation
4. Consider multiple use cases and edge cases
5. Submit policies that others can benefit from

## Support

- **Documentation**: See the main CloakPivot documentation
- **Examples**: Check the `examples/` directory
- **Issues**: Report problems via the project issue tracker
- **Community**: Join discussions about policy development

## License

These policies are provided as examples and starting points. Review and customize them according to your specific requirements and regulatory obligations.