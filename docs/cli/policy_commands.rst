Policy Commands
===============

The policy command group provides comprehensive tools for creating, managing, validating, and testing masking policies.

.. contents::
   :local:
   :depth: 2

Overview
--------

Masking policies define how different types of PII should be handled during the masking process. CloakPivot's policy system supports:

* **Entity-specific strategies**: Different approaches for different PII types
* **Configurable thresholds**: Confidence levels for entity detection
* **Context-aware rules**: Different behavior in headings, tables, etc.
* **Allow/deny lists**: Explicit inclusion/exclusion of specific values
* **Inheritance and composition**: Building complex policies from simpler ones

Policy Sample
-------------

Generate sample policy files for learning and customization.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy sample [OPTIONS]

Description
~~~~~~~~~~~

Creates a comprehensive sample policy file demonstrating all available configuration options, including entity strategies, context rules, and advanced features.

Options
~~~~~~~

.. option:: --output, -o <file>

    Output file path. If not specified, writes to stdout.

    .. code-block:: bash
    
        # Write to file
        cloakpivot policy sample --output my-policy.yaml
        
        # Write to stdout (default)
        cloakpivot policy sample

Examples
~~~~~~~~

.. code-block:: bash

    # Generate sample and save to file
    $ cloakpivot policy sample > sample-policy.yaml
    
    # Generate with explicit output path
    $ cloakpivot policy sample --output healthcare-template.yaml
    Sample policy written to healthcare-template.yaml

Policy Template
---------------

Generate policies from built-in templates optimized for different security levels.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy template TEMPLATE_NAME [OPTIONS]

Arguments
~~~~~~~~~

.. option:: TEMPLATE_NAME

    Template to use for policy generation.
    
    **Choices:**
    
    * ``conservative`` - High security with strict thresholds and extensive masking
    * ``balanced`` - Reasonable security with good usability balance
    * ``permissive`` - Lower security for development and testing environments

Options
~~~~~~~

.. option:: --output, -o <file>

    Output file path. If not specified, writes to stdout.

Examples
~~~~~~~~

.. code-block:: bash

    # Generate balanced policy template
    $ cloakpivot policy template balanced > production-policy.yaml
    ✅ Balanced template written to production-policy.yaml
    
    # Generate conservative template for high-security environments
    $ cloakpivot policy template conservative --output secure-policy.yaml
    ✅ Conservative template written to secure-policy.yaml
    
    # View permissive template
    $ cloakpivot policy template permissive

Template Characteristics
~~~~~~~~~~~~~~~~~~~~~~~~

**Conservative Template:**

* High confidence thresholds (0.8-0.95)
* Aggressive masking strategies (hash, template)
* Strict context rules
* Comprehensive entity coverage

**Balanced Template:**

* Moderate confidence thresholds (0.6-0.8)
* Mix of partial masking and templates
* Reasonable context rules
* Good usability/security balance

**Permissive Template:**

* Lower confidence thresholds (0.4-0.7)
* Partial masking preferred
* Relaxed context rules
* Development-friendly settings

Policy Validate
---------------

Validate policy files for syntax errors, logical consistency, and compatibility.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy validate POLICY_FILE [OPTIONS]

Arguments
~~~~~~~~~

.. option:: POLICY_FILE

    Path to the policy file to validate.

Options
~~~~~~~

.. option:: --verbose, -v

    Show detailed validation information including policy summary and warnings.

Examples
~~~~~~~~

.. code-block:: bash

    # Basic validation
    $ cloakpivot policy validate my-policy.yaml
    ✅ Policy file is valid
    
    # Detailed validation with summary
    $ cloakpivot policy validate healthcare-policy.yaml --verbose
    🔍 Validating policy file: healthcare-policy.yaml
    ✅ Policy file is valid
       Locale: en
       Entity strategies: 8
       Context rules: 4
       Allow list items: 3
       Deny list items: 5

Validation Checks
~~~~~~~~~~~~~~~~~

The validator performs these checks:

1. **YAML Syntax**: Valid YAML structure and formatting
2. **Schema Validation**: Required fields and correct data types
3. **Strategy Consistency**: Valid strategy kinds and parameters
4. **Threshold Ranges**: Confidence thresholds between 0.0 and 1.0
5. **Entity Type Validation**: Known entity types and custom patterns
6. **Context Rule Logic**: Valid context specifications
7. **Circular Dependencies**: No inheritance loops in policy composition

Common Validation Errors
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    $ cloakpivot policy validate broken-policy.yaml
    ❌ Policy validation failed:
       • Invalid strategy kind 'invalid_strategy' at line 15
       • Threshold 1.5 out of range (0.0-1.0) for entity PERSON
       • Missing required field 'kind' in default_strategy
       • Unknown entity type 'CUSTOM_ID' without pattern definition

Policy Test
-----------

Test policies against sample text to preview masking behavior.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy test POLICY_FILE [OPTIONS]

Arguments
~~~~~~~~~

.. option:: POLICY_FILE

    Path to the policy file to test.

Options
~~~~~~~

.. option:: --text, -t <text>

    Test text to analyze. If not provided, uses built-in sample text.

.. option:: --verbose, -v

    Show detailed analysis results including entity detection and strategy application.

Examples
~~~~~~~~

.. code-block:: bash

    # Test with built-in sample text
    $ cloakpivot policy test healthcare-policy.yaml
    📝 Using sample text for testing:
       Contact John Doe at john.doe@example.com or call (555) 123-4567. His SSN is 123-45-6789.
    🎭 Policy test results:
       Default strategy: hash
       Locale: en
       Per-entity strategies: 5
    ℹ️  Full masking test requires document input
       Use: cloakpivot mask <document> --policy <policy> --verbose
    
    # Test with custom text
    $ cloakpivot policy test financial-policy.yaml \
        --text "Account holder: Jane Smith, Card: 4532-1234-5678-9012" \
        --verbose
    📋 Loading policy: financial-policy.yaml
    📊 Configured entity strategies:
       • PERSON: partial (threshold: 0.7)
       • CREDIT_CARD: template (threshold: 0.9)
       • EMAIL_ADDRESS: hash (threshold: 0.75)
    🔍 Detecting PII entities...
    🎭 Policy test results:
       Default strategy: redact
       Locale: en
       Per-entity strategies: 8

Policy Create
-------------

Interactive policy creation wizard with guided prompts and examples.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy create [OPTIONS]

Options
~~~~~~~

.. option:: --output, -o <path>

    Output file path. If not specified, uses ``interactive_policy.yaml``.

.. option:: --template <name>

    Start with a built-in template as the base.
    
    **Choices:** ``conservative``, ``balanced``, ``permissive``

.. option:: --verbose, -v

    Show detailed information during the creation process.

Interactive Flow
~~~~~~~~~~~~~~~~

The policy creation wizard guides you through:

1. **Basic Configuration**: Policy name, description, locale
2. **Template Selection**: Choose starting point or build from scratch
3. **Default Strategy**: Configure fallback masking approach
4. **Entity Configuration**: Set up specific strategies per PII type
5. **Allow/Deny Lists**: Define explicit inclusion/exclusion rules
6. **Validation**: Automatic validation of the created policy

Examples
~~~~~~~~

.. code-block:: bash

    # Interactive policy creation
    $ cloakpivot policy create --output my-custom-policy.yaml
    📋 CloakPivot Interactive Policy Builder
    ==================================================
    This wizard will guide you through creating a custom masking policy.
    Press Ctrl+C at any time to cancel.
    
    🔧 Basic Configuration
    Policy name [my-custom-policy]: Healthcare Policy
    Policy description [Custom masking policy]: HIPAA-compliant policy for patient records
    Locale (language code) [en]: en
    
    📄 Using template: balanced
    Would you like to start with a template? [Y/n]: y
    Choose template [balanced]: balanced
    
    🎭 Default Masking Strategy
    This strategy will be applied to entities without specific configuration.
    Default strategy [redact]: hash
    Hash algorithm [sha256]: sha256
    Truncate hash to length [8]: 8
    
    👤 Entity-Specific Configurations
    Configure specific entity types? [Y/n]: y
    Configure PERSON? [y/N]: y
    
      Configuring PERSON:
      Strategy for PERSON [hash]: template
      Template text [[PERSON]]: [PATIENT]
      Confidence threshold for PERSON (0.0-1.0) [0.8]: 0.85
    
    # ... additional prompts ...
    
    ✅ Policy creation completed successfully!
       Policy file: my-custom-policy.yaml
       Name: Healthcare Policy
       Entities configured: 4
       Allow list items: 2
       Deny list items: 1
    
    🔍 Validating policy...
    ✅ Policy validation successful!

Policy Info
-----------

Display detailed information about a policy file including all configured strategies and rules.

Synopsis
~~~~~~~~

.. code-block:: bash

    cloakpivot policy info POLICY_FILE

Arguments
~~~~~~~~~

.. option:: POLICY_FILE

    Path to the policy file to analyze.

Examples
~~~~~~~~

.. code-block:: bash

    $ cloakpivot policy info healthcare-policy.yaml
    📋 Policy Information: healthcare-policy.yaml
    ==================================================
    Locale: en
    Seed: healthcare-seed-v1
    Min entity length: 2
    
    Default Strategy: hash
      algorithm: sha256
      truncate: 8
    
    Per-Entity Strategies (5):
      • PERSON: template (threshold: 0.85)
        template: [PATIENT]
      • EMAIL_ADDRESS: partial (threshold: 0.75)
        visible_chars: 2
        position: start
      • PHONE_NUMBER: partial (threshold: 0.8)
        visible_chars: 4
        position: end
      • US_SSN: template (threshold: 0.95)
        template: XXX-XX-XXXX
      • MEDICAL_LICENSE: hash (threshold: 0.9)
        algorithm: sha256
        prefix: LIC_
    
    Context Rules (4):
      • heading: disabled
      • table: enabled
      • footer: enabled  
      • header: disabled
    
    Allow List (2 items):
      • Emergency Contact
      • Patient Services
    
    Deny List (2 items):
      • confidential
      • restricted access

Policy File Format
------------------

YAML Structure
~~~~~~~~~~~~~~

CloakPivot policies use YAML format with the following structure:

.. code-block:: yaml

    version: "1.0"
    name: "policy-name"
    description: "Policy description"
    
    # Core settings
    locale: "en"
    seed: "deterministic-seed"
    min_entity_length: 2
    
    # Default strategy for unspecified entities
    default_strategy:
      kind: "redact"
      parameters:
        redact_char: "*"
        preserve_length: true
    
    # Entity-specific configurations
    per_entity:
      PERSON:
        kind: "hash"
        parameters:
          algorithm: "sha256"
          truncate: 8
        threshold: 0.8
        enabled: true
    
    # Global thresholds
    thresholds:
      EMAIL_ADDRESS: 0.7
      CREDIT_CARD: 0.9
    
    # Context-specific rules
    context_rules:
      heading:
        enabled: false
      table:
        enabled: true
        threshold_overrides:
          PERSON: 0.9
    
    # Allow/deny lists
    allow_list:
      - "public information"
      - pattern: ".*@company\\.com$"
    
    deny_list:
      - "confidential"
      - "classified"

Available Strategies
~~~~~~~~~~~~~~~~~~~

**Redact Strategy**

.. code-block:: yaml

    kind: "redact"
    parameters:
      redact_char: "*"        # Character to use for redaction
      preserve_length: true   # Maintain original text length

**Template Strategy**

.. code-block:: yaml

    kind: "template"
    parameters:
      template: "[REDACTED]"   # Replacement text
      preserve_format: false   # Keep original formatting

**Partial Strategy**

.. code-block:: yaml

    kind: "partial"
    parameters:
      visible_chars: 3       # Number of characters to show
      position: "start"      # "start", "end", or "middle"
      format_aware: true     # Preserve formatting structure

**Hash Strategy**

.. code-block:: yaml

    kind: "hash"
    parameters:
      algorithm: "sha256"    # Hash algorithm
      truncate: 8           # Length of hash to keep
      prefix: "HASH_"       # Optional prefix
      per_entity_salt:      # Entity-specific salts
        PERSON: "person_salt"

**Surrogate Strategy**

.. code-block:: yaml

    kind: "surrogate"
    parameters:
      deterministic: true    # Consistent replacements
      format_preserving: true # Maintain format patterns
      locale_aware: true     # Use locale-appropriate surrogates

Best Practices
--------------

Policy Development Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Start with templates**: Use built-in templates as starting points
2. **Iterative testing**: Test policies with sample data before production use
3. **Validation first**: Always validate policies before deployment
4. **Version control**: Track policy changes and maintain version history
5. **Environment-specific**: Use different policies for different environments

Security Considerations
~~~~~~~~~~~~~~~~~~~~~~

1. **Threshold tuning**: Balance false positives vs. privacy protection
2. **Seed management**: Use consistent seeds for deterministic operations
3. **Salt rotation**: Regularly update salts for hash strategies
4. **Access control**: Restrict access to policy files containing sensitive configurations
5. **Audit logging**: Track policy usage and modifications

Performance Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

1. **Appropriate thresholds**: Higher thresholds reduce processing overhead
2. **Strategy selection**: Hash strategies are generally faster than partial masking
3. **Context rules**: Use context rules to skip unnecessary areas
4. **Entity filtering**: Disable unused entity types to improve performance

See Also
--------

* :doc:`mask_command` - Apply policies to documents
* :doc:`../policies/creating_policies` - Detailed policy development guide
* :doc:`../policies/entity_strategies` - Strategy configuration reference
* :doc:`../examples/policy_examples` - Real-world policy examples