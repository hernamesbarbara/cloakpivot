Policy API Reference
====================

The policy API provides comprehensive policy management, validation, and loading capabilities.

.. currentmodule:: cloakpivot.core

Policy Loading and Validation
-----------------------------

.. automodule:: cloakpivot.core.policy_loader
   :members:
   :undoc-members:
   :show-inheritance:

Built-in Policies
-----------------

CloakPivot includes several pre-defined policies for common use cases.

.. autodata:: cloakpivot.CONSERVATIVE_POLICY
   :annotation: = Conservative masking policy with high security

.. autodata:: cloakpivot.TEMPLATE_POLICY
   :annotation: = Template-based masking policy

.. autodata:: cloakpivot.PARTIAL_POLICY
   :annotation: = Partial visibility masking policy

Built-in Strategies
------------------

.. autodata:: cloakpivot.DEFAULT_REDACT
   :annotation: = Default redaction strategy

.. autodata:: cloakpivot.PHONE_TEMPLATE
   :annotation: = Phone number template strategy

.. autodata:: cloakpivot.EMAIL_TEMPLATE
   :annotation: = Email address template strategy

.. autodata:: cloakpivot.SSN_PARTIAL
   :annotation: = SSN partial masking strategy

.. autodata:: cloakpivot.HASH_SHA256
   :annotation: = SHA256 hashing strategy

Policy Examples
---------------

Here are some example policy configurations:

Conservative Policy
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from cloakpivot import MaskingPolicy, Strategy, StrategyKind
    
    conservative = MaskingPolicy(
        locale="en",
        default_strategy=Strategy(kind=StrategyKind.REDACT),
        per_entity={
            "PERSON": Strategy(kind=StrategyKind.HASH, parameters={"algorithm": "sha256"}),
            "EMAIL_ADDRESS": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[EMAIL]"}),
            "PHONE_NUMBER": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PHONE]"}),
        },
        thresholds={
            "PERSON": 0.9,
            "EMAIL_ADDRESS": 0.8,
            "PHONE_NUMBER": 0.8,
            "CREDIT_CARD": 0.95,
        }
    )

Balanced Policy
~~~~~~~~~~~~~~~

.. code-block:: python

    balanced = MaskingPolicy(
        locale="en",
        default_strategy=Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 3}),
        per_entity={
            "PERSON": Strategy(kind=StrategyKind.PARTIAL, parameters={"visible_chars": 2}),
            "EMAIL_ADDRESS": Strategy(
                kind=StrategyKind.PARTIAL, 
                parameters={"visible_chars": 3, "position": "start"}
            ),
            "CREDIT_CARD": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[CARD]"}),
        },
        thresholds={
            "PERSON": 0.7,
            "EMAIL_ADDRESS": 0.6,
            "PHONE_NUMBER": 0.7,
            "CREDIT_CARD": 0.9,
        }
    )

YAML Policy Configuration
------------------------

Policies can also be defined in YAML format:

.. code-block:: yaml

    version: "1.0"
    name: "healthcare-policy"
    description: "HIPAA-compliant masking policy for healthcare documents"
    
    locale: "en"
    seed: "healthcare-seed-v1"
    
    default_strategy:
      kind: "hash"
      parameters:
        algorithm: "sha256"
        truncate: 8
    
    per_entity:
      PERSON:
        kind: "template"
        parameters:
          template: "[PATIENT]"
        threshold: 0.85
        enabled: true
    
      EMAIL_ADDRESS:
        kind: "partial"
        parameters:
          visible_chars: 2
          position: "start"
        threshold: 0.75
        enabled: true
    
      PHONE_NUMBER:
        kind: "partial"
        parameters:
          visible_chars: 4
          position: "end"
        threshold: 0.8
        enabled: true
    
      US_SSN:
        kind: "template"
        parameters:
          template: "XXX-XX-XXXX"
        threshold: 0.95
        enabled: true
    
      MEDICAL_LICENSE:
        kind: "hash"
        parameters:
          algorithm: "sha256"
          prefix: "LIC_"
        threshold: 0.9
        enabled: true
    
    context_rules:
      heading:
        enabled: false
      table:
        enabled: true
        threshold_overrides:
          PERSON: 0.9
    
    allow_list:
      - "Emergency Contact"
      - "Patient Services"
    
    deny_list:
      - "confidential"
      - "restricted access"