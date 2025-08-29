#!/usr/bin/env python3
"""Debug script to test policy loading."""

import json
from pathlib import Path

from cloakpivot.core.policies import MaskingPolicy


def main():
    """Test policy loading."""
    # Create the same policy as test
    policy_content = {
        "locale": "en", 
        "privacy_level": "LOW",
        "entities": {
            "PHONE_NUMBER": {"kind": "TEMPLATE", "parameters": {"template": "[PHONE]"}},
            "EMAIL_ADDRESS": {"kind": "TEMPLATE", "parameters": {"template": "[EMAIL]"}},
            "US_SSN": {"kind": "TEMPLATE", "parameters": {"template": "[SSN]"}},
            "PERSON": {"kind": "TEMPLATE", "parameters": {"template": "[PERSON]"}}
        },
        "thresholds": {
            "PHONE_NUMBER": 0.5,
            "EMAIL_ADDRESS": 0.5,
            "US_SSN": 0.5,
            "PERSON": 0.5
        }
    }
    
    print("=== POLICY CONTENT ===")
    print(json.dumps(policy_content, indent=2))
    
    # Load policy
    print("\n=== LOADING POLICY ===")
    try:
        masking_policy = MaskingPolicy.from_dict(policy_content)
        print(f"✅ Policy loaded successfully")
        print(f"   Locale: {masking_policy.locale}")
        print(f"   Per-entity count: {len(masking_policy.per_entity)}")
        
        print("\n=== PER-ENTITY STRATEGIES ===")
        for entity_type, strategy in masking_policy.per_entity.items():
            print(f"  {entity_type}: {strategy.kind.value} - {strategy.parameters}")
            
        print("\n=== DEFAULT STRATEGY ===")
        print(f"  Kind: {masking_policy.default_strategy.kind.value}")
        print(f"  Parameters: {masking_policy.default_strategy.parameters}")
        
        # Test strategy resolution
        print("\n=== STRATEGY RESOLUTION TEST ===")
        for entity_type in ["PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS", "US_SSN", "UNKNOWN_TYPE"]:
            strategy = masking_policy.get_strategy_for_entity(entity_type)
            print(f"  {entity_type}: {strategy.kind.value} - {strategy.parameters}")
            
    except Exception as e:
        print(f"❌ Policy loading failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()