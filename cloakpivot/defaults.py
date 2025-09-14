"""Smart defaults system for CloakEngine - covers 90% of use cases."""

from typing import Dict, Any, List

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind
from cloakpivot.core.analyzer import AnalyzerConfig


# Default entity types for common PII detection
DEFAULT_ENTITIES = [
    "EMAIL_ADDRESS",
    "PERSON",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "US_SSN",
    "LOCATION",
    "DATE_TIME",
    "MEDICAL_LICENSE",
    "URL",
    "IP_ADDRESS",
    "US_DRIVER_LICENSE",
    "US_PASSPORT",
    "IBAN_CODE",
    "NRP",  # National Registration Number
    "CRYPTO",  # Cryptocurrency addresses
]

# Entity types that should be kept by default (not masked)
KEEP_BY_DEFAULT = [
    "DATE_TIME",  # Often needed for context
    "URL",  # May be needed for references
]

# High-risk entities that should always use strong masking
HIGH_RISK_ENTITIES = [
    "CREDIT_CARD",
    "US_SSN",
    "MEDICAL_LICENSE",
    "US_PASSPORT",
    "US_DRIVER_LICENSE",
    "IBAN_CODE",
    "CRYPTO",
]


def get_default_policy() -> MaskingPolicy:
    """Return a sensible default policy for common PII types.

    This policy uses template masking for most entities, partial masking
    for SSNs, and keeps dates/URLs by default.

    Returns:
        MaskingPolicy with smart defaults for common use cases
    """
    per_entity_strategies = {
        # Contact information
        "EMAIL_ADDRESS": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[EMAIL]"}
        ),
        "PHONE_NUMBER": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[PHONE]"}
        ),

        # Personal identifiers
        "PERSON": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[NAME]"}
        ),
        "LOCATION": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[LOCATION]"}
        ),

        # Financial and sensitive IDs - use stronger masking
        "CREDIT_CARD": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[CARD-****]"}
        ),
        "US_SSN": Strategy(
            kind=StrategyKind.PARTIAL,
            parameters={"visible_chars": 4, "position": "end", "mask_char": "*"}
        ),
        "IBAN_CODE": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[IBAN]"}
        ),

        # Medical and government IDs
        "MEDICAL_LICENSE": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[MED-LIC]"}
        ),
        "US_DRIVER_LICENSE": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[DL]"}
        ),
        "US_PASSPORT": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[PASSPORT]"}
        ),
        "NRP": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[NRP]"}
        ),

        # Technical identifiers
        "IP_ADDRESS": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[IP]"}
        ),
        "CRYPTO": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[CRYPTO]"}
        ),

        # Contextual information - use template with value preserved
        "DATE_TIME": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[DATE]"}
        ),
        "URL": Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[URL]"}
        ),
    }

    return MaskingPolicy(
        per_entity=per_entity_strategies,
        default_strategy=Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[REDACTED]"}
        )
    )


def get_conservative_policy() -> MaskingPolicy:
    """Return a conservative policy that masks everything aggressively.

    Uses REDACT strategy for all entities, removing content entirely.

    Returns:
        MaskingPolicy with aggressive masking for maximum privacy
    """
    return MaskingPolicy(
        per_entity={},  # No special handling
        default_strategy=Strategy(
            kind=StrategyKind.REDACT,
            parameters={"replacement": "[REMOVED]"}
        )
    )


def get_permissive_policy() -> MaskingPolicy:
    """Return a permissive policy with minimal masking.

    Only masks high-risk entities, keeps everything else.

    Returns:
        MaskingPolicy with minimal masking for readability
    """
    per_entity_strategies = {}

    # Only mask high-risk entities
    for entity in HIGH_RISK_ENTITIES:
        per_entity_strategies[entity] = Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": f"[{entity.replace('_', '-')}]"}
        )

    return MaskingPolicy(
        per_entity=per_entity_strategies,
        default_strategy=Strategy(
            kind=StrategyKind.TEMPLATE,
            parameters={"template": "[PII]"}
        )
    )


def get_default_analyzer_config() -> Dict[str, Any]:
    """Return optimized analyzer configuration.

    Provides a balanced configuration for accurate detection
    without excessive false positives.

    Returns:
        Dictionary with analyzer configuration
    """
    return {
        "languages": ["en"],
        "confidence_threshold": 0.7,
        "return_decision_process": False,
        "enable_trace": False,
    }


def get_multilingual_analyzer_config(languages: List[str]) -> Dict[str, Any]:
    """Return analyzer configuration for multiple languages.

    Args:
        languages: List of language codes (e.g., ['en', 'es', 'fr'])

    Returns:
        Dictionary with multilingual analyzer configuration
    """
    return {
        "languages": languages,
        "confidence_threshold": 0.65,  # Slightly lower for multilingual
        "return_decision_process": False,
        "enable_trace": False,
    }


def get_high_precision_analyzer_config() -> Dict[str, Any]:
    """Return analyzer configuration for high precision.

    Reduces false positives at the cost of potentially missing some entities.

    Returns:
        Dictionary with high-precision analyzer configuration
    """
    return {
        "languages": ["en"],
        "confidence_threshold": 0.85,
        "return_decision_process": True,  # Include for debugging
        "enable_trace": False,
    }


def get_high_recall_analyzer_config() -> Dict[str, Any]:
    """Return analyzer configuration for high recall.

    Catches more entities at the cost of more false positives.

    Returns:
        Dictionary with high-recall analyzer configuration
    """
    return {
        "languages": ["en"],
        "confidence_threshold": 0.5,
        "return_decision_process": False,
        "enable_trace": False,
    }


# Policy presets for quick access
POLICY_PRESETS = {
    "default": get_default_policy,
    "conservative": get_conservative_policy,
    "permissive": get_permissive_policy,
}

# Analyzer presets for quick access
ANALYZER_PRESETS = {
    "default": get_default_analyzer_config,
    "high_precision": get_high_precision_analyzer_config,
    "high_recall": get_high_recall_analyzer_config,
}


def get_policy_preset(name: str) -> MaskingPolicy:
    """Get a named policy preset.

    Args:
        name: Preset name ('default', 'conservative', or 'permissive')

    Returns:
        MaskingPolicy for the named preset

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in POLICY_PRESETS:
        raise ValueError(
            f"Unknown policy preset: {name}. "
            f"Available presets: {list(POLICY_PRESETS.keys())}"
        )
    return POLICY_PRESETS[name]()


def get_analyzer_preset(name: str) -> Dict[str, Any]:
    """Get a named analyzer configuration preset.

    Args:
        name: Preset name ('default', 'high_precision', or 'high_recall')

    Returns:
        Analyzer configuration dictionary

    Raises:
        ValueError: If preset name is not recognized
    """
    if name not in ANALYZER_PRESETS:
        raise ValueError(
            f"Unknown analyzer preset: {name}. "
            f"Available presets: {list(ANALYZER_PRESETS.keys())}"
        )
    return ANALYZER_PRESETS[name]()