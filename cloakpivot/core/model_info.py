"""Model characteristics and validation for spaCy model selection.

This module provides information about spaCy model performance characteristics,
validation functions for model availability, and recommendation functions for
selecting appropriate models based on constraints.
"""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


# Performance characteristics for different model sizes
# Based on typical spaCy model behavior and resource requirements
MODEL_CHARACTERISTICS: dict[str, dict[str, Any]] = {
    "small": {
        "memory_mb": 15,
        "load_time_ms": 800,
        "accuracy_score": 0.85,
        "description": "Fast loading, low memory, good accuracy for most use cases",
    },
    "medium": {
        "memory_mb": 50,
        "load_time_ms": 1500,
        "accuracy_score": 0.88,
        "description": "Balanced performance and accuracy",
    },
    "large": {
        "memory_mb": 150,
        "load_time_ms": 3000,
        "accuracy_score": 0.91,
        "description": "Best accuracy, higher resource requirements",
    },
}


# Supported languages for spaCy models
# Based on spaCy's official model releases
SUPPORTED_LANGUAGES: dict[str, dict[str, str]] = {
    "en": {
        "name": "English",
        "small": "en_core_web_sm",
        "medium": "en_core_web_md",
        "large": "en_core_web_lg",
    },
    "es": {
        "name": "Spanish",
        "small": "es_core_news_sm",
        "medium": "es_core_news_md",
        "large": "es_core_news_lg",
    },
    "fr": {
        "name": "French",
        "small": "fr_core_news_sm",
        "medium": "fr_core_news_md",
        "large": "fr_core_news_lg",
    },
    "de": {
        "name": "German",
        "small": "de_core_news_sm",
        "medium": "de_core_news_md",
        "large": "de_core_news_lg",
    },
    "it": {
        "name": "Italian",
        "small": "it_core_news_sm",
        "medium": "it_core_news_md",
        "large": "it_core_news_lg",
    },
    "nl": {
        "name": "Dutch",
        "small": "nl_core_news_sm",
        "medium": "nl_core_news_md",
        "large": "nl_core_news_lg",
    },
    "pt": {
        "name": "Portuguese",
        "small": "pt_core_news_sm",
        "medium": "pt_core_news_md",
        "large": "pt_core_news_lg",
    },
}


def validate_model_availability(language: str, size: str) -> bool:
    """Check if requested model size is available for language.

    Args:
        language: ISO 639-1 language code (e.g., 'en', 'es')
        size: Model size identifier ('small', 'medium', 'large')

    Returns:
        True if the model combination is supported, False otherwise

    Examples:
        >>> validate_model_availability("en", "small")
        True
        >>> validate_model_availability("en", "huge")
        False
        >>> validate_model_availability("zz", "small")
        False
    """
    if not language or not isinstance(language, str):
        logger.warning(f"Invalid language parameter: {language}")
        return False

    if not size or not isinstance(size, str):
        logger.warning(f"Invalid size parameter: {size}")
        return False

    # Check if language is supported
    if language not in SUPPORTED_LANGUAGES:
        logger.debug(f"Language '{language}' not in supported languages")
        return False

    # Check if size is valid
    if size not in MODEL_CHARACTERISTICS:
        logger.debug(f"Size '{size}' not in supported sizes")
        return False

    # Check if specific combination exists
    lang_info = SUPPORTED_LANGUAGES[language]
    if size not in lang_info:
        logger.debug(f"Size '{size}' not available for language '{language}'")
        return False

    logger.debug(f"Model {language}-{size} is available: {lang_info[size]}")
    return True


def get_model_name(language: str, size: str) -> str:
    """Get the full spaCy model name for language and size.

    Args:
        language: ISO 639-1 language code
        size: Model size ('small', 'medium', 'large')

    Returns:
        Full spaCy model name, or fallback pattern for unknown combinations

    Examples:
        >>> get_model_name("en", "small")
        'en_core_web_sm'
        >>> get_model_name("es", "medium")
        'es_core_news_md'
        >>> get_model_name("unknown", "small")
        'unknown_core_web_sm'
    """
    if not validate_model_availability(language, size):
        # Create fallback name using standard pattern
        suffix_map = {"small": "sm", "medium": "md", "large": "lg"}
        suffix = suffix_map.get(size, "sm")
        fallback_name = f"{language}_core_web_{suffix}"
        logger.warning(f"Using fallback model name: {fallback_name}")
        return fallback_name

    return SUPPORTED_LANGUAGES[language][size]


def get_supported_languages() -> list[str]:
    """Get list of supported language codes.

    Returns:
        List of ISO 639-1 language codes with model support
    """
    return list(SUPPORTED_LANGUAGES.keys())


def get_supported_sizes() -> list[str]:
    """Get list of supported model sizes.

    Returns:
        List of supported model size identifiers
    """
    return list(MODEL_CHARACTERISTICS.keys())


def get_model_recommendations(
    memory_limit_mb: Optional[int] = None, speed_priority: bool = False
) -> dict[str, Any]:
    """Get model size recommendations based on constraints.

    Args:
        memory_limit_mb: Maximum memory usage in megabytes (None for no limit)
        speed_priority: Whether to prioritize speed over accuracy

    Returns:
        Dictionary with recommended model size and reasoning

    Examples:
        >>> get_model_recommendations(memory_limit_mb=20)
        {'recommended_size': 'small', 'reason': 'Memory constraint', ...}
        >>> get_model_recommendations(speed_priority=True)
        {'recommended_size': 'small', 'reason': 'Speed priority', ...}
    """
    recommendations = {
        "recommended_size": "small",  # Safe default
        "reason": "Default recommendation",
        "alternatives": [],
        "characteristics": {},
        "warnings": [],
    }

    try:
        # Speed priority recommendation
        if speed_priority:
            recommendations.update(
                {
                    "recommended_size": "small",
                    "reason": "Speed priority - fastest loading and processing",
                    "alternatives": ["medium", "large"],
                    "characteristics": MODEL_CHARACTERISTICS["small"],
                }
            )
            return recommendations

        # Memory constraint recommendation
        if memory_limit_mb is not None:
            suitable_sizes = []

            for size, chars in MODEL_CHARACTERISTICS.items():
                if chars["memory_mb"] <= memory_limit_mb:
                    suitable_sizes.append(size)

            if not suitable_sizes:
                recommendations.update(
                    {
                        "recommended_size": "small",
                        "reason": f"Memory limit {memory_limit_mb}MB too restrictive, "
                        f"using smallest available",
                        "warnings": [
                            f"Recommended model requires "
                            f"{MODEL_CHARACTERISTICS['small']['memory_mb']}MB"
                        ],
                    }
                )
            else:
                # Choose largest size that fits
                size_order = ["large", "medium", "small"]
                best_size = next(size for size in size_order if size in suitable_sizes)

                recommendations.update(
                    {
                        "recommended_size": best_size,
                        "reason": f"Best accuracy within {memory_limit_mb}MB memory limit",
                        "alternatives": [s for s in suitable_sizes if s != best_size],
                        "characteristics": MODEL_CHARACTERISTICS[best_size],
                    }
                )

            return recommendations

        # No constraints - recommend balanced option
        recommendations.update(
            {
                "recommended_size": "medium",
                "reason": "Balanced performance and accuracy for general use",
                "alternatives": ["small", "large"],
                "characteristics": MODEL_CHARACTERISTICS["medium"],
            }
        )

    except Exception as e:
        logger.error(f"Error generating model recommendations: {e}")
        if isinstance(recommendations["warnings"], list):
            recommendations["warnings"].append(f"Error in recommendation: {e}")

    return recommendations


def get_language_info(language: str) -> Optional[dict[str, Any]]:
    """Get detailed information about language support.

    Args:
        language: ISO 639-1 language code

    Returns:
        Dictionary with language info, or None if not supported

    Examples:
        >>> get_language_info("en")
        {'name': 'English', 'small': 'en_core_web_sm', ...}
        >>> get_language_info("zz")
        None
    """
    return SUPPORTED_LANGUAGES.get(language)


def get_all_model_info() -> dict[str, Any]:
    """Get comprehensive model information for debugging and documentation.

    Returns:
        Dictionary with complete model characteristics and language support
    """
    return {
        "characteristics": MODEL_CHARACTERISTICS,
        "supported_languages": SUPPORTED_LANGUAGES,
        "supported_sizes": get_supported_sizes(),
        "total_combinations": len(SUPPORTED_LANGUAGES) * len(MODEL_CHARACTERISTICS),
    }
