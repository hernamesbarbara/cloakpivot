"""Thread-safe singleton loaders for expensive resource initialization.

This module provides cached, thread-safe access to expensive CloakPivot resources
like AnalyzerEngineWrapper instances, DocumentProcessor instances, and
EntityDetectionPipeline instances. Builds upon existing lazy initialization
patterns while providing global caching for improved performance.
"""

import hashlib
import logging
from functools import lru_cache
from threading import Lock
from typing import Optional

from .core.analyzer import AnalyzerConfig, AnalyzerEngineWrapper
from .core.detection import EntityDetectionPipeline
from .core.policies import MaskingPolicy
from .document.processor import DocumentProcessor


class LoaderError(Exception):
    """Base exception for loader-related errors."""
    pass


class ConfigurationError(LoaderError):
    """Raised when loader configuration is invalid."""
    pass


class InitializationError(LoaderError):
    """Raised when resource initialization fails."""
    pass

logger = logging.getLogger(__name__)

# Thread-safe initialization locks
_ANALYZER_LOCK = Lock()
_PROCESSOR_LOCK = Lock()
_PIPELINE_LOCK = Lock()


def _validate_language(language: str) -> None:
    """Validate language parameter.

    Args:
        language: Language code to validate

    Raises:
        ConfigurationError: If language is invalid
    """
    if not isinstance(language, str):
        raise ConfigurationError(f"Language must be a string, got {type(language).__name__}")

    if not language.strip():
        raise ConfigurationError("Language cannot be empty")

    if len(language) < 2 or len(language) > 5:
        raise ConfigurationError(f"Language code '{language}' should be 2-5 characters (ISO format)")


def _validate_confidence(confidence: float) -> None:
    """Validate confidence parameter.

    Args:
        confidence: Confidence value to validate

    Raises:
        ConfigurationError: If confidence is invalid
    """
    if not isinstance(confidence, (int, float)):
        raise ConfigurationError(f"Confidence must be a number, got {type(confidence).__name__}")

    if not (0.0 <= confidence <= 1.0):
        raise ConfigurationError(f"Confidence must be between 0.0 and 1.0, got {confidence}")


def _validate_nlp_engine(nlp_engine_name: str) -> None:
    """Validate NLP engine parameter.

    Args:
        nlp_engine_name: NLP engine name to validate

    Raises:
        ConfigurationError: If NLP engine is invalid
    """
    if not isinstance(nlp_engine_name, str):
        raise ConfigurationError(f"NLP engine must be a string, got {type(nlp_engine_name).__name__}")

    valid_engines = {"spacy", "transformers"}
    if nlp_engine_name not in valid_engines:
        raise ConfigurationError(
            f"Invalid NLP engine '{nlp_engine_name}'. Must be one of: {', '.join(valid_engines)}"
        )


def _generate_config_hash(config: AnalyzerConfig) -> str:
    """Generate stable hash for AnalyzerConfig caching.

    Args:
        config: AnalyzerConfig instance to hash

    Returns:
        Hexadecimal hash string for cache key generation

    Raises:
        ConfigurationError: If config is invalid
    """
    if not isinstance(config, AnalyzerConfig):
        raise ConfigurationError(f"Expected AnalyzerConfig, got {type(config).__name__}")

    try:
        # Create a stable string representation of the config
        config_str = (
            f"{config.language}|{config.min_confidence}|"
            f"{sorted(config.enabled_recognizers or [])}|"
            f"{sorted(config.disabled_recognizers)}|"
            f"{config.nlp_engine_name}|"
            f"{sorted(config.custom_recognizers.keys())}"
        )

        return hashlib.sha256(config_str.encode()).hexdigest()[:32]
    except Exception as e:
        raise ConfigurationError(f"Failed to generate config hash: {e}") from e


def _generate_policy_hash(policy: Optional[MaskingPolicy]) -> str:
    """Generate stable hash for MaskingPolicy caching.

    Args:
        policy: MaskingPolicy instance to hash, or None

    Returns:
        Hexadecimal hash string for cache key generation

    Raises:
        ConfigurationError: If policy is invalid
    """
    if policy is None:
        return "none"

    if not isinstance(policy, MaskingPolicy):
        raise ConfigurationError(f"Expected MaskingPolicy or None, got {type(policy).__name__}")

    try:
        # Create a stable string representation focusing on analyzer-relevant fields
        policy_str = (
            f"{policy.locale}|"
            f"{sorted(policy.thresholds.items()) if policy.thresholds else []}|"
            f"{policy.min_entity_length}"
        )

        return hashlib.sha256(policy_str.encode()).hexdigest()[:32]
    except Exception as e:
        raise ConfigurationError(f"Failed to generate policy hash: {e}") from e


@lru_cache(maxsize=8)
def get_presidio_analyzer(
    language: str = "en",
    config_hash: Optional[str] = None,
    min_confidence: float = 0.5,
    nlp_engine_name: str = "spacy"
) -> AnalyzerEngineWrapper:
    """Get cached Presidio analyzer instance with thread safety.

    This function provides a singleton pattern for AnalyzerEngineWrapper instances,
    using LRU caching to avoid repeated expensive initialization. Thread safety
    is ensured through a module-level lock during initialization.

    Args:
        language: Language code for analysis (ISO 639-1 format)
        config_hash: Optional hash of AnalyzerConfig for cache key
        min_confidence: Minimum confidence threshold for entity detection
        nlp_engine_name: NLP engine to use ('spacy' or 'transformers')

    Returns:
        AnalyzerEngineWrapper instance configured with specified parameters

    Raises:
        ConfigurationError: If parameters are invalid
        InitializationError: If analyzer initialization fails

    Examples:
        >>> # Get default analyzer
        >>> analyzer = get_presidio_analyzer()
        >>>
        >>> # Get analyzer with custom configuration
        >>> analyzer = get_presidio_analyzer(
        ...     language="es",
        ...     min_confidence=0.7
        ... )
    """
    # Validate parameters before attempting creation
    _validate_language(language)
    _validate_confidence(min_confidence)
    _validate_nlp_engine(nlp_engine_name)

    try:
        with _ANALYZER_LOCK:
            # Create configuration
            config = AnalyzerConfig(
                language=language,
                min_confidence=min_confidence,
                nlp_engine_name=nlp_engine_name
            )

            wrapper = AnalyzerEngineWrapper(config)

            logger.info(
                f"Created AnalyzerEngineWrapper for language='{language}', "
                f"confidence={min_confidence}"
            )

            return wrapper

    except Exception as e:
        logger.error(
            f"Failed to create AnalyzerEngineWrapper: {e} "
            f"(language={language}, confidence={min_confidence}, engine={nlp_engine_name})"
        )
        raise InitializationError(f"Failed to initialize analyzer: {e}") from e


def get_presidio_analyzer_from_config(config: AnalyzerConfig) -> AnalyzerEngineWrapper:
    """Get cached Presidio analyzer from AnalyzerConfig instance.

    This is a convenience wrapper around get_presidio_analyzer that generates
    the appropriate cache key from an AnalyzerConfig instance.

    Args:
        config: AnalyzerConfig instance with desired settings

    Returns:
        AnalyzerEngineWrapper instance configured per the provided config

    Raises:
        ConfigurationError: If config is invalid
        InitializationError: If analyzer initialization fails

    Examples:
        >>> config = AnalyzerConfig(language="fr", min_confidence=0.8)
        >>> analyzer = get_presidio_analyzer_from_config(config)
    """
    if config is None:
        raise ConfigurationError("Config cannot be None")

    if not isinstance(config, AnalyzerConfig):
        raise ConfigurationError(f"Expected AnalyzerConfig, got {type(config).__name__}")

    try:
        config_hash = _generate_config_hash(config)

        return get_presidio_analyzer(
            language=config.language,
            config_hash=config_hash,
            min_confidence=config.min_confidence,
            nlp_engine_name=config.nlp_engine_name
        )
    except Exception as e:
        logger.error(f"Failed to create analyzer from config: {e}")
        if isinstance(e, (ConfigurationError, InitializationError)):
            raise
        raise InitializationError(f"Failed to create analyzer from config: {e}") from e


@lru_cache(maxsize=4)
def get_document_processor(enable_chunked: bool = True) -> DocumentProcessor:
    """Get cached document processor instance.

    This function provides a singleton pattern for DocumentProcessor instances,
    using LRU caching to avoid repeated initialization overhead. Thread safety
    is ensured through a module-level lock.

    Args:
        enable_chunked: Whether to enable chunked document processing

    Returns:
        DocumentProcessor instance configured with specified parameters

    Raises:
        ConfigurationError: If parameters are invalid
        InitializationError: If processor initialization fails

    Examples:
        >>> # Get default processor
        >>> processor = get_document_processor()
        >>>
        >>> # Get processor without chunking
        >>> processor = get_document_processor(enable_chunked=False)
    """
    # Validate parameters
    if not isinstance(enable_chunked, bool):
        raise ConfigurationError(f"enable_chunked must be a boolean, got {type(enable_chunked).__name__}")

    try:
        with _PROCESSOR_LOCK:
            processor = DocumentProcessor(
                enable_chunked_processing=enable_chunked
            )

            logger.info(f"Created DocumentProcessor with chunked={enable_chunked}")

            return processor

    except Exception as e:
        logger.error(f"Failed to create DocumentProcessor: {e} (chunked={enable_chunked})")
        raise InitializationError(f"Failed to initialize document processor: {e}") from e


@lru_cache(maxsize=4)
def get_detection_pipeline(
    analyzer_hash: Optional[str] = None,
    policy_hash: Optional[str] = None
) -> EntityDetectionPipeline:
    """Get cached detection pipeline instance.

    This function provides a singleton pattern for EntityDetectionPipeline instances,
    using LRU caching to avoid repeated analyzer initialization. The pipeline uses
    a cached analyzer instance when possible.

    Args:
        analyzer_hash: Optional hash representing analyzer configuration
        policy_hash: Optional hash representing masking policy configuration

    Returns:
        EntityDetectionPipeline instance with cached analyzer

    Raises:
        ConfigurationError: If parameters are invalid
        InitializationError: If pipeline initialization fails

    Examples:
        >>> # Get default pipeline
        >>> pipeline = get_detection_pipeline()
        >>>
        >>> # Get pipeline with cached analyzer
        >>> pipeline = get_detection_pipeline(analyzer_hash="default")
    """
    # Validate parameters
    if analyzer_hash is not None and not isinstance(analyzer_hash, str):
        raise ConfigurationError(f"analyzer_hash must be a string or None, got {type(analyzer_hash).__name__}")

    if policy_hash is not None and not isinstance(policy_hash, str):
        raise ConfigurationError(f"policy_hash must be a string or None, got {type(policy_hash).__name__}")

    try:
        with _PIPELINE_LOCK:
            # Use default analyzer if no specific configuration provided
            analyzer = get_presidio_analyzer()

            pipeline = EntityDetectionPipeline(analyzer=analyzer)

            logger.info("Created EntityDetectionPipeline with cached analyzer")

            return pipeline

    except Exception as e:
        logger.error(f"Failed to create EntityDetectionPipeline: {e}")
        if isinstance(e, (ConfigurationError, InitializationError)):
            raise
        raise InitializationError(f"Failed to initialize detection pipeline: {e}") from e


def get_detection_pipeline_from_policy(policy: MaskingPolicy) -> EntityDetectionPipeline:
    """Get detection pipeline configured from MaskingPolicy.

    This function creates an EntityDetectionPipeline with an analyzer
    configured according to the policy. Each call returns a new pipeline
    instance, but the analyzer may be cached.

    Args:
        policy: MaskingPolicy to derive configuration from

    Returns:
        EntityDetectionPipeline configured according to the policy

    Raises:
        ConfigurationError: If policy is invalid
        InitializationError: If pipeline initialization fails

    Examples:
        >>> policy = MaskingPolicy(locale="en", thresholds={"EMAIL": 0.8})
        >>> pipeline = get_detection_pipeline_from_policy(policy)
    """
    if policy is None:
        raise ConfigurationError("Policy cannot be None")

    if not isinstance(policy, MaskingPolicy):
        raise ConfigurationError(f"Expected MaskingPolicy, got {type(policy).__name__}")

    try:
        # Create analyzer config from policy
        config = AnalyzerConfig.from_policy(policy)
        analyzer = get_presidio_analyzer_from_config(config)

        # Create a new pipeline instance with the configured analyzer
        # Note: We don't cache the pipeline itself since it should be policy-specific
        pipeline = EntityDetectionPipeline(analyzer=analyzer)

        logger.info(f"Created EntityDetectionPipeline from policy (locale={policy.locale})")

        return pipeline

    except Exception as e:
        logger.error(f"Failed to create pipeline from policy: {e} (locale={getattr(policy, 'locale', 'unknown')})")
        if isinstance(e, (ConfigurationError, InitializationError)):
            raise
        raise InitializationError(f"Failed to create pipeline from policy: {e}") from e


def clear_all_caches() -> None:
    """Clear all LRU caches for testing and memory management.

    This function clears all cached instances, forcing fresh initialization
    on the next access. Useful for testing scenarios and memory cleanup.

    Examples:
        >>> # Clear all cached instances
        >>> clear_all_caches()
        >>> # Next calls will create fresh instances
        >>> analyzer = get_presidio_analyzer()
    """
    get_presidio_analyzer.cache_clear()
    get_document_processor.cache_clear()
    get_detection_pipeline.cache_clear()

    logger.info("Cleared all loader caches")


def get_cache_info() -> dict[str, any]:
    """Get cache statistics for monitoring and debugging.

    Returns:
        Dictionary with cache hit/miss statistics for all cached functions

    Examples:
        >>> stats = get_cache_info()
        >>> print(f"Analyzer cache hits: {stats['analyzer']['hits']}")
    """
    return {
        "analyzer": {
            "hits": get_presidio_analyzer.cache_info().hits,
            "misses": get_presidio_analyzer.cache_info().misses,
            "maxsize": get_presidio_analyzer.cache_info().maxsize,
            "currsize": get_presidio_analyzer.cache_info().currsize,
        },
        "processor": {
            "hits": get_document_processor.cache_info().hits,
            "misses": get_document_processor.cache_info().misses,
            "maxsize": get_document_processor.cache_info().maxsize,
            "currsize": get_document_processor.cache_info().currsize,
        },
        "pipeline": {
            "hits": get_detection_pipeline.cache_info().hits,
            "misses": get_detection_pipeline.cache_info().misses,
            "maxsize": get_detection_pipeline.cache_info().maxsize,
            "currsize": get_detection_pipeline.cache_info().currsize,
        },
    }
