"""Deprecation warning system for legacy feature migration."""

import warnings
from functools import wraps
from typing import Any, Callable, Optional


class LegacyDeprecationWarning(UserWarning):
    """Custom warning for legacy feature deprecation."""
    pass


def deprecated_engine(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorator to mark legacy engine usage as deprecated.

    Args:
        func: Function to decorate with deprecation warning

    Returns:
        Wrapped function that emits deprecation warning
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        warnings.warn(
            f"{func.__name__} is using legacy masking engine. "
            f"Consider migrating to Presidio engine for better "
            f"performance and features. Set use_presidio_engine=True "
            f"or CLOAKPIVOT_USE_PRESIDIO_ENGINE=true",
            LegacyDeprecationWarning,
            stacklevel=2
        )
        return func(*args, **kwargs)

    return wrapper


class DeprecationManager:
    """Manages deprecation timeline and warnings."""

    DEPRECATION_SCHEDULE = {
        "2024-Q1": "Legacy engine marked deprecated",
        "2024-Q2": "Presidio engine becomes default",
        "2024-Q3": "Legacy engine requires explicit opt-in",
        "2024-Q4": "Legacy engine removed (major version bump)"
    }
    @classmethod
    def warn_legacy_usage(
        cls,
        component: str,
        alternative: Optional[str] = None,
        stacklevel: int = 3
    ) -> None:
        """Issue standardized deprecation warning.

        Args:
            component: Component being deprecated (e.g., "masking engine")
            alternative: Alternative to use instead
            stacklevel: Stack level for warning attribution
        """
        message = (f"Legacy {component} is deprecated and will be "
                   f"removed in v2.0")
        if alternative:
            message += f". Use {alternative} instead"

        message += (". See migration guide: "
                    "https://cloakpivot.readthedocs.io/migration/")

        warnings.warn(message, LegacyDeprecationWarning,
                      stacklevel=stacklevel)
    
    @classmethod
    def warn_feature_deprecation(
        cls,
        feature: str,
        removal_version: str,
        alternative: Optional[str] = None,
        stacklevel: int = 3
    ) -> None:
        """Warn about a specific feature deprecation.

        Args:
            feature: Feature being deprecated
            removal_version: Version when feature will be removed
            alternative: Alternative feature or approach
            stacklevel: Stack level for warning attribution
        """
        message = (f"{feature} is deprecated and will be removed in "
                   f"version {removal_version}")
        if alternative:
            message += f". Use {alternative} instead"

        warnings.warn(message, LegacyDeprecationWarning,
                      stacklevel=stacklevel)
    
    @classmethod
    def get_deprecation_timeline(cls) -> dict[str, str]:
        """Get the deprecation timeline.

        Returns:
            Dictionary of timeline milestones
        """
        return cls.DEPRECATION_SCHEDULE.copy()
    @classmethod
    def check_deprecation_status(cls, component: str) -> Optional[str]:
        """Check the deprecation status of a component.

        Args:
            component: Component to check

        Returns:
            Deprecation status message or None if not deprecated
        """
        deprecated_components = {
            "legacy_engine": "Use Presidio engine (use_presidio_engine=True)",
            "strategy_applicator": "Use PresidioMaskingAdapter",
            "v1_cloakmap": "Migrate to v2.0 format with Presidio metadata",
            "manual_entity_detection": "Use Presidio AnalyzerEngine",
        }

        return deprecated_components.get(component)
    @classmethod
    def suppress_warnings(cls) -> None:
        """Suppress all legacy deprecation warnings.

        Useful for testing or when migration is not yet possible.
        """
        warnings.filterwarnings("ignore", category=LegacyDeprecationWarning)
    @classmethod
    def reset_warnings(cls) -> None:
        """Reset warning filters to default state."""
        warnings.filterwarnings("default",
                                category=LegacyDeprecationWarning)


def deprecated_parameter(
    param_name: str,
    alternative: Optional[str] = None,
    removal_version: str = "2.0"
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator to mark function parameters as deprecated.

    Args:
        param_name: Name of deprecated parameter
        alternative: Alternative parameter or approach
        removal_version: Version when parameter will be removed

    Returns:
        Decorator function
    """
    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if param_name in kwargs and kwargs[param_name] is not None:
                message = (f"Parameter '{param_name}' is deprecated and "
                           f"will be removed in v{removal_version}")
                if alternative:
                    message += f". Use {alternative} instead"
                warnings.warn(message, LegacyDeprecationWarning,
                              stacklevel=2)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def deprecated_class(
    alternative: Optional[str] = None,
    removal_version: str = "2.0"
) -> Callable[[type], type]:
    """Class decorator to mark entire classes as deprecated.

    Args:
        alternative: Alternative class to use
        removal_version: Version when class will be removed

    Returns:
        Class decorator
    """
    def decorator(cls: type) -> type:
        # Store original __init__ method
        original_init = getattr(cls, '__init__', None)

        def new_init(self: Any, *args: Any, **kwargs: Any) -> None:
            message = (f"{cls.__name__} is deprecated and will be "
                       f"removed in v{removal_version}")
            if alternative:
                message += f". Use {alternative} instead"
            warnings.warn(message, LegacyDeprecationWarning, stacklevel=2)
            # Call original __init__ if it exists
            if original_init is not None:
                original_init(self, *args, **kwargs)

        # Replace __init__ with wrapped version
        setattr(cls, '__init__', new_init)
        return cls

    return decorator