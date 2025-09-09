"""Advanced Presidio integration features for CloakPivot.

This module provides advanced Presidio capabilities including:
- Encryption/decryption workflows with key management
- Operator chaining for complex anonymization sequences
- Ad-hoc recognizer creation without custom classes
- Optimized batch processing with connection pooling
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

# Import all modules at the top level
from .advanced_features import (
    PresidioAdHocRecognizers,
    PresidioEncryptionManager,
    PresidioOperatorChain,
)
from .batch_processor import PresidioBatchProcessor
from .key_management import EnvironmentKeyProvider

if TYPE_CHECKING:
    pass  # No need for TYPE_CHECKING imports since we import them directly


class _SingletonManager:
    """Manages singleton instances without using global statements."""

    def __init__(self) -> None:
        self._encryption_manager: PresidioEncryptionManager | None = None
        self._operator_chain: PresidioOperatorChain | None = None
        self._adhoc_recognizers: PresidioAdHocRecognizers | None = None
        self._batch_processor: PresidioBatchProcessor | None = None

    def get_encryption_manager(self) -> PresidioEncryptionManager:
        """Get or create the singleton encryption manager."""
        if self._encryption_manager is None:
            self._encryption_manager = PresidioEncryptionManager(EnvironmentKeyProvider())
        return self._encryption_manager

    def get_operator_chain(self) -> PresidioOperatorChain:
        """Get or create the singleton operator chain manager."""
        if self._operator_chain is None:
            self._operator_chain = PresidioOperatorChain()
        return self._operator_chain

    def get_adhoc_recognizers(self) -> PresidioAdHocRecognizers:
        """Get or create the singleton ad-hoc recognizer manager."""
        if self._adhoc_recognizers is None:
            self._adhoc_recognizers = PresidioAdHocRecognizers()
        return self._adhoc_recognizers

    def get_batch_processor(self, batch_size: int = 100, parallel_workers: int = 4) -> PresidioBatchProcessor:
        """Get or create a batch processor with specified settings."""
        if (self._batch_processor is None or
            self._batch_processor.batch_size != batch_size or
            self._batch_processor.parallel_workers != parallel_workers):
            self._batch_processor = PresidioBatchProcessor(batch_size, parallel_workers)
        return self._batch_processor


# Create singleton manager instance
_manager = _SingletonManager()


# Public API functions that delegate to the manager
def get_encryption_manager() -> PresidioEncryptionManager:
    """Get or create the singleton encryption manager."""
    return _manager.get_encryption_manager()


def get_operator_chain() -> PresidioOperatorChain:
    """Get or create the singleton operator chain manager."""
    return _manager.get_operator_chain()


def get_adhoc_recognizers() -> PresidioAdHocRecognizers:
    """Get or create the singleton ad-hoc recognizer manager."""
    return _manager.get_adhoc_recognizers()


def get_batch_processor(batch_size: int = 100, parallel_workers: int = 4) -> PresidioBatchProcessor:
    """Get or create a batch processor with specified settings."""
    return _manager.get_batch_processor(batch_size, parallel_workers)


__all__ = [
    "get_encryption_manager",
    "get_operator_chain",
    "get_adhoc_recognizers",
    "get_batch_processor",
    # Also export the classes for direct use if needed
    "PresidioEncryptionManager",
    "PresidioOperatorChain",
    "PresidioAdHocRecognizers",
    "PresidioBatchProcessor",
    "EnvironmentKeyProvider",
]
