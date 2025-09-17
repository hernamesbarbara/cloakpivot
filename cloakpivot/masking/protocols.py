"""Protocol definitions and type-safe dataclasses for masking operations."""

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class OperatorResultLike(Protocol):
    """
    Protocol for objects that behave like Presidio OperatorResult.

    This protocol defines the minimal interface required for operator results,
    eliminating the need for dynamic type checking and hasattr calls.
    """

    entity_type: str
    start: int
    end: int
    operator: str
    text: str


@dataclass
class SyntheticOperatorResult:
    """
    Type-safe dataclass for synthetic operator results.

    This replaces the dynamic class creation in _create_synthetic_result,
    providing better type checking and clearer code.
    """

    entity_type: str
    start: int
    end: int
    operator: str
    text: str
    operator_metadata: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate the result after initialization."""
        if self.start < 0:
            raise ValueError(f"start must be non-negative, got {self.start}")
        if self.end < self.start:
            raise ValueError(f"end ({self.end}) must be >= start ({self.start})")
        if not self.entity_type:
            raise ValueError("entity_type cannot be empty")
        if not self.operator:
            raise ValueError("operator cannot be empty")


@dataclass
class SegmentBoundary:
    """
    Represents a text segment's position in the full document.

    This makes segment boundary tracking more explicit and type-safe.
    """

    segment_index: int
    start: int
    end: int
    node_id: str

    def contains_position(self, pos: int) -> bool:
        """Check if a position falls within this segment."""
        return self.start <= pos < self.end


@dataclass
class ReplacementSpan:
    """
    Represents a text replacement operation.

    Used for efficient O(n) text replacement operations.
    """

    start: int
    end: int
    replacement: str
    entity_type: str

    def __lt__(self, other: object) -> bool:
        """Enable sorting by start position."""
        if not isinstance(other, ReplacementSpan):
            return NotImplemented
        return self.start < other.start
