"""Anchor system for tracking position mappings between original and masked content."""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class AnchorEntry:
    """
    Represents a mapping between original and masked content positions.
    
    This class tracks the precise location of a detected entity in the document
    structure and its corresponding masked replacement, enabling deterministic
    unmasking while maintaining security through checksums.
    
    Attributes:
        node_id: Unique identifier for the docpivot document node
        start: Starting character position within the node's text content
        end: Ending character position within the node's text content  
        entity_type: Type of PII entity (e.g., 'PHONE_NUMBER', 'EMAIL_ADDRESS')
        confidence: Detection confidence score from Presidio (0.0-1.0)
        masked_value: The replacement text that appears in the masked document
        replacement_id: Unique identifier for this replacement (for reverse lookup)
        original_checksum: SHA-256 checksum of the original text (no plaintext stored)
        strategy_used: The masking strategy that was applied
        timestamp: When this anchor was created
        metadata: Additional context information
        
    Examples:
        >>> # Basic anchor for a phone number
        >>> anchor = AnchorEntry(
        ...     node_id="paragraph_0_text_1",
        ...     start=15,
        ...     end=27,
        ...     entity_type="PHONE_NUMBER",
        ...     confidence=0.95,
        ...     masked_value="[PHONE]",
        ...     replacement_id="repl_123456",
        ...     original_checksum="a1b2c3d4...",
        ...     strategy_used="template"
        ... )
        
        >>> # Anchor with partial masking
        >>> anchor = AnchorEntry(
        ...     node_id="table_cell_2_3",
        ...     start=0,
        ...     end=12,
        ...     entity_type="US_SSN",
        ...     confidence=0.88,
        ...     masked_value="***-**-1234",
        ...     replacement_id="repl_789012",
        ...     original_checksum="e5f6g7h8...",
        ...     strategy_used="partial"
        ... )
    """

    node_id: str
    start: int
    end: int
    entity_type: str
    confidence: float
    masked_value: str
    replacement_id: str
    original_checksum: str
    strategy_used: str
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Validate anchor data after initialization."""
        self._validate_positions()
        self._validate_confidence()
        self._validate_checksum()
        self._validate_ids()

        # Set default timestamp if not provided
        if self.timestamp is None:
            object.__setattr__(self, 'timestamp', datetime.utcnow())

        # Initialize empty metadata if not provided
        if self.metadata is None:
            object.__setattr__(self, 'metadata', {})

    def _validate_positions(self) -> None:
        """Validate start and end positions."""
        if not isinstance(self.start, int) or self.start < 0:
            raise ValueError("start position must be a non-negative integer")

        if not isinstance(self.end, int) or self.end < 0:
            raise ValueError("end position must be a non-negative integer")

        if self.end <= self.start:
            raise ValueError("end position must be greater than start position")

    def _validate_confidence(self) -> None:
        """Validate confidence score."""
        if not isinstance(self.confidence, (int, float)):
            raise ValueError("confidence must be a number")

        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")

    def _validate_checksum(self) -> None:
        """Validate original checksum format."""
        if not isinstance(self.original_checksum, str):
            raise ValueError("original_checksum must be a string")

        # Basic SHA-256 hex string validation (64 characters)
        if len(self.original_checksum) != 64:
            raise ValueError("original_checksum should be a 64-character SHA-256 hex string")

        try:
            int(self.original_checksum, 16)
        except ValueError:
            raise ValueError("original_checksum must contain only hexadecimal characters")

    def _validate_ids(self) -> None:
        """Validate node_id and replacement_id."""
        if not isinstance(self.node_id, str) or not self.node_id.strip():
            raise ValueError("node_id must be a non-empty string")

        if not isinstance(self.replacement_id, str) or not self.replacement_id.strip():
            raise ValueError("replacement_id must be a non-empty string")

        if not isinstance(self.entity_type, str) or not self.entity_type.strip():
            raise ValueError("entity_type must be a non-empty string")

        if not isinstance(self.strategy_used, str) or not self.strategy_used.strip():
            raise ValueError("strategy_used must be a non-empty string")

    @property
    def span_length(self) -> int:
        """Get the length of the original text span."""
        return self.end - self.start

    @property
    def replacement_length(self) -> int:
        """Get the length of the masked replacement text."""
        return len(self.masked_value)

    @property
    def length_delta(self) -> int:
        """Get the difference in length between original and replacement."""
        return self.replacement_length - self.span_length

    def verify_original_text(self, original_text: str) -> bool:
        """
        Verify that the provided original text matches the stored checksum.
        
        Args:
            original_text: The original text to verify
            
        Returns:
            True if the text matches the checksum, False otherwise
        """
        computed_checksum = self._compute_checksum(original_text)
        return computed_checksum == self.original_checksum

    def overlaps_with(self, other: "AnchorEntry") -> bool:
        """
        Check if this anchor overlaps with another anchor in the same node.
        
        Args:
            other: Another anchor entry to check against
            
        Returns:
            True if the anchors overlap, False otherwise
        """
        if self.node_id != other.node_id:
            return False

        # Check for any overlap in the position ranges
        return not (self.end <= other.start or other.end <= self.start)

    def contains_position(self, position: int) -> bool:
        """
        Check if a position falls within this anchor's span.
        
        Args:
            position: The position to check
            
        Returns:
            True if the position is within the span, False otherwise
        """
        return self.start <= position < self.end

    def with_metadata(self, **new_metadata: Any) -> "AnchorEntry":
        """
        Create a new anchor with additional metadata.
        
        Args:
            **new_metadata: Key-value pairs to add to metadata
            
        Returns:
            New AnchorEntry with merged metadata
        """
        merged_metadata = {**(self.metadata or {}), **new_metadata}

        return AnchorEntry(
            node_id=self.node_id,
            start=self.start,
            end=self.end,
            entity_type=self.entity_type,
            confidence=self.confidence,
            masked_value=self.masked_value,
            replacement_id=self.replacement_id,
            original_checksum=self.original_checksum,
            strategy_used=self.strategy_used,
            timestamp=self.timestamp,
            metadata=merged_metadata
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert anchor to dictionary for serialization."""
        return {
            "node_id": self.node_id,
            "start": self.start,
            "end": self.end,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "masked_value": self.masked_value,
            "replacement_id": self.replacement_id,
            "original_checksum": self.original_checksum,
            "strategy_used": self.strategy_used,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnchorEntry":
        """Create anchor from dictionary representation."""
        timestamp = None
        if data.get("timestamp"):
            timestamp = datetime.fromisoformat(data["timestamp"])

        return cls(
            node_id=data["node_id"],
            start=data["start"],
            end=data["end"],
            entity_type=data["entity_type"],
            confidence=data["confidence"],
            masked_value=data["masked_value"],
            replacement_id=data["replacement_id"],
            original_checksum=data["original_checksum"],
            strategy_used=data["strategy_used"],
            timestamp=timestamp,
            metadata=data.get("metadata")
        )

    @staticmethod
    def _compute_checksum(text: str) -> str:
        """Compute SHA-256 checksum of text."""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()

    @staticmethod
    def create_replacement_id(entity_type: str, node_id: str, start: int) -> str:
        """
        Create a deterministic but unique replacement ID.
        
        Args:
            entity_type: The type of entity
            node_id: The node identifier
            start: The start position
            
        Returns:
            A unique replacement ID string
        """
        # Create a deterministic ID based on entity context
        base_string = f"{entity_type}:{node_id}:{start}"
        hash_obj = hashlib.md5(base_string.encode('utf-8'))
        return f"repl_{hash_obj.hexdigest()[:8]}"

    @classmethod
    def create_from_detection(
        cls,
        node_id: str,
        start: int,
        end: int,
        entity_type: str,
        confidence: float,
        original_text: str,
        masked_value: str,
        strategy_used: str,
        replacement_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> "AnchorEntry":
        """
        Create an anchor entry from PII detection results.
        
        Args:
            node_id: Document node identifier
            start: Start position in node
            end: End position in node
            entity_type: Type of detected entity
            confidence: Detection confidence score
            original_text: The original text that was detected
            masked_value: The replacement text
            strategy_used: The masking strategy applied
            replacement_id: Optional replacement ID (auto-generated if not provided)
            metadata: Optional additional metadata
            
        Returns:
            New AnchorEntry instance
        """
        if replacement_id is None:
            replacement_id = cls.create_replacement_id(entity_type, node_id, start)

        original_checksum = cls._compute_checksum(original_text)

        return cls(
            node_id=node_id,
            start=start,
            end=end,
            entity_type=entity_type,
            confidence=confidence,
            masked_value=masked_value,
            replacement_id=replacement_id,
            original_checksum=original_checksum,
            strategy_used=strategy_used,
            metadata=metadata
        )


class AnchorIndex:
    """
    Index for efficient lookup and management of anchor entries.
    
    This class provides fast access patterns for anchors by various keys
    and handles operations like conflict detection and position updates.
    """

    def __init__(self, anchors: Optional[List[AnchorEntry]] = None) -> None:
        """
        Initialize the anchor index.
        
        Args:
            anchors: Optional initial list of anchors to index
        """
        self._anchors: List[AnchorEntry] = []
        self._by_replacement_id: Dict[str, AnchorEntry] = {}
        self._by_node_id: Dict[str, List[AnchorEntry]] = {}
        self._by_entity_type: Dict[str, List[AnchorEntry]] = {}

        if anchors:
            for anchor in anchors:
                self.add_anchor(anchor)

    def add_anchor(self, anchor: AnchorEntry) -> None:
        """Add an anchor to the index."""
        if anchor.replacement_id in self._by_replacement_id:
            raise ValueError(f"Duplicate replacement_id: {anchor.replacement_id}")

        self._anchors.append(anchor)
        self._by_replacement_id[anchor.replacement_id] = anchor

        # Index by node_id
        if anchor.node_id not in self._by_node_id:
            self._by_node_id[anchor.node_id] = []
        self._by_node_id[anchor.node_id].append(anchor)

        # Index by entity_type
        if anchor.entity_type not in self._by_entity_type:
            self._by_entity_type[anchor.entity_type] = []
        self._by_entity_type[anchor.entity_type].append(anchor)

    def get_by_replacement_id(self, replacement_id: str) -> Optional[AnchorEntry]:
        """Get anchor by replacement ID."""
        return self._by_replacement_id.get(replacement_id)

    def get_by_node_id(self, node_id: str) -> List[AnchorEntry]:
        """Get all anchors for a specific node ID, sorted by start position."""
        anchors = self._by_node_id.get(node_id, [])
        return sorted(anchors, key=lambda a: a.start)

    def get_by_entity_type(self, entity_type: str) -> List[AnchorEntry]:
        """Get all anchors for a specific entity type."""
        return self._by_entity_type.get(entity_type, [])

    def find_overlapping_anchors(self, anchor: AnchorEntry) -> List[AnchorEntry]:
        """Find all anchors that overlap with the given anchor."""
        node_anchors = self.get_by_node_id(anchor.node_id)
        return [a for a in node_anchors if a != anchor and a.overlaps_with(anchor)]

    def get_anchors_in_range(self, node_id: str, start: int, end: int) -> List[AnchorEntry]:
        """Get all anchors that intersect with the given position range."""
        node_anchors = self.get_by_node_id(node_id)
        return [
            a for a in node_anchors
            if not (a.end <= start or a.start >= end)
        ]

    def get_all_anchors(self) -> List[AnchorEntry]:
        """Get all anchors in the index."""
        return self._anchors.copy()

    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the indexed anchors."""
        entity_counts: Dict[str, int] = {}
        strategy_counts: Dict[str, int] = {}
        total_confidence = 0.0

        for anchor in self._anchors:
            # Count by entity type
            entity_counts[anchor.entity_type] = entity_counts.get(anchor.entity_type, 0) + 1

            # Count by strategy
            strategy_counts[anchor.strategy_used] = strategy_counts.get(anchor.strategy_used, 0) + 1

            # Accumulate confidence
            total_confidence += anchor.confidence

        avg_confidence = round(total_confidence / len(self._anchors), 10) if self._anchors else 0.0

        return {
            "total_anchors": len(self._anchors),
            "unique_nodes": len(self._by_node_id),
            "entity_type_counts": entity_counts,
            "strategy_counts": strategy_counts,
            "average_confidence": avg_confidence
        }
