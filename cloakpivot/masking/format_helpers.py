"""Format-aware masking helpers for partial strategies."""

import hashlib
import logging
import random
from typing import Any

logger = logging.getLogger(__name__)


class FormatPreserver:
    """Helper class for format-aware masking operations."""

    @staticmethod
    def apply_format_aware_partial_masking(
        original_text: str,
        visible_chars: int,
        position: str,
        mask_char: str,
        preserve_delimiters: bool,
        deterministic: bool,
    ) -> str:
        """Apply format-aware partial masking that preserves delimiters and structure."""
        # Detect common delimiters and structural elements
        delimiters = {"-", "_", ".", "@", " ", "(", ")", "+"}

        # Find delimiter positions
        delimiter_positions = []
        if preserve_delimiters:
            for i, char in enumerate(original_text):
                if char in delimiters:
                    delimiter_positions.append(i)

        # Extract maskable characters (non-delimiters)
        maskable_chars = []
        maskable_positions = []
        for i, char in enumerate(original_text):
            if not preserve_delimiters or char not in delimiters:
                maskable_chars.append(char)
                maskable_positions.append(i)

        if not maskable_chars:
            # All characters are delimiters, return as-is
            return original_text

        # Apply partial masking to maskable characters only
        visible_chars = min(visible_chars, len(maskable_chars))

        # Determine which characters to keep visible
        visible_indices = FormatPreserver.select_visible_characters(
            len(maskable_chars), visible_chars, position, deterministic, original_text
        )

        # Build result by preserving delimiters and masking non-visible characters
        result = list(original_text)
        for i, pos in enumerate(maskable_positions):
            if i not in visible_indices:
                result[pos] = mask_char

        return "".join(result)

    @staticmethod
    def select_visible_characters(
        total_chars: int,
        visible_chars: int,
        position: str,
        deterministic: bool,
        original_text: str,
    ) -> set[int]:
        """Select which character indices should remain visible."""
        if visible_chars >= total_chars:
            return set(range(total_chars))

        if position == "start":
            return set(range(visible_chars))
        if position == "end":
            return set(range(total_chars - visible_chars, total_chars))
        if position == "middle":
            # Show chars at both ends
            chars_per_side = visible_chars // 2
            remaining = visible_chars % 2

            start_chars = chars_per_side + remaining
            end_chars = chars_per_side

            if start_chars + end_chars >= total_chars:
                # Fallback to showing alternating characters
                return FormatPreserver.select_alternating_characters(
                    total_chars, visible_chars, deterministic, original_text
                )

            visible_indices: set[int] = set()
            visible_indices.update(range(start_chars))
            visible_indices.update(range(total_chars - end_chars, total_chars))
            return visible_indices
        if position == "random":
            return FormatPreserver.select_random_characters(
                total_chars, visible_chars, deterministic, original_text
            )

        raise ValueError(f"Invalid position for partial strategy: {position}")

    @staticmethod
    def select_alternating_characters(
        total_chars: int,
        visible_chars: int,
        deterministic: bool,
        original_text: str,
    ) -> set[int]:
        """Select alternating characters for visibility."""
        if not deterministic:
            # Non-deterministic alternating
            step = max(1, total_chars // visible_chars)
            return set(range(0, total_chars, step)[:visible_chars])

        # Deterministic alternating based on text content
        hash_seed = hash(original_text) % total_chars
        step = max(1, total_chars // visible_chars)
        visible_indices = set()

        for i in range(visible_chars):
            idx = (hash_seed + i * step) % total_chars
            visible_indices.add(idx)

        return visible_indices

    @staticmethod
    def select_random_characters(
        total_chars: int,
        visible_chars: int,
        deterministic: bool,
        original_text: str,
    ) -> set[int]:
        """Select random characters for visibility."""
        if deterministic:
            # Use text content as seed for deterministic randomness
            local_random = random.Random(hash(original_text))
            indices = list(range(total_chars))
            local_random.shuffle(indices)
            return set(indices[:visible_chars])

        # Non-deterministic random selection
        rng = random.Random()
        indices = list(range(total_chars))
        rng.shuffle(indices)
        return set(indices[:visible_chars])

    @staticmethod
    def preserve_format_in_hash(original_text: str, hash_result: str, prefix: str) -> str:
        """Preserve format structure in hash output."""
        # Detect structural elements in original text
        delimiters = []
        delimiter_positions = []

        for i, char in enumerate(original_text):
            if char in ["-", "_", ".", "@", " ", "(", ")", "+"]:
                delimiters.append(char)
                delimiter_positions.append(i)

        if not delimiters:
            return hash_result

        # Try to maintain similar structure in hash
        result = list(hash_result)

        # Insert delimiters at proportional positions
        hash_len = len(hash_result)
        orig_len = len(original_text)

        for _i, (delimiter, orig_pos) in enumerate(
            zip(delimiters, delimiter_positions, strict=False)
        ):
            # Calculate proportional position in hash
            if orig_len > 0:
                hash_pos = int((orig_pos / orig_len) * hash_len)
                hash_pos = min(hash_pos, len(result) - 1)

                # Insert delimiter if it makes sense
                if hash_pos < len(result) and result[hash_pos].isalnum():
                    result[hash_pos] = delimiter

        return "".join(result)

    @staticmethod
    def build_deterministic_salt(
        base_salt: str,
        per_entity_salt: dict[str, Any],
        entity_type: str,
        original_text: str,
    ) -> str:
        """Build a deterministic salt combining base, per-entity, and content-based salts."""
        salt_components = [base_salt or ""]

        # Add per-entity-type salt for security isolation
        if per_entity_salt and isinstance(per_entity_salt, dict):
            entity_salt = per_entity_salt.get(entity_type, per_entity_salt.get("default", ""))
            salt_components.append(str(entity_salt))

        # Add content-length-based component for additional entropy
        salt_components.append(f"len:{len(original_text)}")

        # Add entity type for separation
        salt_components.append(f"type:{entity_type}")

        return "|".join(salt_components)

    @staticmethod
    def get_hash_algorithm(algorithm: str) -> Any:
        """Get hash algorithm object."""
        if algorithm == "md5":
            return hashlib.md5()
        if algorithm == "sha1":
            return hashlib.sha1()
        if algorithm == "sha256":
            return hashlib.sha256()
        if algorithm == "sha384":
            return hashlib.sha384()
        if algorithm == "sha512":
            return hashlib.sha512()
        raise ValueError(f"Unsupported hash algorithm: {algorithm}")

    @staticmethod
    def apply_consistent_truncation(
        hash_result: str, truncate: int, original_text: str, algorithm: str
    ) -> str:
        """Apply truncation that's consistent for similar-length inputs."""
        if truncate >= len(hash_result):
            return hash_result

        # Use original text characteristics to determine truncation offset
        # This ensures similar inputs get similar hash patterns
        offset = hash(original_text + algorithm) % max(1, len(hash_result) - truncate)
        return hash_result[offset : offset + truncate]