"""Validation utilities for Presidio integration testing."""

import hashlib

from docling_core.types import DoclingDocument

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.strategies import Strategy, StrategyKind


class PresidioIntegrationValidator:
    """Validation utilities for Presidio integration."""

    def validate_round_trip_integrity(
        self,
        original: DoclingDocument,
        restored: DoclingDocument,
        detailed: bool = False
    ) -> tuple[bool, dict[str, any]]:
        """
        Comprehensive round-trip validation.

        Args:
            original: Original document before masking
            restored: Document after mask/unmask round-trip
            detailed: Whether to include detailed comparison results

        Returns:
            Tuple of (is_valid, validation_details)
        """
        validation_results = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }

        # Basic text comparison
        if original._main_text != restored._main_text:
            validation_results["is_valid"] = False
            validation_results["errors"].append("Main text content mismatch")

            if detailed:
                # Find first difference
                for i, (o, r) in enumerate(zip(original._main_text, restored._main_text)):
                    if o != r:
                        validation_results["errors"].append(
                            f"First difference at position {i}: original='{o}', restored='{r}'"
                        )
                        break

                # Check length differences
                len_diff = len(restored._main_text) - len(original._main_text)
                if len_diff != 0:
                    validation_results["errors"].append(
                        f"Length difference: {len_diff} characters"
                    )

        # Validate document structure
        if hasattr(original, 'texts') and hasattr(restored, 'texts'):
            if len(original.texts) != len(restored.texts):
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Text items count mismatch: {len(original.texts)} vs {len(restored.texts)}"
                )
            else:
                for i, (orig_text, rest_text) in enumerate(zip(original.texts, restored.texts)):
                    if orig_text.text != rest_text.text:
                        validation_results["is_valid"] = False
                        validation_results["errors"].append(
                            f"Text item {i} content mismatch"
                        )

        # Validate tables
        if hasattr(original, 'tables') and hasattr(restored, 'tables'):
            if len(original.tables) != len(restored.tables):
                validation_results["is_valid"] = False
                validation_results["errors"].append(
                    f"Table count mismatch: {len(original.tables)} vs {len(restored.tables)}"
                )
            else:
                for i, (orig_table, rest_table) in enumerate(zip(original.tables, restored.tables)):
                    if orig_table.data != rest_table.data:
                        validation_results["is_valid"] = False
                        validation_results["errors"].append(
                            f"Table {i} data mismatch"
                        )

        # Validate key-value items
        if hasattr(original, 'key_value_items') and hasattr(restored, 'key_value_items'):
            if original.key_value_items != restored.key_value_items:
                validation_results["is_valid"] = False
                validation_results["errors"].append("Key-value items mismatch")

        # Calculate metrics
        validation_results["metrics"]["text_length"] = len(original._main_text)
        validation_results["metrics"]["restored_length"] = len(restored._main_text)

        # Hash comparison for exact match
        original_hash = hashlib.sha256(original._main_text.encode()).hexdigest()
        restored_hash = hashlib.sha256(restored._main_text.encode()).hexdigest()
        validation_results["metrics"]["original_hash"] = original_hash
        validation_results["metrics"]["restored_hash"] = restored_hash
        validation_results["metrics"]["hash_match"] = (original_hash == restored_hash)

        return validation_results["is_valid"], validation_results

    def validate_performance_regression(
        self,
        legacy_time: float,
        presidio_time: float,
        max_slowdown_ratio: float = 1.2
    ) -> tuple[bool, dict[str, any]]:
        """
        Ensure Presidio performance is acceptable.

        Args:
            legacy_time: Time taken by legacy engine (seconds)
            presidio_time: Time taken by Presidio engine (seconds)
            max_slowdown_ratio: Maximum acceptable slowdown ratio (default 1.2 = 20% slower)

        Returns:
            Tuple of (is_acceptable, performance_details)
        """
        ratio = presidio_time / legacy_time if legacy_time > 0 else float('inf')
        is_acceptable = ratio <= max_slowdown_ratio

        details = {
            "is_acceptable": is_acceptable,
            "legacy_time": legacy_time,
            "presidio_time": presidio_time,
            "ratio": ratio,
            "max_allowed_ratio": max_slowdown_ratio,
            "percentage_difference": (ratio - 1) * 100,
            "verdict": "PASS" if is_acceptable else "FAIL"
        }

        if not is_acceptable:
            details["recommendation"] = (
                f"Presidio is {ratio:.2f}x slower than legacy. "
                f"Consider optimization or adjusting threshold."
            )

        return is_acceptable, details

    def validate_cloakmap_compatibility(
        self,
        v1_cloakmap: CloakMap,
        v2_cloakmap: CloakMap | None = None
    ) -> tuple[bool, dict[str, any]]:
        """
        Validate CloakMap version compatibility.

        Args:
            v1_cloakmap: Version 1.0 CloakMap
            v2_cloakmap: Optional Version 2.0 CloakMap for comparison

        Returns:
            Tuple of (is_compatible, compatibility_details)
        """
        details = {
            "is_compatible": True,
            "v1_version": getattr(v1_cloakmap, 'version', 'unknown'),
            "issues": [],
            "warnings": []
        }

        # Check v1 CloakMap structure
        if not hasattr(v1_cloakmap, 'transformations'):
            details["is_compatible"] = False
            details["issues"].append("v1 CloakMap missing 'transformations' field")

        # Validate transformation format
        if hasattr(v1_cloakmap, 'transformations'):
            for i, transform in enumerate(v1_cloakmap.transformations):
                required_fields = ['start', 'end', 'entity_type', 'original_text', 'new_text']
                for field in required_fields:
                    if field not in transform:
                        details["is_compatible"] = False
                        details["issues"].append(
                            f"Transformation {i} missing required field '{field}'"
                        )

        # If v2 CloakMap provided, compare
        if v2_cloakmap:
            details["v2_version"] = getattr(v2_cloakmap, 'version', 'unknown')

            # Check if v2 can process v1 transformations
            v1_transform_count = len(v1_cloakmap.transformations) if hasattr(v1_cloakmap, 'transformations') else 0
            v2_transform_count = len(v2_cloakmap.transformations) if hasattr(v2_cloakmap, 'transformations') else 0

            if v1_transform_count != v2_transform_count:
                details["warnings"].append(
                    f"Transformation count differs: v1={v1_transform_count}, v2={v2_transform_count}"
                )

        return details["is_compatible"], details

    def validate_operator_mapping_correctness(
        self,
        strategy: Strategy,
        operator_config: dict[str, any]
    ) -> tuple[bool, dict[str, any]]:
        """
        Validate strategy→operator mapping accuracy.

        Args:
            strategy: The strategy being validated
            operator_config: The operator configuration generated

        Returns:
            Tuple of (is_correct, mapping_details)
        """
        details = {
            "is_correct": True,
            "strategy_kind": strategy.kind,
            "strategy_config": strategy.config,
            "operator_config": operator_config,
            "issues": []
        }

        # Map strategy kind to expected operator type
        expected_mappings = {
            StrategyKind.TEMPLATE: "replace",
            StrategyKind.REDACT: "replace",
            StrategyKind.PARTIAL: "mask",
            StrategyKind.HASH: "hash",
            StrategyKind.SURROGATE: "replace",
            StrategyKind.CUSTOM: "custom",
        }

        expected_operator = expected_mappings.get(strategy.kind)
        actual_operator = operator_config.get("type")

        if expected_operator != actual_operator:
            details["is_correct"] = False
            details["issues"].append(
                f"Expected operator '{expected_operator}' but got '{actual_operator}'"
            )

        # Validate operator parameters based on strategy
        if strategy.kind == StrategyKind.TEMPLATE:
            if "new_value" not in operator_config:
                details["is_correct"] = False
                details["issues"].append("Template strategy missing 'new_value' in operator")

        elif strategy.kind == StrategyKind.PARTIAL:
            if "masking_char" not in operator_config and "chars_to_mask" not in operator_config:
                details["is_correct"] = False
                details["issues"].append("Partial mask strategy missing masking parameters")

        elif strategy.kind == StrategyKind.REDACT:
            if "masking_char" not in operator_config:
                details["is_correct"] = False
                details["issues"].append("Mask strategy missing 'masking_char' parameter")

        return details["is_correct"], details

    def validate_entity_detection_coverage(
        self,
        original_text: str,
        detected_entities: list[dict[str, any]],
        expected_entities: list[dict[str, any]]
    ) -> tuple[bool, dict[str, any]]:
        """
        Validate entity detection coverage and accuracy.

        Args:
            original_text: Original text being analyzed
            detected_entities: Entities detected by the engine
            expected_entities: Expected entities that should be detected

        Returns:
            Tuple of (is_complete, coverage_details)
        """
        details = {
            "is_complete": True,
            "detected_count": len(detected_entities),
            "expected_count": len(expected_entities),
            "coverage_percentage": 0.0,
            "missed_entities": [],
            "false_positives": []
        }

        # Calculate coverage
        if expected_entities:
            matched = 0
            for expected in expected_entities:
                for detected in detected_entities:
                    if (detected.get("start") == expected.get("start") and
                        detected.get("end") == expected.get("end") and
                        detected.get("entity_type") == expected.get("entity_type")):
                        matched += 1
                        break
                else:
                    details["missed_entities"].append(expected)

            details["coverage_percentage"] = (matched / len(expected_entities)) * 100

            if matched < len(expected_entities):
                details["is_complete"] = False

        # Check for false positives
        for detected in detected_entities:
            found = False
            for expected in expected_entities:
                if (detected.get("start") == expected.get("start") and
                    detected.get("end") == expected.get("end")):
                    found = True
                    break

            if not found:
                details["false_positives"].append(detected)

        return details["is_complete"], details

    def validate_masking_consistency(
        self,
        text1: str,
        text2: str,
        policy: any
    ) -> tuple[bool, dict[str, any]]:
        """
        Validate that the same text masked twice produces consistent results.

        Args:
            text1: First masked text
            text2: Second masked text (should be identical)
            policy: The masking policy used

        Returns:
            Tuple of (is_consistent, consistency_details)
        """
        is_consistent = (text1 == text2)

        details = {
            "is_consistent": is_consistent,
            "text1_length": len(text1),
            "text2_length": len(text2),
            "differences": []
        }

        if not is_consistent:
            # Find differences
            for i, (c1, c2) in enumerate(zip(text1, text2)):
                if c1 != c2:
                    details["differences"].append({
                        "position": i,
                        "text1_char": c1,
                        "text2_char": c2
                    })

                    # Limit to first 10 differences
                    if len(details["differences"]) >= 10:
                        details["differences"].append("... and more")
                        break

        return is_consistent, details

    def validate_memory_usage(
        self,
        peak_memory_bytes: int,
        baseline_memory_bytes: int,
        max_increase_ratio: float = 2.0
    ) -> tuple[bool, dict[str, any]]:
        """
        Validate memory usage is within acceptable bounds.

        Args:
            peak_memory_bytes: Peak memory usage during operation
            baseline_memory_bytes: Baseline memory before operation
            max_increase_ratio: Maximum acceptable memory increase ratio

        Returns:
            Tuple of (is_acceptable, memory_details)
        """
        increase = peak_memory_bytes - baseline_memory_bytes
        ratio = peak_memory_bytes / baseline_memory_bytes if baseline_memory_bytes > 0 else float('inf')
        is_acceptable = ratio <= max_increase_ratio

        details = {
            "is_acceptable": is_acceptable,
            "peak_memory_mb": peak_memory_bytes / (1024 * 1024),
            "baseline_memory_mb": baseline_memory_bytes / (1024 * 1024),
            "increase_mb": increase / (1024 * 1024),
            "ratio": ratio,
            "max_allowed_ratio": max_increase_ratio
        }

        if not is_acceptable:
            details["warning"] = f"Memory usage increased by {ratio:.2f}x, exceeding threshold"

        return is_acceptable, details
