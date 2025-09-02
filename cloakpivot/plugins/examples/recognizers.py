"""Example recognizer plugins for demonstration and testing."""

import re
from typing import Any, Optional

from ..base import PluginInfo
from ..recognizers.base import (
    PatternBasedRecognizerPlugin,
)


class CustomPhoneRecognizerPlugin(PatternBasedRecognizerPlugin):
    """
    Example recognizer plugin for detecting phone numbers with custom patterns.

    This demonstrates how to create custom recognizers that extend the
    pattern-based recognizer framework.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="custom_phone_recognizer",
            version="1.0.0",
            description="Enhanced phone number recognizer with international format support",
            author="CloakPivot Team",
            plugin_type="recognizer",
            config_schema={
                "type": "object",
                "properties": {
                    "country_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of country codes to support",
                    },
                    "patterns": {
                        "type": "object",
                        "description": "Custom regex patterns by entity type",
                    },
                    "min_confidence": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Minimum confidence threshold",
                    },
                },
            },
        )

    def _initialize_recognizer(self) -> None:
        """Initialize with default patterns for phone numbers."""
        # Set default configuration if not provided
        default_config = {
            "entity_types": ["PHONE_NUMBER"],
            "supported_languages": ["en"],
            "min_confidence": 0.7,
            "country_codes": ["US", "CA", "GB", "AU"],
            "patterns": {
                "PHONE_NUMBER": [
                    # US/Canada formats
                    r"\b(?:\+1[-.\s]?)?(?:\(?[2-9][0-8][0-9]\)?[-.\s]?)?[2-9][0-9]{2}[-.\s]?[0-9]{4}\b",
                    # International format
                    r"\+[1-9]\d{1,14}\b",
                    # UK format
                    r"\b(?:\+44[-.\s]?)?(?:\(?0\)?[-.\s]?)?[1-9]\d{8,9}\b",
                    # Generic format with separators
                    r"\b(?:\d[-.\s]?){9,14}\d\b",
                ]
            },
        }

        # Update with user configuration
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Call parent initialization to compile patterns
        super()._initialize_recognizer()

    def _calculate_confidence(
        self, match: re.Match[str], entity_type: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Calculate confidence based on phone number validation."""
        match_text = match.group()
        base_confidence = 0.6

        # Length-based confidence
        digits = re.sub(r"\D", "", match_text)
        digit_count = len(digits)

        if digit_count == 10:  # US/Canada local
            base_confidence += 0.2
        elif digit_count == 11 and digits.startswith(
            "1"
        ):  # US/Canada with country code
            base_confidence += 0.3
        elif 7 <= digit_count <= 15:  # International range
            base_confidence += 0.1
        else:
            base_confidence -= 0.2

        # Format validation bonuses
        if re.search(r"\+\d", match_text):  # International prefix
            base_confidence += 0.1

        if re.search(r"\(\d{3}\)", match_text):  # Area code in parentheses
            base_confidence += 0.1

        if re.search(r"\d{3}[-.\s]\d{3}[-.\s]\d{4}", match_text):  # Standard formatting
            base_confidence += 0.1

        # Check for common false positives
        if self._is_likely_false_positive(match_text, entity_type):
            base_confidence -= 0.3

        return max(0.0, min(1.0, base_confidence))

    def _is_likely_false_positive(self, text: str, entity_type: str) -> bool:
        """Check for phone number false positives."""
        # Remove formatting
        digits = re.sub(r"\D", "", text)

        # Common false positive patterns
        false_positive_patterns = [
            r"^0{7,}$",  # All zeros
            r"^1{7,}$",  # All ones
            r"^(\d)\1{6,}$",  # Repeating digit
            r"^123456",  # Sequential numbers
            r"^1234567890$",  # Classic test number
        ]

        for pattern in false_positive_patterns:
            if re.match(pattern, digits):
                return True

        return False


class LicensePlateRecognizerPlugin(PatternBasedRecognizerPlugin):
    """
    Example recognizer plugin for detecting license plates.

    This demonstrates creating a recognizer for a custom entity type
    not covered by standard Presidio recognizers.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="license_plate_recognizer",
            version="1.0.0",
            description="License plate number recognizer for multiple regions",
            author="CloakPivot Team",
            plugin_type="recognizer",
            config_schema={
                "type": "object",
                "properties": {
                    "regions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Regions to support (US, CA, EU, etc.)",
                    },
                    "case_sensitive": {
                        "type": "boolean",
                        "description": "Whether matching should be case sensitive",
                    },
                },
            },
        )

    def _initialize_recognizer(self) -> None:
        """Initialize with license plate patterns."""
        default_config = {
            "entity_types": ["LICENSE_PLATE"],
            "supported_languages": ["en"],
            "min_confidence": 0.8,
            "regions": ["US", "CA"],
            "case_sensitive": False,
            "patterns": {
                "LICENSE_PLATE": [
                    # US formats (3 letters + 3 numbers, etc.)
                    r"\b[A-Z]{2,3}[-\s]?[0-9]{3,4}\b",
                    r"\b[0-9]{3}[-\s]?[A-Z]{2,3}\b",
                    # Canadian formats
                    r"\b[A-Z]{4}[-\s]?[0-9]{3}\b",
                    # European formats (country code + numbers/letters)
                    r"\b[A-Z]{1,3}[-\s]?[0-9A-Z]{2,6}\b",
                    # Generic alphanumeric plates
                    r"\b(?=.*[A-Z])(?=.*[0-9])[A-Z0-9]{5,8}\b",
                ]
            },
        }

        # Update with user configuration
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        # Call parent initialization
        super()._initialize_recognizer()

    def _calculate_confidence(
        self, match: re.Match[str], entity_type: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Calculate confidence for license plate matches."""
        match_text = match.group()
        base_confidence = 0.7

        # Length validation
        clean_text = re.sub(r"[-\s]", "", match_text)
        if 5 <= len(clean_text) <= 8:
            base_confidence += 0.1

        # Character mix validation (letters + numbers)
        has_letters = bool(re.search(r"[A-Z]", match_text.upper()))
        has_numbers = bool(re.search(r"[0-9]", match_text))

        if has_letters and has_numbers:
            base_confidence += 0.1
        else:
            base_confidence -= 0.2

        # Format pattern bonuses
        regions = self.get_config_value("regions", ["US"])

        if "US" in regions:
            # US-style patterns
            if re.match(r"[A-Z]{3}[-\s]?[0-9]{3,4}", match_text.upper()):
                base_confidence += 0.1

        if "CA" in regions:
            # Canadian-style patterns
            if re.match(r"[A-Z]{4}[-\s]?[0-9]{3}", match_text.upper()):
                base_confidence += 0.1

        return max(0.0, min(1.0, base_confidence))


class IPv4AddressRecognizerPlugin(PatternBasedRecognizerPlugin):
    """
    Example recognizer plugin for detecting IPv4 addresses.

    This demonstrates a simple pattern-based recognizer with validation.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="ipv4_address_recognizer",
            version="1.0.0",
            description="IPv4 address recognizer with validation",
            author="CloakPivot Team",
            plugin_type="recognizer",
            config_schema={
                "type": "object",
                "properties": {
                    "include_private": {
                        "type": "boolean",
                        "description": "Include private IP ranges",
                    },
                    "include_localhost": {
                        "type": "boolean",
                        "description": "Include localhost addresses",
                    },
                },
            },
        )

    def _initialize_recognizer(self) -> None:
        """Initialize with IPv4 patterns."""
        default_config = {
            "entity_types": ["IP_ADDRESS"],
            "supported_languages": ["en"],
            "min_confidence": 0.9,
            "include_private": True,
            "include_localhost": True,
            "patterns": {
                "IP_ADDRESS": [
                    # IPv4 pattern with word boundaries
                    r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"
                ]
            },
        }

        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value

        super()._initialize_recognizer()

    def _calculate_confidence(
        self, match: re.Match[str], entity_type: str, context: Optional[dict[str, Any]]
    ) -> float:
        """Calculate confidence with IP address validation."""
        match_text = match.group()

        try:
            # Parse IP address components
            octets = match_text.split(".")
            if len(octets) != 4:
                return 0.0

            # Validate each octet
            for octet_str in octets:
                octet = int(octet_str)
                if octet < 0 or octet > 255:
                    return 0.0

                # Check for leading zeros (invalid in IP addresses)
                if len(octet_str) > 1 and octet_str.startswith("0"):
                    return 0.0

            # Base confidence for valid IP
            base_confidence = 0.9

            # Check configuration filters
            include_private = self.get_config_value("include_private", True)
            include_localhost = self.get_config_value("include_localhost", True)

            first_octet = int(octets[0])
            second_octet = int(octets[1])

            # Check for localhost
            if first_octet == 127:
                return base_confidence if include_localhost else 0.0

            # Check for private ranges
            is_private = (
                first_octet == 10
                or (first_octet == 172 and 16 <= second_octet <= 31)
                or (first_octet == 192 and second_octet == 168)
            )

            if is_private and not include_private:
                return 0.0

            # Reserved ranges (reduce confidence)
            if first_octet in [0, 224, 240, 255]:
                base_confidence -= 0.2

            return max(0.0, min(1.0, base_confidence))

        except (ValueError, IndexError):
            return 0.0
