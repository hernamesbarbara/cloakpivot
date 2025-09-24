"""Template generation helpers for masking strategies."""

import logging

logger = logging.getLogger(__name__)


class TemplateGenerator:
    """Helper class for generating format-preserving templates."""

    @staticmethod
    def generate_format_template(original_text: str, entity_type: str) -> str:
        """Generate a format-preserving template based on the original text structure."""
        if not original_text:
            return "[REDACTED]"

        # Entity-specific format templates
        entity_templates = {
            "PHONE_NUMBER": TemplateGenerator.generate_phone_template,
            "US_SSN": TemplateGenerator.generate_ssn_template,
            "CREDIT_CARD": TemplateGenerator.generate_credit_card_template,
            "EMAIL_ADDRESS": TemplateGenerator.generate_email_template,
        }

        # Try entity-specific template generation first
        if entity_type in entity_templates:
            return entity_templates[entity_type](original_text)

        # Generic format template generation
        return TemplateGenerator.generate_generic_template(original_text)

    @staticmethod
    def generate_phone_template(original_text: str) -> str:
        """Generate format-preserving template for phone numbers."""
        if (
            len(original_text) == 10
            and original_text.isdigit()
            or len(original_text) == 12
            and original_text[3] == "-"
            and original_text[7] == "-"
        ):
            return "XXX-XXX-XXXX"
        if len(original_text) == 14 and original_text.startswith("(") and ")" in original_text:
            return "(XXX) XXX-XXXX"
        if "+" in original_text:
            # International format
            return "+X " + "X" * (len(original_text) - 3)

        # Generic phone template
        return "X" * len([c for c in original_text if c.isdigit()]) + "".join(
            c for c in original_text if not c.isdigit()
        )

    @staticmethod
    def generate_ssn_template(original_text: str) -> str:
        """Generate format-preserving template for SSN."""
        if len(original_text) == 11 and original_text[3] == "-" and original_text[6] == "-":
            return "XXX-XX-XXXX"
        if len(original_text) == 9 and original_text.isdigit():
            return "XXXXXXXXX"

        # Preserve structure but mask digits
        result = ""
        for char in original_text:
            if char.isdigit():
                result += "X"
            else:
                result += char
        return result

    @staticmethod
    def generate_credit_card_template(original_text: str) -> str:
        """Generate format-preserving template for credit cards."""
        if (
            len(original_text) == 16
            and original_text.isdigit()
            or len(original_text) == 19
            and original_text.count("-") == 3
        ):
            return "XXXX-XXXX-XXXX-XXXX"
        if len(original_text) == 19 and original_text.count(" ") == 3:
            return "XXXX XXXX XXXX XXXX"

        # Preserve structure but mask digits
        result = ""
        for char in original_text:
            if char.isdigit():
                result += "X"
            else:
                result += char
        return result

    @staticmethod
    def generate_email_template(original_text: str) -> str:
        """Generate format-preserving template for email addresses."""
        if "@" not in original_text:
            return "[EMAIL]"

        username, domain = original_text.split("@", 1)

        # Preserve username length and domain structure
        username_template = "x" * len(username)

        # Preserve domain structure
        if "." in domain:
            domain_parts = domain.split(".")
            domain_template = ".".join("x" * len(part) for part in domain_parts)
        else:
            domain_template = "x" * len(domain)

        return f"{username_template}@{domain_template}"

    @staticmethod
    def generate_generic_template(original_text: str) -> str:
        """Generate a generic format-preserving template."""
        result = ""
        for char in original_text:
            if char.isalpha():
                result += "X"
            elif char.isdigit():
                result += "#"
            else:
                result += char
        return result

    @staticmethod
    def detect_format_pattern(text: str) -> str:
        """Detect and return the format pattern of the input text."""
        pattern = ""
        for char in text:
            if char.isalpha():
                pattern += "L"  # Letter
            elif char.isdigit():
                pattern += "D"  # Digit
            elif char.isspace():
                pattern += "S"  # Space
            else:
                pattern += "P"  # Punctuation/Special
        return pattern

    @staticmethod
    def merge_template_with_format(
        user_template: str, format_template: str, original_text: str
    ) -> str:
        """Merge user-provided template with format preservation."""
        # If user template contains format placeholders, replace them
        if "{format}" in user_template:
            return user_template.replace("{format}", format_template)

        # If user template is simple (like [PHONE]), append format info
        if user_template.startswith("[") and user_template.endswith("]"):
            base_template = user_template[1:-1]
            return f"[{base_template}:{len(original_text)}]"

        # Otherwise return user template as-is
        return user_template
