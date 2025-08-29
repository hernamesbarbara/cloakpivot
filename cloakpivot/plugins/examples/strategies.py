"""Example strategy plugins for demonstration and testing."""

import hashlib
import random
from typing import Any, Optional

from ..base import PluginInfo
from ..strategies.base import BaseStrategyPlugin, StrategyPluginResult


class ROT13StrategyPlugin(BaseStrategyPlugin):
    """
    Example strategy plugin that applies ROT13 transformation.

    This is a simple reversible transformation useful for demonstration
    and testing purposes.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="rot13_strategy",
            version="1.0.0",
            description="ROT13 character rotation strategy for reversible masking",
            author="CloakPivot Team",
            plugin_type="strategy",
            metadata={
                "reversible": True,
                "deterministic": True,
                "preserves_length": True
            }
        )

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None
    ) -> StrategyPluginResult:
        """Apply ROT13 transformation to the text."""
        try:
            # ROT13 transformation
            result_text = original_text.translate(str.maketrans(
                "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz",
                "NOPQRSTUVWXYZABCDEFGHIJKLMnopqrstuvwxyzabcdefghijklm"
            ))

            return StrategyPluginResult(
                masked_text=result_text,
                execution_time_ms=0.0,  # Will be filled by framework
                metadata={
                    "algorithm": "rot13",
                    "reversible": True,
                    "original_length": len(original_text)
                }
            )

        except Exception as e:
            return StrategyPluginResult(
                masked_text=original_text,
                execution_time_ms=0.0,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )


class UpsideDownStrategyPlugin(BaseStrategyPlugin):
    """
    Example strategy plugin that flips text upside down using Unicode.

    Demonstrates a fun but impractical masking strategy for testing.
    """

    # Character mapping for upside down text
    UPSIDE_DOWN_MAP = str.maketrans(
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
        "ɐqɔpǝɟɓɥᴉɾʞlɯuodbɹsʇnʌʍxʎzAQƆDEℲƃHIſʞ˥WNOԀQᴿSʇNΛMXʎZ0ㄥәろϛ9ㄥ86Ϭ"
    )

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="upside_down_strategy",
            version="1.0.0",
            description="Upside down text transformation using Unicode characters",
            author="CloakPivot Team",
            plugin_type="strategy",
            metadata={
                "reversible": False,
                "deterministic": True,
                "preserves_length": True,
                "unicode_required": True
            }
        )

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None
    ) -> StrategyPluginResult:
        """Apply upside down transformation."""
        try:
            # Transform and reverse the string
            transformed = original_text.translate(self.UPSIDE_DOWN_MAP)
            result_text = transformed[::-1]  # Reverse the string

            return StrategyPluginResult(
                masked_text=result_text,
                execution_time_ms=0.0,
                metadata={
                    "algorithm": "upside_down",
                    "reversible": False,
                    "transformation": "unicode_flip_reverse"
                }
            )

        except Exception as e:
            return StrategyPluginResult(
                masked_text="*" * len(original_text),
                execution_time_ms=0.0,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )


class ColorCodeStrategyPlugin(BaseStrategyPlugin):
    """
    Example strategy plugin that converts text to color codes.

    Demonstrates a creative approach to text masking using color representations.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="color_code_strategy",
            version="1.0.0",
            description="Convert text to hex color codes for creative masking",
            author="CloakPivot Team",
            plugin_type="strategy",
            metadata={
                "reversible": False,
                "deterministic": True,
                "preserves_length": False,
                "output_format": "hex_colors"
            }
        )

    def _validate_strategy_config(self, config: dict[str, Any]) -> bool:
        """Validate color code strategy configuration."""
        color_format = config.get("color_format", "hex")
        if color_format not in ["hex", "rgb", "hsl"]:
            raise ValueError("color_format must be 'hex', 'rgb', or 'hsl'")
        return True

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None
    ) -> StrategyPluginResult:
        """Convert text to color codes."""
        try:
            color_format = self.get_config_value("color_format", "hex")

            # Generate deterministic color from text
            text_hash = hashlib.md5(original_text.encode()).hexdigest()

            if color_format == "hex":
                # Use first 6 characters as hex color
                color_code = f"#{text_hash[:6].upper()}"
            elif color_format == "rgb":
                # Convert to RGB values
                r = int(text_hash[:2], 16)
                g = int(text_hash[2:4], 16)
                b = int(text_hash[4:6], 16)
                color_code = f"rgb({r},{g},{b})"
            else:  # hsl
                # Convert to HSL approximation
                h = int(text_hash[:3], 16) % 360
                s = (int(text_hash[3:5], 16) % 50) + 50  # 50-100%
                l = (int(text_hash[5:7], 16) % 40) + 30  # 30-70%
                color_code = f"hsl({h},{s}%,{l}%)"

            return StrategyPluginResult(
                masked_text=color_code,
                execution_time_ms=0.0,
                metadata={
                    "algorithm": "color_code",
                    "color_format": color_format,
                    "source_hash": text_hash[:8],
                    "original_length": len(original_text)
                }
            )

        except Exception as e:
            return StrategyPluginResult(
                masked_text="[COLOR_ERROR]",
                execution_time_ms=0.0,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )


class WordShuffleStrategyPlugin(BaseStrategyPlugin):
    """
    Example strategy plugin that shuffles words in text.

    Demonstrates a strategy that preserves word structure but changes order.
    """

    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="word_shuffle_strategy",
            version="1.0.0",
            description="Shuffle words in text while preserving word boundaries",
            author="CloakPivot Team",
            plugin_type="strategy",
            metadata={
                "reversible": False,
                "deterministic": True,
                "preserves_word_count": True,
                "preserves_punctuation": True
            }
        )

    def get_supported_entity_types(self) -> Optional[list[str]]:
        """This strategy works best with text-based entities."""
        return ["PERSON", "LOCATION", "ORGANIZATION", "MISC"]

    def _validate_strategy_config(self, config: dict[str, Any]) -> bool:
        """Validate word shuffle configuration."""
        preserve_case = config.get("preserve_case", True)
        if not isinstance(preserve_case, bool):
            raise ValueError("preserve_case must be a boolean")
        return True

    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[dict[str, Any]] = None
    ) -> StrategyPluginResult:
        """Shuffle words in the text."""
        try:
            preserve_case = self.get_config_value("preserve_case", True)

            # Split into words and non-word characters
            import re
            tokens = re.findall(r'\w+|\W+', original_text)

            # Extract just the words
            words = [token for token in tokens if re.match(r'\w+', token)]

            if len(words) <= 1:
                # Not enough words to shuffle meaningfully
                return StrategyPluginResult(
                    masked_text=original_text,
                    execution_time_ms=0.0,
                    metadata={
                        "algorithm": "word_shuffle",
                        "word_count": len(words),
                        "shuffled": False,
                        "reason": "insufficient_words"
                    }
                )

            # Shuffle words deterministically based on original text
            shuffled_words = words.copy()
            shuffle_seed = hash(original_text) % (2**32)
            random.Random(shuffle_seed).shuffle(shuffled_words)

            # Preserve case pattern if requested
            if preserve_case:
                for i, (original, shuffled) in enumerate(zip(words, shuffled_words)):
                    case_pattern = [c.isupper() for c in original]
                    new_word = ""
                    for j, char in enumerate(shuffled.lower()):
                        if j < len(case_pattern) and case_pattern[j]:
                            new_word += char.upper()
                        else:
                            new_word += char
                    shuffled_words[i] = new_word

            # Reconstruct text with shuffled words
            word_iter = iter(shuffled_words)
            result_tokens = []

            for token in tokens:
                if re.match(r'\w+', token):
                    result_tokens.append(next(word_iter))
                else:
                    result_tokens.append(token)

            result_text = ''.join(result_tokens)

            return StrategyPluginResult(
                masked_text=result_text,
                execution_time_ms=0.0,
                metadata={
                    "algorithm": "word_shuffle",
                    "word_count": len(words),
                    "shuffled": True,
                    "preserve_case": preserve_case,
                    "shuffle_seed": shuffle_seed
                }
            )

        except Exception as e:
            return StrategyPluginResult(
                masked_text="[SHUFFLE_ERROR]",
                execution_time_ms=0.0,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
