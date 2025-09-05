"""Tests for MaskingEngine feature flag precedence and behavior."""

import os
from unittest import mock

from cloakpivot.masking.engine import MaskingEngine


class TestFeatureFlags:
    """Test feature flag precedence and configuration for MaskingEngine."""

    def test_precedence_explicit_over_environment(self):
        """Test that explicit parameter has highest precedence."""
        # Set environment variable to True
        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "true"}):
            # Explicit False should override
            engine = MaskingEngine(use_presidio_engine=False)
            assert engine.use_presidio is False

            # Explicit True should be respected
            engine = MaskingEngine(use_presidio_engine=True)
            assert engine.use_presidio is True

        # Set environment variable to False
        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "false"}):
            # Explicit True should override
            engine = MaskingEngine(use_presidio_engine=True)
            assert engine.use_presidio is True

            # Explicit False should be respected
            engine = MaskingEngine(use_presidio_engine=False)
            assert engine.use_presidio is False

    def test_environment_variable_parsing(self):
        """Test various environment variable values are parsed correctly."""
        true_values = ["true", "True", "TRUE", "1", "yes", "Yes", "YES", "on", "On", "ON"]
        false_values = ["false", "False", "FALSE", "0", "no", "No", "NO", "off", "Off", "OFF", "", "invalid"]

        for value in true_values:
            with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": value}):
                engine = MaskingEngine()
                assert engine.use_presidio is True, f"Failed for value: {value}"

        for value in false_values:
            with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": value}):
                engine = MaskingEngine()
                assert engine.use_presidio is False, f"Failed for value: {value}"

    def test_no_environment_variable_defaults_to_legacy(self):
        """Test that absence of environment variable defaults to legacy engine."""
        # Clear the environment variable if it exists
        with mock.patch.dict(os.environ, {}, clear=True):
            # Remove the specific variable if it exists
            os.environ.pop("CLOAKPIVOT_USE_PRESIDIO_ENGINE", None)

            engine = MaskingEngine()
            assert engine.use_presidio is False
            assert engine.strategy_applicator is not None
            assert engine.presidio_adapter is None

    def test_explicit_none_checks_environment(self):
        """Test that explicit None parameter defers to environment variable."""
        # With environment set to true
        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "true"}):
            engine = MaskingEngine(use_presidio_engine=None)
            assert engine.use_presidio is True

        # With environment set to false
        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "false"}):
            engine = MaskingEngine(use_presidio_engine=None)
            assert engine.use_presidio is False

        # With no environment variable
        with mock.patch.dict(os.environ, {}, clear=True):
            os.environ.pop("CLOAKPIVOT_USE_PRESIDIO_ENGINE", None)
            engine = MaskingEngine(use_presidio_engine=None)
            assert engine.use_presidio is False

    def test_engine_initialization_state(self):
        """Test that the correct engine components are initialized."""
        # Legacy engine
        engine = MaskingEngine(use_presidio_engine=False)
        assert engine.use_presidio is False
        assert engine.strategy_applicator is not None
        assert engine.presidio_adapter is None
        assert engine.document_masker is not None

        # Presidio engine
        engine = MaskingEngine(use_presidio_engine=True)
        assert engine.use_presidio is True
        assert engine.strategy_applicator is None
        assert engine.presidio_adapter is not None
        assert engine.document_masker is not None

    def test_all_constructor_params_work_with_feature_flag(self):
        """Test that all constructor parameters work regardless of engine choice."""
        # Test with legacy engine
        engine = MaskingEngine(
            resolve_conflicts=True,
            store_original_text=False,
            use_presidio_engine=False
        )
        assert engine.use_presidio is False
        assert engine.resolve_conflicts is True
        assert engine.store_original_text is False
        assert engine.entity_normalizer is not None

        # Test with Presidio engine
        engine = MaskingEngine(
            resolve_conflicts=True,
            store_original_text=False,
            use_presidio_engine=True
        )
        assert engine.use_presidio is True
        assert engine.resolve_conflicts is True
        assert engine.store_original_text is False
        assert engine.entity_normalizer is not None

    def test_runtime_engine_switch_not_supported(self):
        """Test that engine cannot be switched after initialization."""
        engine = MaskingEngine(use_presidio_engine=False)
        assert engine.use_presidio is False

        # Changing the flag after initialization should not affect the engine
        # This is by design - engine is determined at construction time
        original_value = engine.use_presidio

        # Try to modify (this shouldn't change the actual engine components)
        engine.use_presidio = True

        # The flag might change but the components remain the same
        assert engine.strategy_applicator is not None  # Still initialized
        assert engine.presidio_adapter is None  # Still None

        # Restore for clarity
        engine.use_presidio = original_value

    @mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "invalid_value"})
    def test_invalid_environment_value_defaults_to_legacy(self):
        """Test that invalid environment values default to legacy engine."""
        engine = MaskingEngine()
        assert engine.use_presidio is False

    def test_case_insensitive_environment_values(self):
        """Test that environment variable values are case-insensitive."""
        case_variations = [
            ("TrUe", True),
            ("fAlSe", False),
            ("YeS", True),
            ("nO", False),
            ("oN", True),
            ("OfF", False)
        ]

        for value, expected in case_variations:
            with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": value}):
                engine = MaskingEngine()
                assert engine.use_presidio is expected, f"Failed for value: {value}"

    def test_whitespace_in_environment_values(self):
        """Test that whitespace in environment values is handled."""
        # Note: os.getenv returns the raw string, so we handle this in our logic
        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": " true "}):
            engine = MaskingEngine()
            # Now with .strip(), whitespace should be handled correctly
            assert engine.use_presidio is True

        with mock.patch.dict(os.environ, {"CLOAKPIVOT_USE_PRESIDIO_ENGINE": "  false  "}):
            engine = MaskingEngine()
            assert engine.use_presidio is False
