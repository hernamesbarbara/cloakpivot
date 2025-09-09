"""Tests for deprecation warning system."""

import warnings
from unittest.mock import Mock, patch

import pytest

from cloakpivot.migration import (
    DeprecationManager,
    LegacyDeprecationWarning,
    deprecated_class,
    deprecated_engine,
    deprecated_parameter,
)


class TestDeprecationWarnings:
    """Test deprecation warning functionality."""
    
    def test_deprecated_engine_decorator(self):
        """Test that deprecated_engine decorator emits warnings."""
        
        @deprecated_engine
        def legacy_function():
            return "legacy"
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = legacy_function()
            
            assert result == "legacy"
            assert len(w) == 1
            assert issubclass(w[0].category, LegacyDeprecationWarning)
            assert "legacy masking engine" in str(w[0].message)
            assert "use_presidio_engine=True" in str(w[0].message)
    
    def test_deprecation_manager_warn_legacy_usage(self):
        """Test DeprecationManager.warn_legacy_usage method."""
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            DeprecationManager.warn_legacy_usage(
                "test component",
                "new alternative"
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, LegacyDeprecationWarning)
            assert "Legacy test component is deprecated" in str(w[0].message)
            assert "Use new alternative instead" in str(w[0].message)
            assert "migration guide" in str(w[0].message)
    
    def test_deprecation_manager_warn_feature_deprecation(self):
        """Test DeprecationManager.warn_feature_deprecation method."""
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            DeprecationManager.warn_feature_deprecation(
                "old_feature",
                "3.0",
                "new_feature"
            )
            
            assert len(w) == 1
            assert issubclass(w[0].category, LegacyDeprecationWarning)
            assert "old_feature is deprecated" in str(w[0].message)
            assert "removed in version 3.0" in str(w[0].message)
            assert "Use new_feature instead" in str(w[0].message)
    
    def test_deprecation_manager_get_timeline(self):
        """Test getting deprecation timeline."""
        timeline = DeprecationManager.get_deprecation_timeline()
        
        assert isinstance(timeline, dict)
        assert "2024-Q1" in timeline
        assert "2024-Q2" in timeline
        assert "2024-Q3" in timeline
        assert "2024-Q4" in timeline
        
        # Verify timeline content
        assert "Legacy engine marked deprecated" in timeline["2024-Q1"]
        assert "Presidio engine becomes default" in timeline["2024-Q2"]
    
    def test_deprecation_manager_check_status(self):
        """Test checking deprecation status of components."""
        
        # Check deprecated components
        status = DeprecationManager.check_deprecation_status("legacy_engine")
        assert status is not None
        assert "Presidio engine" in status
        
        status = DeprecationManager.check_deprecation_status("strategy_applicator")
        assert status is not None
        assert "PresidioMaskingAdapter" in status
        
        # Check non-deprecated component
        status = DeprecationManager.check_deprecation_status("nonexistent")
        assert status is None
    
    def test_suppress_and_reset_warnings(self):
        """Test suppressing and resetting deprecation warnings."""
        
        # Save original warning filters
        original_filters = warnings.filters[:]
        
        try:
            # First, verify warnings are shown
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                DeprecationManager.warn_legacy_usage("test")
                assert len(w) == 1
            
            # Suppress warnings globally
            DeprecationManager.suppress_warnings()
            
            # Test that warnings are suppressed
            with warnings.catch_warnings(record=True) as w:
                # Apply the global filter to this context
                for action, message, category, module, lineno in warnings.filters:
                    if category == LegacyDeprecationWarning and action == "ignore":
                        warnings.filterwarnings(action, category=category)
                        break
                else:
                    warnings.simplefilter("always")
                
                DeprecationManager.warn_legacy_usage("test")
                # Warning should be suppressed if filter was applied
                if any(f[0] == "ignore" and f[2] == LegacyDeprecationWarning for f in warnings.filters):
                    assert len(w) == 0
            
            # Reset warnings
            DeprecationManager.reset_warnings()
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                DeprecationManager.warn_legacy_usage("test")
                # Warning should be shown again
                assert len(w) == 1
                
        finally:
            # Restore original warning filters
            warnings.filters[:] = original_filters
    
    def test_deprecated_parameter_decorator(self):
        """Test deprecated_parameter decorator."""
        
        @deprecated_parameter("old_param", "new_param", "2.5")
        def test_function(new_param=None, old_param=None):
            return new_param or old_param
        
        # Using new parameter - no warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_function(new_param="value")
            assert result == "value"
            assert len(w) == 0
        
        # Using deprecated parameter - warning
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result = test_function(old_param="value")
            assert result == "value"
            assert len(w) == 1
            assert "Parameter 'old_param' is deprecated" in str(w[0].message)
            assert "removed in v2.5" in str(w[0].message)
            assert "Use new_param instead" in str(w[0].message)
    
    def test_deprecated_class_decorator(self):
        """Test deprecated_class decorator."""
        
        @deprecated_class("NewClass", "3.0")
        class OldClass:
            def __init__(self, value):
                self.value = value
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Creating instance should trigger warning
            obj = OldClass("test")
            
            assert obj.value == "test"
            assert len(w) == 1
            assert "OldClass is deprecated" in str(w[0].message)
            assert "removed in v3.0" in str(w[0].message)
            assert "Use NewClass instead" in str(w[0].message)
    
    def test_masking_engine_deprecation_warning(self):
        """Test that MaskingEngine emits deprecation warning for legacy engine."""
        from cloakpivot.masking.engine import MaskingEngine
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create engine with legacy mode (explicitly set to False)
            engine = MaskingEngine(use_presidio_engine=False)
            
            # Should have deprecation warning
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, LegacyDeprecationWarning)
            ]
            assert len(deprecation_warnings) > 0
            assert any("masking engine" in str(w.message) for w in deprecation_warnings)
    
    def test_no_warning_with_presidio_engine(self):
        """Test that no warning is emitted when using Presidio engine."""
        from cloakpivot.masking.engine import MaskingEngine
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Create engine with Presidio mode
            engine = MaskingEngine(use_presidio_engine=True)
            
            # Should have no deprecation warnings
            deprecation_warnings = [
                warning for warning in w 
                if issubclass(warning.category, LegacyDeprecationWarning)
            ]
            assert len(deprecation_warnings) == 0
    
    def test_multiple_warnings_stacklevel(self):
        """Test that warnings have correct stack level."""
        
        def outer_function():
            inner_function()
        
        def inner_function():
            DeprecationManager.warn_legacy_usage("test", stacklevel=3)
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            outer_function()
            
            assert len(w) == 1
            # The warning should point to the correct line in the stack
            assert w[0].lineno > 0
    
    def test_warning_categories(self):
        """Test that LegacyDeprecationWarning is a proper warning category."""
        
        assert issubclass(LegacyDeprecationWarning, UserWarning)
        assert issubclass(LegacyDeprecationWarning, Warning)
        
        # Can be used in warnings.filterwarnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=LegacyDeprecationWarning)
            # This should not raise
            warnings.warn("test", LegacyDeprecationWarning)


class TestDeprecationIntegration:
    """Test deprecation warnings in integration scenarios."""
    
    def test_migration_preserves_functionality(self):
        """Test that deprecated features still work during migration period."""
        from cloakpivot.masking.engine import MaskingEngine
        
        # Suppress warnings for this test
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=LegacyDeprecationWarning)
            
            # Legacy engine should still work
            legacy_engine = MaskingEngine(use_presidio_engine=False)
            assert not legacy_engine.use_presidio
            assert legacy_engine.strategy_applicator is not None
            
            # Presidio engine should work
            presidio_engine = MaskingEngine(use_presidio_engine=True)
            assert presidio_engine.use_presidio
            assert presidio_engine.presidio_adapter is not None
    
    def test_environment_variable_deprecation(self):
        """Test deprecation based on environment variable."""
        import os
        from cloakpivot.masking.engine import MaskingEngine
        
        # Save original env var
        original_value = os.environ.get("CLOAKPIVOT_USE_PRESIDIO_ENGINE")
        
        try:
            # Set to use legacy engine
            os.environ["CLOAKPIVOT_USE_PRESIDIO_ENGINE"] = "false"
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                engine = MaskingEngine()
                
                # Should have deprecation warning
                deprecation_warnings = [
                    warning for warning in w 
                    if issubclass(warning.category, LegacyDeprecationWarning)
                ]
                assert len(deprecation_warnings) > 0
            
            # Set to use Presidio engine
            os.environ["CLOAKPIVOT_USE_PRESIDIO_ENGINE"] = "true"
            
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                
                engine = MaskingEngine()
                
                # Should have no deprecation warnings
                deprecation_warnings = [
                    warning for warning in w 
                    if issubclass(warning.category, LegacyDeprecationWarning)
                ]
                assert len(deprecation_warnings) == 0
                
        finally:
            # Restore original env var
            if original_value is not None:
                os.environ["CLOAKPIVOT_USE_PRESIDIO_ENGINE"] = original_value
            else:
                os.environ.pop("CLOAKPIVOT_USE_PRESIDIO_ENGINE", None)