"""Unit tests for cloakpivot.core.model_info module."""

from cloakpivot.core.types.model_info import MODEL_CHARACTERISTICS


class TestModelInfo:
    """Test model_info module."""

    def test_model_characteristics_exists(self):
        """Test that MODEL_CHARACTERISTICS is defined."""
        assert MODEL_CHARACTERISTICS is not None
        assert isinstance(MODEL_CHARACTERISTICS, dict)

    def test_small_model_characteristics(self):
        """Test small model characteristics."""
        assert "small" in MODEL_CHARACTERISTICS
        small = MODEL_CHARACTERISTICS["small"]
        assert "memory_mb" in small
        assert "load_time_ms" in small
        assert "accuracy_score" in small
        assert "description" in small

    def test_medium_model_characteristics(self):
        """Test medium model characteristics."""
        assert "medium" in MODEL_CHARACTERISTICS
        medium = MODEL_CHARACTERISTICS["medium"]
        assert "memory_mb" in medium
        assert "load_time_ms" in medium
        assert "accuracy_score" in medium
        assert "description" in medium

    def test_large_model_characteristics(self):
        """Test large model characteristics."""
        assert "large" in MODEL_CHARACTERISTICS
        large = MODEL_CHARACTERISTICS["large"]
        assert "memory_mb" in large
        assert "load_time_ms" in large
        assert "accuracy_score" in large
        assert "description" in large

    def test_model_characteristics_types(self):
        """Test that model characteristics have correct types."""
        for model_size, chars in MODEL_CHARACTERISTICS.items():
            assert isinstance(model_size, str)
            assert isinstance(chars, dict)
            assert isinstance(chars.get("memory_mb", 0), (int, float))
            assert isinstance(chars.get("load_time_ms", 0), (int, float))
            assert isinstance(chars.get("accuracy_score", 0), (int, float))
            assert isinstance(chars.get("description", ""), str)

    def test_model_characteristics_values(self):
        """Test that model characteristics have reasonable values."""
        # Small should use less memory than large
        small_mem = MODEL_CHARACTERISTICS["small"]["memory_mb"]
        large_mem = MODEL_CHARACTERISTICS["large"]["memory_mb"]
        assert small_mem < large_mem

    def test_all_models_have_required_fields(self):
        """Test that all models have required fields."""
        required_fields = ["memory_mb", "load_time_ms", "accuracy_score", "description"]
        for model_size, chars in MODEL_CHARACTERISTICS.items():
            for field in required_fields:
                assert field in chars, f"Model {model_size} missing field {field}"
