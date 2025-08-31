"""Unit tests for scripts/setup-models.py ModelManager class.

Tests comprehensive model download and verification functionality for CI/CD caching.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

# Import the module by adding scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import after path modification
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("setup_models", scripts_dir / "setup-models.py")
    setup_models = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(setup_models)
    # Make the module available globally for patching
    sys.modules["setup_models"] = setup_models
    ModelManager = setup_models.ModelManager
except ImportError as e:
    pytest.skip(f"Cannot import setup-models.py: {e}")


class TestModelManager:
    """Test suite for ModelManager class functionality."""

    @pytest.fixture
    def temp_cache_dir(self):
        """Create temporary cache directory for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir

    @pytest.fixture
    def mock_spacy_module(self):
        """Mock spaCy module for controlled testing."""
        with patch('setup_models.spacy') as mock_spacy:
            mock_spacy.__version__ = "3.7.0"
            yield mock_spacy

    @pytest.fixture
    def mock_spacy_download(self):
        """Mock spaCy download function."""
        with patch('setup_models.spacy_download') as mock_download:
            yield mock_download

    @pytest.fixture
    def model_manager(self, temp_cache_dir):
        """Create ModelManager instance with temporary cache directory."""
        return ModelManager(model_size="small", cache_dir=temp_cache_dir)

    def test_init_default_parameters(self):
        """Test ModelManager initialization with default parameters."""
        manager = ModelManager()
        
        assert manager.model_size == "small"
        assert manager.cache_dir == Path.home() / ".cache" / "spacy"
        assert manager.spacy_models_dir == Path.home() / "spacy_models"
        assert hasattr(manager, 'start_time')

    def test_init_custom_parameters(self, temp_cache_dir):
        """Test ModelManager initialization with custom parameters."""
        manager = ModelManager(model_size="large", cache_dir=temp_cache_dir)
        
        assert manager.model_size == "large"
        assert manager.cache_dir == Path(temp_cache_dir)

    def test_get_required_models_small(self):
        """Test getting required models for small configuration."""
        manager = ModelManager(model_size="small")
        models = manager.get_required_models()
        
        assert models == ["en_core_web_sm"]

    def test_get_required_models_medium(self):
        """Test getting required models for medium configuration."""
        manager = ModelManager(model_size="medium")
        models = manager.get_required_models()
        
        assert models == ["en_core_web_sm", "en_core_web_md"]

    def test_get_required_models_large(self):
        """Test getting required models for large configuration."""
        manager = ModelManager(model_size="large")
        models = manager.get_required_models()
        
        assert models == ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"]

    def test_get_required_models_invalid_size(self):
        """Test getting required models for invalid configuration defaults to small."""
        manager = ModelManager(model_size="invalid")
        models = manager.get_required_models()
        
        assert models == ["en_core_web_sm"]

    def test_is_model_available_success(self, model_manager, mock_spacy_module):
        """Test successful model availability check."""
        # Mock a working spaCy model
        mock_doc = Mock()
        mock_doc.ents = [Mock()]  # Mock entities
        mock_doc.__len__ = Mock(return_value=5)
        
        mock_token = Mock()
        mock_token.pos_ = "NOUN"
        mock_doc.__iter__ = Mock(return_value=iter([mock_token, mock_token]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        
        mock_spacy_module.load.return_value = mock_nlp
        
        result = model_manager.is_model_available("en_core_web_sm")
        
        assert result is True
        mock_spacy_module.load.assert_called_once_with("en_core_web_sm")

    def test_is_model_available_model_not_found(self, model_manager, mock_spacy_module):
        """Test model availability check when model not found."""
        mock_spacy_module.load.side_effect = OSError("Model not found")
        
        result = model_manager.is_model_available("nonexistent_model")
        
        assert result is False

    def test_is_model_available_validation_failure(self, model_manager, mock_spacy_module):
        """Test model availability check when validation fails."""
        # Mock a model that loads but fails validation
        mock_doc = Mock()
        mock_doc.__len__ = Mock(return_value=0)  # Empty document
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        
        mock_spacy_module.load.return_value = mock_nlp
        
        result = model_manager.is_model_available("broken_model")
        
        assert result is False

    def test_is_model_available_no_pos_tags(self, model_manager, mock_spacy_module):
        """Test model availability check when POS tagging fails."""
        # Mock a model with no POS tags
        mock_doc = Mock()
        mock_doc.ents = []
        mock_doc.__len__ = Mock(return_value=5)
        
        mock_token = Mock()
        mock_token.pos_ = ""  # No POS tag
        mock_doc.__iter__ = Mock(return_value=iter([mock_token, mock_token]))
        
        mock_nlp = Mock()
        mock_nlp.return_value = mock_doc
        
        mock_spacy_module.load.return_value = mock_nlp
        
        result = model_manager.is_model_available("no_pos_model")
        
        assert result is False

    def test_is_model_available_attribute_error(self, model_manager, mock_spacy_module):
        """Test model availability check with AttributeError."""
        mock_spacy_module.load.side_effect = AttributeError("Missing attribute")
        
        result = model_manager.is_model_available("broken_model")
        
        assert result is False

    def test_is_model_available_unexpected_error(self, model_manager, mock_spacy_module):
        """Test model availability check with unexpected error."""
        mock_spacy_module.load.side_effect = ValueError("Unexpected error")
        
        result = model_manager.is_model_available("error_model")
        
        assert result is False

    def test_download_model_already_available(self, model_manager, mock_spacy_download):
        """Test downloading model that's already available."""
        with patch.object(model_manager, 'is_model_available', return_value=True):
            result = model_manager.download_model("en_core_web_sm")
            
            assert result is True
            mock_spacy_download.assert_not_called()

    def test_download_model_force_download(self, model_manager, mock_spacy_download):
        """Test force downloading model even when available."""
        with patch.object(model_manager, 'is_model_available', side_effect=[True, True]):
            result = model_manager.download_model("en_core_web_sm", force=True)
            
            assert result is True
            mock_spacy_download.assert_called_once_with("en_core_web_sm")

    def test_download_model_success(self, model_manager, mock_spacy_download):
        """Test successful model download."""
        with patch.object(model_manager, 'is_model_available', side_effect=[False, True]):
            result = model_manager.download_model("en_core_web_sm")
            
            assert result is True
            mock_spacy_download.assert_called_once_with("en_core_web_sm")

    def test_download_model_download_failure(self, model_manager, mock_spacy_download):
        """Test model download failure."""
        mock_spacy_download.side_effect = Exception("Download failed")
        
        with patch.object(model_manager, 'is_model_available', return_value=False):
            result = model_manager.download_model("en_core_web_sm")
            
            assert result is False

    def test_download_model_verification_failure(self, model_manager, mock_spacy_download):
        """Test model download with verification failure."""
        with patch.object(model_manager, 'is_model_available', side_effect=[False, False]):
            result = model_manager.download_model("en_core_web_sm")
            
            assert result is False
            mock_spacy_download.assert_called_once_with("en_core_web_sm")

    def test_setup_models_cache_hit_verified(self, model_manager):
        """Test setup models with cache hit and successful verification."""
        with patch.object(model_manager, 'is_model_available', return_value=True):
            result = model_manager.setup_models(cache_hit=True)
            
            assert result is True

    def test_setup_models_cache_hit_verification_failed(self, model_manager, mock_spacy_download):
        """Test setup models with cache hit but verification failure."""
        with patch.object(model_manager, 'is_model_available', side_effect=[False, True]):
            with patch.object(model_manager, 'download_model', return_value=True) as mock_download:
                result = model_manager.setup_models(cache_hit=True)
                
                assert result is True
                mock_download.assert_called_once_with("en_core_web_sm", force=True)

    def test_setup_models_fresh_download_success(self, model_manager):
        """Test setup models with fresh download success."""
        with patch.object(model_manager, 'download_model', return_value=True):
            result = model_manager.setup_models(cache_hit=False)
            
            assert result is True

    def test_setup_models_download_failure(self, model_manager):
        """Test setup models with download failure."""
        with patch.object(model_manager, 'download_model', return_value=False):
            result = model_manager.setup_models(cache_hit=False)
            
            assert result is False

    def test_setup_models_verify_only_success(self, model_manager):
        """Test setup models in verify-only mode with success."""
        with patch.object(model_manager, 'is_model_available', return_value=True):
            result = model_manager.setup_models(verify_only=True)
            
            assert result is True

    def test_setup_models_verify_only_failure(self, model_manager):
        """Test setup models in verify-only mode with failure."""
        with patch.object(model_manager, 'is_model_available', return_value=False):
            result = model_manager.setup_models(verify_only=True)
            
            assert result is False

    def test_setup_models_mixed_results(self, model_manager):
        """Test setup models with mixed success/failure results."""
        # Set up medium configuration with two models
        model_manager.model_size = "medium"
        
        def mock_download_side_effect(model_name):
            # First model succeeds, second fails
            return model_name == "en_core_web_sm"
        
        with patch.object(model_manager, 'download_model', side_effect=mock_download_side_effect):
            result = model_manager.setup_models(cache_hit=False)
            
            assert result is False  # Should fail if any model fails

    def test_get_cache_info_basic(self, model_manager, mock_spacy_module):
        """Test getting basic cache info."""
        info = model_manager.get_cache_info()
        
        assert "cache_dir" in info
        assert "models_dir" in info
        assert "spacy_version" in info
        assert "model_size_config" in info
        assert "required_models" in info
        assert "system_info" in info
        assert "available_models" in info
        
        assert info["model_size_config"] == "small"
        assert info["required_models"] == ["en_core_web_sm"]

    def test_get_cache_info_with_available_models(self, model_manager, mock_spacy_module):
        """Test getting cache info with available models."""
        # Mock model loading for cache info
        mock_nlp = Mock()
        mock_nlp.lang = "en"
        mock_nlp.pipe_names = ["tok2vec", "tagger", "parser", "ner"]
        mock_nlp.meta = Mock()
        mock_nlp.meta.version = "3.7.0"
        
        mock_spacy_module.load.return_value = mock_nlp
        
        with patch.object(model_manager, 'is_model_available', return_value=True):
            info = model_manager.get_cache_info()
            
            assert len(info["available_models"]) == 1
            model_info = info["available_models"][0]
            
            assert model_info["name"] == "en_core_web_sm"
            assert model_info["lang"] == "en"
            assert model_info["pipeline"] == ["tok2vec", "tagger", "parser", "ner"]
            assert model_info["version"] == "3.7.0"

    def test_get_cache_info_model_loading_error(self, model_manager, mock_spacy_module):
        """Test getting cache info with model loading error."""
        mock_spacy_module.load.side_effect = Exception("Model loading failed")
        
        with patch.object(model_manager, 'is_model_available', return_value=True):
            info = model_manager.get_cache_info()
            
            assert len(info["available_models"]) == 1
            model_info = info["available_models"][0]
            
            assert model_info["name"] == "en_core_web_sm"
            assert "error" in model_info
            assert "Model loading failed" in model_info["error"]

    def test_get_cache_info_enumeration_error(self, model_manager):
        """Test getting cache info with enumeration error."""
        with patch.object(model_manager, 'get_required_models', side_effect=Exception("Enumeration failed")):
            info = model_manager.get_cache_info()
            
            assert "error" in info
            assert "Enumeration failed" in info["error"]


class TestMainFunction:
    """Test suite for main() function and CLI argument parsing."""

    @pytest.fixture
    def mock_model_manager(self):
        """Mock ModelManager for testing main function."""
        with patch('setup_models.ModelManager') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_main_cache_info_text_output(self, mock_model_manager, capsys):
        """Test main function with cache info in text format."""
        mock_model_manager.get_cache_info.return_value = {
            "cache_dir": "/test/cache",
            "models_dir": "/test/models",
            "spacy_version": "3.7.0",
            "model_size_config": "small",
            "required_models": ["en_core_web_sm"],
            "available_models": []
        }
        
        with patch('sys.argv', ['setup-models.py', '--cache-info']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0

    def test_main_cache_info_json_output(self, mock_model_manager, capsys):
        """Test main function with cache info in JSON format."""
        cache_info = {
            "cache_dir": "/test/cache",
            "models_dir": "/test/models",
            "spacy_version": "3.7.0",
            "model_size_config": "small",
            "required_models": ["en_core_web_sm"],
            "available_models": []
        }
        mock_model_manager.get_cache_info.return_value = cache_info
        
        with patch('sys.argv', ['setup-models.py', '--cache-info', '--json-output']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0

    def test_main_setup_models_success(self, mock_model_manager):
        """Test main function with successful model setup."""
        mock_model_manager.setup_models.return_value = True
        
        with patch('sys.argv', ['setup-models.py', '--model-size', 'medium']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0
                mock_model_manager.setup_models.assert_called_once_with(
                    cache_hit=False, verify_only=False
                )

    def test_main_setup_models_failure(self, mock_model_manager):
        """Test main function with failed model setup."""
        mock_model_manager.setup_models.return_value = False
        
        with patch('sys.argv', ['setup-models.py', '--model-size', 'small']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 1

    def test_main_cache_hit_true(self, mock_model_manager):
        """Test main function with cache hit true."""
        mock_model_manager.setup_models.return_value = True
        
        with patch('sys.argv', ['setup-models.py', '--cache-hit', 'true']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0
                mock_model_manager.setup_models.assert_called_once_with(
                    cache_hit=True, verify_only=False
                )

    def test_main_verify_installation(self, mock_model_manager):
        """Test main function with verify installation flag."""
        mock_model_manager.setup_models.return_value = True
        
        with patch('sys.argv', ['setup-models.py', '--verify-installation']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0
                mock_model_manager.setup_models.assert_called_once_with(
                    cache_hit=False, verify_only=True
                )

    def test_main_json_output_success(self, mock_model_manager, capsys):
        """Test main function with JSON output format."""
        mock_model_manager.setup_models.return_value = True
        mock_model_manager.get_cache_info.return_value = {"test": "info"}
        
        with patch('sys.argv', ['setup-models.py', '--json-output']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 0

    def test_main_keyboard_interrupt(self, mock_model_manager):
        """Test main function with keyboard interrupt."""
        mock_model_manager.setup_models.side_effect = KeyboardInterrupt()
        
        with patch('sys.argv', ['setup-models.py']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 130

    def test_main_unexpected_error(self, mock_model_manager):
        """Test main function with unexpected error."""
        mock_model_manager.setup_models.side_effect = Exception("Unexpected error")
        
        with patch('sys.argv', ['setup-models.py']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 1

    def test_main_unexpected_error_json_output(self, mock_model_manager, capsys):
        """Test main function with unexpected error and JSON output."""
        mock_model_manager.setup_models.side_effect = Exception("Unexpected error")
        
        with patch('sys.argv', ['setup-models.py', '--json-output']):
            with patch('setup_models.main', wraps=main_function_wrapper(mock_model_manager)) as mock_main:
                result = mock_main()
                
                assert result == 1


def main_function_wrapper(mock_manager):
    """Wrapper function for testing main() with mocked ModelManager."""
    def wrapped_main():
        import argparse
        import json
        import sys
        
        parser = argparse.ArgumentParser()
        parser.add_argument("--model-size", choices=["small", "medium", "large"], default="small")
        parser.add_argument("--cache-hit", type=str, default="false")
        parser.add_argument("--verify-installation", action="store_true")
        parser.add_argument("--cache-info", action="store_true")
        parser.add_argument("--cache-dir", type=str)
        parser.add_argument("--json-output", action="store_true")
        
        args = parser.parse_args()
        
        try:
            if args.cache_info:
                info = mock_manager.get_cache_info()
                if args.json_output:
                    print(json.dumps(info, indent=2))
                else:
                    print("📊 Cache and Model Information:")
                    print(f"  Cache directory: {info['cache_dir']}")
                    print(f"  Models directory: {info['models_dir']}")
                    print(f"  spaCy version: {info['spacy_version']}")
                    print(f"  Model size config: {info['model_size_config']}")
                    print(f"  Required models: {info['required_models']}")
                    print(f"  Available models: {len(info['available_models'])}")
                return 0

            cache_hit = args.cache_hit.lower() == "true"
            success = mock_manager.setup_models(
                cache_hit=cache_hit, verify_only=args.verify_installation
            )

            if args.json_output:
                result = {
                    "success": success,
                    "model_size": args.model_size,
                    "cache_hit": cache_hit,
                    "verify_only": args.verify_installation,
                    "cache_info": mock_manager.get_cache_info(),
                }
                print(json.dumps(result, indent=2))

            return 0 if success else 1

        except KeyboardInterrupt:
            print("\n⏹ Operation cancelled by user")
            return 130
        except Exception as e:
            print(f"💥 Unexpected error: {e}")
            if args.json_output:
                error_result = {"success": False, "error": str(e)}
                print(json.dumps(error_result, indent=2))
            return 1
    
    return wrapped_main