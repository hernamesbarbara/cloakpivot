#!/usr/bin/env python3
"""Intelligent model download and verification manager for CI/CD caching.

This script provides cache-aware model management for spaCy and HuggingFace models,
optimizing CI/CD pipeline performance through intelligent caching strategies.
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

try:
    import spacy
    from spacy.cli.download import download as spacy_download
except ImportError:
    print("ERROR: spaCy not installed. Run 'pip install spacy' first.")
    sys.exit(1)


class ModelManager:
    """Intelligent model download and verification manager with caching awareness."""

    def __init__(self, model_size: str = "small", cache_dir: Optional[str] = None):
        self.model_size = model_size
        self.cache_dir = Path(cache_dir or Path.home() / ".cache" / "spacy")
        self.spacy_models_dir = Path.home() / "spacy_models"
        self.start_time = time.time()

    def get_required_models(self) -> list[str]:
        """Get list of models required for current configuration."""
        base_models = {
            "small": ["en_core_web_sm"],
            "medium": ["en_core_web_sm", "en_core_web_md"],
            "large": ["en_core_web_sm", "en_core_web_md", "en_core_web_lg"],
        }
        return base_models.get(self.model_size, ["en_core_web_sm"])

    def is_model_available(self, model_name: str) -> bool:
        """Check if model is available and functional with comprehensive validation."""
        try:
            print(f"  Verifying model {model_name}...")
            nlp = spacy.load(model_name)

            # Comprehensive validation tests
            test_text = (
                "This is a test document for validation. John Smith works at Microsoft."
            )
            doc = nlp(test_text)

            # Verify basic pipeline components work
            if len(doc) == 0:
                print(f"  ✗ Model {model_name}: Failed basic tokenization")
                return False

            # Test NER if available
            if doc.ents:
                print(f"  ✓ Model {model_name}: NER working ({len(doc.ents)} entities)")

            # Test POS tagging
            pos_tags = [token.pos_ for token in doc if token.pos_]
            if not pos_tags:
                print(f"  ✗ Model {model_name}: POS tagging not working")
                return False

            print(f"  ✓ Model {model_name}: All components validated")
            return True

        except (OSError, AttributeError) as e:
            print(f"  ✗ Model {model_name}: Validation failed - {e}")
            return False
        except Exception as e:
            print(f"  ✗ Model {model_name}: Unexpected validation error - {e}")
            return False

    def download_model(self, model_name: str, force: bool = False) -> bool:
        """Download model with progress indication and verification."""
        if not force and self.is_model_available(model_name):
            print(f"✓ Model {model_name} already available and functional")
            return True

        print(f"📦 Downloading model: {model_name}")
        download_start = time.time()

        try:
            # Use spacy download command
            spacy_download(model_name)

            download_time = time.time() - download_start
            print(f"  Download completed in {download_time:.1f}s")

            # Verify installation immediately after download
            if self.is_model_available(model_name):
                print(f"✓ Model {model_name} downloaded and verified successfully")
                return True
            else:
                print(f"✗ Model {model_name} download failed verification")
                return False

        except Exception as e:
            download_time = time.time() - download_start
            print(f"✗ Failed to download {model_name} after {download_time:.1f}s: {e}")
            return False

    def setup_models(self, cache_hit: bool = False, verify_only: bool = False) -> bool:
        """Setup all required models with caching awareness."""
        required_models = self.get_required_models()

        print(f"🚀 Setting up models for size: {self.model_size}")
        print(f"📋 Required models: {required_models}")
        print(f"💾 Cache hit detected: {cache_hit}")
        print(f"📁 Cache directory: {self.cache_dir}")

        if cache_hit and not verify_only:
            print("🔍 Cache hit detected - verifying cached models...")

        success_count = 0
        failed_models = []

        for model_name in required_models:
            print(f"\n--- Processing {model_name} ---")

            if cache_hit and not verify_only:
                # Verify cached model first
                if self.is_model_available(model_name):
                    print(f"✓ Cached model {model_name} verified and ready")
                    success_count += 1
                else:
                    print(
                        f"⚠ Cached model {model_name} verification failed, re-downloading..."
                    )
                    if self.download_model(model_name, force=True):
                        success_count += 1
                    else:
                        failed_models.append(model_name)
            else:
                # Fresh download or verification only
                if verify_only:
                    if self.is_model_available(model_name):
                        success_count += 1
                    else:
                        failed_models.append(model_name)
                else:
                    if self.download_model(model_name):
                        success_count += 1
                    else:
                        failed_models.append(model_name)

        total_time = time.time() - self.start_time

        print("\n🏁 Model setup summary:")
        print(f"  ✓ Successful: {success_count}/{len(required_models)} models")
        print(f"  ⏱ Total time: {total_time:.1f}s")

        if failed_models:
            print(f"  ✗ Failed models: {failed_models}")
            return False
        else:
            print(f"  🎉 All {success_count} models ready for use")
            return True

    def get_cache_info(self) -> dict:
        """Get comprehensive information about cached models and system state."""
        info = {
            "cache_dir": str(self.cache_dir),
            "models_dir": str(self.spacy_models_dir),
            "available_models": [],
            "spacy_version": spacy.__version__,
            "model_size_config": self.model_size,
            "required_models": self.get_required_models(),
            "system_info": {
                "python_version": sys.version,
                "platform": sys.platform,
            },
        }

        # Check available models
        try:
            for model_name in self.get_required_models():
                if self.is_model_available(model_name):
                    try:
                        nlp = spacy.load(model_name)
                        model_info = {
                            "name": model_name,
                            "lang": nlp.lang,
                            "pipeline": nlp.pipe_names,
                            "version": getattr(nlp.meta, "version", "unknown"),
                        }
                        info["available_models"].append(model_info)
                    except Exception as e:
                        info["available_models"].append(
                            {"name": model_name, "error": str(e)}
                        )
        except Exception as e:
            info["error"] = f"Failed to enumerate models: {e}"

        return info


def main():
    parser = argparse.ArgumentParser(
        description="Setup models for CloakPivot testing with intelligent caching",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --model-size small --cache-hit true
  %(prog)s --model-size medium --verify-installation
  %(prog)s --cache-info
        """,
    )

    parser.add_argument(
        "--model-size",
        choices=["small", "medium", "large"],
        default="small",
        help="Model size to download (default: small)",
    )
    parser.add_argument(
        "--cache-hit",
        type=str,
        default="false",
        help="Whether cache was hit (true/false, default: false)",
    )
    parser.add_argument(
        "--verify-installation",
        action="store_true",
        help="Verify model installation without downloading",
    )
    parser.add_argument(
        "--cache-info",
        action="store_true",
        help="Display comprehensive cache and model information",
    )
    parser.add_argument("--cache-dir", type=str, help="Custom cache directory path")
    parser.add_argument(
        "--json-output", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    try:
        manager = ModelManager(model_size=args.model_size, cache_dir=args.cache_dir)

        if args.cache_info:
            info = manager.get_cache_info()
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

                for model in info["available_models"]:
                    if "error" in model:
                        print(f"    ✗ {model['name']}: {model['error']}")
                    else:
                        print(
                            f"    ✓ {model['name']} (v{model.get('version', '?')}) - {model.get('pipeline', [])}"
                        )
            return 0

        cache_hit = args.cache_hit.lower() == "true"
        success = manager.setup_models(
            cache_hit=cache_hit, verify_only=args.verify_installation
        )

        if args.json_output:
            result = {
                "success": success,
                "model_size": args.model_size,
                "cache_hit": cache_hit,
                "verify_only": args.verify_installation,
                "cache_info": manager.get_cache_info(),
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


if __name__ == "__main__":
    sys.exit(main())
