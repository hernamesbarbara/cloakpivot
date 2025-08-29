"""Main CLI entry point for CloakPivot."""

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Optional, TextIO

import click

from cloakpivot import __version__

if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

    from ..core.detection import DocumentAnalysisResult
    from ..core.policies import MaskingPolicy
    from ..masking.engine import MaskingResult

    try:
        from docling_core.types import DoclingDocument
    except ImportError:
        DoclingDocument = None


@click.group()
@click.version_option(version=__version__, prog_name="cloakpivot")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """CloakPivot: PII masking/unmasking on top of DocPivot and Presidio.

    CloakPivot enables reversible document masking while preserving
    structure and formatting using DocPivot and Presidio.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose


def _validate_mask_arguments(
    output_path: Optional[Path], cloakmap: Optional[Path]
) -> None:
    """Validate mask command arguments."""
    if not output_path and not cloakmap:
        raise click.ClickException(
            "Must specify either --out for masked output or --cloakmap for CloakMap output"
        )


def _set_default_paths(
    input_path: Path,
    output_path: Optional[Path],
    cloakmap: Optional[Path],
    output_format: str,
) -> tuple[Path, Path]:
    """Set default output paths if not specified."""
    if not output_path:
        output_path = input_path.with_suffix(f".masked.{output_format}.json")
    if not cloakmap:
        cloakmap = input_path.with_suffix(".cloakmap.json")
    return output_path, cloakmap


def _load_masking_policy(policy: Optional[Path], verbose: bool) -> "MaskingPolicy":
    """Load masking policy from file or use default with enhanced inheritance support."""
    from ..core.policies import MaskingPolicy
    from ..core.policy_loader import (
        PolicyInheritanceError,
        PolicyLoader,
        PolicyValidationError,
    )

    if policy:
        if verbose:
            click.echo(f"📋 Loading policy: {policy}")

        try:
            # Try new enhanced policy loader first
            loader = PolicyLoader()
            masking_policy = loader.load_policy(policy)
            if verbose:
                click.echo("✓ Enhanced policy loaded successfully")
                if hasattr(masking_policy, "name") or policy.name.endswith(
                    (".yaml", ".yml")
                ):
                    click.echo("   (with inheritance and validation support)")
        except (PolicyValidationError, PolicyInheritanceError) as e:
            click.echo(f"⚠️  Enhanced policy loading failed: {e}")
            click.echo("   Falling back to basic policy loading")
            # Fall back to basic loading
            try:
                import yaml

                with open(policy, encoding="utf-8") as f:
                    policy_data = yaml.safe_load(f)
                masking_policy = MaskingPolicy.from_dict(policy_data)
                if verbose:
                    click.echo("✓ Basic policy loaded successfully")
            except ImportError:
                click.echo("⚠️  PyYAML not installed, using default policy")
                masking_policy = MaskingPolicy()
            except Exception as e:
                click.echo(f"⚠️  Failed to load policy file: {e}")
                click.echo("   Using default policy")
                masking_policy = MaskingPolicy()
        except ImportError:
            click.echo("⚠️  Required dependencies not available for enhanced policies")
            # Fall back to basic loading
            try:
                import yaml

                with open(policy, encoding="utf-8") as f:
                    policy_data = yaml.safe_load(f)
                masking_policy = MaskingPolicy.from_dict(policy_data)
                if verbose:
                    click.echo("✓ Basic policy loaded successfully")
            except ImportError:
                click.echo("⚠️  PyYAML not installed, using default policy")
                masking_policy = MaskingPolicy()
            except Exception as e:
                click.echo(f"⚠️  Failed to load policy file: {e}")
                click.echo("   Using default policy")
                masking_policy = MaskingPolicy()
        except Exception as e:
            click.echo(f"⚠️  Unexpected error loading policy: {e}")
            click.echo("   Using default policy")
            masking_policy = MaskingPolicy()
    else:
        masking_policy = MaskingPolicy()
    return masking_policy


def _perform_entity_detection(
    document: "DoclingDocument", masking_policy: "MaskingPolicy", verbose: bool
) -> "DocumentAnalysisResult":
    """Perform PII entity detection on document."""
    from ..core.detection import EntityDetectionPipeline

    if verbose:
        click.echo("🔍 Detecting PII entities")

    detection_pipeline = EntityDetectionPipeline()

    with click.progressbar(length=1, label="Analyzing document for PII") as progress:
        detection_result = detection_pipeline.analyze_document(document, masking_policy)
        progress.update(1)

    if verbose:
        click.echo(f"✓ Detected {detection_result.total_entities} entities")
        for entity_type, count in detection_result.entity_breakdown.items():
            click.echo(f"  {entity_type}: {count}")

    if detection_result.total_entities == 0:
        click.echo("ℹ️  No PII entities detected in document")
        if not click.confirm("Continue with masking anyway?"):
            raise click.Abort()

    return detection_result


def _prepare_entities_for_masking(
    detection_result: "DocumentAnalysisResult",
) -> list["RecognizerResult"]:
    """Convert detection results to format expected by masking engine."""
    from presidio_analyzer import RecognizerResult

    entities = []
    for segment_result in detection_result.segment_results:
        for entity in segment_result.entities:
            recognizer_result = RecognizerResult(
                entity_type=entity.entity_type,
                start=entity.start + segment_result.segment.start_offset,
                end=entity.end + segment_result.segment.start_offset,
                score=entity.confidence,
            )
            entities.append(recognizer_result)
    return entities


def _perform_masking(
    document: "DoclingDocument",
    entities: list["RecognizerResult"],
    masking_policy: "MaskingPolicy",
    verbose: bool,
) -> "MaskingResult":
    """Perform the actual masking operation."""
    from ..document.extractor import TextExtractor
    from ..masking.engine import MaskingEngine

    masking_engine = MaskingEngine()
    text_extractor = TextExtractor()
    text_segments = text_extractor.extract_text_segments(document)

    if verbose:
        click.echo(f"📝 Extracted {len(text_segments)} text segments")

    with click.progressbar(length=1, label="Masking PII entities") as progress:
        masking_result = masking_engine.mask_document(
            document=document,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments,
        )
        progress.update(1)

    if verbose:
        click.echo(f"✓ Masked {len(masking_result.cloakmap.anchors)} entities")

    return masking_result


def _save_masked_document(
    masking_result: "MaskingResult", output_path: Path, verbose: bool
) -> None:
    """Save the masked document to file."""
    if verbose:
        click.echo(f"💾 Saving masked document: {output_path}")

    with click.progressbar(length=1, label="Saving masked document") as progress:
        from docpivot import LexicalDocSerializer

        serializer = LexicalDocSerializer()
        serialized_content = serializer.serialize(masking_result.masked_document)

        with open(output_path, "w", encoding="utf-8") as f:
            f.write(serialized_content)
        progress.update(1)


def _save_cloakmap(
    masking_result: "MaskingResult", cloakmap: Path, verbose: bool
) -> None:
    """Save the CloakMap to file."""
    import json

    if verbose:
        click.echo(f"🗺️  Saving CloakMap: {cloakmap}")

    with click.progressbar(length=1, label="Saving CloakMap") as progress:
        with open(cloakmap, "w", encoding="utf-8") as f:
            json.dump(masking_result.cloakmap.to_dict(), f, indent=2, default=str)
        progress.update(1)


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--out",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output path or format (lexical, markdown, html)",
)
@click.option(
    "--cloakmap", type=click.Path(path_type=Path), help="Path to save the CloakMap file"
)
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Path to masking policy file",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["lexical", "docling", "markdown", "html"]),
    default="lexical",
    help="Output format",
)
@click.option("--lang", default="en", help="Language code for analysis")
@click.option(
    "--min-score",
    type=float,
    default=0.5,
    help="Minimum confidence score for PII detection",
)
@click.option("--encrypt", is_flag=True, help="Encrypt the CloakMap")
@click.option("--key-id", help="Key ID for encryption")
@click.pass_context
def mask(
    ctx: click.Context,
    input_path: Path,
    output_path: Optional[Path],
    cloakmap: Optional[Path],
    policy: Optional[Path],
    output_format: str,
    lang: str,
    min_score: float,
    encrypt: bool,
    key_id: Optional[str],
) -> None:
    """Mask PII in a document while preserving structure.

    Takes a document as INPUT_PATH and creates a masked version with
    a CloakMap for later unmasking.

    Example:
        cloakpivot mask document.pdf --out masked.json --cloakmap map.json
    """
    verbose = ctx.obj["verbose"]

    try:
        # Import required modules
        import json

        from ..core.detection import EntityDetectionPipeline
        from ..core.policies import MaskingPolicy
        from ..document.processor import DocumentProcessor
        from ..masking.engine import MaskingEngine

        if verbose:
            click.echo(f"🔍 Loading document: {input_path}")

        # Validate input arguments
        if not output_path and not cloakmap:
            raise click.ClickException(
                "Must specify either --out for masked output or --cloakmap for CloakMap output"
            )

        # Set default output paths if not specified
        if not output_path:
            output_path = input_path.with_suffix(f".masked.{output_format}.json")
        if not cloakmap:
            cloakmap = input_path.with_suffix(".cloakmap.json")

        # Load document
        with click.progressbar(length=1, label="Loading document") as progress:
            processor = DocumentProcessor()
            document = processor.load_document(input_path, validate=True)
            progress.update(1)

        if verbose:
            click.echo(f"✓ Loaded document: {document.name}")
            click.echo(f"  Text items: {len(document.texts)}")
            click.echo(f"  Tables: {len(document.tables)}")

        # Load or create masking policy
        if policy:
            if verbose:
                click.echo(f"📋 Loading policy: {policy}")
            try:
                import yaml

                with open(policy, encoding="utf-8") as f:
                    policy_data = yaml.safe_load(f)
                masking_policy = MaskingPolicy.from_dict(policy_data)
                if verbose:
                    click.echo("✓ Custom policy loaded successfully")
            except ImportError:
                click.echo("⚠️  PyYAML not installed, using default policy")
                masking_policy = MaskingPolicy()
            except Exception as e:
                click.echo(f"⚠️  Failed to load policy file: {e}")
                click.echo("   Using default policy")
                masking_policy = MaskingPolicy()
        else:
            masking_policy = MaskingPolicy()  # Use default policy

        # Initialize detection pipeline
        if verbose:
            click.echo(
                f"🔍 Detecting PII entities (min_score: {min_score}, lang: {lang})"
            )

        detection_pipeline = EntityDetectionPipeline()

        # Detect entities with progress
        with click.progressbar(
            length=1, label="Analyzing document for PII"
        ) as progress:
            detection_result = detection_pipeline.analyze_document(
                document, masking_policy
            )
            progress.update(1)

        if verbose:
            click.echo(f"✓ Detected {detection_result.total_entities} entities")
            for entity_type, count in detection_result.entity_breakdown.items():
                click.echo(f"  {entity_type}: {count}")

        if detection_result.total_entities == 0:
            click.echo("ℹ️  No PII entities detected in document")
            if not click.confirm("Continue with masking anyway?"):
                raise click.Abort()

        # Initialize masking engine
        masking_engine = MaskingEngine()

        # Extract text segments (needed for masking engine)
        from ..document.extractor import TextExtractor

        text_extractor = TextExtractor()
        text_segments = text_extractor.extract_text_segments(document)

        if verbose:
            click.echo(f"📝 Extracted {len(text_segments)} text segments")

        # Convert detection results to RecognizerResult format expected by masking engine
        from presidio_analyzer import RecognizerResult

        entities = []
        for segment_result in detection_result.segment_results:
            for entity in segment_result.entities:
                # Convert to RecognizerResult
                recognizer_result = RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start
                    + segment_result.segment.start_offset,  # Convert to global offset
                    end=entity.end + segment_result.segment.start_offset,
                    score=entity.confidence,
                )
                entities.append(recognizer_result)

        # Perform masking
        with click.progressbar(length=1, label="Masking PII entities") as progress:
            masking_result = masking_engine.mask_document(
                document=document,
                entities=entities,
                policy=masking_policy,
                text_segments=text_segments,
            )
            progress.update(1)

        if verbose:
            click.echo(f"✓ Masked {len(masking_result.cloakmap.anchors)} entities")

        # Save masked document
        if verbose:
            click.echo(f"💾 Saving masked document: {output_path}")

        with click.progressbar(length=1, label="Saving masked document") as progress:
            # Use docpivot serializer to save the document
            from docpivot import LexicalDocSerializer

            serializer = LexicalDocSerializer(masking_result.masked_document)
            result = serializer.serialize()
            serialized_content = result.text

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(serialized_content)
            progress.update(1)

        # Save CloakMap
        if verbose:
            click.echo(f"🗺️  Saving CloakMap: {cloakmap}")

        with click.progressbar(length=1, label="Saving CloakMap") as progress:
            with open(cloakmap, "w", encoding="utf-8") as f:
                json.dump(masking_result.cloakmap.to_dict(), f, indent=2, default=str)
            progress.update(1)

        # Success message
        click.echo("✅ Masking completed successfully!")
        click.echo(f"   Masked document: {output_path}")
        click.echo(f"   CloakMap: {cloakmap}")
        if masking_result.stats:
            click.echo(
                f"   Entities processed: {masking_result.stats['total_entities_masked']}"
            )

    except ImportError as e:
        raise click.ClickException(f"Missing required dependency: {e}") from e
    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}") from e
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Masking failed: {e}") from e


@cli.command()
@click.argument("masked_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--cloakmap",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to the CloakMap file",
)
@click.option(
    "--out",
    "-o",
    "output_path",
    type=click.Path(path_type=Path),
    help="Output path or format",
)
@click.option(
    "--verify-only", is_flag=True, help="Only verify the CloakMap without unmasking"
)
@click.pass_context
def unmask(
    ctx: click.Context,
    masked_path: Path,
    cloakmap: Path,
    output_path: Optional[Path],
    verify_only: bool,
) -> None:
    """Unmask a previously masked document using its CloakMap.

    Takes a masked document and its CloakMap to restore the original.

    Example:
        cloakpivot unmask masked.json --cloakmap map.json --out original.json
    """
    verbose = ctx.obj["verbose"]

    try:
        # Import required modules
        import json

        from ..core.cloakmap import CloakMap
        from ..document.processor import DocumentProcessor
        from ..unmasking.engine import UnmaskingEngine

        if verbose:
            click.echo(f"🔓 Loading masked document: {masked_path}")
            click.echo(f"🗺️  Loading CloakMap: {cloakmap}")
            if verify_only:
                click.echo("🔍 Verification mode only")

        # Set default output path if not specified
        if not output_path and not verify_only:
            output_path = masked_path.with_suffix(".restored.json")

        # Load CloakMap first for validation
        with click.progressbar(length=1, label="Loading CloakMap") as progress:
            with open(cloakmap, encoding="utf-8") as f:
                cloakmap_data = json.load(f)
            cloakmap_obj = CloakMap.from_dict(cloakmap_data)
            progress.update(1)

        if verbose:
            click.echo(f"✓ Loaded CloakMap with {len(cloakmap_obj.anchors)} anchors")
            click.echo(f"  Document ID: {cloakmap_obj.doc_id}")
            click.echo(f"  Version: {cloakmap_obj.version}")

        # Load masked document
        with click.progressbar(length=1, label="Loading masked document") as progress:
            processor = DocumentProcessor()
            masked_document = processor.load_document(masked_path, validate=True)
            progress.update(1)

        if verbose:
            click.echo(f"✓ Loaded masked document: {masked_document.name}")

        # Initialize unmasking engine
        unmasking_engine = UnmaskingEngine()

        # Verify document compatibility
        if verbose:
            click.echo("🔍 Verifying document compatibility")

        try:
            # Attempt unmasking or verification
            with click.progressbar(length=1, label="Processing unmasking") as progress:
                unmasking_result = unmasking_engine.unmask_document(
                    masked_document=masked_document,
                    cloakmap=cloakmap_obj,
                    verify_integrity=True,
                )
                progress.update(1)

            # Show statistics
            if verbose and unmasking_result.stats:
                click.echo("✓ Unmasking completed")
                stats = unmasking_result.stats
                click.echo(f"  Success rate: {stats.get('success_rate', 0):.1f}%")
                click.echo(f"  Anchors resolved: {stats.get('resolved_anchors', 0)}")
                click.echo(f"  Anchors failed: {stats.get('failed_anchors', 0)}")

            # Check integrity
            if unmasking_result.integrity_report:
                integrity = unmasking_result.integrity_report
                if integrity.get("valid", False):
                    if verbose:
                        click.echo("✅ Integrity verification passed")
                else:
                    click.echo("⚠️  Integrity verification failed:")
                    for issue in integrity.get("issues", []):
                        click.echo(f"   - {issue}")

            if verify_only:
                # Verification mode - just report results
                if (
                    unmasking_result.integrity_report
                    and unmasking_result.integrity_report.get("valid", False)
                ):
                    click.echo("✅ CloakMap verification successful")
                    click.echo(f"   Document: {masked_path}")
                    click.echo(f"   CloakMap: {cloakmap}")
                    click.echo(f"   Anchors: {len(cloakmap_obj.anchors)}")
                else:
                    click.echo("❌ CloakMap verification failed")
                    raise click.ClickException("Verification failed - see issues above")
            else:
                # Save restored document
                if output_path is None:
                    raise click.ClickException(
                        "Output path is required for unmasking (use --out option)"
                    )

                if verbose:
                    click.echo(f"💾 Saving restored document: {output_path}")

                with click.progressbar(
                    length=1, label="Saving restored document"
                ) as progress:
                    # Use docpivot serializer to save the restored document
                    from docpivot import LexicalDocSerializer

                    serializer = LexicalDocSerializer(
                        unmasking_result.restored_document
                    )
                    result = serializer.serialize()
                    serialized_content = result.text

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(serialized_content)
                    progress.update(1)

                # Success message
                click.echo("✅ Unmasking completed successfully!")
                click.echo(f"   Masked document: {masked_path}")
                click.echo(f"   CloakMap: {cloakmap}")
                click.echo(f"   Restored document: {output_path}")
                if unmasking_result.stats:
                    click.echo(
                        f"   Entities restored: {unmasking_result.stats.get('resolved_anchors', 0)}"
                    )

        except Exception as unmasking_error:
            # Handle unmasking-specific errors
            error_msg = str(unmasking_error)
            if "compatibility" in error_msg.lower():
                click.echo("❌ Document-CloakMap compatibility issue")
                click.echo("   The CloakMap may not match the provided masked document")
            elif "anchor" in error_msg.lower():
                click.echo("❌ Anchor resolution failed")
                click.echo(
                    "   Some replacement tokens could not be located in the document"
                )
            else:
                click.echo(f"❌ Unmasking failed: {error_msg}")

            if verbose:
                import traceback

                click.echo(f"\nError details:\n{traceback.format_exc()}")

            raise click.ClickException(
                f"Unmasking operation failed: {error_msg}"
            ) from None

    except ImportError as e:
        raise click.ClickException(f"Missing required dependency: {e}") from e
    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}") from e
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid CloakMap file format: {e}") from e
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Unmasking failed: {e}") from e


@cli.group()
def policy() -> None:
    """Manage masking policies."""
    pass


@policy.command("sample")
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)",
)
def policy_sample(output: TextIO) -> None:
    """Generate a sample masking policy file.

    Example:
        cloakpivot policy sample > policy.yaml
    """
    sample_policy = """# CloakPivot Enhanced Masking Policy Configuration
# This sample demonstrates the comprehensive policy configuration system with
# inheritance support, per-entity configuration, and advanced features

version: "1.0"
name: "sample-policy"
description: "Sample policy demonstrating all available configuration options"

# Optional: inherit from base template (uncomment to enable)
# extends: "templates/balanced.yaml"

# Core configuration
locale: "en"
seed: "sample-policy-seed-v1"  # For deterministic operations
min_entity_length: 2

# Default strategy applied to entities without specific overrides
default_strategy:
  kind: "redact"
  parameters:
    redact_char: "*"
    preserve_length: true

# Per-entity type configuration with full options
per_entity:
  PERSON:
    kind: "hash"
    parameters:
      algorithm: "sha256"
      truncate: 8
      prefix: "PERSON_"
      per_entity_salt:
        PERSON: "person_salt_v1"
    threshold: 0.75
    enabled: true

  EMAIL_ADDRESS:
    kind: "partial"
    parameters:
      visible_chars: 3
      position: "start"
      format_aware: true
      preserve_delimiters: true
      deterministic: true
    threshold: 0.7
    enabled: true

  PHONE_NUMBER:
    kind: "partial"
    parameters:
      visible_chars: 4
      position: "end"
      format_aware: true
      preserve_delimiters: true
    threshold: 0.8
    enabled: true

  CREDIT_CARD:
    kind: "template"
    parameters:
      template: "[CARD-REDACTED]"
    threshold: 0.9
    enabled: true

  US_SSN:
    kind: "template"
    parameters:
      template: "XXX-XX-XXXX"
      preserve_format: true
    threshold: 0.9
    enabled: true

# Global thresholds (can be overridden in per_entity)
thresholds:
  IP_ADDRESS: 0.6
  URL: 0.6

# Context-specific masking rules
context_rules:
  heading:
    enabled: false  # Don't mask in document headings

  table:
    enabled: true
    threshold_overrides:
      PERSON: 0.8  # Higher threshold in tables

  footer:
    enabled: true

  header:
    enabled: false

# Allow list - values that should never be masked
allow_list:
  - "support@company.com"
  - "Company Name"
  # Pattern-based matching (requires enhanced loader)
  - pattern: ".*@company\\.com$"

# Deny list - values that should always be masked regardless of confidence
deny_list:
  - "confidential"
  - "internal use only"

# Locale-specific configuration (requires enhanced loader)
locales:
  "es":
    recognizers: ["SPANISH_DNI", "SPANISH_PHONE"]
    entity_overrides:
      PERSON:
        threshold: 0.75

  "fr":
    recognizers: ["FRENCH_CNI"]
    entity_overrides:
      PERSON:
        threshold: 0.75

# Policy composition settings (for inheritance)
policy_composition:
  merge_strategy: "override"  # override, merge, strict
  validation_level: "warn"    # strict, warn, permissive

# Note: This enhanced format requires the PolicyLoader class
# For basic compatibility, use simple key-value format without nested structure
"""
    output.write(sample_policy)
    if output != sys.stdout:
        click.echo(f"Sample policy written to {output.name}")


@policy.command("validate")
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed validation information"
)
def policy_validate(policy_file: Path, verbose: bool) -> None:
    """Validate a policy file for errors and compatibility.

    Example:
        cloakpivot policy validate my-policy.yaml
    """
    try:
        from ..core.policy_loader import PolicyLoader

        if verbose:
            click.echo(f"🔍 Validating policy file: {policy_file}")

        loader = PolicyLoader()
        errors = loader.validate_policy_file(policy_file)

        if not errors:
            click.echo("✅ Policy file is valid")
            if verbose:
                # Load and show policy summary
                policy = loader.load_policy(policy_file)
                click.echo(f"   Locale: {policy.locale}")
                click.echo(f"   Entity strategies: {len(policy.per_entity)}")
                click.echo(f"   Context rules: {len(policy.context_rules)}")
                click.echo(f"   Allow list items: {len(policy.allow_list)}")
                click.echo(f"   Deny list items: {len(policy.deny_list)}")
        else:
            click.echo("❌ Policy validation failed:")
            for error in errors:
                click.echo(f"   • {error}")
            raise click.ClickException("Policy validation failed")

    except ImportError:
        click.echo("⚠️  Enhanced policy validation requires pydantic")
        click.echo("   Install with: pip install pydantic")
        raise click.ClickException("Missing required dependency") from None


@policy.command("template")
@click.argument(
    "template_name", type=click.Choice(["conservative", "balanced", "permissive"])
)
@click.option(
    "--output",
    "-o",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)",
)
def policy_template(template_name: str, output: TextIO) -> None:
    """Generate a policy file from a built-in template.

    Available templates:
    - conservative: High security with strict thresholds
    - balanced: Reasonable security with good usability
    - permissive: Low security for development/testing

    Example:
        cloakpivot policy template balanced > my-policy.yaml
    """
    import pkg_resources

    try:
        template_path = f"policies/templates/{template_name}.yaml"
        template_content = pkg_resources.resource_string(
            "cloakpivot", template_path
        ).decode("utf-8")
        output.write(template_content)

        if output != sys.stdout:
            click.echo(f"✅ {template_name.title()} template written to {output.name}")

    except FileNotFoundError:
        # Fall back to reading from file system if package resource doesn't work
        from pathlib import Path

        template_file = (
            Path(__file__).parent.parent
            / "policies"
            / "templates"
            / f"{template_name}.yaml"
        )

        if template_file.exists():
            with open(template_file, encoding="utf-8") as f:
                template_content = f.read()
            output.write(template_content)

            if output != sys.stdout:
                click.echo(
                    f"✅ {template_name.title()} template written to {output.name}"
                )
        else:
            raise click.ClickException(
                f"Template '{template_name}' not found"
            ) from None
    except Exception as e:
        raise click.ClickException(f"Failed to load template: {e}") from e


@policy.command("test")
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
@click.option("--text", "-t", help="Test text to analyze with the policy")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis results")
def policy_test(policy_file: Path, text: Optional[str], verbose: bool) -> None:
    """Test a policy against sample text to see masking behavior.

    Example:
        cloakpivot policy test policy.yaml --text "John Doe's email is john@example.com"
    """
    try:
        from ..core.policy_loader import PolicyLoader

        # Load policy
        if verbose:
            click.echo(f"📋 Loading policy: {policy_file}")

        loader = PolicyLoader()
        policy = loader.load_policy(policy_file)

        # Use sample text if none provided
        if not text:
            text = "Contact John Doe at john.doe@example.com or call (555) 123-4567. His SSN is 123-45-6789."
            click.echo("📝 Using sample text for testing:")
            click.echo(f"   {text}")

        # Detect entities
        if verbose:
            click.echo("🔍 Detecting PII entities...")

        # Create a simple document-like structure for testing
        class TestDocument:
            def __init__(self, text: str):
                self.text = text

        # This is a simplified test - in practice would use full document masking pipeline
        click.echo("🎭 Policy test results:")
        click.echo(f"   Default strategy: {policy.default_strategy.kind.value}")
        click.echo(f"   Locale: {policy.locale}")
        click.echo(f"   Per-entity strategies: {len(policy.per_entity)}")

        if verbose:
            click.echo("📊 Configured entity strategies:")
            for entity_type, strategy in policy.per_entity.items():
                threshold = policy.thresholds.get(entity_type, 0.5)
                click.echo(
                    f"   • {entity_type}: {strategy.kind.value} (threshold: {threshold})"
                )

        click.echo("ℹ️  Full masking test requires document input")
        click.echo("   Use: cloakpivot mask <document> --policy <policy> --verbose")

    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Policy test failed: {e}") from e


@policy.command("info")
@click.argument("policy_file", type=click.Path(exists=True, path_type=Path))
def policy_info(policy_file: Path) -> None:
    """Show detailed information about a policy file.

    Example:
        cloakpivot policy info my-policy.yaml
    """
    try:
        from ..core.policy_loader import PolicyLoader

        loader = PolicyLoader()
        policy = loader.load_policy(policy_file)

        click.echo(f"📋 Policy Information: {policy_file.name}")
        click.echo("=" * 50)

        # Basic info
        click.echo(f"Locale: {policy.locale}")
        if policy.seed:
            click.echo(f"Seed: {policy.seed}")
        click.echo(f"Min entity length: {policy.min_entity_length}")

        # Default strategy
        click.echo(f"\nDefault Strategy: {policy.default_strategy.kind.value}")
        if policy.default_strategy.parameters:
            for key, value in policy.default_strategy.parameters.items():
                click.echo(f"  {key}: {value}")

        # Per-entity strategies
        if policy.per_entity:
            click.echo(f"\nPer-Entity Strategies ({len(policy.per_entity)}):")
            for entity_type, strategy in policy.per_entity.items():
                threshold = policy.thresholds.get(entity_type, "default")
                click.echo(
                    f"  • {entity_type}: {strategy.kind.value} (threshold: {threshold})"
                )
                if strategy.parameters:
                    for key, value in strategy.parameters.items():
                        click.echo(f"    {key}: {value}")

        # Context rules
        if policy.context_rules:
            click.echo(f"\nContext Rules ({len(policy.context_rules)}):")
            for context, rules in policy.context_rules.items():
                enabled = rules.get("enabled", True)
                status = "enabled" if enabled else "disabled"
                click.echo(f"  • {context}: {status}")

        # Allow/deny lists
        if policy.allow_list:
            click.echo(f"\nAllow List ({len(policy.allow_list)} items):")
            for item in list(policy.allow_list)[:5]:  # Show first 5
                click.echo(f"  • {item}")
            if len(policy.allow_list) > 5:
                click.echo(f"  ... and {len(policy.allow_list) - 5} more")

        if policy.deny_list:
            click.echo(f"\nDeny List ({len(policy.deny_list)} items):")
            for item in list(policy.deny_list)[:5]:  # Show first 5
                click.echo(f"  • {item}")
            if len(policy.deny_list) > 5:
                click.echo(f"  ... and {len(policy.deny_list) - 5} more")

    except Exception as e:
        raise click.ClickException(f"Failed to read policy info: {e}") from e


@cli.group()
def diagnostics() -> None:
    """Generate diagnostic reports from masking operations."""
    pass


@diagnostics.command("analyze")
@click.argument("masked_file", type=click.Path(exists=True, path_type=Path))
@click.argument("cloakmap_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output path for diagnostic report",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["json", "html", "markdown"]),
    default="html",
    help="Report format (default: html)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def diagnostics_analyze(
    ctx: click.Context,
    masked_file: Path,
    cloakmap_file: Path,
    output: Optional[Path],
    report_format: str,
    verbose: bool,
) -> None:
    """Analyze masking results and generate diagnostic reports.

    Analyzes a masked document and its CloakMap to generate comprehensive
    diagnostic reports with statistics, coverage analysis, and recommendations.

    Example:
        cloakpivot diagnostics analyze masked.json map.json -o report.html
    """
    try:
        import json

        from ..core.cloakmap import CloakMap
        from ..diagnostics import (
            CoverageAnalyzer,
            DiagnosticReporter,
            DiagnosticsCollector,
            ReportData,
            ReportFormat,
        )
        from ..document.extractor import TextExtractor
        from ..document.processor import DocumentProcessor

        verbose = verbose or ctx.obj.get("verbose", False)

        if verbose:
            click.echo("📊 Analyzing masking results")
            click.echo(f"   Masked file: {masked_file}")
            click.echo(f"   CloakMap: {cloakmap_file}")

        # Validate file existence and readability
        if not masked_file.exists():
            raise click.ClickException(f"Masked file does not exist: {masked_file}")
        if not cloakmap_file.exists():
            raise click.ClickException(f"CloakMap file does not exist: {cloakmap_file}")

        # Load CloakMap with comprehensive error handling
        try:
            with click.progressbar(length=1, label="Loading CloakMap") as progress:
                with open(cloakmap_file, encoding="utf-8") as f:
                    cloakmap_data = json.load(f)
                
                # Validate CloakMap structure
                if not isinstance(cloakmap_data, dict):
                    raise click.ClickException(f"Invalid CloakMap format: expected JSON object, got {type(cloakmap_data).__name__}")
                
                if "anchors" not in cloakmap_data:
                    raise click.ClickException("Invalid CloakMap: missing 'anchors' field")
                
                cloakmap = CloakMap.from_dict(cloakmap_data)
                progress.update(1)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in CloakMap file: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise click.ClickException(f"Corrupted CloakMap structure: {e}") from e

        if verbose:
            click.echo(f"✓ Loaded CloakMap with {len(cloakmap.anchors)} anchors")

        # Load masked document with error handling
        try:
            with click.progressbar(length=1, label="Loading masked document") as progress:
                processor = DocumentProcessor()
                masked_document = processor.load_document(masked_file, validate=True)
                progress.update(1)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in masked document: {e}") from e
        except (ValueError, TypeError) as e:
            raise click.ClickException(f"Invalid document format: {e}") from e

        # Extract text segments for coverage analysis
        try:
            text_extractor = TextExtractor()
            text_segments = text_extractor.extract_text_segments(masked_document)
        except (AttributeError, KeyError) as e:
            raise click.ClickException(f"Failed to extract text segments: {e}") from e

        if verbose:
            click.echo(f"✓ Extracted {len(text_segments)} text segments")

        # Create mock MaskResult for statistics collection
        from ..core.results import (
            MaskResult,
            OperationStatus,
            PerformanceMetrics,
            ProcessingStats,
        )

        mock_result = MaskResult(
            status=OperationStatus.SUCCESS,
            masked_document=masked_document,
            cloakmap=cloakmap,
            stats=ProcessingStats(
                total_entities_found=len(cloakmap.anchors),
                entities_masked=len(cloakmap.anchors),
                entities_skipped=0,
                entities_failed=0,
            ),
            performance=PerformanceMetrics(),
        )

        # Collect diagnostics with error handling
        try:
            with click.progressbar(
                length=1, label="Analyzing coverage and statistics"
            ) as progress:
                collector = DiagnosticsCollector()
                coverage_analyzer = CoverageAnalyzer()

                # Generate statistics and coverage
                statistics = collector.collect_masking_statistics(mock_result)
                coverage = coverage_analyzer.analyze_document_coverage(
                    text_segments, cloakmap.anchors
                )
                performance = collector.collect_performance_metrics(mock_result)
                diagnostics_info = collector.collect_processing_diagnostics(mock_result)

                # Generate recommendations
                recommendations = coverage_analyzer.generate_recommendations(coverage)

                progress.update(1)
        except (AttributeError, KeyError, ValueError) as e:
            raise click.ClickException(f"Failed to analyze diagnostics: {e}") from e

        # Create report data
        report_data = ReportData(
            statistics=statistics,
            coverage=coverage,
            performance=performance,
            diagnostics=diagnostics_info,
            document_metadata={
                "name": masked_file.name,
                "size_bytes": masked_file.stat().st_size,
                "cloakmap_file": cloakmap_file.name,
            },
            recommendations=recommendations,
        )

        # Generate report with error handling
        try:
            reporter = DiagnosticReporter()
            format_enum = {
                "json": ReportFormat.JSON,
                "html": ReportFormat.HTML,
                "markdown": ReportFormat.MARKDOWN,
            }[report_format]

            # Set default output path
            if not output:
                output = masked_file.with_suffix(f".diagnostics.{report_format}")

            # Validate output directory
            output_dir = output.parent
            if not output_dir.exists():
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                except OSError as e:
                    raise click.ClickException(f"Cannot create output directory {output_dir}: {e}") from e

            with click.progressbar(
                length=1, label=f"Generating {report_format.upper()} report"
            ) as progress:
                reporter.save_report(report_data, output, format_enum)
                progress.update(1)

            # Show summary
            summary = reporter.generate_summary(report_data)
        except PermissionError as e:
            raise click.ClickException(f"Permission denied writing to {output}: {e}") from e
        except OSError as e:
            raise click.ClickException(f"Failed to write report file: {e}") from e
        except (KeyError, AttributeError, ValueError) as e:
            raise click.ClickException(f"Failed to generate report: {e}") from e

        click.echo("✅ Diagnostic analysis completed!")
        click.echo(f"   Report saved: {output}")
        click.echo(f"   Entities processed: {summary['entities']['detected']}")
        click.echo(f"   Coverage rate: {summary['coverage']['rate']:.1%}")
        click.echo(f"   Issues found: {summary['issues']['total_issues']}")

        if verbose and recommendations:
            click.echo("\n💡 Recommendations:")
            for rec in recommendations[:3]:  # Show first 3 recommendations
                click.echo(f"   • {rec}")
            if len(recommendations) > 3:
                click.echo(
                    f"   ... and {len(recommendations) - 3} more (see full report)"
                )

    except ImportError as e:
        raise click.ClickException(f"Missing required dependency: {e}") from e
    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}") from e
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Diagnostic analysis failed: {e}") from e


@diagnostics.command("summary")
@click.argument("cloakmap_file", type=click.Path(exists=True, path_type=Path))
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def diagnostics_summary(ctx: click.Context, cloakmap_file: Path, verbose: bool) -> None:
    """Show a quick summary of masking results from a CloakMap.

    Provides a concise overview of masking statistics without generating
    a full diagnostic report.

    Example:
        cloakpivot diagnostics summary map.json
    """
    try:
        import json

        from ..core.cloakmap import CloakMap

        verbose = verbose or ctx.obj.get("verbose", False)

        # Validate file existence and readability
        if not cloakmap_file.exists():
            raise click.ClickException(f"CloakMap file does not exist: {cloakmap_file}")

        # Load CloakMap with comprehensive error handling
        try:
            with open(cloakmap_file, encoding="utf-8") as f:
                cloakmap_data = json.load(f)
            
            # Validate CloakMap structure
            if not isinstance(cloakmap_data, dict):
                raise click.ClickException(f"Invalid CloakMap format: expected JSON object, got {type(cloakmap_data).__name__}")
            
            if "anchors" not in cloakmap_data:
                raise click.ClickException("Invalid CloakMap: missing 'anchors' field")
            
            cloakmap = CloakMap.from_dict(cloakmap_data)
        except json.JSONDecodeError as e:
            raise click.ClickException(f"Invalid JSON in CloakMap file: {e}") from e
        except (KeyError, ValueError, TypeError) as e:
            raise click.ClickException(f"Corrupted CloakMap structure: {e}") from e

        # Display summary
        click.echo(f"📊 CloakMap Summary: {cloakmap_file.name}")
        click.echo("=" * 50)

        click.echo(f"Document ID: {cloakmap.doc_id}")
        click.echo(f"Total Anchors: {cloakmap.anchor_count}")
        click.echo(f"CloakMap Version: {cloakmap.version}")
        click.echo(f"Created: {cloakmap.created_at}")

        # Entity breakdown
        entity_counts = cloakmap.entity_count_by_type
        if entity_counts:
            click.echo("\nEntity Breakdown:")
            for entity_type, count in sorted(entity_counts.items()):
                click.echo(f"  • {entity_type}: {count}")

        # Strategy breakdown
        strategy_counts = {}
        for anchor in cloakmap.anchors:
            strategy = anchor.strategy_used
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        if strategy_counts:
            click.echo("\nStrategy Usage:")
            for strategy, count in sorted(strategy_counts.items()):
                click.echo(f"  • {strategy}: {count}")

        if verbose:
            # Show additional details
            click.echo("\nAdditional Details:")
            click.echo(f"  Document Hash: {cloakmap.doc_hash}")
            if hasattr(cloakmap, "policy_snapshot") and cloakmap.policy_snapshot:
                click.echo("  Policy Snapshot: Available")

            # Show sample anchors
            if cloakmap.anchors:
                click.echo("\nSample Anchors (first 3):")
                for i, anchor in enumerate(cloakmap.anchors[:3]):
                    click.echo(
                        f"  {i + 1}. {anchor.entity_type} at {anchor.node_id}:{anchor.start}-{anchor.end}"
                    )

    except ImportError as e:
        raise click.ClickException(f"Missing required dependency: {e}") from e
    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}") from e
    except PermissionError as e:
        raise click.ClickException(f"Permission denied reading file: {e}") from e
    except Exception as e:
        if verbose:
            import traceback
            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Failed to read CloakMap summary: {e}") from e


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
