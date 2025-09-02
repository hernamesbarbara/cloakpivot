"""Main CLI entry point for CloakPivot."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, TextIO, cast

import click

from cloakpivot import __version__
from cloakpivot.core.types import DoclingDocument

from .batch import batch

# Error messages
ERROR_MASK_ARGS = (
    "Must specify either --out for masked output or --cloakmap for CloakMap output"
)


class DocDocumentLike(Protocol):
    name: str
    texts: Any
    tables: Any


if TYPE_CHECKING:
    from presidio_analyzer import RecognizerResult

    from cloakpivot.core.detection import DocumentAnalysisResult
    from cloakpivot.core.policies import MaskingPolicy
    from cloakpivot.masking.engine import MaskingResult


@click.group()
@click.version_option(version=__version__, prog_name="cloakpivot")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--quiet", "-q", is_flag=True, help="Suppress all non-error output")
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Configuration file path",
)
@click.pass_context
def cli(ctx: click.Context, verbose: bool, quiet: bool, config: Path | None) -> None:
    """CloakPivot: PII masking/unmasking on top of DocPivot and Presidio.

    CloakPivot enables reversible document masking while preserving
    structure and formatting using DocPivot and Presidio.

    Shell completion is available for bash, zsh, and fish.
    To enable completion, run one of:

    \b
    # For bash
    _CLOAKPIVOT_COMPLETE=bash_source cloakpivot > ~/.cloakpivot-complete.bash
    source ~/.cloakpivot-complete.bash

    \b
    # For zsh
    _CLOAKPIVOT_COMPLETE=zsh_source cloakpivot > ~/.cloakpivot-complete.zsh
    source ~/.cloakpivot-complete.zsh

    \b
    # For fish
    _CLOAKPIVOT_COMPLETE=fish_source cloakpivot > ~/.config/fish/completions/cloakpivot.fish
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose and not quiet
    ctx.obj["quiet"] = quiet

    # Load configuration file if provided
    if config:
        _load_config_file(ctx, config)


def _load_config_file(ctx: click.Context, config_path: Path) -> None:
    """Load configuration from YAML file."""
    try:
        import yaml

        with open(config_path, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if not isinstance(config_data, dict):
            raise click.ClickException(
                f"Invalid configuration file format: {config_path}"
            )

        # Store config data in context for use by commands
        ctx.obj["config"] = config_data

        # Apply global configuration options
        if "verbose" in config_data and not ctx.obj.get("verbose"):
            ctx.obj["verbose"] = config_data["verbose"]

        if "quiet" in config_data and not ctx.obj.get("quiet"):
            ctx.obj["quiet"] = config_data["quiet"]

    except ImportError:
        raise click.ClickException(
            "PyYAML is required for configuration files. Install with: pip install pyyaml"
        ) from None
    except Exception as e:
        raise click.ClickException(f"Failed to load configuration file: {e}") from e


def _validate_mask_arguments(output_path: Path | None, cloakmap: Path | None) -> None:
    """Validate mask command arguments."""
    if not output_path and not cloakmap:
        raise click.ClickException(ERROR_MASK_ARGS)


def _set_default_paths(
    input_path: Path,
    output_path: Path | None,
    cloakmap: Path | None,
    output_format: str,
) -> tuple[Path, Path]:
    """Set default output paths if not specified."""
    if not output_path:
        output_path = input_path.with_suffix(f".masked.{output_format}.json")
    if not cloakmap:
        cloakmap = input_path.with_suffix(".cloakmap.json")
    return output_path, cloakmap


def _try_enhanced_policy_loading(policy: Path, verbose: bool) -> MaskingPolicy | None:
    """Try to load policy using enhanced policy loader."""
    try:
        from cloakpivot.core.policy_loader import PolicyLoader

        loader = PolicyLoader()
        masking_policy = loader.load_policy(policy)
        if verbose:
            click.echo("✓ Enhanced policy loaded successfully")
            if hasattr(masking_policy, "name") or policy.name.endswith(
                (".yaml", ".yml")
            ):
                click.echo("   (with inheritance and validation support)")
        return masking_policy
    except ImportError:
        if verbose:
            click.echo("⚠️  Required dependencies not available for enhanced policies")
        return None
    except Exception as e:
        if verbose:
            click.echo(f"⚠️  Enhanced policy loading failed: {e}")
            click.echo("   Falling back to basic policy loading")
        return None


def _try_basic_policy_loading(policy: Path, verbose: bool) -> MaskingPolicy:
    """Try to load policy using basic YAML loading."""
    from cloakpivot.core.policies import MaskingPolicy

    try:
        import yaml

        with open(policy, encoding="utf-8") as f:
            policy_data = yaml.safe_load(f)
        masking_policy = MaskingPolicy.from_dict(policy_data)
        if verbose:
            click.echo("✓ Basic policy loaded successfully")
        return masking_policy
    except ImportError:
        if verbose:
            click.echo("⚠️  PyYAML not installed, using default policy")
        return MaskingPolicy()
    except Exception as e:
        if verbose:
            click.echo(f"⚠️  Failed to load policy file: {e}")
            click.echo("   Using default policy")
        return MaskingPolicy()


def _load_masking_policy(policy: Path | None, verbose: bool) -> MaskingPolicy:
    """Load masking policy from file or use default."""
    from cloakpivot.core.policies import MaskingPolicy

    if not policy:
        return MaskingPolicy()

    if verbose:
        click.echo(f"📋 Loading policy: {policy}")

    # Try enhanced loading first
    enhanced_policy = _try_enhanced_policy_loading(policy, verbose)
    if enhanced_policy is not None:
        return enhanced_policy

    # Fall back to basic loading
    return _try_basic_policy_loading(policy, verbose)


def _perform_entity_detection(
    document: DocDocumentLike,
    masking_policy: MaskingPolicy,
    verbose: bool,
    quiet: bool = False,
) -> DocumentAnalysisResult:
    """Perform PII entity detection on document."""
    from cloakpivot.core.detection import EntityDetectionPipeline

    if verbose and not quiet:
        click.echo("🔍 Detecting PII entities")

    detection_pipeline = EntityDetectionPipeline()
    doc_dl = cast("DoclingDocument", document)

    if not quiet:
        with click.progressbar(
            length=1, label="Analyzing document for PII"
        ) as progress:
            detection_result = detection_pipeline.analyze_document(
                doc_dl, masking_policy
            )
            progress.update(1)
    else:
        detection_result = detection_pipeline.analyze_document(doc_dl, masking_policy)

    if verbose and not quiet:
        click.echo(f"✓ Detected {detection_result.total_entities} entities")
        for entity_type, count in detection_result.entity_breakdown.items():
            click.echo(f"  {entity_type}: {count}")

    if detection_result.total_entities == 0 and not quiet:
        click.echo("ℹ️  No PII entities detected in document")
        if not click.confirm("Continue with masking anyway?"):
            raise click.Abort()

    return detection_result


def _prepare_entities_for_masking(
    detection_result: DocumentAnalysisResult,
) -> list[RecognizerResult]:
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
    document: DocDocumentLike,
    entities: list[RecognizerResult],
    masking_policy: MaskingPolicy,
    verbose: bool,
) -> MaskingResult:
    """Perform the actual masking operation."""
    from cloakpivot.document.extractor import TextExtractor
    from cloakpivot.masking.engine import MaskingEngine

    masking_engine = MaskingEngine()
    text_extractor = TextExtractor()
    doc_dl = cast("DoclingDocument", document)
    text_segments = text_extractor.extract_text_segments(doc_dl)

    if verbose:
        click.echo(f"📝 Extracted {len(text_segments)} text segments")

    with click.progressbar(length=1, label="Masking PII entities") as progress:
        masking_result = masking_engine.mask_document(
            document=doc_dl,
            entities=entities,
            policy=masking_policy,
            text_segments=text_segments,
        )
        progress.update(1)

    if verbose:
        click.echo(f"✓ Masked {len(masking_result.cloakmap.anchors)} entities")

    return masking_result


def _save_masked_document(
    masking_result: MaskingResult, output_path: Path, verbose: bool
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
    masking_result: MaskingResult, cloakmap: Path, verbose: bool
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
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--out",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output path or format (lexical, markdown, html)",
)
@click.option("--cloakmap", type=click.Path(), help="Path to save the CloakMap file")
@click.option(
    "--policy",
    type=click.Path(exists=True),
    help="Path to masking policy file",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["lexical", "docling", "markdown", "md", "html", "doctags"]),
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
    output_path: Path | None,
    cloakmap: Path | None,
    policy: Path | None,
    output_format: str,
    lang: str,
    min_score: float,
    encrypt: bool,
    key_id: str | None,
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

        from cloakpivot.core.detection import EntityDetectionPipeline
        from cloakpivot.core.policies import MaskingPolicy
        from cloakpivot.document.processor import DocumentProcessor
        from cloakpivot.masking.engine import MaskingEngine

        # Convert string paths to Path objects
        input_path = Path(input_path)
        output_path = Path(output_path) if output_path else None
        cloakmap = Path(cloakmap) if cloakmap else None
        policy = Path(policy) if policy else None

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

        # Load document and detect input format
        with click.progressbar(length=1, label="Loading document") as progress:
            processor = DocumentProcessor()
            document = processor.load_document(input_path, validate=True)

            # Detect input document format for round-trip preservation
            from cloakpivot.formats.serialization import CloakPivotSerializer

            format_serializer = CloakPivotSerializer()
            detected_input_format = format_serializer.detect_format(input_path)

            if verbose:
                click.echo(
                    f"📋 Detected input format: {detected_input_format or 'unknown'}"
                )

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

        # Initialize masking engine with conflict resolution enabled
        masking_engine = MaskingEngine(resolve_conflicts=True)

        # Extract text segments (needed for masking engine)
        from cloakpivot.document.extractor import TextExtractor

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
                original_format=detected_input_format,
            )
            progress.update(1)

        if verbose:
            click.echo(f"✓ Masked {len(masking_result.cloakmap.anchors)} entities")

        # Save masked document
        if verbose:
            click.echo(f"💾 Saving masked document: {output_path}")

        with click.progressbar(length=1, label="Saving masked document") as progress:
            # Use CloakPivot's enhanced serializer system
            from cloakpivot.formats.serialization import CloakPivotSerializer

            serializer = CloakPivotSerializer()

            # For round-trip fidelity, preserve the original format when detected
            # Only use the explicit output_format if it differs from detected format
            preserve_format = detected_input_format or output_format

            result = serializer.serialize_document(
                masking_result.masked_document, preserve_format
            )

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(result.content)
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
@click.argument("masked_path", type=click.Path(exists=True))
@click.option(
    "--cloakmap",
    required=True,
    type=click.Path(exists=True),
    help="Path to the CloakMap file",
)
@click.option(
    "--out",
    "-o",
    "output_path",
    type=click.Path(),
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
    output_path: Path | None,
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

        from cloakpivot.core.cloakmap import CloakMap
        from cloakpivot.document.processor import DocumentProcessor
        from cloakpivot.unmasking.engine import UnmaskingEngine

        # Convert string paths to Path objects
        masked_path = Path(masked_path)
        cloakmap = Path(cloakmap)
        output_path = Path(output_path) if output_path else None

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
                    # Use CloakPivot's enhanced serializer system
                    from cloakpivot.formats.serialization import CloakPivotSerializer

                    serializer = CloakPivotSerializer()

                    # Try to restore to original format from CloakMap metadata
                    original_format = cloakmap_obj.metadata.get("original_format")
                    if original_format:
                        output_format = original_format
                        if verbose:
                            click.echo(
                                f"🔄 Restoring to original format: {original_format}"
                            )
                    else:
                        # Fallback to detecting from file extension or default to lexical
                        output_format = (
                            serializer.detect_format(output_path) or "lexical"
                        )
                        if verbose:
                            click.echo(
                                f"⚠️  No original format found, using: {output_format}"
                            )

                    result = serializer.serialize_document(
                        unmasking_result.restored_document, output_format
                    )

                    with open(output_path, "w", encoding="utf-8") as f:
                        f.write(result.content)
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
@click.argument("policy_file", type=click.Path(exists=True))
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed validation information"
)
def policy_validate(policy_file: Path, verbose: bool) -> None:
    """Validate a policy file for errors and compatibility.

    Example:
        cloakpivot policy validate my-policy.yaml
    """
    try:
        from cloakpivot.core.policy_loader import PolicyLoader

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
@click.argument("policy_file", type=click.Path(exists=True))
@click.option("--text", "-t", help="Test text to analyze with the policy")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed analysis results")
def policy_test(policy_file: Path, text: str | None, verbose: bool) -> None:
    """Test a policy against sample text to see masking behavior.

    Example:
        cloakpivot policy test policy.yaml --text "John Doe's email is john@example.com"
    """
    try:
        from cloakpivot.core.policy_loader import PolicyLoader

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


@policy.command("create")
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: interactive_policy.yaml)",
)
@click.option(
    "--template",
    type=click.Choice(["conservative", "balanced", "permissive"]),
    help="Start with a built-in template",
)
@click.option(
    "--verbose", "-v", is_flag=True, help="Show detailed information during creation"
)
@click.pass_context
def policy_create(
    ctx: click.Context, output: Path | None, template: str | None, verbose: bool
) -> None:
    """Create a new masking policy through interactive prompts.

    Guides you through policy configuration with prompts and provides
    templates and examples during creation.

    Example:
        cloakpivot policy create --output my-policy.yaml
        cloakpivot policy create --template balanced
    """
    try:
        verbose = verbose or ctx.obj.get("verbose", False)

        if verbose:
            click.echo("🎯 Starting interactive policy creation")

        # Convert string path to Path object and set default if not specified
        if output:
            output = Path(output)
        else:
            output = Path("interactive_policy.yaml")

        # Check if output file exists and warn user
        if output.exists():
            if not click.confirm(f"File {output} already exists. Overwrite?"):
                raise click.Abort()

        click.echo("📋 CloakPivot Interactive Policy Builder")
        click.echo("=" * 50)
        click.echo(
            "This wizard will guide you through creating a custom masking policy."
        )
        click.echo("Press Ctrl+C at any time to cancel.\n")

        # Step 1: Basic Configuration
        click.echo("🔧 Basic Configuration")
        policy_name = click.prompt("Policy name", default="my-custom-policy")
        description = click.prompt(
            "Policy description", default="Custom masking policy"
        )
        locale = click.prompt("Locale (language code)", default="en")

        # Step 2: Choose starting template or start from scratch
        if template:
            click.echo(f"\n📄 Using template: {template}")
            use_template = True
        else:
            use_template = click.confirm(
                "\nWould you like to start with a template?", default=True
            )
            if use_template:
                template = click.prompt(
                    "Choose template",
                    type=click.Choice(["conservative", "balanced", "permissive"]),
                    default="balanced",
                )

        # Step 3: Default strategy configuration
        click.echo("\n🎭 Default Masking Strategy")
        click.echo(
            "This strategy will be applied to entities without specific configuration."
        )

        strategy_choices = ["redact", "template", "hash", "partial"]
        default_strategy = click.prompt(
            "Default strategy", type=click.Choice(strategy_choices), default="redact"
        )

        strategy_params = {}
        if default_strategy == "redact":
            strategy_params["redact_char"] = click.prompt(
                "Redaction character", default="*"
            )
            strategy_params["preserve_length"] = click.confirm(
                "Preserve original length?", default=True
            )
        elif default_strategy == "template":
            strategy_params["template"] = click.prompt(
                "Template text", default="[REDACTED]"
            )
        elif default_strategy == "hash":
            strategy_params["algorithm"] = click.prompt(
                "Hash algorithm", type=click.Choice(["sha256", "md5"]), default="sha256"
            )
            strategy_params["truncate"] = int(
                click.prompt("Truncate hash to length", default="8")
            )
        elif default_strategy == "partial":
            strategy_params["visible_chars"] = int(
                click.prompt("Number of visible characters", default="3")
            )
            strategy_params["position"] = click.prompt(
                "Position of visible chars",
                type=click.Choice(["start", "end"]),
                default="start",
            )

        # Step 4: Per-entity configurations
        click.echo("\n👤 Entity-Specific Configurations")
        common_entities = [
            "PERSON",
            "EMAIL_ADDRESS",
            "PHONE_NUMBER",
            "CREDIT_CARD",
            "US_SSN",
            "IP_ADDRESS",
        ]
        per_entity = {}

        configure_entities = click.confirm(
            "Configure specific entity types?", default=True
        )
        if configure_entities:
            for entity_type in common_entities:
                if click.confirm(f"Configure {entity_type}?", default=False):
                    click.echo(f"\n  Configuring {entity_type}:")
                    entity_strategy = click.prompt(
                        f"  Strategy for {entity_type}",
                        type=click.Choice(strategy_choices),
                        default=default_strategy,
                    )

                    entity_params = {}
                    if entity_strategy == "redact":
                        entity_params["redact_char"] = click.prompt(
                            "  Redaction character", default="*"
                        )
                        entity_params["preserve_length"] = click.confirm(
                            "  Preserve original length?", default=True
                        )
                    elif entity_strategy == "template":
                        suggested_template = f"[{entity_type.replace('_', '-')}]"
                        entity_params["template"] = click.prompt(
                            "  Template text", default=suggested_template
                        )
                    elif entity_strategy == "hash":
                        entity_params["algorithm"] = click.prompt(
                            "  Hash algorithm",
                            type=click.Choice(["sha256", "md5"]),
                            default="sha256",
                        )
                        entity_params["truncate"] = int(
                            click.prompt("  Truncate hash to length", default="8")
                        )
                    elif entity_strategy == "partial":
                        entity_params["visible_chars"] = int(
                            click.prompt("  Number of visible characters", default="3")
                        )
                        entity_params["position"] = click.prompt(
                            "  Position of visible chars",
                            type=click.Choice(["start", "end"]),
                            default="start",
                        )

                    threshold = float(
                        click.prompt(
                            f"  Confidence threshold for {entity_type} (0.0-1.0)",
                            default="0.8",
                        )
                    )

                    per_entity[entity_type] = {
                        "kind": entity_strategy,
                        "parameters": entity_params,
                        "threshold": threshold,
                        "enabled": True,
                    }

        # Step 5: Allow/Deny Lists
        click.echo("\n📝 Allow and Deny Lists")
        allow_list = []
        deny_list = []

        if click.confirm("Add items to allow list (never masked)?", default=False):
            click.echo("Enter items one by one (empty line to finish):")
            while True:
                item = click.prompt(
                    "Allow list item (empty to finish)", default="", show_default=False
                )
                if not item:
                    break
                allow_list.append(item)

        if click.confirm("Add items to deny list (always masked)?", default=False):
            click.echo("Enter items one by one (empty line to finish):")
            while True:
                item = click.prompt(
                    "Deny list item (empty to finish)", default="", show_default=False
                )
                if not item:
                    break
                deny_list.append(item)

        # Step 6: Generate policy content
        click.echo(f"\n📄 Generating policy file: {output}")

        from datetime import datetime

        policy_content = f"""# CloakPivot Interactive Policy Configuration
# Generated by CloakPivot Policy Builder
# Created: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

version: "1.0"
name: "{policy_name}"
description: "{description}"
locale: "{locale}"

# Default strategy applied to entities without specific configuration
default_strategy:
  kind: "{default_strategy}"
  parameters:
"""

        for key, value in strategy_params.items():
            if isinstance(value, bool):
                policy_content += f"    {key}: {str(value).lower()}\n"
            elif isinstance(value, str):
                policy_content += f'    {key}: "{value}"\n'
            else:
                policy_content += f"    {key}: {value}\n"

        # Add per-entity configurations
        if per_entity:
            policy_content += "\n# Per-entity type configuration\nper_entity:\n"
            for entity_type, config in per_entity.items():
                policy_content += f"  {entity_type}:\n"
                policy_content += f'    kind: "{config["kind"]}"\n'
                policy_content += "    parameters:\n"
                for key, value in config["parameters"].items():
                    if isinstance(value, bool):
                        policy_content += f"      {key}: {str(value).lower()}\n"
                    elif isinstance(value, str):
                        policy_content += f'      {key}: "{value}"\n'
                    else:
                        policy_content += f"      {key}: {value}\n"
                policy_content += f"    threshold: {config['threshold']}\n"
                policy_content += f"    enabled: {str(config['enabled']).lower()}\n\n"

        # Add allow/deny lists
        if allow_list:
            policy_content += "# Items that should never be masked\nallow_list:\n"
            for item in allow_list:
                policy_content += f'  - "{item}"\n'
            policy_content += "\n"

        if deny_list:
            policy_content += "# Items that should always be masked\ndeny_list:\n"
            for item in deny_list:
                policy_content += f'  - "{item}"\n'
            policy_content += "\n"

        # Add basic context rules
        policy_content += """# Context-specific masking rules
context_rules:
  heading:
    enabled: false  # Don't mask in document headings

  table:
    enabled: true

  footer:
    enabled: true

  header:
    enabled: false

# Basic configuration options
min_entity_length: 2
"""

        # Save the policy file
        with open(output, "w", encoding="utf-8") as f:
            f.write(policy_content)

        click.echo("✅ Policy creation completed successfully!")
        click.echo(f"   Policy file: {output}")
        click.echo(f"   Name: {policy_name}")
        click.echo(f"   Entities configured: {len(per_entity)}")
        click.echo(f"   Allow list items: {len(allow_list)}")
        click.echo(f"   Deny list items: {len(deny_list)}")

        if verbose:
            click.echo("\n📊 Policy Summary:")
            click.echo(f"   Default strategy: {default_strategy}")
            click.echo(f"   Locale: {locale}")
            if per_entity:
                click.echo("   Configured entities:")
                for entity_type, config in per_entity.items():
                    click.echo(
                        f"     • {entity_type}: {config['kind']} (threshold: {config['threshold']})"
                    )

        # Offer to validate the created policy
        if click.confirm(
            "\nWould you like to validate the created policy?", default=True
        ):
            click.echo("🔍 Validating policy...")
            try:
                from cloakpivot.core.policy_loader import PolicyLoader

                loader = PolicyLoader()
                errors = loader.validate_policy_file(output)

                if not errors:
                    click.echo("✅ Policy validation successful!")
                else:
                    click.echo("❌ Policy validation found issues:")
                    for error in errors:
                        click.echo(f"   • {error}")
            except ImportError:
                click.echo("⚠️  Enhanced validation requires pydantic")
                click.echo("   Basic YAML structure appears valid")
            except Exception as e:
                click.echo(f"⚠️  Validation failed: {e}")

        click.echo("\n💡 Next steps:")
        click.echo(f"   • Test your policy: cloakpivot policy test {output}")
        click.echo(
            f"   • Use in masking: cloakpivot mask document.pdf --policy {output}"
        )
        click.echo(f"   • View policy details: cloakpivot policy info {output}")

    except click.Abort:
        click.echo("\n❌ Policy creation cancelled")
        raise
    except KeyboardInterrupt:
        click.echo("\n❌ Policy creation interrupted")
        raise click.Abort() from None
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Policy creation failed: {e}") from e


@policy.command("info")
@click.argument("policy_file", type=click.Path(exists=True))
def policy_info(policy_file: Path) -> None:
    """Show detailed information about a policy file.

    Example:
        cloakpivot policy info my-policy.yaml
    """
    try:
        from cloakpivot.core.policy_loader import PolicyLoader

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
@click.argument("masked_file", type=click.Path(exists=True))
@click.argument("cloakmap_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
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
    output: Path | None,
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

        from cloakpivot.core.cloakmap import CloakMap
        from cloakpivot.diagnostics import (
            CoverageAnalyzer,
            DiagnosticReporter,
            DiagnosticsCollector,
            ReportData,
            ReportFormat,
        )
        from cloakpivot.document.extractor import TextExtractor
        from cloakpivot.document.processor import DocumentProcessor

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
                    raise click.ClickException(
                        f"Invalid CloakMap format: expected JSON object, got {type(cloakmap_data).__name__}"
                    )

                if "anchors" not in cloakmap_data:
                    raise click.ClickException(
                        "Invalid CloakMap: missing 'anchors' field"
                    )

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
            with click.progressbar(
                length=1, label="Loading masked document"
            ) as progress:
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
        from cloakpivot.core.results import (
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
                    raise click.ClickException(
                        f"Cannot create output directory {output_dir}: {e}"
                    ) from e

            with click.progressbar(
                length=1, label=f"Generating {report_format.upper()} report"
            ) as progress:
                reporter.save_report(report_data, output, format_enum)
                progress.update(1)

            # Show summary
            summary = reporter.generate_summary(report_data)
        except PermissionError as e:
            raise click.ClickException(
                f"Permission denied writing to {output}: {e}"
            ) from e
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
@click.argument("cloakmap_file", type=click.Path(exists=True))
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

        from cloakpivot.core.cloakmap import CloakMap

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
                raise click.ClickException(
                    f"Invalid CloakMap format: expected JSON object, got {type(cloakmap_data).__name__}"
                )

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
        strategy_counts: dict[str, int] = {}
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


@cli.group()
def format() -> None:
    """Manage document format conversion and detection."""
    pass


@format.command("convert")
@click.argument("input_path", type=click.Path(exists=True))
@click.option(
    "--to",
    "target_format",
    required=True,
    type=click.Choice(["lexical", "docling", "markdown", "md", "html", "doctags"]),
    help="Target output format",
)
@click.option(
    "--out",
    "-o",
    "output_path",
    type=click.Path(),
    help="Output file path (auto-generated if not specified)",
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def format_convert(
    ctx: click.Context,
    input_path: Path,
    target_format: str,
    output_path: Path | None,
    verbose: bool,
) -> None:
    """Convert a document from one format to another.

    Supports conversion between all supported formats while preserving
    document structure and content.

    Example:
        cloakpivot format convert document.lexical.json --to markdown
        cloakpivot format convert document.md --to html --out output.html
    """
    try:
        from cloakpivot.formats.serialization import (
            CloakPivotSerializer,
            SerializationError,
        )

        verbose = verbose or ctx.obj.get("verbose", False)

        if verbose:
            click.echo(f"🔄 Converting document: {input_path}")
            click.echo(f"   Target format: {target_format}")

        # Initialize serializer
        serializer = CloakPivotSerializer()

        # Detect input format
        input_format = serializer.detect_format(input_path)
        if input_format:
            if verbose:
                click.echo(f"   Detected input format: {input_format}")
        else:
            click.echo(f"⚠️  Could not detect input format for {input_path}")

        # Perform conversion
        with click.progressbar(length=1, label="Converting document") as progress:
            result = serializer.convert_format(
                input_path=input_path,
                output_format=target_format,
                output_path=output_path,
            )
            progress.update(1)

        # Report results
        actual_output_path = output_path or result.metadata.get(
            "output_path", "output file"
        )
        click.echo("✅ Format conversion completed!")
        click.echo(f"   Input: {input_path} ({input_format or 'unknown'})")
        click.echo(f"   Output: {actual_output_path} ({target_format})")
        click.echo(f"   Size: {result.size_kb:.1f} KB")

        if verbose and result.metadata:
            click.echo("\n📊 Conversion Details:")
            click.echo(
                f"   Document name: {result.metadata.get('document_name', 'N/A')}"
            )
            click.echo(f"   Text items: {result.metadata.get('document_texts', 'N/A')}")
            click.echo(f"   Tables: {result.metadata.get('document_tables', 'N/A')}")

    except SerializationError as e:
        if verbose:
            click.echo(f"Context: {e.context}")
        raise click.ClickException(f"Format conversion failed: {e}") from e
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Conversion failed: {e}") from e


@format.command("detect")
@click.argument("file_path", type=click.Path(exists=True))
@click.option("--verbose", "-v", is_flag=True, help="Show detailed format information")
@click.pass_context
def format_detect(ctx: click.Context, file_path: Path, verbose: bool) -> None:
    """Detect the format of a document file.

    Analyzes file extension, naming conventions, and content to determine
    the document format.

    Example:
        cloakpivot format detect document.lexical.json
        cloakpivot format detect unknown_document.json --verbose
    """
    try:
        from cloakpivot.formats.serialization import CloakPivotSerializer

        verbose = verbose or ctx.obj.get("verbose", False)

        serializer = CloakPivotSerializer()

        # Detect format
        detected_format = serializer.detect_format(file_path)

        if detected_format:
            click.echo(f"📄 File: {file_path}")
            click.echo(f"🔍 Detected format: {detected_format}")

            if verbose:
                # Get format information
                format_info = serializer.get_format_info(detected_format)
                click.echo("\n📋 Format Details:")
                click.echo(f"   Supported: {format_info['supported']}")
                click.echo(f"   Text format: {format_info['is_text_format']}")
                click.echo(f"   JSON format: {format_info['is_json_format']}")
                click.echo(f"   Extensions: {', '.join(format_info['extensions'])}")
                click.echo(
                    f"   Suggested extension: {format_info['suggested_extension']}"
                )

        else:
            click.echo(f"❓ Could not detect format for: {file_path}")
            if verbose:
                click.echo("\n💡 Supported formats:")
                for fmt in serializer.supported_formats:
                    info = serializer.get_format_info(fmt)
                    click.echo(f"   • {fmt}: {', '.join(info['extensions'])}")

    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Format detection failed: {e}") from e


@format.command("list")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed format information")
def format_list(verbose: bool) -> None:
    """List all supported document formats.

    Shows available formats for input and output operations.

    Example:
        cloakpivot format list
        cloakpivot format list --verbose
    """
    try:
        from cloakpivot.formats.serialization import CloakPivotSerializer

        serializer = CloakPivotSerializer()
        formats = serializer.supported_formats

        click.echo(f"📋 Supported Formats ({len(formats)}):")

        for fmt in sorted(formats):
            if verbose:
                info = serializer.get_format_info(fmt)
                extensions = (
                    ", ".join(info["extensions"]) if info["extensions"] else "none"
                )
                format_type = []
                if info["is_text_format"]:
                    format_type.append("text")
                if info["is_json_format"]:
                    format_type.append("json")
                type_str = f" ({', '.join(format_type)})" if format_type else ""
                click.echo(f"   • {fmt}{type_str}")
                click.echo(f"     Extensions: {extensions}")
                click.echo(f"     Suggested: {info['suggested_extension']}")
            else:
                click.echo(f"   • {fmt}")

    except Exception as e:
        raise click.ClickException(f"Failed to list formats: {e}") from e


@cli.command()
@click.argument("doc1", type=click.Path(exists=True))
@click.argument("doc2", type=click.Path(exists=True))
@click.option(
    "--cloakmap1",
    type=click.Path(exists=True),
    help="CloakMap for first document (for masking strategy analysis)",
)
@click.option(
    "--cloakmap2",
    type=click.Path(exists=True),
    help="CloakMap for second document (for masking strategy analysis)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output path for diff report (default: diff_report.html)",
)
@click.option(
    "--format",
    "report_format",
    type=click.Choice(["text", "html", "json"]),
    default="text",
    help="Diff report format (default: text)",
)
@click.option(
    "--show-context",
    type=int,
    default=3,
    help="Number of context lines to show around differences",
)
@click.option("--verbose", "-v", is_flag=True, help="Show detailed diff information")
@click.pass_context
def diff(
    ctx: click.Context,
    doc1: Path,
    doc2: Path,
    cloakmap1: Path | None,
    cloakmap2: Path | None,
    output: Path | None,
    report_format: str,
    show_context: int,
    verbose: bool,
) -> None:
    """Compare two documents and show differences in masking approaches.

    Compares two documents (masked or unmasked) and shows differences in content,
    masking strategies, and coverage. Useful for comparing different policy
    applications or document versions.

    Examples:
        cloakpivot diff doc1.json doc2.json
        cloakpivot diff masked1.json masked2.json --cloakmap1 map1.json --cloakmap2 map2.json
        cloakpivot diff doc1.json doc2.json --format html --output report.html
    """
    try:
        import difflib
        import json

        verbose = verbose or ctx.obj.get("verbose", False)

        # Convert string paths to Path objects
        doc1 = Path(doc1)
        doc2 = Path(doc2)
        cloakmap1 = Path(cloakmap1) if cloakmap1 else None
        cloakmap2 = Path(cloakmap2) if cloakmap2 else None
        output = Path(output) if output else None

        if verbose:
            click.echo("📊 Comparing documents")
            click.echo(f"   Document 1: {doc1}")
            click.echo(f"   Document 2: {doc2}")
            if cloakmap1:
                click.echo(f"   CloakMap 1: {cloakmap1}")
            if cloakmap2:
                click.echo(f"   CloakMap 2: {cloakmap2}")

        # Load documents
        with click.progressbar(length=1, label="Loading documents") as progress:
            from cloakpivot.document.processor import DocumentProcessor

            processor = DocumentProcessor()
            document1 = processor.load_document(doc1, validate=True)
            document2 = processor.load_document(doc2, validate=True)
            progress.update(1)

        if verbose:
            click.echo(f"✓ Loaded document 1: {document1.name}")
            click.echo(f"✓ Loaded document 2: {document2.name}")

        # Load CloakMaps if provided
        cloakmap1_obj = None
        cloakmap2_obj = None

        if cloakmap1 or cloakmap2:
            with click.progressbar(length=1, label="Loading CloakMaps") as progress:
                from cloakpivot.core.cloakmap import CloakMap

                if cloakmap1:
                    with open(cloakmap1, encoding="utf-8") as f:
                        cloakmap1_data = json.load(f)
                    cloakmap1_obj = CloakMap.from_dict(cloakmap1_data)

                if cloakmap2:
                    with open(cloakmap2, encoding="utf-8") as f:
                        cloakmap2_data = json.load(f)
                    cloakmap2_obj = CloakMap.from_dict(cloakmap2_data)

                progress.update(1)

        # Extract text content for comparison
        from cloakpivot.document.extractor import TextExtractor

        text_extractor = TextExtractor()
        text1_segments = text_extractor.extract_text_segments(document1)
        text2_segments = text_extractor.extract_text_segments(document2)

        # Combine text segments into full text for comparison
        text1_full = "\n".join(segment.text for segment in text1_segments)
        text2_full = "\n".join(segment.text for segment in text2_segments)

        text1_lines = text1_full.splitlines()
        text2_lines = text2_full.splitlines()

        if verbose:
            click.echo(f"📝 Document 1 has {len(text1_lines)} lines")
            click.echo(f"📝 Document 2 has {len(text2_lines)} lines")

        # Generate text diff
        with click.progressbar(length=1, label="Generating diff") as progress:
            differ = difflib.unified_diff(
                text1_lines,
                text2_lines,
                fromfile=str(doc1),
                tofile=str(doc2),
                n=show_context,
                lineterm="",
            )
            diff_lines = list(differ)
            progress.update(1)

        # Analyze masking differences if CloakMaps provided
        masking_analysis: dict[str, Any] | None = None
        if cloakmap1_obj or cloakmap2_obj:
            masking_analysis = {}

            if cloakmap1_obj:
                masking_analysis["doc1_entities"] = len(cloakmap1_obj.anchors)
                masking_analysis["doc1_strategies"] = {}
                for anchor in cloakmap1_obj.anchors:
                    strategy = anchor.strategy_used
                    masking_analysis["doc1_strategies"][strategy] = (
                        masking_analysis["doc1_strategies"].get(strategy, 0) + 1
                    )
                masking_analysis["doc1_entity_types"] = (
                    cloakmap1_obj.entity_count_by_type
                )

            if cloakmap2_obj:
                masking_analysis["doc2_entities"] = len(cloakmap2_obj.anchors)
                masking_analysis["doc2_strategies"] = {}
                for anchor in cloakmap2_obj.anchors:
                    strategy = anchor.strategy_used
                    masking_analysis["doc2_strategies"][strategy] = (
                        masking_analysis["doc2_strategies"].get(strategy, 0) + 1
                    )
                masking_analysis["doc2_entity_types"] = (
                    cloakmap2_obj.entity_count_by_type
                )

        # Generate report based on format
        if report_format == "text":
            _generate_text_diff_report(
                diff_lines, masking_analysis, output, doc1, doc2, verbose
            )
        elif report_format == "html":
            _generate_html_diff_report(
                text1_lines, text2_lines, masking_analysis, output, doc1, doc2, verbose
            )
        elif report_format == "json":
            _generate_json_diff_report(
                diff_lines, masking_analysis, output, doc1, doc2, verbose
            )

        # Show summary
        changes = sum(
            1
            for line in diff_lines
            if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
        )

        click.echo("✅ Document comparison completed!")
        click.echo(f"   Document 1: {doc1}")
        click.echo(f"   Document 2: {doc2}")
        click.echo(f"   Changes detected: {changes // 2 if changes > 0 else 0}")

        if masking_analysis:
            entities1 = masking_analysis.get("doc1_entities", 0)
            entities2 = masking_analysis.get("doc2_entities", 0)
            click.echo(f"   Entities in doc1: {entities1}")
            click.echo(f"   Entities in doc2: {entities2}")
            click.echo(f"   Entity difference: {abs(entities1 - entities2)}")

        if output:
            click.echo(f"   Report saved: {output}")

    except ImportError as e:
        raise click.ClickException(f"Missing required dependency: {e}") from e
    except FileNotFoundError as e:
        raise click.ClickException(f"File not found: {e}") from e
    except json.JSONDecodeError as e:
        raise click.ClickException(f"Invalid JSON in file: {e}") from e
    except Exception as e:
        if verbose:
            import traceback

            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Document comparison failed: {e}") from e


def _generate_text_diff_report(
    diff_lines: list[str],
    masking_analysis: dict[str, Any] | None,
    output: Path | None,
    doc1: Path,
    doc2: Path,
    verbose: bool,
) -> None:
    """Generate text-based diff report."""
    if not output:
        output = Path("diff_report.txt")

    with open(output, "w", encoding="utf-8") as f:
        f.write("Document Comparison Report\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Document 1: {doc1}\n")
        f.write(f"Document 2: {doc2}\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        if masking_analysis:
            f.write("Masking Analysis:\n")
            f.write("-" * 20 + "\n")

            if "doc1_entities" in masking_analysis:
                f.write(f"Document 1 entities: {masking_analysis['doc1_entities']}\n")
                if masking_analysis.get("doc1_strategies"):
                    f.write("  Strategies used:\n")
                    for strategy, count in masking_analysis["doc1_strategies"].items():
                        f.write(f"    {strategy}: {count}\n")

            if "doc2_entities" in masking_analysis:
                f.write(f"Document 2 entities: {masking_analysis['doc2_entities']}\n")
                if masking_analysis.get("doc2_strategies"):
                    f.write("  Strategies used:\n")
                    for strategy, count in masking_analysis["doc2_strategies"].items():
                        f.write(f"    {strategy}: {count}\n")
            f.write("\n")

        f.write("Content Differences:\n")
        f.write("-" * 20 + "\n")

        if diff_lines:
            for line in diff_lines:
                f.write(line + "\n")
        else:
            f.write("No differences found.\n")


def _generate_html_diff_report(
    text1_lines: list[str],
    text2_lines: list[str],
    masking_analysis: dict[str, Any] | None,
    output: Path | None,
    doc1: Path,
    doc2: Path,
    verbose: bool,
) -> None:
    """Generate HTML-based diff report with highlighting."""
    import difflib
    from datetime import datetime

    if not output:
        output = Path("diff_report.html")

    # Generate HTML diff
    differ = difflib.HtmlDiff()
    html_diff = differ.make_file(
        text1_lines,
        text2_lines,
        fromdesc=str(doc1),
        todesc=str(doc2),
        context=True,
        numlines=3,
    )

    # Enhanced HTML with masking analysis
    enhanced_html = f"""<!DOCTYPE html>
<html>
<head>
    <title>CloakPivot Document Comparison</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f5f5f5; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .analysis {{ background-color: #e6f3ff; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
        .diff-container {{ border: 1px solid #ddd; border-radius: 5px; }}
        table {{ width: 100%; border-collapse: collapse; }}
        .diff_header {{ background-color: #f0f0f0; }}
        .diff_next {{ background-color: #c0c0c0; }}
        .diff_add {{ background-color: #aaffaa; }}
        .diff_chg {{ background-color: #ffff77; }}
        .diff_sub {{ background-color: #ffaaaa; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>📊 CloakPivot Document Comparison</h1>
        <p><strong>Document 1:</strong> {doc1}</p>
        <p><strong>Document 2:</strong> {doc2}</p>
        <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
    </div>
"""

    if masking_analysis:
        enhanced_html += '<div class="analysis">\n<h2>🎭 Masking Analysis</h2>\n'

        if "doc1_entities" in masking_analysis:
            enhanced_html += "<h3>Document 1</h3>\n"
            enhanced_html += f"<p><strong>Entities:</strong> {masking_analysis['doc1_entities']}</p>\n"
            if masking_analysis.get("doc1_strategies"):
                enhanced_html += "<p><strong>Strategies:</strong></p>\n<ul>\n"
                for strategy, count in masking_analysis["doc1_strategies"].items():
                    enhanced_html += f"<li>{strategy}: {count}</li>\n"
                enhanced_html += "</ul>\n"

        if "doc2_entities" in masking_analysis:
            enhanced_html += "<h3>Document 2</h3>\n"
            enhanced_html += f"<p><strong>Entities:</strong> {masking_analysis['doc2_entities']}</p>\n"
            if masking_analysis.get("doc2_strategies"):
                enhanced_html += "<p><strong>Strategies:</strong></p>\n<ul>\n"
                for strategy, count in masking_analysis["doc2_strategies"].items():
                    enhanced_html += f"<li>{strategy}: {count}</li>\n"
                enhanced_html += "</ul>\n"

        enhanced_html += "</div>\n"

    enhanced_html += '<div class="diff-container">\n<h2>📄 Content Differences</h2>\n'
    enhanced_html += html_diff[
        html_diff.find("<table") : html_diff.rfind("</table>") + 8
    ]
    enhanced_html += "</div>\n</body>\n</html>"

    with open(output, "w", encoding="utf-8") as f:
        f.write(enhanced_html)


def _generate_json_diff_report(
    diff_lines: list[str],
    masking_analysis: dict[str, Any] | None,
    output: Path | None,
    doc1: Path,
    doc2: Path,
    verbose: bool,
) -> None:
    """Generate JSON-based diff report."""
    import json
    from datetime import datetime

    if not output:
        output = Path("diff_report.json")

    report_data = {
        "comparison": {
            "document1": str(doc1),
            "document2": str(doc2),
            "timestamp": datetime.now().isoformat(),
            "tool": "CloakPivot",
        },
        "differences": {
            "lines": diff_lines,
            "change_count": sum(
                1
                for line in diff_lines
                if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
            ),
        },
    }

    if masking_analysis:
        report_data["masking_analysis"] = masking_analysis

    with open(output, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, default=str)


@cli.command(hidden=True)
@click.argument("shell", type=click.Choice(["bash", "zsh", "fish"]))
def completion(shell: str) -> None:
    """Generate shell completion script.

    This is used internally by the shell completion system.
    Use the instructions in --help for manual setup.
    """
    import os

    shell_complete = f"_{(cli.name or 'cloakpivot').upper()}_COMPLETE"
    if shell == "bash":
        os.environ[shell_complete] = "bash_source"
    elif shell == "zsh":
        os.environ[shell_complete] = "zsh_source"
    elif shell == "fish":
        os.environ[shell_complete] = "fish_source"

    # This will output the completion script
    cli.main(standalone_mode=False)


# Add command groups
cli.add_command(batch)


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
