"""Main CLI entry point for CloakPivot."""

import sys
from pathlib import Path
from typing import Optional, TextIO

import click

from cloakpivot import __version__


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


def _validate_mask_arguments(output_path: Optional[Path], cloakmap: Optional[Path]) -> None:
    """Validate mask command arguments."""
    if not output_path and not cloakmap:
        raise click.ClickException(
            "Must specify either --out for masked output or --cloakmap for CloakMap output"
        )


def _set_default_paths(input_path: Path, output_path: Optional[Path],
                      cloakmap: Optional[Path], output_format: str) -> tuple[Path, Path]:
    """Set default output paths if not specified."""
    if not output_path:
        output_path = input_path.with_suffix(f".masked.{output_format}.json")
    if not cloakmap:
        cloakmap = input_path.with_suffix(".cloakmap.json")
    return output_path, cloakmap


def _load_masking_policy(policy: Optional[Path], verbose: bool):
    """Load masking policy from file or use default."""
    from ..core.policies import MaskingPolicy

    if policy:
        if verbose:
            click.echo(f"📋 Loading policy: {policy}")
        try:
            import yaml
            with open(policy, encoding='utf-8') as f:
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
        masking_policy = MaskingPolicy()
    return masking_policy


def _perform_entity_detection(document, masking_policy, verbose: bool):
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


def _prepare_entities_for_masking(detection_result):
    """Convert detection results to format expected by masking engine."""
    from presidio_analyzer import RecognizerResult

    entities = []
    for segment_result in detection_result.segment_results:
        for entity in segment_result.entities:
            recognizer_result = RecognizerResult(
                entity_type=entity.entity_type,
                start=entity.start + segment_result.segment.start_offset,
                end=entity.end + segment_result.segment.start_offset,
                score=entity.confidence
            )
            entities.append(recognizer_result)
    return entities


def _perform_masking(document, entities, masking_policy, verbose: bool):
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
            text_segments=text_segments
        )
        progress.update(1)

    if verbose:
        click.echo(f"✓ Masked {len(masking_result.cloakmap.anchors)} entities")

    return masking_result


def _save_masked_document(masking_result, output_path: Path, verbose: bool) -> None:
    """Save the masked document to file."""
    if verbose:
        click.echo(f"💾 Saving masked document: {output_path}")

    with click.progressbar(length=1, label="Saving masked document") as progress:
        from docpivot import LexicalDocSerializer
        serializer = LexicalDocSerializer()
        serialized_content = serializer.serialize(masking_result.masked_document)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(serialized_content)
        progress.update(1)


def _save_cloakmap(masking_result, cloakmap: Path, verbose: bool) -> None:
    """Save the CloakMap to file."""
    import json

    if verbose:
        click.echo(f"🗺️  Saving CloakMap: {cloakmap}")

    with click.progressbar(length=1, label="Saving CloakMap") as progress:
        with open(cloakmap, 'w', encoding='utf-8') as f:
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
                with open(policy, encoding='utf-8') as f:
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
            click.echo(f"🔍 Detecting PII entities (min_score: {min_score}, lang: {lang})")

        detection_pipeline = EntityDetectionPipeline()

        # Detect entities with progress
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
                    start=entity.start + segment_result.segment.start_offset,  # Convert to global offset
                    end=entity.end + segment_result.segment.start_offset,
                    score=entity.confidence
                )
                entities.append(recognizer_result)

        # Perform masking
        with click.progressbar(length=1, label="Masking PII entities") as progress:
            masking_result = masking_engine.mask_document(
                document=document,
                entities=entities,
                policy=masking_policy,
                text_segments=text_segments
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
            serializer = LexicalDocSerializer()
            serialized_content = serializer.serialize(masking_result.masked_document)

            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(serialized_content)
            progress.update(1)

        # Save CloakMap
        if verbose:
            click.echo(f"🗺️  Saving CloakMap: {cloakmap}")

        with click.progressbar(length=1, label="Saving CloakMap") as progress:
            with open(cloakmap, 'w', encoding='utf-8') as f:
                json.dump(masking_result.cloakmap.to_dict(), f, indent=2, default=str)
            progress.update(1)

        # Success message
        click.echo("✅ Masking completed successfully!")
        click.echo(f"   Masked document: {output_path}")
        click.echo(f"   CloakMap: {cloakmap}")
        if masking_result.stats:
            click.echo(f"   Entities processed: {masking_result.stats['total_entities_masked']}")

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
            with open(cloakmap, encoding='utf-8') as f:
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
                    verify_integrity=True
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
                if integrity.get('valid', False):
                    if verbose:
                        click.echo("✅ Integrity verification passed")
                else:
                    click.echo("⚠️  Integrity verification failed:")
                    for issue in integrity.get('issues', []):
                        click.echo(f"   - {issue}")

            if verify_only:
                # Verification mode - just report results
                if unmasking_result.integrity_report and unmasking_result.integrity_report.get('valid', False):
                    click.echo("✅ CloakMap verification successful")
                    click.echo(f"   Document: {masked_path}")
                    click.echo(f"   CloakMap: {cloakmap}")
                    click.echo(f"   Anchors: {len(cloakmap_obj.anchors)}")
                else:
                    click.echo("❌ CloakMap verification failed")
                    raise click.ClickException("Verification failed - see issues above")
            else:
                # Save restored document
                if verbose:
                    click.echo(f"💾 Saving restored document: {output_path}")

                with click.progressbar(length=1, label="Saving restored document") as progress:
                    # Use docpivot serializer to save the restored document
                    from docpivot import LexicalDocSerializer
                    serializer = LexicalDocSerializer()
                    serialized_content = serializer.serialize(unmasking_result.restored_document)

                    with open(output_path, 'w', encoding='utf-8') as f:
                        f.write(serialized_content)
                    progress.update(1)

                # Success message
                click.echo("✅ Unmasking completed successfully!")
                click.echo(f"   Masked document: {masked_path}")
                click.echo(f"   CloakMap: {cloakmap}")
                click.echo(f"   Restored document: {output_path}")
                if unmasking_result.stats:
                    click.echo(f"   Entities restored: {unmasking_result.stats.get('resolved_anchors', 0)}")

        except Exception as unmasking_error:
            # Handle unmasking-specific errors
            error_msg = str(unmasking_error)
            if "compatibility" in error_msg.lower():
                click.echo("❌ Document-CloakMap compatibility issue")
                click.echo("   The CloakMap may not match the provided masked document")
            elif "anchor" in error_msg.lower():
                click.echo("❌ Anchor resolution failed")
                click.echo("   Some replacement tokens could not be located in the document")
            else:
                click.echo(f"❌ Unmasking failed: {error_msg}")

            if verbose:
                import traceback
                click.echo(f"\nError details:\n{traceback.format_exc()}")

            raise click.ClickException(f"Unmasking operation failed: {error_msg}") from None

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
    sample_policy = """# CloakPivot Masking Policy Configuration
# This is a sample policy that demonstrates available options

# Default masking strategy applied to all entities
default_strategy:
  kind: "redact"  # redact, template, hash, surrogate, partial, custom
  template: "[REDACTED]"

# Language and locale settings
locale: "en"
confidence_threshold: 0.5

# Per-entity type strategies
per_entity:
  PERSON:
    kind: "template"
    template: "[PERSON-{id}]"

  EMAIL_ADDRESS:
    kind: "partial"
    show_prefix: 2
    show_suffix: 0
    mask_char: "*"

  PHONE_NUMBER:
    kind: "partial"
    show_prefix: 0
    show_suffix: 4
    mask_char: "X"

  CREDIT_CARD:
    kind: "hash"
    algorithm: "sha256"
    salt: "your-secret-salt"

  SSN:
    kind: "surrogate"
    format_preserving: true

# Advanced options
custom_recognizers: []
allow_list: []
deny_list: []

# Security settings
encryption:
  enabled: false
  key_id: null

# Audit and logging
audit_trail: true
log_level: "INFO"
"""
    output.write(sample_policy)
    if output != sys.stdout:
        click.echo(f"Sample policy written to {output.name}")


def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
