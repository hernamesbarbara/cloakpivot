#!/usr/bin/env python3
"""Simplified CloakPivot CLI - Core mask/unmask functionality only."""

import sys
from pathlib import Path

import click
from docling.document_converter import DocumentConverter

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.engine import CloakEngine


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CloakPivot - PII masking and unmasking for DoclingDocument instances."""
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: input_masked.ext)",
)
@click.option(
    "--cloakmap",
    "-c",
    type=click.Path(path_type=Path),
    help="CloakMap output file (default: input.cloakmap.json)",
)
@click.option(
    "--policy",
    "-p",
    type=click.Path(exists=True, path_type=Path),
    help="Path to masking policy YAML file",
)
@click.option(
    "--confidence",
    "-t",
    type=float,
    default=0.7,
    help="Minimum confidence threshold for entity detection (0.0-1.0)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json", "text"]),
    default="markdown",
    help="Output format for the masked document",
)
@click.pass_context
def mask(
    ctx: click.Context,
    input_file: Path,
    output: Path | None,
    cloakmap: Path | None,
    policy: Path | None,
    confidence: float,
    format: str,
) -> None:
    """Mask PII in a document and generate a CloakMap for reversal."""
    # Convert document
    click.echo(f"Converting document: {input_file}")
    converter = DocumentConverter()
    result = converter.convert(input_file)
    doc = result.document

    # Load policy if provided
    masking_policy = None
    if policy:
        import yaml

        with open(policy) as f:
            policy_data = yaml.safe_load(f)
        masking_policy = MaskingPolicy(**policy_data)

    # Create CloakEngine with configuration
    if masking_policy:
        engine = CloakEngine(default_policy=masking_policy, confidence_threshold=confidence)
    else:
        engine = CloakEngine(confidence_threshold=confidence)

    # Mask the document using CloakEngine's simple API
    click.echo("Detecting and masking PII entities...")
    mask_result = engine.mask_document(doc)

    # Save masked document
    if not output:
        output = input_file.parent / f"{input_file.stem}_masked{input_file.suffix}"

    click.echo(f"Saving masked document to: {output}")
    if format == "markdown":
        output.write_text(mask_result.document.export_to_markdown())
    elif format == "json":
        import json

        output.write_text(json.dumps(mask_result.document.export_to_dict(), indent=2))
    else:  # text
        output.write_text(mask_result.document.export_to_text())

    # Save CloakMap
    if not cloakmap:
        cloakmap = input_file.parent / f"{input_file.stem}.cloakmap.json"

    click.echo(f"Saving CloakMap to: {cloakmap}")
    mask_result.cloakmap.save_to_file(cloakmap)

    click.echo(
        f"✓ Found {mask_result.entities_found} entities, masked {mask_result.entities_masked}"
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.argument("cloakmap_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file path (default: input_unmasked.ext)",
)
@click.option(
    "--format",
    "-f",
    type=click.Choice(["markdown", "json", "text"]),
    default="markdown",
    help="Output format for the unmasked document",
)
@click.pass_context
def unmask(
    ctx: click.Context,
    input_file: Path,
    cloakmap_file: Path,
    output: Path | None,
    format: str,
) -> None:
    """Unmask a document using a CloakMap to restore original PII."""
    # Convert masked document
    click.echo(f"Loading masked document: {input_file}")
    converter = DocumentConverter()
    result = converter.convert(input_file)
    doc = result.document

    # Load CloakMap
    click.echo(f"Loading CloakMap: {cloakmap_file}")
    cloakmap = CloakMap.load_from_file(cloakmap_file)

    # Use CloakEngine for unmasking
    engine = CloakEngine()
    unmasked_doc = engine.unmask_document(doc, cloakmap)

    # Save unmasked document
    if not output:
        output = input_file.parent / f"{input_file.stem}_unmasked{input_file.suffix}"

    click.echo(f"Saving unmasked document to: {output}")
    if format == "markdown":
        output.write_text(unmasked_doc.export_to_markdown())
    elif format == "json":
        import json

        output.write_text(json.dumps(unmasked_doc.export_to_dict(), indent=2))
    else:  # text
        output.write_text(unmasked_doc.export_to_text())

    click.echo(f"✓ Restored {len(cloakmap.anchors)} PII entities")


@cli.command()
def version() -> None:
    """Show CloakPivot version."""
    from cloakpivot import __version__

    click.echo(f"CloakPivot v{__version__}")


def main() -> int:
    """Main entry point."""
    try:
        cli()
        return 0
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
