#!/usr/bin/env python3
"""Simplified CloakPivot CLI - Core mask/unmask functionality only."""

import sys
from pathlib import Path

import click
from docling.document_converter import DocumentConverter

from cloakpivot.core.types.cloakmap import CloakMap
from cloakpivot.core.policies.policies import MaskingPolicy
from cloakpivot.engine import CloakEngine


@click.group()
@click.pass_context
def cli(ctx: click.Context) -> None:
    """CloakPivot - PII masking and unmasking for DoclingDocument instances."""
    ctx.ensure_object(dict)


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    help="Output file path (default: input_masked.ext)",
)
@click.option(
    "--cloakmap",
    "-c",
    type=click.Path(),
    help="CloakMap output file (default: input.cloakmap.json)",
)
@click.option(
    "--policy",
    "-p",
    type=click.Path(exists=True),
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
    input_file: str,
    output: str | None,
    cloakmap: str | None,
    policy: str | None,
    confidence: float,
    format: str,
) -> None:
    """Mask PII in a document and generate a CloakMap for reversal."""
    # Convert string paths to Path objects
    input_path = Path(input_file)
    output_path = Path(output) if output else None
    cloakmap_path = Path(cloakmap) if cloakmap else None
    policy_path = Path(policy) if policy else None

    # Convert document
    click.echo(f"Converting document: {input_path}")
    converter = DocumentConverter()
    result = converter.convert(input_path)
    doc = result.document

    # Load policy if provided
    masking_policy = None
    if policy_path:
        import yaml

        with policy_path.open() as f:
            policy_data = yaml.safe_load(f)
        masking_policy = MaskingPolicy(**policy_data)

    # Create CloakEngine with configuration
    analyzer_config = {"confidence_threshold": confidence}
    if masking_policy:
        engine = CloakEngine(analyzer_config=analyzer_config, default_policy=masking_policy)
    else:
        engine = CloakEngine(analyzer_config=analyzer_config)

    # Mask the document using CloakEngine's simple API
    click.echo("Detecting and masking PII entities...")
    mask_result = engine.mask_document(doc)

    # Save masked document
    if not output_path:
        output_path = input_path.parent / f"{input_path.stem}_masked{input_path.suffix}"

    click.echo(f"Saving masked document to: {output_path}")
    if format == "markdown":
        output_path.write_text(mask_result.document.export_to_markdown())
    elif format == "json":
        import json

        output_path.write_text(json.dumps(mask_result.document.export_to_dict(), indent=2))
    else:  # text
        output_path.write_text(mask_result.document.export_to_text())

    # Save CloakMap
    if not cloakmap_path:
        cloakmap_path = input_path.parent / f"{input_path.stem}.cloakmap.json"

    click.echo(f"Saving CloakMap to: {cloakmap_path}")
    mask_result.cloakmap.save_to_file(cloakmap_path)

    click.echo(
        f"✓ Found {mask_result.entities_found} entities, masked {mask_result.entities_masked}"
    )


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("cloakmap_file", type=click.Path(exists=True))
@click.option(
    "--output",
    "-o",
    type=click.Path(),
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
    input_file: str,
    cloakmap_file: str,
    output: str | None,
    format: str,
) -> None:
    """Unmask a document using a CloakMap to restore original PII."""
    # Convert string paths to Path objects
    input_path = Path(input_file)
    cloakmap_file_path = Path(cloakmap_file)
    output_path = Path(output) if output else None

    # Convert masked document
    click.echo(f"Loading masked document: {input_path}")
    converter = DocumentConverter()
    result = converter.convert(input_path)
    doc = result.document

    # Load CloakMap
    click.echo(f"Loading CloakMap: {cloakmap_file_path}")
    cloakmap = CloakMap.load_from_file(cloakmap_file_path)

    # Use CloakEngine for unmasking
    engine = CloakEngine()
    unmasked_doc = engine.unmask_document(doc, cloakmap)

    # Save unmasked document
    if not output_path:
        output_path = input_path.parent / f"{input_path.stem}_unmasked{input_path.suffix}"

    click.echo(f"Saving unmasked document to: {output_path}")
    if format == "markdown":
        output_path.write_text(unmasked_doc.export_to_markdown())
    elif format == "json":
        import json

        output_path.write_text(json.dumps(unmasked_doc.export_to_dict(), indent=2))
    else:  # text
        output_path.write_text(unmasked_doc.export_to_text())

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
