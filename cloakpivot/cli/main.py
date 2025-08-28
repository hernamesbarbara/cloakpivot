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
    if ctx.obj["verbose"]:
        click.echo(f"Masking document: {input_path}")
        click.echo(f"Output format: {output_format}")
        click.echo(f"Language: {lang}")
        click.echo(f"Min score: {min_score}")

    # TODO: Implement masking functionality
    click.echo("🔒 Document masking is not yet implemented.")
    click.echo(f"Input: {input_path}")
    if output_path:
        click.echo(f"Output: {output_path}")
    if cloakmap:
        click.echo(f"CloakMap: {cloakmap}")

    sys.exit(1)


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
    if ctx.obj["verbose"]:
        click.echo(f"Unmasking document: {masked_path}")
        click.echo(f"CloakMap: {cloakmap}")
        if verify_only:
            click.echo("Verification mode only")

    # TODO: Implement unmasking functionality
    click.echo("🔓 Document unmasking is not yet implemented.")
    click.echo(f"Masked: {masked_path}")
    click.echo(f"CloakMap: {cloakmap}")
    if output_path:
        click.echo(f"Output: {output_path}")

    sys.exit(1)


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
