"""CLI commands for batch processing operations."""

import sys
from pathlib import Path
from typing import Optional

import click

from ..core.batch import BatchConfig, BatchOperationType, BatchProcessor
from ..core.policies import MaskingPolicy


@click.group()
def batch() -> None:
    """Batch processing operations for multiple documents.

    Batch operations support processing multiple files with pattern matching,
    progress reporting, error isolation, and configurable resource management.
    """
    pass


def _validate_patterns(patterns: list[str]) -> list[str]:
    """Validate and normalize input patterns."""
    if not patterns:
        raise click.ClickException("At least one input pattern must be specified")

    # Expand patterns to handle shell-style globbing
    validated_patterns = []
    for pattern in patterns:
        # Convert to absolute path if relative
        pattern_path = Path(pattern)
        if not pattern_path.is_absolute():
            pattern = str(Path.cwd() / pattern)
        validated_patterns.append(pattern)

    return validated_patterns


def _load_masking_policy(policy_file: Optional[Path], verbose: bool) -> Optional[MaskingPolicy]:
    """Load masking policy from file if specified."""
    if not policy_file:
        return None

    if verbose:
        click.echo(f"📋 Loading policy: {policy_file}")

    try:
        from ..core.policy_loader import (
            PolicyInheritanceError,
            PolicyLoader,
            PolicyValidationError,
        )

        # Try enhanced policy loader first
        try:
            loader = PolicyLoader()
            masking_policy = loader.load_policy(policy_file)
            if verbose:
                click.echo("✓ Enhanced policy loaded successfully")
            return masking_policy
        except (PolicyValidationError, PolicyInheritanceError, ImportError) as e:
            if verbose:
                click.echo(f"⚠️  Enhanced policy loading failed: {e}")
                click.echo("   Falling back to basic policy loading")

            # Fall back to basic loading
            import yaml

            with open(policy_file, encoding="utf-8") as f:
                policy_data = yaml.safe_load(f)
            masking_policy = MaskingPolicy.from_dict(policy_data)
            if verbose:
                click.echo("✓ Basic policy loaded successfully")
            return masking_policy

    except ImportError:
        click.echo("⚠️  PyYAML not installed, cannot load policy file")
        raise click.ClickException("Missing required dependency: PyYAML") from None
    except Exception as e:
        click.echo(f"⚠️  Failed to load policy file: {e}")
        raise click.ClickException(f"Policy loading failed: {e}") from None


@batch.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option(
    "--out-dir", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for masked documents"
)
@click.option(
    "--cloakmap-dir",
    type=click.Path(path_type=Path),
    help="Directory for CloakMap files (default: same as output directory)"
)
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Path to masking policy file"
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["lexical", "docling", "markdown", "html"]),
    default="lexical",
    help="Output format (default: lexical)"
)
@click.option(
    "--max-workers", "-w",
    type=int,
    default=4,
    help="Maximum number of worker threads (default: 4)"
)
@click.option(
    "--max-files",
    type=int,
    help="Maximum number of files to process in this batch"
)
@click.option(
    "--max-retries",
    type=int,
    default=2,
    help="Maximum retry attempts for failed files (default: 2)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files"
)
@click.option(
    "--preserve-structure",
    is_flag=True,
    default=True,
    help="Preserve directory structure in output (default: enabled)"
)
@click.option(
    "--throttle-delay",
    type=float,
    default=0.0,
    help="Delay between file operations in seconds (default: 0.0)"
)
@click.option(
    "--max-memory",
    type=float,
    help="Maximum memory usage in MB (will cancel batch if exceeded)"
)
@click.option(
    "--validate/--no-validate",
    default=True,
    help="Validate outputs after processing (default: enabled)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def mask(
    ctx: click.Context,
    patterns: tuple[str, ...],
    out_dir: Path,
    cloakmap_dir: Optional[Path],
    policy: Optional[Path],
    output_format: str,
    max_workers: int,
    max_files: Optional[int],
    max_retries: int,
    overwrite: bool,
    preserve_structure: bool,
    throttle_delay: float,
    max_memory: Optional[float],
    validate: bool,
    verbose: bool,
) -> None:
    """Batch mask PII in multiple documents.

    Process multiple documents matching the specified PATTERNS and create
    masked versions with CloakMaps for later unmasking.

    Examples:
        # Mask all PDFs in current directory
        cloakpivot batch mask "*.pdf" --out-dir ./masked

        # Mask documents in specific directory with custom policy
        cloakpivot batch mask "data/**/*.json" --out-dir ./output --policy policy.yaml

        # High-throughput batch with more workers
        cloakpivot batch mask "docs/**/*" --out-dir ./masked --max-workers 8
    """
    verbose = verbose or (ctx.obj and ctx.obj.get("verbose", False))

    try:
        # Validate inputs
        pattern_list = _validate_patterns(list(patterns))

        if not out_dir:
            raise click.ClickException("Output directory must be specified")

        # Set default cloakmap directory
        if not cloakmap_dir:
            cloakmap_dir = out_dir

        # Load policy if specified
        masking_policy = _load_masking_policy(policy, verbose)

        # Create configuration
        config = BatchConfig(
            operation_type=BatchOperationType.MASK,
            input_patterns=pattern_list,
            output_directory=out_dir,
            cloakmap_directory=cloakmap_dir,
            max_workers=max_workers,
            max_files_per_batch=max_files,
            max_retries=max_retries,
            overwrite_existing=overwrite,
            preserve_directory_structure=preserve_structure,
            throttle_delay_ms=throttle_delay * 1000,
            max_memory_mb=max_memory,
            output_format=output_format,
            masking_policy=masking_policy,
            validate_outputs=validate,
            verbose_logging=verbose,
        )

        if verbose:
            click.echo("🚀 Starting batch masking operation")
            click.echo(f"   Patterns: {pattern_list}")
            click.echo(f"   Output directory: {out_dir}")
            click.echo(f"   CloakMap directory: {cloakmap_dir}")
            click.echo(f"   Workers: {max_workers}")
            click.echo(f"   Format: {output_format}")
            if max_files:
                click.echo(f"   File limit: {max_files}")

        # Create batch processor and run
        processor = BatchProcessor(config)

        try:
            result = processor.process_batch()

            # Report final results
            if result.failed_files > 0:
                click.echo(f"⚠️  {result.failed_files} files failed processing:")
                for file_result in result.file_results:
                    if file_result.error:
                        click.echo(f"   • {file_result.file_path.name}: {file_result.error}")

            if result.success_rate < 100.0:
                raise click.ClickException("Batch processing completed with failures")

        except KeyboardInterrupt:
            click.echo("\n🛑 Batch processing cancelled by user")
            processor.cancel()
            import sys
            sys.exit(130)  # SIGINT

    except Exception as e:
        if verbose:
            import traceback
            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Batch masking failed: {e}") from e


@batch.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option(
    "--cloakmap-dir",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Directory containing CloakMap files"
)
@click.option(
    "--out-dir", "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for restored documents"
)
@click.option(
    "--max-workers", "-w",
    type=int,
    default=4,
    help="Maximum number of worker threads (default: 4)"
)
@click.option(
    "--max-files",
    type=int,
    help="Maximum number of files to process in this batch"
)
@click.option(
    "--max-retries",
    type=int,
    default=2,
    help="Maximum retry attempts for failed files (default: 2)"
)
@click.option(
    "--overwrite",
    is_flag=True,
    help="Overwrite existing output files"
)
@click.option(
    "--preserve-structure",
    is_flag=True,
    default=True,
    help="Preserve directory structure in output (default: enabled)"
)
@click.option(
    "--verify-integrity/--no-verify",
    default=True,
    help="Verify CloakMap integrity before unmasking (default: enabled)"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def unmask(
    ctx: click.Context,
    patterns: tuple[str, ...],
    cloakmap_dir: Path,
    out_dir: Path,
    max_workers: int,
    max_files: Optional[int],
    max_retries: int,
    overwrite: bool,
    preserve_structure: bool,
    verify_integrity: bool,
    verbose: bool,
) -> None:
    """Batch unmask previously masked documents.

    Process multiple masked documents matching the specified PATTERNS and
    restore them using their corresponding CloakMaps.

    Examples:
        # Unmask all masked files in directory
        cloakpivot batch unmask "masked/*.json" --cloakmap-dir ./cloakmaps --out-dir ./restored

        # Unmask with integrity verification disabled (faster)
        cloakpivot batch unmask "masked/**/*.json" --cloakmap-dir ./maps --out-dir ./restored --no-verify
    """
    verbose = verbose or (ctx.obj and ctx.obj.get("verbose", False))

    try:
        # Validate inputs
        pattern_list = _validate_patterns(list(patterns))

        # Create configuration
        config = BatchConfig(
            operation_type=BatchOperationType.UNMASK,
            input_patterns=pattern_list,
            output_directory=out_dir,
            cloakmap_directory=cloakmap_dir,
            max_workers=max_workers,
            max_files_per_batch=max_files,
            max_retries=max_retries,
            overwrite_existing=overwrite,
            preserve_directory_structure=preserve_structure,
            validate_outputs=verify_integrity,
            verbose_logging=verbose,
        )

        if verbose:
            click.echo("🔓 Starting batch unmasking operation")
            click.echo(f"   Patterns: {pattern_list}")
            click.echo(f"   CloakMap directory: {cloakmap_dir}")
            click.echo(f"   Output directory: {out_dir}")
            click.echo(f"   Workers: {max_workers}")
            if max_files:
                click.echo(f"   File limit: {max_files}")

        # Create batch processor and run
        processor = BatchProcessor(config)

        try:
            result = processor.process_batch()

            # Report final results
            if result.failed_files > 0:
                click.echo(f"⚠️  {result.failed_files} files failed processing:")
                for file_result in result.file_results:
                    if file_result.error:
                        click.echo(f"   • {file_result.file_path.name}: {file_result.error}")

            if result.success_rate < 100.0:
                raise click.ClickException("Batch processing completed with failures")

        except KeyboardInterrupt:
            click.echo("\n🛑 Batch processing cancelled by user")
            processor.cancel()
            import sys
            sys.exit(130)  # SIGINT

    except Exception as e:
        if verbose:
            import traceback
            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Batch unmasking failed: {e}") from e


@batch.command()
@click.argument("patterns", nargs=-1, required=True)
@click.option(
    "--out-dir", "-o",
    type=click.Path(path_type=Path),
    help="Output directory for analysis results (optional)"
)
@click.option(
    "--policy",
    type=click.Path(exists=True, path_type=Path),
    help="Path to masking policy file for analysis configuration"
)
@click.option(
    "--max-workers", "-w",
    type=int,
    default=4,
    help="Maximum number of worker threads (default: 4)"
)
@click.option(
    "--max-files",
    type=int,
    help="Maximum number of files to process in this batch"
)
@click.option(
    "--summary-only",
    is_flag=True,
    help="Show only summary statistics, don't save detailed analysis files"
)
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.pass_context
def analyze(
    ctx: click.Context,
    patterns: tuple[str, ...],
    out_dir: Optional[Path],
    policy: Optional[Path],
    max_workers: int,
    max_files: Optional[int],
    summary_only: bool,
    verbose: bool,
) -> None:
    """Batch analyze documents for PII without masking.

    Process multiple documents matching the specified PATTERNS and analyze
    them for PII entities, generating statistics and reports.

    Examples:
        # Analyze all documents and show summary
        cloakpivot batch analyze "docs/**/*.pdf" --summary-only

        # Analyze and save detailed results
        cloakpivot batch analyze "data/*.json" --out-dir ./analysis

        # Analyze with custom detection policy
        cloakpivot batch analyze "**/*.txt" --policy analysis-policy.yaml --out-dir ./results
    """
    verbose = verbose or (ctx.obj and ctx.obj.get("verbose", False))

    try:
        # Validate inputs
        pattern_list = _validate_patterns(list(patterns))

        # Load policy if specified
        masking_policy = _load_masking_policy(policy, verbose)

        # Create configuration
        config = BatchConfig(
            operation_type=BatchOperationType.ANALYZE,
            input_patterns=pattern_list,
            output_directory=out_dir if not summary_only else None,
            max_workers=max_workers,
            max_files_per_batch=max_files,
            masking_policy=masking_policy,
            verbose_logging=verbose,
        )

        if verbose:
            click.echo("🔍 Starting batch analysis operation")
            click.echo(f"   Patterns: {pattern_list}")
            if out_dir and not summary_only:
                click.echo(f"   Output directory: {out_dir}")
            click.echo(f"   Workers: {max_workers}")
            if max_files:
                click.echo(f"   File limit: {max_files}")

        # Create batch processor and run
        processor = BatchProcessor(config)

        try:
            result = processor.process_batch()

            # Show analysis summary
            click.echo("\n📊 Batch Analysis Summary")
            click.echo("=" * 50)
            click.echo(f"Total files processed: {result.total_files}")
            click.echo(f"Successful analyses: {result.successful_files}")
            click.echo(f"Failed analyses: {result.failed_files}")
            click.echo(f"Total entities found: {result.total_entities_processed}")
            click.echo(f"Success rate: {result.success_rate:.1f}%")
            click.echo(f"Processing time: {result.duration_ms / 1000:.1f} seconds")
            click.echo(f"Throughput: {result.throughput_files_per_second:.1f} files/sec")

            # Show entity breakdown if we have detailed results
            if result.file_results:

                if verbose:
                    click.echo("\n📋 Per-File Results:")
                    for file_result in result.file_results[:10]:  # Show first 10
                        status_icon = "✅" if file_result.status.value == "completed" else "❌"
                        click.echo(
                            f"   {status_icon} {file_result.file_path.name}: "
                            f"{file_result.entities_processed} entities"
                        )
                    if len(result.file_results) > 10:
                        click.echo(f"   ... and {len(result.file_results) - 10} more files")

            # Report failures
            if result.failed_files > 0:
                click.echo(f"\n⚠️  Failed Files ({result.failed_files}):")
                for file_result in result.file_results:
                    if file_result.error:
                        click.echo(f"   • {file_result.file_path.name}: {file_result.error}")

                raise click.ClickException("Batch processing completed with failures")

        except KeyboardInterrupt:
            click.echo("\n🛑 Batch analysis cancelled by user")
            processor.cancel()
            import sys
            sys.exit(130)  # SIGINT

    except Exception as e:
        if verbose:
            import traceback
            click.echo(f"Error details:\n{traceback.format_exc()}")
        raise click.ClickException(f"Batch analysis failed: {e}") from e


@batch.command()
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "yaml", "text"]),
    default="text",
    help="Output format for the sample configuration (default: text)"
)
@click.option(
    "--output", "-o",
    type=click.File("w"),
    default="-",
    help="Output file (default: stdout)"
)
def config_sample(output_format: str, output) -> None:
    """Generate a sample batch processing configuration.

    This command generates example configurations showing all available
    batch processing options and their default values.

    Example:
        cloakpivot batch config-sample --format yaml > batch-config.yaml
    """
    if output_format == "yaml":
        sample_config = """# CloakPivot Batch Processing Configuration Sample

# Operation settings
operation_type: "mask"  # mask, unmask, analyze
input_patterns:
  - "documents/**/*.pdf"
  - "data/**/*.json"

# Output settings
output_directory: "./output"
cloakmap_directory: "./cloakmaps"  # Optional: defaults to output_directory
output_format: "lexical"  # lexical, docling, markdown, html
preserve_directory_structure: true
overwrite_existing: false

# Processing settings
max_workers: 4
max_files_per_batch: null  # No limit
max_retries: 2
retry_delay_seconds: 1.0

# Resource management
max_memory_mb: null  # No limit
throttle_delay_ms: 0.0

# Validation and quality
validate_outputs: true
verbose_logging: false

# Policy configuration (for mask and analyze operations)
# masking_policy_file: "./policy.yaml"
"""
    elif output_format == "json":
        sample_config = """{
  "operation_type": "mask",
  "input_patterns": [
    "documents/**/*.pdf",
    "data/**/*.json"
  ],
  "output_directory": "./output",
  "cloakmap_directory": "./cloakmaps",
  "output_format": "lexical",
  "preserve_directory_structure": true,
  "overwrite_existing": false,
  "max_workers": 4,
  "max_files_per_batch": null,
  "max_retries": 2,
  "retry_delay_seconds": 1.0,
  "max_memory_mb": null,
  "throttle_delay_ms": 0.0,
  "validate_outputs": true,
  "verbose_logging": false
}"""
    else:  # text format
        sample_config = """CloakPivot Batch Processing Configuration Options
================================================

Core Operation Settings:
  operation_type        Type of batch operation (mask, unmask, analyze)
  input_patterns        List of glob patterns for input files
  output_directory      Directory for processed output files
  cloakmap_directory    Directory for CloakMap files (mask/unmask operations)

Processing Configuration:
  max_workers           Number of parallel worker threads (default: 4)
  max_files_per_batch   Limit on files processed per batch (default: unlimited)
  max_retries           Retry attempts for failed files (default: 2)
  retry_delay_seconds   Delay between retries in seconds (default: 1.0)

Resource Management:
  max_memory_mb         Memory limit in MB (batch cancelled if exceeded)
  throttle_delay_ms     Delay between file operations (default: 0.0)

Output and Quality:
  output_format         Format for output files (lexical, docling, markdown, html)
  preserve_structure    Maintain directory structure in output (default: true)
  overwrite_existing    Overwrite existing output files (default: false)
  validate_outputs      Validate outputs after processing (default: true)

Logging and Monitoring:
  verbose_logging       Enable detailed progress logging (default: false)

Example CLI Usage:
  # Basic batch masking
  cloakpivot batch mask "docs/**/*.pdf" --out-dir ./masked

  # High-throughput processing
  cloakpivot batch mask "data/*.json" --out-dir ./output --max-workers 8

  # With resource limits
  cloakpivot batch mask "large-docs/*" --out-dir ./masked --max-memory 2048

  # Analysis only (no masking)
  cloakpivot batch analyze "**/*.txt" --summary-only
"""

    output.write(sample_config)
    if output != sys.stdout:
        click.echo(f"Sample configuration written to {output.name}")
