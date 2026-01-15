#!/usr/bin/env python3
"""PII masking and unmasking for DoclingDocument instances.

Usage:
  cloakpivot mask <input_file> [options]
  cloakpivot unmask <input_file> <cloakmap_file> [options]
  cloakpivot version
  cloakpivot (-h | --help)

Examples:
  cloakpivot mask data/pdf/email.pdf
  cloakpivot mask data/pdf/email.pdf -o masked.md -c email.cloakmap.json
  cloakpivot mask data/pdf/email.pdf --confidence 0.8 --format json
  cloakpivot unmask masked.md email.cloakmap.json
  cloakpivot unmask masked.md email.cloakmap.json -o restored.md
  cloakpivot version

Options:
  -h, --help             Show this help message.
  -o, --output=<file>    Output file path.
  -c, --cloakmap=<file>  CloakMap output file (mask only).
  -p, --policy=<file>    Path to masking policy YAML file (mask only).
  -t, --confidence=<n>   Minimum confidence threshold 0.0-1.0 [default: 0.7].
  -f, --format=<fmt>     Output format: markdown, json, text [default: markdown].
"""

import sys
from pathlib import Path

from docopt import docopt

FORMAT_EXTENSIONS = {"markdown": ".md", "json": ".json", "text": ".txt"}


def mask(
    input_file: str,
    output: str | None,
    cloakmap: str | None,
    policy: str | None,
    confidence: float,
    format: str,
) -> None:
    """Mask PII in a document and generate a CloakMap for reversal."""
    from docling.document_converter import DocumentConverter

    from cloakpivot.engine import CloakEngine

    input_path = Path(input_file)
    output_path = Path(output) if output else None
    cloakmap_path = Path(cloakmap) if cloakmap else None
    policy_path = Path(policy) if policy else None

    print(f"Converting document: {input_path}")
    converter = DocumentConverter()
    result = converter.convert(input_path)
    doc = result.document

    masking_policy = None
    if policy_path:
        from cloakpivot.core.policies.policy_loader import PolicyLoader

        loader = PolicyLoader()
        masking_policy = loader.load_policy(policy_path)

    analyzer_config = {"confidence_threshold": confidence}
    if masking_policy:
        engine = CloakEngine(analyzer_config=analyzer_config, default_policy=masking_policy)
    else:
        engine = CloakEngine(analyzer_config=analyzer_config)

    print("Detecting and masking PII entities...")
    mask_result = engine.mask_document(doc)

    if not output_path:
        ext = FORMAT_EXTENSIONS[format]
        output_path = input_path.parent / f"{input_path.stem}_masked{ext}"

    print(f"Saving masked document to: {output_path}")
    if format == "markdown":
        output_path.write_text(mask_result.document.export_to_markdown())
    elif format == "json":
        import json

        output_path.write_text(json.dumps(mask_result.document.export_to_dict(), indent=2))
    else:
        output_path.write_text(mask_result.document.export_to_text())

    if not cloakmap_path:
        cloakmap_path = input_path.parent / f"{input_path.stem}.cloakmap.json"

    print(f"Saving CloakMap to: {cloakmap_path}")
    mask_result.cloakmap.save_to_file(cloakmap_path)

    print(f"✓ Found {mask_result.entities_found} entities, masked {mask_result.entities_masked}")


def unmask(
    input_file: str,
    cloakmap_file: str,
    output: str | None,
    format: str,
) -> None:
    """Unmask a document using a CloakMap to restore original PII."""
    from docling.document_converter import DocumentConverter

    from cloakpivot.core.types.cloakmap import CloakMap
    from cloakpivot.engine import CloakEngine

    input_path = Path(input_file)
    cloakmap_file_path = Path(cloakmap_file)
    output_path = Path(output) if output else None

    print(f"Loading masked document: {input_path}")
    converter = DocumentConverter()
    result = converter.convert(input_path)
    doc = result.document

    print(f"Loading CloakMap: {cloakmap_file_path}")
    cloakmap = CloakMap.load_from_file(cloakmap_file_path)

    engine = CloakEngine()
    unmasked_doc = engine.unmask_document(doc, cloakmap)

    if not output_path:
        ext = FORMAT_EXTENSIONS[format]
        output_path = input_path.parent / f"{input_path.stem}_unmasked{ext}"

    print(f"Saving unmasked document to: {output_path}")
    if format == "markdown":
        output_path.write_text(unmasked_doc.export_to_markdown())
    elif format == "json":
        import json

        output_path.write_text(json.dumps(unmasked_doc.export_to_dict(), indent=2))
    else:
        output_path.write_text(unmasked_doc.export_to_text())

    print(f"✓ Restored {len(cloakmap.anchors)} PII entities")


def version() -> None:
    """Show CloakPivot version."""
    from cloakpivot import __version__

    print(f"CloakPivot v{__version__}")


def main() -> int:
    """Main entry point."""
    args = docopt(__doc__)

    # Validate input file exists
    input_file = args["<input_file>"]
    if input_file and not Path(input_file).exists():
        print(f"Error: Input file not found: {input_file}", file=sys.stderr)
        return 1

    # Validate cloakmap file exists (for unmask)
    cloakmap_file = args["<cloakmap_file>"]
    if cloakmap_file and not Path(cloakmap_file).exists():
        print(f"Error: CloakMap file not found: {cloakmap_file}", file=sys.stderr)
        return 1

    # Validate format choice
    format_val = args["--format"]
    if format_val not in ("markdown", "json", "text"):
        print(f"Error: Invalid format '{format_val}'. Choose: markdown, json, text", file=sys.stderr)
        return 1

    # Validate confidence threshold
    try:
        confidence = float(args["--confidence"])
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("out of range")
    except ValueError:
        print(f"Error: Confidence must be a number between 0.0 and 1.0", file=sys.stderr)
        return 1

    try:
        if args["mask"]:
            mask(
                input_file=input_file,
                output=args["--output"],
                cloakmap=args["--cloakmap"],
                policy=args["--policy"],
                confidence=confidence,
                format=format_val,
            )
        elif args["unmask"]:
            unmask(
                input_file=input_file,
                cloakmap_file=cloakmap_file,
                output=args["--output"],
                format=format_val,
            )
        elif args["version"]:
            version()
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
