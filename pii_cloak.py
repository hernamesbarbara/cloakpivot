#!/usr/bin/env python
"""
PII Cloak - Mask personally identifiable information in PDF documents

Usage:
    pii-cloak mask --infile=<path> --outdir=<dir> [--strategy=<strategy>]
    pii-cloak -h | --help
    pii-cloak --version

Options:
    -h --help                   Show this screen
    --version                   Show version
    --infile=<path>             Path to the input PDF file
    --outdir=<dir>              Directory to save output files
    --strategy=<strategy>       Masking strategy: 'redact' (default) or 'replace' [default: redact]

Examples:
    pii-cloak mask --infile data/pdf/email.pdf --outdir data/md/
    pii-cloak mask --infile data/pdf/email.pdf --outdir data/md/ --strategy replace
"""

import os
import sys
import json
from pathlib import Path
from typing import Tuple, Dict
from docopt import docopt
from docling.document_converter import DocumentConverter
from cloakpivot import CloakEngine
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.core.strategies import Strategy, StrategyKind

__version__ = "0.1.0"

def build_policy(strategy: str) -> MaskingPolicy:
    """Return a MaskingPolicy for 'redact' or 'replace'."""
    if strategy == "replace":
        return MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.SURROGATE,
                parameters={"seed": "testcloak-consistent"},
            ),
            seed="testcloak-consistent",
        )

    # 'redact' default + entity overrides
    tmpl = Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[REDACTED]"})
    policy = MaskingPolicy(default_strategy=tmpl)

    entity_templates = {
        "EMAIL_ADDRESS": "[EMAIL]",
        "PERSON": "[NAME]",
        "DATE_TIME": "[DATE]",
        "PHONE_NUMBER": "[PHONE]",
        "LOCATION": "[LOCATION]",
        "CREDIT_CARD": "[CARD-****]",
    }
    for label, token in entity_templates.items():
        policy = policy.with_entity_strategy(
            label,
            Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": token}),
        )
    return policy

def render_markdown(doc, *, title: str, entities_found: int, entities_masked: int) -> str:
    """Return markdown for a Docling-like document, with graceful fallback."""
    if hasattr(doc, "export_to_markdown"):
        md = doc.export_to_markdown()
        if md:
            return md

    # Fallback: simple stitched markdown
    parts = [f"# {title}", "", f"**Entities Found:** {entities_found}", f"**Entities Masked:** {entities_masked}", ""]
    if getattr(doc, "texts", None):
        for t in doc.texts:
            txt = getattr(t, "text", "")
            if txt:
                parts.append(txt)
                parts.append("")
    if getattr(doc, "tables", None):
        for tbl in doc.tables:
            if hasattr(tbl, "to_markdown"):
                parts.append(tbl.to_markdown())
                parts.append("")
    return "\n".join(parts).strip() + "\n"

def build_output_paths(outdir: Path, infile: Path, strategy: str
                       ) -> Tuple[Path, Path, Path, Path]:
    """Return (base_dir, unmasked_md, masked_md, cloakmap_json).

    base_dir: <outdir>/<input_base>/
    Filenames:
      - unmasked.md
      - masked.<strategy>.md
      - cloakmap.<strategy>.json
    """
    base_dir = outdir / infile.stem
    base_dir.mkdir(parents=True, exist_ok=True)

    unmasked_md = base_dir / "unmasked.md"
    masked_md = base_dir / f"masked.{strategy}.md"
    cloakmap_json = base_dir / f"cloakmap.{strategy}.json"
    return base_dir, unmasked_md, masked_md, cloakmap_json

def write_text(path: Path, content: str) -> None:
    """Write UTF-8 text to a file."""
    path.write_text(content, encoding="utf-8")

def write_json(path: Path, payload: dict) -> None:
    """Write pretty JSON to a file."""
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

def write_manifest(path: Path, *, infile: Path, strategy: str,
                   unmasked: Path, masked: Path, cloakmap: Path,
                   entities_found: int, entities_masked: int, version: str) -> None:
    """Write a minimal manifest that points to the latest outputs."""
    payload: Dict[str, object] = {
        "tool": "pii-cloak",
        "version": version,
        "input_pdf": str(infile),
        "strategy": strategy,
        "counts": {"entities_found": entities_found, "entities_masked": entities_masked},
        "artifacts": {
            "unmasked_md": str(unmasked.name),
            "masked_md": str(masked.name),
            "cloakmap_json": str(cloakmap.name),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

def process_pdf(infile, outdir, strategy="redact"):
    """Process a PDF file to detect and mask PII."""

    infile_path = Path(infile)
    if not infile_path.exists():
        raise FileNotFoundError(f"Input file not found: {infile}")

    if not infile_path.suffix.lower() == '.pdf':
        raise ValueError(f"Input file must be a PDF: {infile}")


    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    base_name = infile_path.stem

    print(f"Processing: {infile}")
    print(f"Output directory: {outdir}")
    print(f"Strategy: {strategy}")

    print("\n1. Converting PDF to DoclingDocument...")
    converter = DocumentConverter()
    result = converter.convert(str(infile_path))
    docling_doc = result.document
    print(f"   ✓ PDF converted successfully")

    print("\n2. Detecting and masking PII...")

    policy = build_policy(strategy)
    cloak_engine = CloakEngine(default_policy=policy)
    mask_result = cloak_engine.mask_document(docling_doc)

    print(f"   ✓ Found {mask_result.entities_found} PII entities")
    print(f"   ✓ Masked {mask_result.entities_masked} entities")

    print("\n3. Exporting documents to markdown...")

    unmasked_md = render_markdown(
        docling_doc,
        title="Unmasked Document",
        entities_found=mask_result.entities_found,
        entities_masked=mask_result.entities_masked,
    )
    masked_md = render_markdown(
        mask_result.document,
        title=f"Masked Document (Strategy: {strategy})",
        entities_found=mask_result.entities_found,
        entities_masked=mask_result.entities_masked,
    )

    base_dir, unmasked_path, masked_path, cloakmap_path = build_output_paths(
        outdir_path, infile_path, strategy
    )
    write_text(unmasked_path, unmasked_md)
    write_text(masked_path, masked_md)
    print(f"   ✓ Saved unmasked document: {unmasked_path}")
    print(f"   ✓ Saved masked document: {masked_path}")

    cloakmap_obj = mask_result.cloakmap
    cloakmap_dict = (
        cloakmap_obj.to_dict() if hasattr(cloakmap_obj, "to_dict")
        else vars(cloakmap_obj) if hasattr(cloakmap_obj, "__dict__")
        else {"entities_found": mask_result.entities_found, "entities_masked": mask_result.entities_masked}
    )
    write_json(cloakmap_path, cloakmap_dict)
    print(f"   ✓ Saved cloakmap: {cloakmap_path}")

    # Write manifest
    manifest_path = base_dir / f"manifest.{strategy}.json"
    write_manifest(
        manifest_path,
        infile=infile_path,
        strategy=strategy,
        unmasked=unmasked_path,
        masked=masked_path,
        cloakmap=cloakmap_path,
        entities_found=mask_result.entities_found,
        entities_masked=mask_result.entities_masked,
        version=__version__,
    )
    print(f"   ✓ Saved manifest: {manifest_path}")

    print(f"\n✅ Processing complete!")
    print(f"   Output files in: {base_dir}")
    print(f"   - {unmasked_path.name}")
    print(f"   - {masked_path.name}")
    print(f"   - {cloakmap_path.name}")
    print(f"   - {manifest_path.name}")

def main():
    """Main entry point for the CLI."""
    arguments = docopt(__doc__, version=f"PII Cloak {__version__}")

    if arguments['mask']:
        try:
            strategy = arguments.get('--strategy', 'redact')
            if strategy not in ['redact', 'replace']:
                print(f"\n❌ Error: Invalid strategy '{strategy}'. Choose 'redact' or 'replace'.")
                return 1

            process_pdf(
                infile=arguments['--infile'],
                outdir=arguments['--outdir'],
                strategy=strategy
            )
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
