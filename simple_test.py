#!/usr/bin/env python3
"""Simple test to debug the issue by printing to files."""

import json
import sys
import tempfile
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem
from click.testing import CliRunner

from cloakpivot.cli.main import cli


def main():
    """Run simple debug test."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        debug_log = workspace / "debug.log"
        
        # Create document content
        doc_content = """Employee Information
===================

Personal Details:
Name: John Smith
Phone: 555-123-4567
Email: john.smith@company.com
SSN: 123-45-6789

Contact Information:
Emergency Contact: Jane Smith
Phone: 555-987-6543
Email: jane.smith@personal.com"""

        # Create DoclingDocument
        doc = DoclingDocument(name="test_document")
        text_item = TextItem(
            text=doc_content,
            self_ref="#/texts/0",
            label="text",
            orig=doc_content
        )
        doc.texts = [text_item]
        
        # Save document
        doc_file = workspace / "employee_info.json"
        doc_file.write_text(json.dumps(doc.model_dump(), indent=2))
        
        # Create policy
        policy_content = {
            "locale": "en",
            "privacy_level": "MEDIUM",
            "entities": {
                "PHONE_NUMBER": {"kind": "PHONE_TEMPLATE"},
                "EMAIL_ADDRESS": {"kind": "EMAIL_TEMPLATE"},
                "US_SSN": {"kind": "SURROGATE_SECURE"},
                "PERSON": {"kind": "TEMPLATE", "parameters": {"auto_generate": True}}
            },
            "thresholds": {
                "PHONE_NUMBER": 0.7,
                "EMAIL_ADDRESS": 0.8,
                "US_SSN": 0.9,
                "PERSON": 0.8
            }
        }
        
        policy_file = workspace / "test_policy.json"
        policy_file.write_text(json.dumps(policy_content, indent=2))
        
        # Output files
        masked_file = workspace / "masked_output.json"
        cloakmap_file = workspace / "cloakmap.json"
        unmasked_file = workspace / "unmasked_output.json"
        
        # CLI runner
        cli_runner = CliRunner()
        
        with open(debug_log, 'w') as log:
            # Step 1: Mask
            log.write("=== MASKING ===\n")
            mask_result = cli_runner.invoke(cli, [
                "mask",
                str(doc_file),
                "--out", str(masked_file),
                "--policy", str(policy_file),
                "--cloakmap", str(cloakmap_file)
            ])
            
            log.write(f"Mask exit code: {mask_result.exit_code}\n")
            log.write(f"Mask output: {mask_result.output}\n")
            log.write(f"Exception: {mask_result.exception}\n\n")
            
            if mask_result.exit_code != 0:
                log.write("MASKING FAILED - ABORTING\n")
                return
            
            # Step 2: Unmask
            log.write("=== UNMASKING ===\n")
            unmask_result = cli_runner.invoke(cli, [
                "unmask",
                str(masked_file),
                "--out", str(unmasked_file),
                "--cloakmap", str(cloakmap_file)
            ])
            
            log.write(f"Unmask exit code: {unmask_result.exit_code}\n")
            log.write(f"Unmask output: {unmask_result.output}\n")
            log.write(f"Exception: {unmask_result.exception}\n\n")
            
            if unmask_result.exit_code != 0:
                log.write("UNMASKING FAILED - ABORTING\n")
                return
                
            # Step 3: Compare
            log.write("=== COMPARISON ===\n")
            original_content = doc_file.read_text()
            unmasked_content = unmasked_file.read_text()
            
            log.write(f"Original length: {len(original_content)}\n")
            log.write(f"Unmasked length: {len(unmasked_content)}\n")
            log.write(f"Equal: {original_content.strip() == unmasked_content.strip()}\n\n")
            
            if original_content.strip() != unmasked_content.strip():
                log.write("CONTENT MISMATCH!\n")
                log.write("\nOriginal first 500 chars:\n")
                log.write(repr(original_content[:500]))
                log.write("\n\nUnmasked first 500 chars:\n")
                log.write(repr(unmasked_content[:500]))
                log.write("\n\n")
        
        # Read debug log and print it
        debug_content = debug_log.read_text()
        print(debug_content)
        
        # Copy debug log to a persistent location
        persistent_log = Path("/tmp/cloakpivot_debug.log")
        persistent_log.write_text(debug_content)
        print(f"\nDebug log saved to: {persistent_log}")


if __name__ == "__main__":
    main()