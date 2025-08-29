#!/usr/bin/env python3
"""Debug script to examine the CloakMap contents."""

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
    """Examine CloakMap contents."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        debug_log = workspace / "cloakmap_debug.log"
        
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
        doc_content_json = json.dumps(doc.model_dump(), indent=2)
        doc_file.write_text(doc_content_json)
        
        # Create simple policy  
        policy_content = {
            "locale": "en", 
            "privacy_level": "LOW",
            "entities": {
                "PHONE_NUMBER": {"kind": "TEMPLATE", "parameters": {"template": "[PHONE]"}},
                "EMAIL_ADDRESS": {"kind": "TEMPLATE", "parameters": {"template": "[EMAIL]"}},
                "US_SSN": {"kind": "TEMPLATE", "parameters": {"template": "[SSN]"}},
                "PERSON": {"kind": "TEMPLATE", "parameters": {"template": "[PERSON]"}}
            },
            "thresholds": {
                "PHONE_NUMBER": 0.5,
                "EMAIL_ADDRESS": 0.5,
                "US_SSN": 0.5,
                "PERSON": 0.5
            }
        }
        
        policy_file = workspace / "test_policy.json"
        policy_file.write_text(json.dumps(policy_content, indent=2))
        
        # Output files
        masked_file = workspace / "masked_output.json"
        cloakmap_file = workspace / "cloakmap.json"
        
        # CLI runner
        cli_runner = CliRunner()
        
        # Step 1: Mask only
        mask_result = cli_runner.invoke(cli, [
            "mask",
            str(doc_file),
            "--out", str(masked_file),
            "--policy", str(policy_file),
            "--cloakmap", str(cloakmap_file)
        ])
        
        with open(debug_log, 'w') as log:
            log.write("=== MASKING RESULT ===\n")
            log.write(f"Exit code: {mask_result.exit_code}\n")
            log.write(f"Output: {mask_result.output}\n")
            
            if mask_result.exit_code == 0 and cloakmap_file.exists():
                # Read and analyze CloakMap
                cloakmap_content = cloakmap_file.read_text()
                cloakmap_data = json.loads(cloakmap_content)
                
                log.write("\n=== CLOAKMAP ANALYSIS ===\n")
                log.write(f"Doc ID: {cloakmap_data.get('doc_id')}\n")
                log.write(f"Version: {cloakmap_data.get('version')}\n")
                log.write(f"Anchor count: {len(cloakmap_data.get('anchors', []))}\n")
                log.write(f"Metadata: {cloakmap_data.get('metadata', {})}\n\n")
                
                log.write("=== ANCHOR DETAILS ===\n")
                for i, anchor in enumerate(cloakmap_data.get('anchors', [])):
                    log.write(f"Anchor {i+1}:\n")
                    log.write(f"  ID: {anchor.get('replacement_id')}\n")
                    log.write(f"  Node ID: {anchor.get('node_id')}\n")
                    log.write(f"  Entity Type: {anchor.get('entity_type')}\n")
                    log.write(f"  Original: {repr(anchor.get('original_value'))}\n")
                    log.write(f"  Masked: {repr(anchor.get('masked_value'))}\n")
                    log.write(f"  Position: {anchor.get('start')}-{anchor.get('end')}\n")
                    log.write(f"  Strategy: {anchor.get('strategy_used')}\n")
                    log.write(f"  Segment: {anchor.get('segment_info', {})}\n")
                    log.write("\n")
                
                # Read and analyze masked document
                masked_content = masked_file.read_text()
                masked_data = json.loads(masked_content)
                
                log.write("\n=== MASKED DOCUMENT ANALYSIS ===\n")
                log.write(f"Document type: {type(masked_data)}\n")
                log.write(f"Keys: {list(masked_data.keys()) if isinstance(masked_data, dict) else 'N/A'}\n")
                
                if isinstance(masked_data, dict):
                    if "schema_name" in masked_data:
                        log.write(f"Schema: {masked_data['schema_name']}\n")
                        log.write(f"Texts count: {len(masked_data.get('texts', []))}\n")
                        if masked_data.get('texts'):
                            log.write(f"First text: {repr(masked_data['texts'][0].get('text', '')[:100])}\n")
                    elif "root" in masked_data:
                        log.write("Format: Lexical JSON\n")
                        root = masked_data.get('root', {})
                        log.write(f"Root type: {root.get('type')}\n")
                        log.write(f"Children: {len(root.get('children', []))}\n")
                
            else:
                log.write("MASKING FAILED OR NO CLOAKMAP CREATED\n")
        
        # Read debug log and print it
        debug_content = debug_log.read_text()
        print(debug_content)
        
        # Copy debug log to a persistent location
        persistent_log = Path("/tmp/cloakmap_debug.log")
        persistent_log.write_text(debug_content)
        print(f"\nDebug log saved to: {persistent_log}")


if __name__ == "__main__":
    main()