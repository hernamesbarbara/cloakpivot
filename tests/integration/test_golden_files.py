"""Golden file regression tests for CloakPivot.

These tests ensure that masking output remains consistent across code changes.
Golden files contain expected outputs that are compared against current results.
"""

import json
import pytest
from pathlib import Path
from typing import Any, Dict

from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem

from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.masking.engine import MaskingEngine
from cloakpivot.unmasking.engine import UnmaskingEngine
from tests.utils.assertions import (
    assert_document_structure_preserved,
    assert_masking_result_valid,
    assert_round_trip_fidelity
)
from tests.utils.masking_helpers import mask_document_with_detection


class TestGoldenFiles:
    """Test suite for golden file regression testing."""
    
    @pytest.fixture
    def golden_files_dir(self) -> Path:
        """Directory containing golden files."""
        return Path(__file__).parent.parent / "fixtures" / "golden_files"
    
    @pytest.fixture
    def test_documents_dir(self) -> Path:
        """Directory containing test documents."""
        return Path(__file__).parent.parent / "fixtures" / "documents"
    
    @pytest.fixture
    def test_policies_dir(self) -> Path:
        """Directory containing test policies."""
        return Path(__file__).parent.parent / "fixtures" / "policies"
    
    def create_golden_file_if_missing(self, golden_path: Path, content: Any) -> None:
        """Create golden file if it doesn't exist (for initial test creation)."""
        if not golden_path.exists():
            golden_path.parent.mkdir(parents=True, exist_ok=True)
            with open(golden_path, 'w', encoding='utf-8') as f:
                if isinstance(content, str):
                    f.write(content)
                else:
                    json.dump(content, f, indent=2, ensure_ascii=False)
    
    def load_golden_file(self, golden_path: Path) -> Any:
        """Load golden file content."""
        with open(golden_path, 'r', encoding='utf-8') as f:
            if golden_path.suffix == '.json':
                return json.load(f)
            else:
                return f.read()
    
    def create_document_from_file(self, file_path: Path) -> DoclingDocument:
        """Create DoclingDocument from text file."""
        text_content = file_path.read_text(encoding='utf-8')
        doc = DoclingDocument(name=file_path.stem)
        
        text_item = TextItem(
            text=text_content,
            self_ref="#/texts/0",
            label="text",
            orig=text_content
        )
        doc.texts = [text_item]
        return doc
    
    @pytest.mark.golden
    def test_employee_record_basic_masking(
        self,
        golden_files_dir: Path,
        test_documents_dir: Path,
        basic_masking_policy: MaskingPolicy
    ):
        """Test masking of employee record with basic policy."""
        # Load test document
        doc_path = test_documents_dir / "sample_employee_record.txt"
        document = self.create_document_from_file(doc_path)
        
        # Apply masking
        result = mask_document_with_detection(document, basic_masking_policy)
        
        # Prepare golden file data
        golden_data = {
            "masked_text": result.masked_document.texts[0].text,
            "entity_count": len(result.cloakmap.anchors),
            "policy_hash": result.cloakmap.doc_hash,
            "document_structure": {
                "name": result.masked_document.name,
                "text_items_count": len(result.masked_document.texts)
            }
        }
        
        # Compare with golden file
        golden_path = golden_files_dir / "employee_record_basic_masking.json"
        
        if not golden_path.exists():
            # Create golden file for first run
            self.create_golden_file_if_missing(golden_path, golden_data)
            pytest.skip("Golden file created - run test again to validate")
        
        expected_data = self.load_golden_file(golden_path)
        
        # Validate core structure matches
        assert golden_data["entity_count"] == expected_data["entity_count"]
        assert golden_data["document_structure"] == expected_data["document_structure"]
        
        # Note: We don't compare exact masked text due to randomness in some strategies
        # Instead, we validate that the same number of entities were processed
    
    @pytest.mark.golden
    def test_medical_report_strict_masking(
        self,
        golden_files_dir: Path,
        test_documents_dir: Path,
        strict_masking_policy: MaskingPolicy
    ):
        """Test masking of medical report with strict policy."""
        # Load test document
        doc_path = test_documents_dir / "medical_report.txt"
        document = self.create_document_from_file(doc_path)
        
        # Apply masking
        result = mask_document_with_detection(document, strict_masking_policy)
        
        # Validate result
        assert_masking_result_valid(result)
        assert_document_structure_preserved(document, result.masked_document)
        
        # Golden file comparison
        golden_data = {
            "entity_count": result.cloakmap.anchor_count,
            "document_name": result.masked_document.name,
            "text_items_count": len(result.masked_document.texts),
            "policy_privacy_level": str(strict_masking_policy.privacy_level)
        }
        
        golden_path = golden_files_dir / "medical_report_strict_masking.json"
        
        if not golden_path.exists():
            self.create_golden_file_if_missing(golden_path, golden_data)
            pytest.skip("Golden file created - run test again to validate")
        
        expected_data = self.load_golden_file(golden_path)
        assert golden_data == expected_data
    
    @pytest.mark.golden
    def test_round_trip_golden_validation(
        self,
        golden_files_dir: Path,
        test_documents_dir: Path,
        basic_masking_policy: MaskingPolicy
    ):
        """Test round-trip masking/unmasking produces consistent results."""
        # Test with multiple documents
        doc_files = ["sample_employee_record.txt", "medical_report.txt"]
        
        for doc_file in doc_files:
            doc_path = test_documents_dir / doc_file
            if not doc_path.exists():
                continue
                
            original_document = self.create_document_from_file(doc_path)
            
            # Mask document
            mask_result = mask_document_with_detection(original_document, basic_masking_policy)
            
            # Unmask document
            unmasking_engine = UnmaskingEngine()
            unmask_result = unmasking_engine.unmask_document(
                mask_result.masked_document,
                mask_result.cloakmap
            )
            
            # Validate round-trip fidelity
            assert_round_trip_fidelity(
                original_document,
                mask_result.masked_document,
                unmask_result.unmasked_document,
                mask_result.cloakmap
            )
            
            # Golden file for round-trip validation
            golden_data = {
                "document_name": doc_path.stem,
                "original_char_count": len(original_document.texts[0].text),
                "masked_char_count": len(mask_result.masked_document.texts[0].text),
                "unmasked_char_count": len(unmask_result.unmasked_document.texts[0].text),
                "entity_mappings_count": len(mask_result.cloakmap.entity_mappings),
                "round_trip_success": True
            }
            
            golden_path = golden_files_dir / f"round_trip_{doc_path.stem}.json"
            
            if not golden_path.exists():
                self.create_golden_file_if_missing(golden_path, golden_data)
                continue
                
            expected_data = self.load_golden_file(golden_path)
            
            # Validate key metrics match golden file
            assert golden_data["original_char_count"] == expected_data["original_char_count"]
            assert golden_data["round_trip_success"] == expected_data["round_trip_success"]
            # Allow some variation in entity count due to confidence thresholds
            assert abs(golden_data["entity_mappings_count"] - expected_data["entity_mappings_count"]) <= 2
    
    @pytest.mark.golden
    def test_format_specific_golden_files(
        self,
        golden_files_dir: Path,
        basic_masking_policy: MaskingPolicy
    ):
        """Test masking behavior with format-specific challenges."""
        # Test cases for specific formatting challenges
        test_cases = [
            {
                "name": "table_format",
                "text": """
Employee Data Table
===================
| Name        | Phone       | Email              | SSN         |
|-------------|-------------|--------------------|-------------|
| John Smith  | 555-123-456 | john@company.com   | 123-45-6789 |
| Jane Doe    | 555-987-654 | jane@company.com   | 987-65-4321 |
| Bob Johnson | 555-555-555 | bob@company.com    | 111-22-3333 |
""",
            },
            {
                "name": "nested_structure",
                "text": """
Report: Employee Contact Information
-----------------------------------
Section 1: Management Team
  - Manager: Alice Smith
    Phone: (555) 123-4567
    Email: alice.smith@company.com
    
  - Assistant Manager: Bob Johnson
    Phone: (555) 234-5678
    Email: bob.johnson@company.com

Section 2: Development Team
  - Lead Developer: Charlie Brown
    Phone: (555) 345-6789
    Email: charlie.brown@company.com
    SSN: 123-45-6789 (for payroll)
""",
            }
        ]
        
        for test_case in test_cases:
            # Create document
            document = DoclingDocument(name=test_case["name"])
            text_item = TextItem(
                text=test_case["text"],
                self_ref="#/texts/0",
                label="text",
                orig=test_case["text"]
            )
            document.texts = [text_item]
            
            # Apply masking
            result = mask_document_with_detection(document, basic_masking_policy)
            
            # Create golden file data focusing on structure preservation
            lines_original = test_case["text"].strip().split('\n')
            lines_masked = result.masked_document.texts[0].text.strip().split('\n')
            
            golden_data = {
                "test_case": test_case["name"],
                "line_count_preserved": len(lines_original) == len(lines_masked),
                "entity_mappings": len(result.cloakmap.entity_mappings),
                "structure_markers": {
                    "has_table_separators": "|" in result.masked_document.texts[0].text,
                    "has_section_headers": "Section" in result.masked_document.texts[0].text,
                    "has_indentation": any(line.startswith("  ") for line in lines_masked),
                }
            }
            
            golden_path = golden_files_dir / f"format_{test_case['name']}.json"
            
            if not golden_path.exists():
                self.create_golden_file_if_missing(golden_path, golden_data)
                continue
                
            expected_data = self.load_golden_file(golden_path)
            
            # Validate structural preservation
            assert golden_data["line_count_preserved"] == expected_data["line_count_preserved"]
            assert golden_data["structure_markers"] == expected_data["structure_markers"]
            
    @pytest.mark.golden
    def test_regression_detection(
        self,
        golden_files_dir: Path,
        test_documents_dir: Path,
        basic_masking_policy: MaskingPolicy
    ):
        """Test that changes in masking behavior are detected."""
        # This test helps detect unintended changes in masking behavior
        
        # Create a standardized test document
        test_text = """
        Test Document for Regression Detection
        =====================================
        
        Contact: John Smith
        Phone: (555) 123-4567
        Email: john.smith@example.com
        SSN: 123-45-6789
        
        This document is used to detect regressions in the masking system.
        Any changes to the masking behavior should be intentional and documented.
        """
        
        document = DoclingDocument(name="regression_test")
        text_item = TextItem(
            text=test_text,
            self_ref="#/texts/0",
            label="text",
            orig=test_text
        )
        document.texts = [text_item]
        
        # Apply masking
        result = mask_document_with_detection(document, basic_masking_policy)
        
        # Create regression detection metrics
        golden_data = {
            "version": "1.0",  # Increment when intentional changes are made
            "test_name": "regression_detection",
            "metrics": {
                "entities_detected": len(result.cloakmap.entity_mappings),
                "masked_length": len(result.masked_document.texts[0].text),
                "cloakmap_size": len(str(result.cloakmap.entity_mappings)),
                "has_phone_masking": any("PHONE" in str(mapping) for mapping in result.cloakmap.entity_mappings.values()),
                "has_email_masking": any("EMAIL" in str(mapping) for mapping in result.cloakmap.entity_mappings.values()),
            }
        }
        
        golden_path = golden_files_dir / "regression_detection.json"
        
        if not golden_path.exists():
            self.create_golden_file_if_missing(golden_path, golden_data)
            pytest.skip("Regression golden file created - run test again to validate")
        
        expected_data = self.load_golden_file(golden_path)
        
        # Check for regressions
        if expected_data["metrics"] != golden_data["metrics"]:
            # Detailed comparison for debugging
            for key, expected_value in expected_data["metrics"].items():
                actual_value = golden_data["metrics"][key]
                if actual_value != expected_value:
                    pytest.fail(
                        f"Regression detected in '{key}': "
                        f"expected {expected_value}, got {actual_value}. "
                        f"If this change is intentional, update the golden file and version."
                    )