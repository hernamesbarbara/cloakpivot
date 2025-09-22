#!/usr/bin/env python3
"""Generate sample test data for CloakPivot examples.

This script creates sample files with synthetic PII for testing:
1. A simple text file with PII
2. A DoclingDocument JSON file
3. Instructions for obtaining a test PDF
"""

import json
from pathlib import Path
from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TextItem


CLOAKPIVOT_ROOT = Path(__file__).parent.parent
DATA_DIR = CLOAKPIVOT_ROOT / "data"


def create_sample_text_file():
    """Create a sample text file with synthetic PII."""
    data_dir = Path("data/text")
    data_dir.mkdir(parents=True, exist_ok=True)

    content = """CONFIDENTIAL EMPLOYEE RECORD
============================

Employee Information:
- Name: John Michael Smith
- Employee ID: EMP-2024-0156
- Email: john.smith@techcorp.com
- Phone: (555) 123-4567
- Mobile: +1-555-987-6543

Personal Details:
- SSN: 123-45-6789
- Date of Birth: January 15, 1985
- Address: 123 Main Street, Suite 400
           San Francisco, CA 94105

Emergency Contact:
- Name: Jane Elizabeth Doe
- Relationship: Spouse
- Phone: (555) 234-5678
- Email: jane.doe@email.com

Banking Information:
- Account Number: 1234567890
- Routing Number: 021000021
- Credit Card: 4111-1111-1111-1111

Medical Information:
- Policy Number: MED-789456123
- Blood Type: O+
- Physician: Dr. Sarah Johnson
- Physician Phone: (555) 345-6789

Notes:
Employee has been with the company since March 2020.
Performance review scheduled for December 15, 2024.
Direct supervisor: Robert Williams (robert.williams@techcorp.com)
"""

    file_path = data_dir / "employee_record.txt"
    file_path.write_text(content)
    print(f"✓ Created: {file_path}")
    return file_path


def create_sample_docling_json():
    """Create a sample DoclingDocument JSON file."""
    data_dir = Path("data/docling")
    data_dir.mkdir(parents=True, exist_ok=True)

    # Create a DoclingDocument with various PII
    doc = DoclingDocument(name="financial_report.pdf")

    # Add text items with different types of PII
    doc.texts = [
        TextItem(
            text="2024 Q3 Financial Report",
            self_ref="#/texts/0",
            label="text",
            orig="2024 Q3 Financial Report"
        ),
        TextItem(
            text="Prepared by: Michael Anderson (CFO)",
            self_ref="#/texts/1",
            label="text",
            orig="Prepared by: Michael Anderson (CFO)"
        ),
        TextItem(
            text="Contact: m.anderson@techcorp.com or (555) 456-7890",
            self_ref="#/texts/2",
            label="text",
            orig="Contact: m.anderson@techcorp.com or (555) 456-7890"
        ),
        TextItem(
            text="Key Clients:",
            self_ref="#/texts/3",
            label="text",
            orig="Key Clients:"
        ),
        TextItem(
            text="- Acme Corp (Contact: Lisa Chen, lisa@acme.com, 555-111-2222)",
            self_ref="#/texts/4",
            label="text",
            orig="- Acme Corp (Contact: Lisa Chen, lisa@acme.com, 555-111-2222)"
        ),
        TextItem(
            text="- Global Industries (Contact: David Park, dpark@global.com)",
            self_ref="#/texts/5",
            label="text",
            orig="- Global Industries (Contact: David Park, dpark@global.com)"
        ),
        TextItem(
            text="Banking Details: Account #9876543210, Routing #123456789",
            self_ref="#/texts/6",
            label="text",
            orig="Banking Details: Account #9876543210, Routing #123456789"
        ),
        TextItem(
            text="Tax ID: 12-3456789, Registration: REG-2024-789456",
            self_ref="#/texts/7",
            label="text",
            orig="Tax ID: 12-3456789, Registration: REG-2024-789456"
        ),
        TextItem(
            text="For internal use only. Questions? Contact hr@techcorp.com",
            self_ref="#/texts/8",
            label="text",
            orig="For internal use only. Questions? Contact hr@techcorp.com"
        )
    ]

    # Save as JSON
    file_path = data_dir / "financial_report.docling.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(doc.export_to_dict(), f, indent=2)

    print(f"✓ Created: {file_path}")
    return file_path


def create_pdf_instructions():
    """Create instructions for obtaining a test PDF."""
    data_dir = Path("data/pdf")
    data_dir.mkdir(parents=True, exist_ok=True)

    instructions = """# Test PDF Files

To test PDF processing, you have several options:

## Option 1: Create a simple PDF from text

You can convert the sample text file to PDF using various tools:

### On macOS:
```bash
# Using textutil and cupsfilter
textutil -convert html ../text/employee_record.txt -output temp.html
cupsfilter temp.html > sample.pdf
rm temp.html
```

### Using Python (requires reportlab):
```python
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter

c = canvas.Canvas("sample.pdf", pagesize=letter)
with open("../text/employee_record.txt") as f:
    y = 750
    for line in f:
        c.drawString(100, y, line.strip())
        y -= 15
        if y < 50:
            c.showPage()
            y = 750
c.save()
```

## Option 2: Use any existing PDF

Copy any PDF file (without real PII!) to this directory:
```bash
cp ~/path/to/your/test.pdf sample.pdf
```

## Option 3: Download a sample PDF

Many websites offer sample PDFs for testing:
- https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
- https://www.africau.edu/images/default/sample.pdf

```bash
# Download a sample
curl -o sample.pdf https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf
```

## Important

⚠️ **Never use PDFs containing real personal information for testing!**
"""

    file_path = data_dir / "README.md"
    file_path.write_text(instructions)
    print(f"✓ Created: {file_path}")
    return file_path


def main():
    """Generate all test data files."""
    print("=" * 60)
    print("CloakPivot Test Data Generator")
    print("=" * 60)
    print("\nGenerating synthetic test data...\n")

    # Create sample files
    text_file = create_sample_text_file()
    docling_file = create_sample_docling_json()
    pdf_instructions = create_pdf_instructions()

    print("\n" + "=" * 60)
    print("✅ Test data generated successfully!")
    print("=" * 60)
    print("\nCreated files:")
    print(f"  • {text_file}")
    print(f"  • {docling_file}")
    print(f"  • {pdf_instructions}")
    print("\nYou can now run the examples:")
    print("  python examples/simple_usage.py")
    print("  python examples/advanced_usage.py")
    print("  python examples/docling_integration.py")
    print("  python examples/pdf_workflow.py  (requires PDF - see data/pdf/README.md)")
    print("\n⚠️  Remember: Never commit files with real PII to the repository!")


if __name__ == "__main__":
    main()