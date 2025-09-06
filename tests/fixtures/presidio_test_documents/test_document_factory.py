"""Factory for creating test documents for Presidio integration testing."""


from docling_core.types import DoclingDocument
from docling_core.types.doc.document import TableItem, TextItem


class TestDocumentFactory:

    def _get_document_text(self, document: DoclingDocument) -> str:
        """Helper to get text from document, handling both formats."""
        if hasattr(document, '_main_text'):
            return document._main_text
        elif document.texts:
            return document.texts[0].text
        return ""

    def _set_document_text(self, document: DoclingDocument, text: str) -> None:
        """Helper to set text in document, handling both formats."""
        from docling_core.types.doc.document import TextItem
        # Create proper TextItem
        text_item = TextItem(
            text=text,
            self_ref="#/texts/0",
            label="text",
            orig=text
        )
        document.texts = [text_item]
        # Also set _main_text for backward compatibility
        document._main_text = text

    """Factory for creating various types of test documents."""

    @staticmethod
    def create_document_with_phone_numbers() -> DoclingDocument:
        """Create a simple document with phone numbers."""
        doc = DoclingDocument(name="phone_test.txt")
        doc._main_text = """
        Contact Information:
        - Main Office: (555) 123-4567
        - Mobile: +1-555-987-6543
        - Fax: 555.234.5678
        - International: +44 20 7123 4567
        - Toll Free: 1-800-555-0123
        """
        return doc

    @staticmethod
    def create_document_with_mixed_entities() -> DoclingDocument:
        """Create a complex document with multiple entity types."""
        doc = DoclingDocument(name="mixed_entities.txt")
        doc._main_text = """
        === CONFIDENTIAL EMPLOYEE FILE ===

        Employee: Jennifer Martinez Rodriguez
        ID: EMP-2024-98765
        Department: Human Resources

        Personal Information:
        - Date of Birth: April 23, 1987
        - SSN: 456-78-9123
        - Driver's License: CA-DL-987654
        - Passport: US123456789

        Contact Details:
        - Work Email: j.martinez@company.com
        - Personal Email: jenny.m.rodriguez@gmail.com
        - Office Phone: (555) 234-5678 ext. 1234
        - Mobile: +1 (555) 987-6543
        - Address: 789 Oak Boulevard, Suite 456, Los Angeles, CA 90001

        Financial Information:
        - Annual Salary: $98,500
        - Bank: Bank of America
        - Account: ****3456
        - Routing: 123456789
        - Credit Card: 4532-9876-5432-1098 (Company Card)
        - 401k Plan: 401K-JMR-2024

        Medical Records:
        - Insurance Provider: Blue Cross Blue Shield
        - Policy Number: BCBS-987654321
        - Group Number: GRP-123456
        - Primary Care Physician: Dr. Michael Chen, MD
        - Physician License: MED-CA-54321
        - Prescriptions:
          * Metformin 500mg (Diabetes)
          * Lisinopril 10mg (Blood Pressure)
        - Blood Type: AB+
        - Allergies: Penicillin, Shellfish

        Security Information:
        - Security Clearance: Confidential (C-2024-5678)
        - Badge Number: BADGE-98765
        - Access Code: AC-1234-5678
        - VPN Token: VPN-JMR-2024
        - Background Check: Completed 01/15/2024 (BC-2024-9876)

        Emergency Contacts:
        1. Carlos Rodriguez (Husband)
           - Phone: (555) 345-6789
           - Email: c.rodriguez@email.com

        2. Maria Martinez (Mother)
           - Phone: +52 55 1234 5678
           - Email: maria.martinez@email.mx

        Performance History:
        - 2023 Review: 4.7/5.0 - Exceeds Expectations
        - 2022 Review: 4.2/5.0 - Meets Expectations
        - Promotion Date: March 1, 2023
        - Next Review: December 15, 2024

        Training & Certifications:
        - SHRM-CP Certification: CERT-SHRM-123456
        - ISO 9001 Lead Auditor: ISO-LA-987654
        - Six Sigma Green Belt: 6S-GB-456789

        Travel Information:
        - TSA PreCheck: TSA123456789
        - Global Entry: GE987654321
        - Frequent Flyer: AA-123456789, UA-987654321

        IT Assets:
        - Laptop Serial: LT-2024-98765
        - Monitor Serial: MON-2024-54321
        - Phone IMEI: 123456789012345

        Notes:
        Employee is eligible for senior management track.
        Scheduled for leadership training in Q3 2024.
        Mentor: David Thompson (david.thompson@company.com)
        """

        # Add text items
        doc.texts.append(TextItem(
            text="Jennifer Martinez Rodriguez - Employee Record",
            prov=[{"page_no": 1, "bbox": {"l": 0, "t": 0, "r": 200, "b": 20}}]
        ))

        # Add table with sensitive data
        table_data = [
            ["Category", "Type", "Value", "Status"],
            ["Personal", "SSN", "456-78-9123", "Verified"],
            ["Contact", "Email", "j.martinez@company.com", "Active"],
            ["Financial", "Salary", "$98,500", "Current"],
            ["Medical", "Insurance", "BCBS-987654321", "Active"],
            ["Security", "Clearance", "C-2024-5678", "Valid"],
        ]

        doc.tables.append(TableItem(
            data=table_data,
            prov=[{"page_no": 1, "bbox": {"l": 0, "t": 100, "r": 400, "b": 300}}]
        ))

        # Add key-value pairs
        doc.key_value_items.extend([
            {"key": "Employee Name", "value": "Jennifer Martinez Rodriguez"},
            {"key": "Employee ID", "value": "EMP-2024-98765"},
            {"key": "Department", "value": "Human Resources"},
            {"key": "Salary", "value": "$98,500"},
            {"key": "Security Clearance", "value": "Confidential"},
        ])

        return doc

    @staticmethod
    def create_large_document(pages: int = 100) -> DoclingDocument:
        """Create a large document for performance testing."""
        doc = DoclingDocument(name=f"large_document_{pages}pages.txt")

        text_parts = []
        for page in range(pages):
            for record in range(10):  # 10 records per page
                record_id = page * 10 + record
                text_parts.append(f"""
                === RECORD {record_id:06d} ===

                Customer Information:
                - Name: Customer_{record_id:06d} Johnson
                - ID: CUST-{record_id:08d}
                - Email: customer{record_id}@example.com
                - Phone: 555-{record_id % 10000:04d}
                - Mobile: +1 (555) {(record_id * 7) % 1000:03d}-{(record_id * 13) % 10000:04d}

                Personal Details:
                - DOB: {(record_id % 12) + 1:02d}/{(record_id % 28) + 1:02d}/{1950 + (record_id % 50)}
                - SSN: {record_id % 1000:03d}-{(record_id * 3) % 100:02d}-{(record_id * 7) % 10000:04d}
                - Driver's License: DL-{record_id:08d}

                Address:
                - Street: {record_id} Main Street
                - City: City_{record_id % 100}
                - State: State_{record_id % 50:02d}
                - ZIP: {10000 + record_id % 90000:05d}

                Financial:
                - Credit Card: 4111-1111-{record_id % 10000:04d}-{(record_id * 11) % 10000:04d}
                - Bank Account: ****{record_id % 10000:04d}
                - Annual Income: ${50000 + (record_id * 1000) % 100000}

                Medical:
                - Insurance ID: INS-{record_id:08d}
                - Policy: POL-{record_id:08d}
                - Provider: Provider_{record_id % 20}

                Notes: Customer has been active since {2020 - (record_id % 10)}.
                Last transaction on {(record_id % 12) + 1:02d}/{(record_id % 28) + 1:02d}/2024.
                Loyalty status: {['Bronze', 'Silver', 'Gold', 'Platinum'][record_id % 4]}

                """)

        doc._main_text = "\n".join(text_parts)
        return doc

    @staticmethod
    def create_unicode_document() -> DoclingDocument:
        """Create a document with Unicode and special characters."""
        doc = DoclingDocument(name="unicode_test.txt")
        doc._main_text = """
        === INTERNATIONAL EMPLOYEE REGISTRY ===

        European Office:
        - François Müller (München)
          Email: françois.müller@société.de
          Phone: +49 89 1234 5678
          Salary: €75,000

        - José García Pérez (Madrid)
          Email: josé.garcía@compañía.es
          Phone: +34 91 234 5678
          Salary: €68,000

        - Søren Østergaard (København)
          Email: søren.østergaard@virksomhed.dk
          Phone: +45 33 12 34 56
          Salary: kr 550,000

        Asian Office:
        - 田中太郎 (Tanaka Tarō)
          Email: tanaka@会社.jp
          Phone: +81 3 1234 5678
          Salary: ¥8,500,000

        - 김민수 (Kim Min-su)
          Email: kim@회사.kr
          Phone: +82 2 1234 5678
          Salary: ₩85,000,000

        - 王小明 (Wang Xiaoming)
          Email: wang@公司.cn
          Phone: +86 10 1234 5678
          Salary: ¥580,000

        Middle East Office:
        - محمد الأحمد (Mohammed Al-Ahmad)
          Email: mohammed@شركة.ae
          Phone: +971 4 123 4567
          Salary: د.إ 250,000

        - יוסף כהן (Yosef Cohen)
          Email: yosef@חברה.il
          Phone: +972 3 123 4567
          Salary: ₪ 350,000

        Special Characters & Symbols:
        - Email: user@café-société.fr
        - Currencies: $, €, £, ¥, ₹, ₽, ₿
        - Math: α, β, γ, δ, ∑, ∏, ∫, ∞, √, ≈, ≠, ≤, ≥
        - Emojis: 😀 📧 ☎️ 💼 🏢 💰 🌍 ✈️
        - Quotes: "English" 'single' «French» „German" 「Japanese」
        - Punctuation: ¡Hola! ¿Cómo estás? — Em dash – En dash
        """
        return doc

    @staticmethod
    def create_edge_case_document() -> DoclingDocument:
        """Create a document with edge cases and overlapping entities."""
        doc = DoclingDocument(name="edge_cases.txt")
        doc._main_text = """
        Edge Case Scenarios:

        1. Back-to-back entities:
           Email:john@example.comPhone:555-1234SSN:123-45-6789

        2. Entities within entities:
           Contact Dr. John Smith, MD at john.smith.md@hospital.org
           License: MED-JS-12345-SMITH

        3. Partial overlaps:
           John Smith's email (john@smith.com) and phone (555-JOHN)

        4. Special formatting:
           SSN:123 45 6789 or 123456789 or 123-45-6789
           Phone: 5551234, 555-1234, (555) 123-4567, +1-555-123-4567

        5. Ambiguous text:
           Call 911 (emergency) or 555-1234 (office)
           Account 123456789 (checking) or 987654321 (savings)

        6. International formats:
           UK: +44 20 7123 4567 or 020 7123 4567
           France: +33 1 23 45 67 89 or 01 23 45 67 89

        7. Entities at boundaries:
        john@example.com
        555-1234

        8. Very long entity:
           Email: this.is.a.very.long.email.address.that.might.cause.issues@subdomain.example.company.org

        9. Repeated entities:
           John Smith called John Smith about John Smith's account
           555-1234, again 555-1234, and once more 555-1234

        10. Mixed case and spacing:
            JOHN@EXAMPLE.COM, john@example.com, JoHn@ExAmPlE.cOm
            5 5 5 - 1 2 3 4, 555-1234, 5551234
        """
        return doc

    @staticmethod
    def create_legacy_cloakmap_document() -> DoclingDocument:
        """Create a document that would generate a v1.0 style CloakMap."""
        doc = DoclingDocument(name="legacy_format.txt")
        doc._main_text = """
        Legacy Format Document (v1.0 Compatible):

        Customer: Jane Doe
        Account: 987654321
        Email: jane.doe@oldformat.com
        Phone: (555) 999-8888
        SSN: 987-65-4321

        This document uses the older format that should be
        compatible with CloakMap v1.0 transformations.
        """
        return doc

    @staticmethod
    def create_empty_document() -> DoclingDocument:
        """Create an empty document."""
        doc = DoclingDocument(name="empty.txt")
        doc._main_text = ""
        return doc

    @staticmethod
    def create_whitespace_only_document() -> DoclingDocument:
        """Create a document with only whitespace."""
        doc = DoclingDocument(name="whitespace.txt")
        doc._main_text = "   \t\n\n\t   \n   "
        return doc

    @staticmethod
    def create_single_entity_document(entity_type: str = "EMAIL") -> DoclingDocument:
        """Create a document with a single entity for focused testing."""
        doc = DoclingDocument(name=f"single_{entity_type.lower()}.txt")

        entity_examples = {
            "EMAIL": "Contact us at support@example.com for assistance.",
            "PHONE": "Call our office at 555-1234 for more information.",
            "SSN": "The patient's SSN is 123-45-6789 for insurance.",
            "CREDIT_CARD": "Payment with card 4111-1111-1111-1111 was processed.",
            "PERSON": "Please contact John Smith regarding this matter.",
            "DATE": "The meeting is scheduled for January 15, 2024.",
            "LOCATION": "Our office is located in New York City.",
        }

        doc._main_text = entity_examples.get(entity_type, "No example available.")
        return doc


def create_test_documents_fixture() -> dict[str, DoclingDocument]:
    """
    Create a comprehensive set of test documents.

    Returns:
        Dictionary of test documents by category
    """
    factory = TestDocumentFactory()

    return {
        "simple_phone": factory.create_document_with_phone_numbers(),
        "complex_mixed": factory.create_document_with_mixed_entities(),
        "large_100pages": factory.create_large_document(100),
        "unicode_content": factory.create_unicode_document(),
        "edge_cases": factory.create_edge_case_document(),
        "legacy_format": factory.create_legacy_cloakmap_document(),
        "empty": factory.create_empty_document(),
        "whitespace_only": factory.create_whitespace_only_document(),
        "single_email": factory.create_single_entity_document("EMAIL"),
        "single_phone": factory.create_single_entity_document("PHONE"),
        "single_ssn": factory.create_single_entity_document("SSN"),
        "single_credit_card": factory.create_single_entity_document("CREDIT_CARD"),
        "single_person": factory.create_single_entity_document("PERSON"),
    }
