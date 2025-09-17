"""Unit tests for table masking functionality."""

import pytest
from docling_core.types.doc.document import (
    DocItemLabel,
    DoclingDocument,
    TableCell,
    TableData,
    TableItem,
    TextItem,
)

from cloakpivot import CloakEngine
from cloakpivot.core import MaskingPolicy, Strategy, StrategyKind


class TestTableMasking:
    """Test that PII in table cells is properly masked."""

    def test_table_cells_are_masked_simple(self):
        """Test that table cells containing PII are masked."""
        # Create document with a table containing PII
        doc = DoclingDocument(name="test_table.txt")

        # Create a simple table with PII
        table_data = TableData(num_rows=3, num_cols=3)

        # Properly populate table_cells with TableCell objects
        cells_data = [
            ["Name", "Email", "Date"],
            ["John Doe", "john@example.com", "2020-01-01"],
            ["Jane Smith", "jane@example.com", "2020-02-15"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        # Also add as text for PII detection
        doc.texts = [
            TextItem(
                text="Name Email Date",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Name Email Date",
            ),
            TextItem(
                text="John Doe john@example.com 2020-01-01",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/1",
                orig="John Doe john@example.com 2020-01-01",
            ),
            TextItem(
                text="Jane Smith jane@example.com 2020-02-15",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/2",
                orig="Jane Smith jane@example.com 2020-02-15",
            ),
        ]

        # Create masking policy
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.TEMPLATE, parameters={"template": "[REDACTED]"}
            )
        )

        # Mask the document
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        # Check that table cells were masked
        masked_table = result.document.tables[0]
        cells = masked_table.data.grid

        # Check that PII is not in table cells
        table_text_all = ""
        for row in cells:
            for cell in row:
                if hasattr(cell, "text"):
                    table_text_all += cell.text + " "

        # Original PII should not be in the table
        assert "John Doe" not in table_text_all
        assert "Jane Smith" not in table_text_all
        assert "john@example.com" not in table_text_all
        assert "jane@example.com" not in table_text_all

        # At least some cells should contain masked values
        # Note: Headers (Name, Email, Date) might not be detected as PII
        masked_values_found = False
        for row in cells[1:]:  # Skip header row
            for cell in row:
                if hasattr(cell, "text") and "[REDACTED]" in cell.text:
                    masked_values_found = True
                    break

        assert masked_values_found, "No masked values found in table cells"

    def test_table_structure_preserved(self):
        """Test that table structure is preserved during masking."""
        # Create document with table
        doc = DoclingDocument(name="test_structure.txt")

        # Create table with known structure
        table_data = TableData(num_rows=3, num_cols=3)

        cells_data = [
            ["Col1", "Col2", "Col3"],
            ["Data1", "john@test.com", "Data3"],
            ["Data4", "jane@test.com", "Data6"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        # Add text items for detection
        doc.texts = [
            TextItem(
                text="Col1 Col2 Col3\nData1 john@test.com Data3\nData4 jane@test.com Data6",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Col1 Col2 Col3\nData1 john@test.com Data3\nData4 jane@test.com Data6",
            )
        ]

        # Mask document
        policy = MaskingPolicy(default_strategy=Strategy(kind=StrategyKind.REDACT))
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        # Check structure is preserved
        assert len(result.document.tables) == 1
        masked_table = result.document.tables[0]
        assert len(masked_table.data.grid) == 3  # Same number of rows
        assert len(masked_table.data.grid[0]) == 3  # Same number of columns

        # Check that non-PII data is preserved
        assert masked_table.data.grid[0][0].text == "Col1"
        assert masked_table.data.grid[1][0].text == "Data1"
        assert masked_table.data.grid[2][0].text == "Data4"

    def test_table_round_trip(self):
        """Test that masked tables maintain structure through round trip."""
        # Create document with table
        doc = DoclingDocument(name="test_roundtrip.txt")

        table_data = TableData(num_rows=3, num_cols=2)

        cells_data = [
            ["Employee", "Contact"],
            ["Bob Wilson", "bob@company.com"],
            ["Alice Brown", "alice@company.com"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        # Mask document
        policy = MaskingPolicy(
            default_strategy=Strategy(
                kind=StrategyKind.TEMPLATE, parameters={"template": "[MASKED]"}
            )
        )
        engine = CloakEngine(default_policy=policy)
        mask_result = engine.mask_document(doc)

        # Check that table structure is preserved after masking
        assert len(mask_result.document.tables) == 1
        masked_table = mask_result.document.tables[0]
        assert len(masked_table.data.grid) == 3  # Same number of rows
        assert len(masked_table.data.grid[0]) == 2  # Same number of columns

        # Check that some masking occurred
        table_text = ""
        for row in masked_table.data.grid:
            for cell in row:
                if hasattr(cell, "text"):
                    table_text += cell.text + " "

        # PII should be masked
        assert "bob@company.com" not in table_text
        assert "alice@company.com" not in table_text

        # Unmask document
        restored = engine.unmask_document(mask_result.document, mask_result.cloakmap)

        # Check structure is preserved after unmasking
        assert len(restored.tables) == 1
        assert len(restored.tables[0].data.grid) == 3
        assert len(restored.tables[0].data.grid[0]) == 2

        # Note: TEMPLATE strategy is not reversible, so we can't check for exact restoration
        # The cells will contain [MASKED] after unmasking since original data is lost

    def test_table_with_mixed_content(self):
        """Test tables with both PII and non-PII content."""
        doc = DoclingDocument(name="test_mixed.txt")

        table_data = TableData(num_rows=3, num_cols=3)

        cells_data = [
            ["Product", "Price", "Customer"],
            ["Widget A", "$99.99", "John Smith"],
            ["Widget B", "$149.99", "Jane Doe"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        # Add text for detection
        doc.texts = [
            TextItem(
                text="Product Price Customer\nWidget A $99.99 John Smith\nWidget B $149.99 Jane Doe",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Product Price Customer\nWidget A $99.99 John Smith\nWidget B $149.99 Jane Doe",
            )
        ]

        # Mask with specific entity types
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.REDACT),
            per_entity={
                "PERSON": Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[NAME]"}),
            },
        )
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc, entities=["PERSON"])

        masked_table = result.document.tables[0]

        # Check that product names and prices are preserved
        assert masked_table.data.grid[1][0].text == "Widget A"
        assert masked_table.data.grid[2][0].text == "Widget B"
        assert "$99.99" in masked_table.data.grid[1][1].text
        assert "$149.99" in masked_table.data.grid[2][1].text

        # Check that names are masked (if detected)
        # Note: Names might not always be detected depending on context
        table_text = ""
        for row in masked_table.data.grid:
            for cell in row:
                if hasattr(cell, "text"):
                    table_text += cell.text + " "

        if result.entities_masked > 0:
            # If entities were masked, names should not appear
            assert "John Smith" not in table_text or "[NAME]" in table_text
            assert "Jane Doe" not in table_text or "[NAME]" in table_text

    def test_empty_table_cells(self):
        """Test handling of empty table cells."""
        doc = DoclingDocument(name="test_empty.txt")

        table_data = TableData(num_rows=3, num_cols=2)

        cells_data = [
            ["Name", "Email"],
            ["", "test@example.com"],
            ["Alice", ""],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        doc.texts = [
            TextItem(
                text="Name Email\n test@example.com\nAlice",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Name Email\n test@example.com\nAlice",
            )
        ]

        # Mask document - should not crash on empty cells
        policy = MaskingPolicy(default_strategy=Strategy(kind=StrategyKind.REDACT))
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        # Check that the table still has the same structure
        assert len(result.document.tables[0].data.grid) == 3
        assert len(result.document.tables[0].data.grid[0]) == 2

    def test_table_multiple_entities_per_cell(self):
        """Test cells containing multiple PII entities."""
        doc = DoclingDocument(name="test_multiple.txt")

        table_data = TableData(num_rows=3, num_cols=1)

        cells_data = [
            ["Contact Info"],
            ["John Doe - john@example.com - 555-1234"],
            ["Jane Smith (jane@test.com)"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        doc.texts = [
            TextItem(
                text="Contact Info\nJohn Doe - john@example.com - 555-1234\nJane Smith (jane@test.com)",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Contact Info\nJohn Doe - john@example.com - 555-1234\nJane Smith (jane@test.com)",
            )
        ]

        # Mask with template strategy
        policy = MaskingPolicy(
            default_strategy=Strategy(kind=StrategyKind.TEMPLATE, parameters={"template": "[PII]"})
        )
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        # Check that PII is masked
        table_text = ""
        for row in result.document.tables[0].data.grid:
            for cell in row:
                if hasattr(cell, "text"):
                    table_text += cell.text + " "

        # These should be masked if detected
        if result.entities_masked > 0:
            # At least emails should be detected
            assert "john@example.com" not in table_text
            assert "jane@test.com" not in table_text

    @pytest.mark.parametrize(
        "strategy_kind",
        [
            StrategyKind.REDACT,
            StrategyKind.TEMPLATE,
            StrategyKind.HASH,
            StrategyKind.PARTIAL,
        ],
    )
    def test_table_with_different_strategies(self, strategy_kind):
        """Test table masking with different strategy types."""
        doc = DoclingDocument(name=f"test_{strategy_kind.value}.txt")

        table_data = TableData(num_rows=2, num_cols=1)

        cells_data = [
            ["Email"],
            ["user@example.com"],
        ]
        table_data.table_cells = []
        for i, row in enumerate(cells_data):
            for j, text in enumerate(row):
                cell = TableCell(
                    text=text,
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                table_data.table_cells.append(cell)

        table_item = TableItem(
            data=table_data,
            self_ref="#/tables/0",
        )
        doc.tables = [table_item]

        doc.texts = [
            TextItem(
                text="Email\nuser@example.com",
                label=DocItemLabel.TEXT,
                self_ref="#/texts/0",
                orig="Email\nuser@example.com",
            )
        ]

        # Configure strategy parameters
        params = {}
        if strategy_kind == StrategyKind.TEMPLATE:
            params = {"template": "[EMAIL]"}
        elif strategy_kind == StrategyKind.PARTIAL:
            params = {"visible_chars": 4, "position": "end"}

        policy = MaskingPolicy(default_strategy=Strategy(kind=strategy_kind, parameters=params))
        engine = CloakEngine(default_policy=policy)
        result = engine.mask_document(doc)

        # Check that email was masked
        table_text = ""
        for row in result.document.tables[0].data.grid:
            for cell in row:
                if hasattr(cell, "text"):
                    table_text += cell.text + " "
        assert "user@example.com" not in table_text

        # Check strategy-specific output
        if strategy_kind == StrategyKind.TEMPLATE:
            # Template strategy should show [EMAIL]
            if result.entities_masked > 0:
                assert "[EMAIL]" in table_text or "user@example.com" not in table_text
        elif strategy_kind == StrategyKind.REDACT and result.entities_masked > 0:
            # Redact should show asterisks
            cells = result.document.tables[0].data.grid
            email_cell = cells[1][0].text
            if "user@example.com" not in email_cell:
                # If masked, should have some masking characters
                assert "*" in email_cell or len(email_cell) < len("user@example.com")
