# Test PDF Files

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
