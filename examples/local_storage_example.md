# PDF Masking with LocalStorage Example

## Purpose

This example demonstrates how CloakPivot's LocalStorage backend manages multiple document formats and their relationships through systematic file organization and metadata tracking. It showcases the complete workflow from PDF ingestion to masked document storage, highlighting CloakPivot's built-in conventions for file naming, directory organization, and metadata management.

## Key Features Demonstrated

1. **LocalStorage Backend Configuration**: Proper setup with security permissions and auto-directory creation
2. **Naming Conventions**: How CloakPivot systematically names different document variations
3. **Metadata Sidecars**: `.meta` files that track document relationships and processing stages
4. **CloakMap Usage**: Proper creation and storage of CloakMaps for reversible masking
5. **Directory Organization**: Hierarchical storage structure preserving source file paths
6. **Document Tracking**: Using `doc_id` to link related files across formats
7. **Query Capabilities**: Finding related documents through metadata queries

## Input Files

- **Primary Input**: `data/pdf/email.pdf`
  - The only hardcoded filename in the script
  - All other filenames are derived from this input using CloakPivot's naming conventions

## Output Files and Structure

The script generates the following file structure under `data/storage/`:

```
data/storage/
└── pdf/
    ├── email.docling.cmap      # Original DoclingDocument
    ├── email.docling.meta      # Metadata for original
    ├── email.masked.docling.cmap  # Masked DoclingDocument
    ├── email.masked.docling.meta  # Metadata for masked
    ├── email.cloakmap.cmap     # Standalone CloakMap for unmasking
    └── email.cloakmap.meta     # Metadata for CloakMap
```

### File Contents Explained

1. **`email.docling.cmap` & `.meta`**
   - **Purpose**: Stores the ORIGINAL DoclingDocument (unmasked)
   - **CloakMap**: Contains original DoclingDocument JSON with empty anchors list
   - **Metadata**: Marks stage="original", includes binary_hash from PDF

2. **`email.masked.docling.cmap` & `.meta`**
   - **Purpose**: Stores the MASKED DoclingDocument
   - **CloakMap**: Contains masked DoclingDocument JSON with populated anchors
   - **Anchors**: Each anchor maps original text → masked replacement (e.g., "john@email.com" → "[EMAIL]")
   - **Metadata**: Marks stage="masked", references original file

3. **`email.cloakmap.cmap` & `.meta`**
   - **Purpose**: Standalone CloakMap for REVERSING the masking
   - **CloakMap**: Contains ONLY the mapping data (no document content)
   - **Use Case**: This is the "key" to unmask the masked document
   - **Metadata**: Marks stage="mapping", purpose="reversible_unmasking"

## Script Logic Flow

### 1. Initialization
```python
workflow = DocumentWorkflow(storage_base="data/storage")
```
- Creates LocalStorage backend with secure permissions (0o600)
- Enables automatic directory creation
- Configures file extensions (.cmap, .meta)

### 2. PDF Processing and Document ID Generation
```python
dl_doc = converter.convert(pdf_path).document
doc_id = f"{dl_doc.name}_{dl_doc.origin.binary_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
```
- Converts PDF to DoclingDocument using Docling
- Extracts structured text and metadata
- Uses DocumentOrigin's binary_hash (computed from PDF content)
- Creates unique doc_id combining: filename + content hash + timestamp
- Ensures both content-based identification and processing uniqueness

### 3. Storage Key Generation
```python
docling_key = self.get_storage_key(pdf_path, "docling")
# Results in: "pdf/email.docling"
```
- Preserves directory structure relative to `data/`
- Adds descriptive suffixes for each format
- No hardcoded names - all derived from input path

### 4. Original Document Storage
```python
original_cloakmap = CloakMap.create(
    doc_id=doc_id,
    doc_hash=self._calculate_hash(docling_bytes),
    anchors=[],  # No masking yet
    metadata={...}
)
storage.save(key=docling_key, cloakmap=original_cloakmap, metadata={...})
```
- Creates pseudo-CloakMap for original document
- Saves with metadata about source and stage
- Generates `.meta` sidecar automatically

### 5. PII Detection
```python
analyzer = AnalyzerEngine()
text_segments = extractor.extract_text_segments(dl_doc)
# Detect entities and adjust coordinates
```
- Uses Presidio for entity detection
- Adjusts positions to global document coordinates

### 6. Document Masking
```python
policy = MaskingPolicy(
    per_entity={
        "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
        "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
        # ... other entity types
    },
    default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
)
mask_result = masking_engine.mask_document(
    document=dl_doc,
    entities=all_entities,
    policy=policy,
    text_segments=text_segments
)
```
- Uses `per_entity` dict to map entity types to strategies
- TEMPLATE strategy performs simple string replacement (no template files needed)
- Generates CloakMap with anchor entries for each masked entity
- Returns MaskingResult with masked_document, cloakmap, and optional stats

### 7. Masked Document Storage
```python
masked_docling_key = self.get_storage_key(pdf_path, "masked.docling")
# Results in: "pdf/email.masked.docling"
```
- Follows naming convention with `.masked.` infix
- Stores with updated CloakMap containing anchors
- Links to original via metadata

### 8. Standalone CloakMap Storage
```python
cloakmap_key = self.get_storage_key(pdf_path, "cloakmap")
storage.save(key=cloakmap_key, cloakmap=mask_result.cloakmap, ...)
```
- Saves CloakMap separately for unmasking operations
- Contains all anchor entries for reversibility

### 9. Metadata Queries
```python
for key in storage.list_keys():
    meta_path = storage._get_metadata_path(key)
    if metadata.get("doc_id") == doc_id:
        # Found related file
```
- Demonstrates finding all files for a document
- Uses metadata sidecars for efficient queries

## Naming Convention Rules

The script follows CloakPivot's default naming conventions:

1. **Base name preservation**: Original filename stem is preserved
2. **Format suffixes**: Added to indicate document format
   - `.docling` - DoclingDocument format
   - `.masked.docling` - Masked DoclingDocument
   - `.cloakmap` - Standalone CloakMap
3. **Directory preservation**: Source directory structure maintained
4. **No version numbers**: Timestamps in doc_id handle versioning

## Metadata Sidecar Structure

Each `.meta` file contains:

```json
{
  "key": "pdf/email.docling",
  "size_bytes": 8192,
  "content_hash": "sha256:abc123...",
  "created_at": "2024-01-15T10:30:00",
  "modified_at": "2024-01-15T10:30:00",
  "doc_id": "email_10251300040603033583_20240115_103000",
  "version": "1.0",
  "anchor_count": 0,
  "is_encrypted": false,
  "backend_type": "local_filesystem",
  "backend_metadata": {
    "format": "docling.json",
    "source": "data/pdf/email.pdf",
    "stage": "original",
    "binary_hash": 10251300040603033583
  }
}
```

## Running the Example

```bash
# Ensure input PDF exists
ls data/pdf/email.pdf

# Run the example
python examples/pdf_masking_with_localstorage.py

# Examine the generated storage structure
tree data/storage/

# View a metadata file
cat data/storage/pdf/email.docling.meta | jq .
```

## Key Takeaways

1. **No Hardcoded Filenames**: Only the input PDF path is specified; all other names are derived
2. **Content-Based Identification**: Uses DocumentOrigin's binary_hash for PDF content identification
3. **Unique Processing IDs**: Combines binary_hash with timestamp for deduplication and tracking
4. **Systematic Organization**: Files are organized hierarchically matching source structure
5. **Metadata Tracking**: Every stored item has a `.meta` sidecar for relationship tracking
6. **Document Relationships**: All files share a `doc_id` for easy correlation
7. **Reversible Masking**: Standalone CloakMap enables unmasking operations
8. **Storage Abstraction**: Same code works with S3, GCS, or database backends
9. **Query Capabilities**: Metadata enables finding related files without parsing content

## Extension Points

The example can be extended to:

- Add encryption to sensitive CloakMaps
- Implement versioning with timestamps
- Support multiple masking policies
- Add cloud storage backends
- Implement batch processing
- Add audit logging
- Support format conversion (lexical.json)