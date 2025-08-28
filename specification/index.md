## CloakPivot: PII Masking/Unmasking on Top of DocPivot and Presidio — Product Requirements Document (PRD)

### 1. Overview

- **Problem**: Organizations need to share and analyze documents while protecting PII. Existing redaction often destroys layout/structure, making the document hard to read or to round-trip back to the original.
- **Solution**: CloakPivot is a Python package that leverages `docpivot` for robust document parsing/serialization and `presidio` for PII detection/anonymization. It enables two-way conversion between an original (unmasked) document and a masked version, preserving structure and formatting to allow deterministic unmasking later.
- **Outcomes**:
  - Produce masked views for safe sharing while preserving document fidelity.
  - Round-trip from masked back to original using secure mapping, without re-running recognition.
  - Support batch operations, policy-driven masking, and auditable artifacts.

### 2. Goals and Non-Goals

- **Goals**:
  - Provide high-level APIs to mask/unmask documents with preserved structure using `docpivot` document models.
  - Use `presidio` RecognizerResult entities for detection and configurable anonymization strategies per entity type.
  - Maintain a reversible mapping: masked token ↔ original span, with cryptographic integrity and optional encryption at rest.
  - Support multiple docpivot formats as I/O: `docling.json`, `lexical.json`, and serialized outputs like markdown/html.
  - Minimize formatting drift across conversions; preserve element boundaries and styles.
  - Offer policies (per entity type) and locale-aware recognizer configuration.
  - Provide CLI and Python APIs; enable streaming/batch processing.

- **Non-Goals**:
  - Creating new OCR or layout extraction (use upstream sources/docling pipeline).
  - Implementing custom PII models beyond configuring Presidio (advanced ML out of scope).
  - Guaranteeing perfect layout pixel parity for rasterized exports (focus on structural fidelity in supported formats).

### 3. Users and Use Cases

- **Users**: Data privacy teams, MLOps/data engineers, compliance officers, platform integrators.
- **Use cases**:
  - Share masked datasets while enabling exact unmasking for incident review.
  - Pre-mask documents for LLM ingestion and later unmask specific passages with audit.
  - Policy testing: evaluate impact of different masking policies without losing structure.
  - Pseudonymization pipelines with consistent reversible placeholders across corpora.

### 4. Key Concepts

- **Entity**: In Presidio, an entity is a span identifying PII, represented by `RecognizerResult` with fields like `start`, `end`, `entity_type`, `score`.
- **Masking Policy**: Rules mapping `entity_type` → anonymization strategy (e.g., redact, replace template, hash, surrogate).
- **CloakMap**: A secure mapping artifact that stores original spans, masked replacements, and structural anchors to allow deterministic unmasking. May be encrypted and signed.
- **Anchors**: Stable references to positions in the `docpivot` document model (node IDs, offsets within text nodes) to survive serialization/deserialization.

### 5. System Architecture

- **Inputs**: Supported by `docpivot` readers via `ReaderFactory` and `FormatRegistry`:
  - `docling.json` and `lexical.json` as primary structured inputs.
  - Optional import from markdown/html if/when reader support exists (extensible via plugin).

- **Core Flow (Mask)**:
  1. Load document using `docpivot.load_document(...)` to get a `DoclingDocument`.
  2. Traverse text-bearing nodes; extract plain text segments with structural anchors (node path/ID, offset ranges).
  3. Run Presidio analyzer to obtain `RecognizerResult` entities per segment.
  4. Normalize/merge overlapping entities; resolve conflicts; apply policy.
  5. Generate masked text segments; update the `DoclingDocument` while preserving node boundaries and styles.
  6. Emit outputs: masked document (any serializer via `SerializerProvider`) and `CloakMap` sidecar.

- **Core Flow (Unmask)**:
  1. Load masked document with `docpivot`.
  2. Load and verify `CloakMap` (signature/integrity, decryption as needed).
  3. Re-apply original spans using anchors and replacement tokens to reconstruct the original `DoclingDocument`.
  4. Serialize to desired output or `docling.json` for storage.

- **CloakMap Contents** (versioned JSON):
  - doc metadata: source hash, timestamps, doc name, doc model version.
  - crypto: optional envelope (algorithm, key-id), signature.
  - anchors: list of entries [{ node_id, start, end, entity_type, policy_id, masked_value, original_value_checksum, replacement_id }]
  - policy snapshot: effective rules at masking time.
  - index: quick lookups by `replacement_id` and by node_id.

### 6. Product Requirements

- **R1: Fidelity Preservation**
  - Edits must occur at text run level without altering block hierarchy or styles beyond replaced spans.
  - Round-trip masked→unmasked must produce text-equivalent content; structure-equivalent within docpivot model.

- **R2: Presidio Integration**
  - Support configurable analyzers, languages/locales, recognizer registry extensions, and confidence thresholds.
  - Accept and emit raw `RecognizerResult` entries in diagnostics.

- **R3: Policies**
  - Define policies as composable rules: default + per-entity overrides.
  - Strategies: fixed token (e.g., "[PHONE]"), partial masking (e.g., last 4 visible), hashing (SHA-256 with salt), deterministic surrogate (format-preserving), or custom callback.
  - Support allow/deny lists and contextual constraints (e.g., within headings).

- **R4: Security and Integrity**
  - Optional AES-GCM encryption of `CloakMap`; keys managed externally.
  - HMAC/SHA-256 signatures; verify before unmasking.
  - Store only checksums of originals unless policy allows plaintext escrow.

- **R5: Performance and Scale**
  - Process 100–500 page documents within practical time and memory budgets.
  - Batch mode with progress reporting and error isolation per file.
  - Options for parallel analysis at section or page granularity.

- **R6: Extensibility**
  - Use `docpivot` `FormatRegistry`/`PluginManager` to support new formats.
  - Allow custom policy strategies and custom Presidio recognizers.

- **R7: Observability**
  - Structured logs; per-entity counters; redaction coverage metrics.
  - Debug mode emits annotated overlays and a diff report.

### 7. API Design

- **Python API**
  - `mask_document(input_path: str | Path, output_format: str = "lexical", policy: MaskingPolicy = ..., analyzer: Optional[AnalyzerEngine] = None, cloakmap_path: Optional[str|Path] = None, **kwargs) -> MaskResult`
    - Loads via `docpivot.load_document`, applies policy, returns masked content and writes `CloakMap`.
  - `unmask_document(masked_path: str | Path, cloakmap_path: str | Path, output_format: str = "lexical", **kwargs) -> UnmaskResult`
    - Loads masked doc and `CloakMap`, restores originals, returns serialized output.
  - `mask_docling(doc: DoclingDocument, policy: MaskingPolicy, ...) -> tuple[DoclingDocument, CloakMap]`
  - `unmask_docling(doc: DoclingDocument, cloakmap: CloakMap, ...) -> DoclingDocument`

- **Core Types**
  - `MaskingPolicy`
    - fields: `default_strategy`, `per_entity: dict[str, Strategy]`, `thresholds: dict[str, float]`, `locale`, `seed/salt`, `custom_callbacks`.
  - `Strategy`
    - `kind` in {`redact`, `template`, `hash`, `surrogate`, `partial`, `custom`} with parameters.
  - `CloakMap`
    - fields: `version`, `doc_id`, `doc_hash`, `anchors`, `policy_snapshot`, `crypto`, `signature`.
  - `MaskResult`
    - fields: `doc: DoclingDocument`, `cloakmap_path`, `stats`, `diagnostics`.
  - `UnmaskResult`
    - fields: `doc: DoclingDocument`, `restored_stats`, `diagnostics`.

- **CLI** (`cloakpivot`)
  - `cloakpivot mask <input> --out <path|format> --cloakmap <path> [--policy <file>] [--format lexical|markdown|html] [--lang en] [--min-score 0.5] [--encrypt --key-id <id>]`
  - `cloakpivot unmask <masked> --cloakmap <path> --out <path|format> [--verify-only]`
  - `cloakpivot policy sample > policy.yaml`

### 8. Interactions with docpivot

- Use `docpivot.workflows.load_document` for detection/loading and `SerializerProvider` for output (`markdown`, `html`, `lexical`).
- Use `FormatRegistry`/`PluginManager` to support additional formats.
- Manipulate `DoclingDocument` text nodes conservatively, maintaining `GroupItem`/`NodeItem` boundaries and styles.
- Optionally provide a `CloakPivotSerializer` for emitting side-by-side masked previews (diagnostic only).

### 9. Interactions with Presidio

- Depend on `presidio-analyzer` and optionally `presidio-anonymizer` for standard strategies.
- Configure recognizers, languages, and thresholds via `MaskingPolicy`.
- Return raw `RecognizerResult` in diagnostics for traceability.

### 10. Data Model and Storage

- `CloakMap` default format: versioned JSON with optional JWE-like envelope for encryption.
- Optional storage backends: local file, S3/GCS via pluggable IO layer.
- Reference by content hash to ensure the `CloakMap` matches the masked document instance.

### 11. Security & Privacy

- Never store raw originals in `CloakMap` unless explicitly permitted; store salted checksums or encrypted payloads.
- Provide key rotation hooks; fail-closed if verification fails.
- PII is only present in memory during masking/unmasking; recommend ephemeral processing.

### 12. Performance Requirements

- Target latency: ≤ 2x `docpivot` serialization time for medium documents (≤ 200 pages) with default policies.
- Memory: streaming or chunked processing of large text nodes.
- Parallel execution parameter for multi-core machines.

### 13. Error Handling

- Deterministic behavior for overlapping/adjacent entities; stable ordering and tie-breaks by confidence and entity priority.
- Partial failure isolation per node/section; emit diagnostics without aborting the entire file when safe.
- Clear exceptions mirroring `docpivot` patterns: configuration, validation, transformation errors.

### 14. Telemetry & Reporting

- Stats: total entities found/masked per type, coverage rate, average score.
- Reports: JSON diagnostics, optional HTML preview overlay with highlights.

### 15. Testing Strategy

- Unit tests for policy engines, anchor mapping, and round-trip integrity.
- Golden files for representative formats (docling/lexical) covering edge cases (overlaps, nested styles, tables, headers/footers).
- Fuzzing for anchor drift across serialize/deserialize cycles.

### 16. Sample Data for Development & Testing
- Find realistic document data in the data/ directory
- Documents in PDF format are in data/pdf/
- Documents described in various json dialects are in data/json/ 
- Json dialect is indicated by file naming convention *.lexical.json and *.docling.json

### 17. Deliverables

- Python package `cloakpivot` with:
  - Core API (`mask_document`, `unmask_document`, policy types, CloakMap I/O).
  - CLI with mask/unmask commands.
  - Integration with `docpivot` and Presidio; examples and docs.
  - Reference policy samples and quickstart notebook.

### 18. Milestones

- M1: Minimal mask/unmask on `docling.json` with CloakMap and CLI.
- M2: Add `lexical` serializer round-trip and policy features (partial, hash, surrogate).
- M3: Security hardening (encryption/signature), performance tuning, diagnostics overlay.
- M4: Plugins and custom recognizers, batch workflows, storage backends.

### 19. Dependencies

- `docpivot` (located in `docpivot/` directory in this repo) and its transitive `docling_core` serializers.
- `presidio-analyzer` (and optionally `presidio-anonymizer`).
- Crypto library for HMAC/AES-GCM (e.g., `cryptography`).

### 20. Open Questions

- Preferred default output format for masked distribution (`lexical json` vs `docling json` vs. `markdown`): probably leaning towards docling json or lexical json.
- Whether to support lossy exports (PDF/PNG) with visual overlays as diagnostics only: TBD
- Policy DSL format: YAML vs JSON; schema versioning strategy: TBD
