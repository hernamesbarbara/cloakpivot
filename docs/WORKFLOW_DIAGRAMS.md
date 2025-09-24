# CloakPivot Workflow Diagrams

This document provides visual representations of CloakPivot's document processing workflows, showing how documents flow through the system from input to masked output and back to original content.

## Core Document Processing Workflow

### PDF → JSON → Masked JSON → Masked Markdown

```mermaid
graph TD
    A[PDF Document] --> B[Docling DocumentConverter]
    B --> C[DoclingDocument JSON]
    C --> D[CloakEngine.mask_document]
    
    D --> E[TextExtractor]
    E --> F[Text Segments]
    
    F --> G[Presidio AnalyzerEngine]
    G --> H[Detected Entities]
    
    H --> I[MaskingEngine]
    I --> J[Strategy Application]
    J --> K[Masked DoclingDocument]
    
    I --> L[CloakMap Generation]
    L --> M[CloakMap JSON]
    
    K --> N[DocPivot Conversion]
    N --> O[Masked Markdown]
    
    style A fill:#e1f5fe
    style C fill:#f3e5f5
    style K fill:#fff3e0
    style M fill:#e8f5e8
    style O fill:#fff9c4
```

### Detailed Processing Steps

```mermaid
sequenceDiagram
    participant User
    participant CloakEngine
    participant Docling
    participant TextExtractor
    participant Presidio
    participant MaskingEngine
    participant CloakMap
    participant DocPivot

    User->>CloakEngine: mask_document(pdf_doc)
    CloakEngine->>Docling: Convert PDF
    Docling-->>CloakEngine: DoclingDocument
    
    CloakEngine->>TextExtractor: extract_text_segments()
    TextExtractor-->>CloakEngine: TextSegments[]
    
    CloakEngine->>Presidio: analyze(text, entities)
    Presidio-->>CloakEngine: RecognizerResult[]
    
    CloakEngine->>MaskingEngine: mask_document()
    MaskingEngine->>MaskingEngine: Apply strategies per entity
    MaskingEngine-->>CloakEngine: MaskingResult
    
    MaskingEngine->>CloakMap: Generate mapping
    CloakMap-->>CloakEngine: CloakMap
    
    CloakEngine-->>User: MaskResult{document, cloakmap}
    
    User->>DocPivot: to_markdown(masked_doc)
    DocPivot-->>User: Masked Markdown
```

## Unmasking Workflow

### Masked JSON + CloakMap → Original JSON → Original Markdown

```mermaid
graph TD
    A[Masked DoclingDocument] --> B[CloakEngine.unmask_document]
    C[CloakMap] --> B
    
    B --> D[UnmaskingEngine]
    D --> E[Anchor Processing]
    E --> F[Text Replacement]
    F --> G[Original DoclingDocument]
    
    G --> H[DocPivot Conversion]
    H --> I[Original Markdown]
    
    style A fill:#fff3e0
    style C fill:#e8f5e8
    style G fill:#f3e5f5
    style I fill:#e1f5fe
```

## Strategy Application Flow

### How Different PII Types Are Processed

```mermaid
graph TD
    A[Detected Entity] --> B{Policy Check}
    B -->|PHONE_NUMBER| C[Phone Strategy]
    B -->|EMAIL_ADDRESS| D[Email Strategy]
    B -->|CREDIT_CARD| E[Credit Card Strategy]
    B -->|PERSON| F[Person Strategy]
    B -->|Other| G[Default Strategy]
    
    C --> H[Apply Template/Partial/Hash]
    D --> H
    E --> H
    F --> H
    G --> H
    
    H --> I[Generate Replacement]
    I --> J[Create Anchor Entry]
    J --> K[Store in CloakMap]
    
    style A fill:#e3f2fd
    style B fill:#f1f8e9
    style H fill:#fff8e1
    style K fill:#e8f5e8
```

## CloakMap Structure and Usage

### CloakMap Components

```mermaid
graph TD
    A[CloakMap] --> B[Document Metadata]
    A --> C[Anchor Entries]
    A --> D[Policy Snapshot]
    A --> E[Security Features]
    
    B --> B1[doc_id]
    B --> B2[doc_hash]
    B --> B3[created_at]
    
    C --> C1[AnchorEntry[]]
    C1 --> C2[original_text]
    C1 --> C3[masked_value]
    C1 --> C4[position_info]
    C1 --> C5[entity_type]
    
    D --> D1[strategies_used]
    D --> D2[thresholds]
    D --> D3[locale]
    
    E --> E1[encryption]
    E --> E2[signatures]
    E --> E3[presidio_metadata]
    
    style A fill:#e8f5e8
    style C1 fill:#fff3e0
```

## CLI Workflow

### Command Line Usage Flow

```mermaid
graph TD
    A[cloakpivot mask document.pdf] --> B[Load PDF]
    B --> C[Initialize CloakEngine]
    C --> D[Process Document]
    D --> E[Generate Outputs]
    E --> F[masked.md]
    E --> G[document.cloakmap.json]
    
    H[cloakpivot unmask masked.md document.cloakmap.json] --> I[Load Masked Doc & CloakMap]
    I --> J[Initialize CloakEngine]
    J --> K[Unmask Document]
    K --> L[restored.md]
    
    style F fill:#fff9c4
    style G fill:#e8f5e8
    style L fill:#e1f5fe
```

## Builder Pattern Configuration

### CloakEngine.builder() Flow

```mermaid
graph TD
    A[CloakEngine.builder()] --> B[CloakEngineBuilder]
    
    B --> C[.with_languages(['en', 'es'])]
    C --> D[.with_confidence_threshold(0.9)]
    D --> E[.with_custom_policy(policy)]
    E --> F[.with_conflict_resolution(config)]
    F --> G[.build()]
    
    G --> H[Configured CloakEngine]
    
    style A fill:#e3f2fd
    style H fill:#f3e5f5
```

## Error Handling and Validation

### Document Processing Error Flow

```mermaid
graph TD
    A[Document Input] --> B{Valid Format?}
    B -->|No| C[DocumentFormatError]
    B -->|Yes| D[Text Extraction]
    
    D --> E{Extraction Success?}
    E -->|No| F[TextExtractionError]
    E -->|Yes| G[Entity Detection]
    
    G --> H{Detection Success?}
    H -->|No| I[EntityDetectionError]
    H -->|Yes| J[Masking Process]
    
    J --> K{Masking Success?}
    K -->|No| L[MaskingError]
    K -->|Yes| M[Success Result]
    
    style C fill:#ffebee
    style F fill:#ffebee
    style I fill:#ffebee
    style L fill:#ffebee
    style M fill:#e8f5e8
```

## Performance Optimizations (PR-013)

### Optimized Processing Pipeline

```mermaid
graph TD
    A[Text Processing] --> B[O(n) Algorithm]
    B --> C[List Concatenation]
    C --> D[Efficient Replacement]
    
    E[Strategy Mapping] --> F[LRU Cache]
    F --> G[128 Entry Limit]
    G --> H[2x-5x Speedup]
    
    I[Document Building] --> J[Batch Operations]
    J --> K[Memory Efficiency]
    
    style B fill:#c8e6c9
    style F fill:#c8e6c9
    style J fill:#c8e6c9
```

## Integration Patterns

### Common Usage Patterns

```mermaid
graph TD
    A[Healthcare Documents] --> B[Conservative Policy]
    C[Development Testing] --> D[Permissive Policy]
    E[Financial Reports] --> F[Template Policy]
    G[Legal Documents] --> H[Custom Policy]
    
    B --> I[High Confidence Thresholds]
    D --> J[Low Confidence Thresholds]
    F --> K[Format-Preserving Templates]
    H --> L[Context-Aware Rules]
    
    style I fill:#ffcdd2
    style J fill:#dcedc8
    style K fill:#fff3e0
    style L fill:#e1f5fe
```

---

## Legend

- 🔵 **Blue**: Input documents and original content
- 🟣 **Purple**: DoclingDocument JSON format
- 🟠 **Orange**: Masked content
- 🟢 **Green**: CloakMap and mapping data
- 🟡 **Yellow**: Output formats (Markdown, etc.)
- 🔴 **Red**: Error states and validation failures

## Key Concepts

### Document Flow States
1. **Original**: Unprocessed document with PII
2. **Analyzed**: Entities detected but not yet masked
3. **Masked**: PII replaced with strategies, CloakMap generated
4. **Exported**: Converted to target format (Markdown, JSON, etc.)
5. **Restored**: Original content restored using CloakMap

### Processing Guarantees
- **Reversibility**: Perfect restoration using CloakMap
- **Structure Preservation**: Document layout and formatting maintained
- **Security**: Optional encryption and signing of CloakMaps
- **Performance**: Optimized algorithms for large documents
- **Flexibility**: Configurable policies and strategies per entity type