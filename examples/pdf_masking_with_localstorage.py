#!/usr/bin/env python3
"""
PDF Masking with LocalStorage Example

Demonstrates CloakPivot's local storage features for managing multiple
document formats and their relationships through systematic file organization.
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from docling.document_converter import DocumentConverter
from docling_core.types import DoclingDocument
from presidio_analyzer import AnalyzerEngine

from cloakpivot import (
    MaskingEngine,
    TextExtractor,
    MaskingPolicy,
    CloakMap,
    Strategy,
    StrategyKind
)
from cloakpivot.storage import LocalStorage, StorageRegistry
from cloakpivot.storage.backends.base import StorageMetadata


class DocumentWorkflow:
    """Manages document processing workflow with local storage."""
    
    def __init__(self, storage_base: str = "data/storage"):
        """Initialize workflow with storage backend."""
        self.storage_base = Path(storage_base)
        
        # Initialize LocalStorage with proper configuration
        self.storage = LocalStorage(
            base_path=str(self.storage_base),
            config={
                "create_dirs": True,  # Auto-create directories
                "file_extension": ".cmap",  # CloakMap files
                "metadata_extension": ".meta",  # Metadata sidecars
                "ensure_permissions": 0o600,  # Secure file permissions
                "remove_empty_dirs": True  # Clean up empty directories
            }
        )
        
        # Document converter and analyzers
        self.converter = DocumentConverter()
        self.analyzer = AnalyzerEngine()
        self.extractor = TextExtractor()
        
        # Track document relationships
        self.doc_registry = {}
        
    def get_storage_key(self, original_path: Path, suffix: str = "") -> str:
        """
        Generate storage key following CloakPivot naming conventions.
        
        Conventions:
        - Preserves directory structure relative to data root
        - Adds descriptive suffixes for different formats
        - Maintains consistent naming across transformations
        """
        # Get relative path components
        rel_path = original_path.relative_to(Path("data"))
        base_name = original_path.stem
        
        # Build hierarchical key
        if suffix:
            key = f"{rel_path.parent}/{base_name}.{suffix}"
        else:
            key = f"{rel_path.parent}/{base_name}"
            
        return key
    
    def process_pdf(self, pdf_path: Path) -> dict:
        """
        Complete PDF processing workflow with storage management.
        
        Returns:
            Dictionary containing all generated file paths and metadata
        """
        print(f"{'='*60}")
        print(f"PDF Masking with LocalStorage Workflow")
        print(f"{'='*60}\n")
        
        # Step 1: Convert PDF to DoclingDocument
        print(f"📄 Converting PDF: {pdf_path}")
        dl_doc = self.converter.convert(pdf_path).document
        print(f"  ✓ Converted to DoclingDocument")
        print(f"  - Document name: {dl_doc.name}")
        print(f"  - Text items: {len(dl_doc.texts)}")
        print(f"  - Binary hash: {dl_doc.origin.binary_hash}\n")
        
        # Generate unique document ID using DocumentOrigin's binary_hash
        # The binary_hash is computed by Docling from the PDF file content
        # We combine it with timestamp for processing uniqueness
        # Format: filename_hash_timestamp
        doc_id = f"{dl_doc.name}_{dl_doc.origin.binary_hash}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        results = {
            "doc_id": doc_id,
            "original_pdf": str(pdf_path),
            "storage_base": str(self.storage_base),
            "files_created": [],
            "metadata_files": [],
            "cloakmap_files": [],
            "binary_hash": dl_doc.origin.binary_hash
        }
        
        # Step 2: Save original DoclingDocument to storage
        print("💾 Saving original DoclingDocument to LocalStorage")
        
        # Create storage key for original docling
        docling_key = self.get_storage_key(pdf_path, "docling")
        
        # Save docling JSON with metadata
        docling_json = dl_doc.model_dump(mode='json', by_alias=True)
        docling_bytes = json.dumps(docling_json, indent=2).encode('utf-8')
        
        # Create a CloakMap that embeds the document in metadata
        original_cloakmap = CloakMap.create(
            doc_id=doc_id,
            doc_hash=self._calculate_hash(docling_bytes),
            anchors=[],  # No masking yet
            metadata={
                "document_type": "docling",
                "source_pdf": str(pdf_path),
                "conversion_timestamp": datetime.now().isoformat(),
                "is_masked": False,
                "document_content": docling_json  # Embed the actual document content
            }
        )
        
        # Save using storage backend
        storage_metadata = self.storage.save(
            key=docling_key,
            cloakmap=original_cloakmap,
            metadata={
                "format": "docling.json",
                "source": str(pdf_path),
                "stage": "original",
                "binary_hash": dl_doc.origin.binary_hash,
                "origin_filename": dl_doc.origin.filename,
                "origin_mimetype": dl_doc.origin.mimetype
            }
        )
        
        print(f"  ✓ Saved to: {self.storage_base / docling_key}.cmap")
        print(f"  ✓ Metadata sidecar: {self.storage_base / docling_key}.meta")
        print(f"  - Storage key: {docling_key}")
        print(f"  - Content hash: {storage_metadata.content_hash[:16]}...")
        print(f"  - Size: {storage_metadata.size_bytes} bytes\n")
        
        results["files_created"].append(f"{docling_key}.cmap")
        results["metadata_files"].append(f"{docling_key}.meta")
        
        # Step 3: Detect PII entities
        print("🔍 Detecting PII entities")
        text_segments = self.extractor.extract_text_segments(dl_doc)
        
        all_entities = []
        entity_types_found = set()
        
        for segment in text_segments:
            segment_entities = self.analyzer.analyze(text=segment.text, language="en")
            
            # Adjust positions to global coordinates
            for entity in segment_entities:
                from presidio_analyzer import RecognizerResult
                adjusted_entity = RecognizerResult(
                    entity_type=entity.entity_type,
                    start=entity.start + segment.start_offset,
                    end=entity.end + segment.start_offset,
                    score=entity.score,
                    analysis_explanation=entity.analysis_explanation,
                )
                all_entities.append(adjusted_entity)
                entity_types_found.add(entity.entity_type)
        
        print(f"  ✓ Found {len(all_entities)} PII entities")
        if entity_types_found:
            print(f"  - Types: {', '.join(sorted(entity_types_found))}\n")
        
        # Step 4: Create masking policy
        print("📋 Creating masking policy")
        policy = MaskingPolicy(
            per_entity={
                "EMAIL_ADDRESS": Strategy(StrategyKind.TEMPLATE, {"template": "[EMAIL]"}),
                "PERSON": Strategy(StrategyKind.TEMPLATE, {"template": "[NAME]"}),
                "PHONE_NUMBER": Strategy(StrategyKind.TEMPLATE, {"template": "[PHONE]"}),
                "LOCATION": Strategy(StrategyKind.TEMPLATE, {"template": "[LOCATION]"}),
                "DATE_TIME": Strategy(StrategyKind.TEMPLATE, {"template": "[DATE]"}),
                "CREDIT_CARD": Strategy(StrategyKind.TEMPLATE, {"template": "[CARD]"}),
                "US_SSN": Strategy(StrategyKind.PARTIAL, {"visible_chars": 4, "position": "end"}),
            },
            default_strategy=Strategy(StrategyKind.TEMPLATE, {"template": "[REDACTED]"})
        )
        print(f"  ✓ Policy configured with {len(policy.per_entity)} entity types\n")
        
        # Step 5: Mask the document
        print("🔒 Masking document")
        
        # Create custom conflict resolution config to fix the adjacency grouping bug
        from cloakpivot.core.normalization import ConflictResolutionConfig
        conflict_config = ConflictResolutionConfig(
            merge_threshold_chars=0  # Don't group adjacent entities
        )
        
        masking_engine = MaskingEngine(
            resolve_conflicts=True,
            conflict_resolution_config=conflict_config
        )
        mask_result = masking_engine.mask_document(
            document=dl_doc,
            entities=all_entities,
            policy=policy,
            text_segments=text_segments
        )
        
        print(f"  ✓ Masking complete")
        print(f"  - CloakMap anchors: {len(mask_result.cloakmap.anchors)}")
        if mask_result.stats:
            print(f"  - Stats: {mask_result.stats}")
        print()
        
        # Step 6: Save masked DoclingDocument
        print("💾 Saving masked DoclingDocument to LocalStorage")
        
        masked_docling_key = self.get_storage_key(pdf_path, "masked.docling")
        
        # Save masked document with its CloakMap
        masked_json = mask_result.masked_document.model_dump(mode='json', by_alias=True)
        masked_bytes = json.dumps(masked_json, indent=2).encode('utf-8')
        
        # Create CloakMap for masked document with embedded content
        masked_cloakmap = CloakMap.create(
            doc_id=doc_id,
            doc_hash=self._calculate_hash(masked_bytes),
            anchors=mask_result.cloakmap.anchors,
            metadata={
                "document_type": "docling",
                "source_pdf": str(pdf_path),
                "masking_timestamp": datetime.now().isoformat(),
                "is_masked": True,
                "original_key": docling_key,
                "entities_masked": len(all_entities),
                "entity_types": list(entity_types_found),
                "document_content": masked_json  # Embed the masked document content
            }
        )
        
        masked_metadata = self.storage.save(
            key=masked_docling_key,
            cloakmap=masked_cloakmap,
            metadata={
                "format": "masked.docling.json",
                "source": str(pdf_path),
                "stage": "masked",
                "original_key": docling_key
            }
        )
        
        print(f"  ✓ Saved to: {self.storage_base / masked_docling_key}.cmap")
        print(f"  ✓ Metadata sidecar: {self.storage_base / masked_docling_key}.meta")
        print(f"  - Storage key: {masked_docling_key}")
        print(f"  - Content hash: {masked_metadata.content_hash[:16]}...\n")
        
        results["files_created"].append(f"{masked_docling_key}.cmap")
        results["metadata_files"].append(f"{masked_docling_key}.meta")
        results["cloakmap_files"].append(f"{masked_docling_key}.cmap")
        
        # Step 7: Save standalone CloakMap for unmasking
        print("🗺️ Saving standalone CloakMap for reversibility")
        
        cloakmap_key = self.get_storage_key(pdf_path, "cloakmap")
        
        cloakmap_metadata = self.storage.save(
            key=cloakmap_key,
            cloakmap=mask_result.cloakmap,
            metadata={
                "format": "cloakmap",
                "source": str(pdf_path),
                "stage": "mapping",
                "original_key": docling_key,
                "masked_key": masked_docling_key,
                "purpose": "reversible_unmasking"
            }
        )
        
        print(f"  ✓ Saved to: {self.storage_base / cloakmap_key}.cmap")
        print(f"  ✓ Metadata sidecar: {self.storage_base / cloakmap_key}.meta")
        print(f"  - Anchor count: {len(mask_result.cloakmap.anchors)}")
        print(f"  - Can unmask: Yes\n")
        
        results["files_created"].append(f"{cloakmap_key}.cmap")
        results["metadata_files"].append(f"{cloakmap_key}.meta")
        results["cloakmap_files"].append(f"{cloakmap_key}.cmap")
        
        # Step 8: Demonstrate storage organization
        print("📁 Storage Organization")
        self._show_storage_structure()
        
        # Step 9: Demonstrate metadata queries
        print("\n🔎 Demonstrating Metadata Queries")
        self._demonstrate_metadata_queries(doc_id)
        
        # Step 10: List all related files
        print("\n📋 All Related Files")
        all_keys = self.storage.list_keys(prefix="pdf/")
        for key in all_keys:
            if pdf_path.stem in key:
                print(f"  - {key}")
                # Load and display metadata
                meta_path = self.storage._get_metadata_path(key)
                if meta_path.exists():
                    with open(meta_path) as f:
                        meta = json.load(f)
                        print(f"    Doc ID: {meta.get('doc_id', 'N/A')}")
                        print(f"    Stage: {meta.get('backend_metadata', {}).get('stage', 'N/A')}")
        
        results["total_files"] = len(results["files_created"])
        results["total_metadata"] = len(results["metadata_files"])
        
        return results
    
    def _calculate_hash(self, content: bytes) -> str:
        """Calculate SHA256 hash of content."""
        import hashlib
        return hashlib.sha256(content).hexdigest()
    
    def _show_storage_structure(self):
        """Display the storage directory structure."""
        import os
        
        print(f"  Storage base: {self.storage_base}")
        print("  Structure:")
        
        # Walk through storage directory
        for root, dirs, files in os.walk(self.storage_base):
            root = Path(root)
            level = len(root.relative_to(self.storage_base).parts)
            indent = "  " * (level + 1)
            print(f"{indent}{root.name}/")
            
            # Show files
            sub_indent = "  " * (level + 2)
            for file in sorted(files):
                file_path = root / file
                size = file_path.stat().st_size
                
                if file.endswith('.cmap'):
                    print(f"{sub_indent}├── {file} ({size} bytes) [CloakMap]")
                elif file.endswith('.meta'):
                    print(f"{sub_indent}├── {file} ({size} bytes) [Metadata]")
                else:
                    print(f"{sub_indent}├── {file} ({size} bytes)")
    
    def _demonstrate_metadata_queries(self, doc_id: str):
        """Demonstrate metadata sidecar usage."""
        print(f"  Finding all files for document ID: {doc_id}")
        
        related_files = []
        all_keys = self.storage.list_keys()
        
        for key in all_keys:
            meta_path = self.storage._get_metadata_path(key)
            if meta_path.exists():
                with open(meta_path) as f:
                    metadata = json.load(f)
                    if metadata.get("doc_id") == doc_id:
                        related_files.append({
                            "key": key,
                            "stage": metadata.get("backend_metadata", {}).get("stage"),
                            "format": metadata.get("backend_metadata", {}).get("format"),
                            "size": metadata.get("size_bytes"),
                            "created": metadata.get("created_at")
                        })
        
        print(f"  ✓ Found {len(related_files)} related files:")
        for file_info in related_files:
            print(f"    - {file_info['key']}")
            print(f"      Stage: {file_info['stage']}, Format: {file_info['format']}")


def main():
    """Main execution function."""
    # Input PDF path
    pdf_path = Path("data/pdf/email.pdf")
    
    if not pdf_path.exists():
        print(f"❌ Error: Input PDF not found: {pdf_path}")
        print("Please ensure the file exists at the specified location.")
        sys.exit(1)
    
    # Initialize workflow
    workflow = DocumentWorkflow(storage_base="data/storage")
    
    # Process the PDF
    results = workflow.process_pdf(pdf_path)
    
    # Summary
    print(f"\n{'='*60}")
    print("✅ Workflow Complete!")
    print(f"{'='*60}")
    print(f"Document ID: {results['doc_id']}")
    print(f"Files created: {results['total_files']}")
    print(f"Metadata files: {results['total_metadata']}")
    print(f"Storage location: {results['storage_base']}")
    
    print("\n📚 Key Concepts Demonstrated:")
    print("  1. LocalStorage backend with automatic directory creation")
    print("  2. Systematic naming conventions for document variations")
    print("  3. Metadata sidecar files (.meta) for each stored item")
    print("  4. CloakMap usage for tracking masked entities")
    print("  5. Document relationship tracking via doc_id")
    print("  6. Hierarchical storage organization (pdf/email.*)")
    print("  7. Reversible masking with standalone CloakMap")


if __name__ == "__main__":
    main()