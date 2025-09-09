"""Optimized batch processing for high-volume Presidio operations."""

import base64
import hashlib
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from presidio_analyzer import AnalyzerEngine
from presidio_analyzer import RecognizerResult as AnalyzerRecognizerResult
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult as AnonymizerRecognizerResult

from cloakpivot.core.cloakmap import CloakMap
from cloakpivot.core.policies import MaskingPolicy
from cloakpivot.masking.engine import MaskingResult


class PresidioBatchProcessor:
    """Optimized batch processing for high-volume operations."""

    def __init__(self, batch_size: int = 100, parallel_workers: int = 4, default_mask: str = "[REDACTED]"):
        """Initialize batch processor.

        Args:
            batch_size: Number of items to process in each batch
            parallel_workers: Number of parallel workers for processing
            default_mask: Default masking string for redacted content
        """
        self.batch_size = batch_size
        self.parallel_workers = parallel_workers
        self.default_mask = default_mask
        self.anonymizer_pool = self._create_anonymizer_pool()
        self.analyzer_pool = self._create_analyzer_pool()
        self._processing_stats: dict[str, Any] = {
            "total_processed": 0,
            "total_time_ms": 0,
            "batch_count": 0,
            "errors": []
        }

    def _create_anonymizer_pool(self) -> list[AnonymizerEngine]:
        """Create pool of AnonymizerEngine instances for parallel processing.

        Returns:
            List of AnonymizerEngine instances
        """
        return [AnonymizerEngine() for _ in range(self.parallel_workers)]

    def _create_analyzer_pool(self) -> list[AnalyzerEngine]:
        """Create pool of AnalyzerEngine instances for parallel processing.

        Returns:
            List of AnalyzerEngine instances
        """
        return [AnalyzerEngine() for _ in range(self.parallel_workers)]

    def _create_batches(self, documents: list[Any]) -> list[list[Any]]:
        """Group documents into optimal batches.

        Args:
            documents: List of documents to batch

        Returns:
            List of document batches
        """
        batches = []
        for i in range(0, len(documents), self.batch_size):
            batch = documents[i:i + self.batch_size]
            batches.append(batch)
        return batches

    def _process_single_document(
        self,
        document: Any,
        policy: MaskingPolicy,
        analyzer: AnalyzerEngine,
        anonymizer: AnonymizerEngine
    ) -> MaskingResult:
        """Process a single document.

        Args:
            document: Document to process
            policy: Masking policy to apply
            analyzer: Analyzer engine to use
            anonymizer: Anonymizer engine to use

        Returns:
            MaskingResult for the document
        """
        try:
            # Extract text from document
            text = self._extract_text(document)

            # Analyze for PII
            analyzer_results = analyzer.analyze(text=text, language="en")

            # Convert policy to operators
            operators = self._policy_to_operators(policy)

            # Apply anonymization - need to convert analyzer results for anonymizer
            # The anonymizer expects its own RecognizerResult type
            anonymizer_recognizer_results = [
                AnonymizerRecognizerResult(
                    entity_type=r.entity_type,
                    start=r.start,
                    end=r.end,
                    score=r.score
                )
                for r in analyzer_results
            ]

            anonymizer_result = anonymizer.anonymize(
                text=text,
                analyzer_results=anonymizer_recognizer_results,
                operators=operators
            )

            # Create CloakMap
            cloakmap = self._create_cloakmap(
                document,
                analyzer_results,
                anonymizer_result
            )

            # Apply masking to document
            masked_document = self._apply_masking(document, anonymizer_result)

            from cloakpivot.masking.engine import MaskingResult
            return MaskingResult(
                masked_document=masked_document,
                cloakmap=cloakmap,
                stats={"success": True}
            )

        except Exception as e:
            # Return error result
            from cloakpivot.core.cloakmap import CloakMap
            from cloakpivot.masking.engine import MaskingResult

            # Create an empty CloakMap with required fields
            doc_id_val = getattr(document, "id", None) or getattr(document, "name", "unknown")
            doc_id = str(doc_id_val) if doc_id_val is not None else "unknown"
            # Generate a hash for the error case
            error_hash = hashlib.sha256(f"error_{doc_id}_{str(e)}".encode()).hexdigest()[:8]
            empty_cloakmap = CloakMap(
                doc_id=doc_id,
                version="2.0",
                doc_hash=error_hash,
                anchors=[]
            )

            return MaskingResult(
                masked_document=document,
                cloakmap=empty_cloakmap,
                stats={"success": False, "error": str(e)}
            )

    def _extract_text(self, document: Any) -> str:
        """Extract text from document.

        Args:
            document: Document to extract text from

        Returns:
            Extracted text
        """
        # Check if it's a DoclingDocument
        if hasattr(document, "export_to_text"):
            try:
                text_result = document.export_to_text()
                return str(text_result) if text_result is not None else ""
            except Exception:
                pass

        # Check for text attribute first
        try:
            if hasattr(document, "text"):
                text = document.text
                # Ensure it's a string (handle MagicMock and other types)
                if isinstance(text, str):
                    return text
                else:
                    return str(text)
        except Exception:
            pass

        # Check for content attribute
        try:
            if hasattr(document, "content"):
                content = document.content
                if isinstance(content, str):
                    return content
                else:
                    return str(content)
        except Exception:
            pass

        return str(document)

    def _policy_to_operators(self, policy: MaskingPolicy) -> dict[str, OperatorConfig]:
        """Convert masking policy to Presidio operators.

        Args:
            policy: Masking policy

        Returns:
            Dictionary of operators
        """
        operators = {}

        # Default operator for all entity types
        operators["DEFAULT"] = OperatorConfig(
            operator_name="replace",
            params={"new_value": self.default_mask}
        )

        # Add specific operators from policy if available
        if hasattr(policy, "per_entity") and policy.per_entity:
            for entity_type, strategy in policy.per_entity.items():
                if strategy.kind.name == "REDACT":
                    operators[entity_type] = OperatorConfig(
                        operator_name="redact"
                    )
                elif strategy.kind.name == "HASH":
                    operators[entity_type] = OperatorConfig(
                        operator_name="hash",
                        params={"hash_type": "sha256"}
                    )
                elif strategy.kind.name == "PARTIAL":
                    operators[entity_type] = OperatorConfig(
                        operator_name="mask",
                        params={
                            "masking_char": "*",
                            "chars_to_mask": 4,
                            "from_end": False
                        }
                    )
                elif strategy.kind.name == "TEMPLATE":
                    operators[entity_type] = OperatorConfig(
                        operator_name="replace",
                        params={"new_value": f"[{entity_type}]"}
                    )
                else:
                    operators[entity_type] = OperatorConfig(
                        operator_name="replace",
                        params={"new_value": f"[{entity_type}]"}
                    )

        # Handle default strategy
        if hasattr(policy, "default_strategy") and policy.default_strategy:
            strategy = policy.default_strategy
            if strategy.kind.name == "REDACT":
                operators["DEFAULT"] = OperatorConfig(
                    operator_name="redact"
                )
            elif strategy.kind.name == "HASH":
                operators["DEFAULT"] = OperatorConfig(
                    operator_name="hash",
                    params={"hash_type": "sha256"}
                )
            elif strategy.kind.name == "MASK":
                operators["DEFAULT"] = OperatorConfig(
                    operator_name="mask",
                    params={
                        "masking_char": "*",
                        "chars_to_mask": 4,
                        "from_end": False
                    }
                )

        return operators

    def _create_cloakmap(
        self,
        document: Any,
        analyzer_results: list[AnalyzerRecognizerResult],
        anonymizer_result: Any
    ) -> CloakMap:
        """Create CloakMap from processing results.

        Args:
            document: Original document
            analyzer_results: PII detection results
            anonymizer_result: Anonymization results

        Returns:
            CloakMap instance
        """
        from cloakpivot.core.anchors import AnchorEntry

        # Extract text from document
        text = self._extract_text(document)

        # Create anchors from analyzer results
        anchors = []
        for result in analyzer_results:
            # Generate checksum for the original text
            original_text = text[result.start:result.end]
            salt = base64.b64encode(hashlib.sha256(f"{result.entity_type}_{result.start}".encode()).digest()[:8]).decode()
            checksum = hashlib.sha256((original_text + salt).encode()).hexdigest()

            anchor = AnchorEntry(
                node_id="doc_root",  # Default node for batch processing
                start=result.start,
                end=result.end,
                entity_type=result.entity_type,
                confidence=result.score,
                masked_value=f"[{result.entity_type}]",  # Default masking
                replacement_id=f"{result.entity_type}_{result.start}_{result.end}",
                original_checksum=checksum,
                checksum_salt=salt,
                strategy_used="batch_processing",
                metadata={
                    "recognition_metadata": getattr(result, 'recognition_metadata', {})
                }
            )
            anchors.append(anchor)

        # Create CloakMap with anchors
        doc_id_val = getattr(document, "id", None) or getattr(document, "name", "unknown")
        doc_id = str(doc_id_val) if doc_id_val is not None else "unknown"
        # Generate a hash for the document
        # text is already guaranteed to be a string from _extract_text
        doc_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        cloakmap = CloakMap(
            doc_id=doc_id,
            version="2.0",
            doc_hash=doc_hash,
            anchors=anchors
        )

        return cloakmap

    def _apply_masking(self, document: Any, anonymizer_result: Any) -> Any:
        """Apply masking results to document.

        Args:
            document: Original document
            anonymizer_result: Anonymization results

        Returns:
            Masked document
        """
        # Create a copy of the document
        from copy import deepcopy
        masked_doc = deepcopy(document) if hasattr(document, '__dict__') else type('Document', (), {})()

        # Apply masked text
        masked_text = anonymizer_result.text if hasattr(anonymizer_result, "text") else str(anonymizer_result)
        
        # Handle different document types
        if hasattr(masked_doc, 'texts') and masked_doc.texts and len(masked_doc.texts) > 0:
            # DoclingDocument - update the first text item
            masked_doc.texts[0].text = masked_text
        elif hasattr(masked_doc, 'text') or not hasattr(masked_doc, 'texts'):
            # Simple document object or mock with text attribute
            masked_doc.text = masked_text

        # Copy other document attributes
        if hasattr(document, "metadata"):
            masked_doc.metadata = document.metadata

        return masked_doc

    def _process_batch(
        self,
        batch: list[Any],
        policy: MaskingPolicy,
        worker_id: int
    ) -> list[MaskingResult]:
        """Process a batch of documents.

        Args:
            batch: Batch of documents to process
            policy: Masking policy to apply
            worker_id: ID of the worker processing this batch

        Returns:
            List of MaskingResults
        """
        results = []

        # Get engines for this worker
        analyzer = self.analyzer_pool[worker_id % len(self.analyzer_pool)]
        anonymizer = self.anonymizer_pool[worker_id % len(self.anonymizer_pool)]

        for document in batch:
            result = self._process_single_document(
                document, policy, analyzer, anonymizer
            )
            results.append(result)

        return results

    def process_document_batch(
        self,
        documents: list[Any],
        policy: MaskingPolicy,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[MaskingResult]:
        """Process multiple documents efficiently with connection pooling.

        Args:
            documents: List of documents to process
            policy: Masking policy to apply
            progress_callback: Optional callback for progress updates

        Returns:
            List of MaskingResults
        """
        start_time = time.time()

        # Group documents into optimal batches
        batches = self._create_batches(documents)
        results = []

        with ThreadPoolExecutor(max_workers=self.parallel_workers) as executor:
            futures = []

            for i, batch in enumerate(batches):
                future = executor.submit(self._process_batch, batch, policy, i)
                futures.append((i, future))

            completed_batches = 0
            # Iterate through futures as they complete
            for future in as_completed([f for _, f in futures]):
                batch_results = future.result()
                results.extend(batch_results)
                completed_batches += 1

                if progress_callback:
                    progress_callback(completed_batches, len(batches))

        # Update statistics
        elapsed_ms = int((time.time() - start_time) * 1000)
        self._processing_stats["total_processed"] += len(documents)
        self._processing_stats["total_time_ms"] += elapsed_ms
        self._processing_stats["batch_count"] += len(batches)

        return results

    def get_statistics(self) -> dict[str, Any]:
        """Get processing statistics.

        Returns:
            Dictionary of processing statistics
        """
        stats = self._processing_stats.copy()

        if stats["total_processed"] > 0:
            stats["avg_time_per_doc_ms"] = stats["total_time_ms"] / stats["total_processed"]
        else:
            stats["avg_time_per_doc_ms"] = 0

        if stats["batch_count"] > 0:
            stats["avg_batch_size"] = stats["total_processed"] / stats["batch_count"]
        else:
            stats["avg_batch_size"] = 0

        return stats

    def reset_statistics(self) -> None:
        """Reset processing statistics."""
        self._processing_stats = {
            "total_processed": 0,
            "total_time_ms": 0,
            "batch_count": 0,
            "errors": []
        }

    # Public methods for testing
    def create_batches(self, documents: list[Any]) -> list[list[Any]]:
        """Public wrapper for _create_batches for testing.

        Args:
            documents: List of documents to batch

        Returns:
            List of document batches
        """
        return self._create_batches(documents)

    def extract_text_from_document(self, document: Any) -> str:
        """Public wrapper for _extract_text for testing.

        Args:
            document: Document to extract text from

        Returns:
            Extracted text
        """
        return self._extract_text(document)

    def policy_to_operators(self, policy: MaskingPolicy) -> dict[str, Any]:
        """Public wrapper for _policy_to_operators for testing.

        Args:
            policy: Masking policy

        Returns:
            Dictionary of operators
        """
        return self._policy_to_operators(policy)

    def create_cloakmap(
        self,
        document: Any,
        analyzer_results: list[AnalyzerRecognizerResult],
        anonymizer_result: Any
    ) -> CloakMap:
        """Public wrapper for _create_cloakmap for testing.

        Args:
            document: Original document
            analyzer_results: PII detection results
            anonymizer_result: Anonymization results

        Returns:
            CloakMap instance
        """
        return self._create_cloakmap(document, analyzer_results, anonymizer_result)

    def process_single_document(
        self,
        document: Any,
        policy: MaskingPolicy,
        analyzer: Optional[AnalyzerEngine] = None,
        anonymizer: Optional[AnonymizerEngine] = None
    ) -> MaskingResult:
        """Public wrapper for _process_single_document for testing.

        Args:
            document: Document to process
            policy: Masking policy to apply
            analyzer: Analyzer engine to use (optional)
            anonymizer: Anonymizer engine to use (optional)

        Returns:
            MaskingResult for the document
        """
        if analyzer is None:
            analyzer = self.analyzer_pool[0]
        if anonymizer is None:
            anonymizer = self.anonymizer_pool[0]
        return self._process_single_document(document, policy, analyzer, anonymizer)

    def process_parallel(
        self,
        documents: list[Any],
        policy: MaskingPolicy,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[MaskingResult]:
        """Process documents in parallel.

        Args:
            documents: List of documents to process
            policy: Masking policy to apply
            progress_callback: Optional callback for progress updates

        Returns:
            List of MaskingResults
        """
        return self.process_document_batch(documents, policy, progress_callback)

    def process_text_batch(
        self,
        texts: list[str],
        policy: MaskingPolicy,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> list[tuple[str, CloakMap]]:
        """Process multiple text strings efficiently.

        Args:
            texts: List of text strings to process
            policy: Masking policy to apply
            progress_callback: Optional callback for progress updates

        Returns:
            List of tuples (masked_text, cloakmap)
        """
        # Convert texts to simple documents
        documents = []
        for i, text in enumerate(texts):
            doc = type('Document', (), {})()
            doc.text = text
            doc.id = f"text_{i}"
            documents.append(doc)

        # Process as documents
        results = self.process_document_batch(documents, policy, progress_callback)

        # Extract text and cloakmap from results
        output = []
        for i, result in enumerate(results):
            if hasattr(result, 'stats') and result.stats and result.stats.get('success'):
                # Extract text from DoclingDocument or simple document
                if hasattr(result.masked_document, 'texts') and result.masked_document.texts:
                    masked_text = result.masked_document.texts[0].text
                else:
                    masked_text = result.masked_document.text
                output.append((masked_text, result.cloakmap))
            elif hasattr(result, 'masked_document'):
                # Extract masked text even if success is not defined
                if hasattr(result.masked_document, 'texts') and result.masked_document.texts:
                    masked_text = result.masked_document.texts[0].text
                else:
                    masked_text = getattr(result.masked_document, 'text', texts[i])
                output.append((masked_text, result.cloakmap))
            else:
                # Fallback for failed processing - create empty CloakMap
                empty_cloakmap = CloakMap(document_id=f"text_{i}")
                output.append((texts[i] if i < len(texts) else "", empty_cloakmap))

        return output

    def optimize_for_batch_size(self, estimated_document_count: int) -> int:
        """Optimize batch size based on estimated document count.

        Args:
            estimated_document_count: Estimated number of documents to process

        Returns:
            Optimized batch size
        """
        # Simple optimization heuristic
        if estimated_document_count < 100:
            return min(10, estimated_document_count)
        elif estimated_document_count < 1000:
            return 50
        elif estimated_document_count < 10000:
            return 100
        else:
            return 200
