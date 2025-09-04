"""CloakMap enhancer for Presidio metadata integration."""

import logging
from typing import Any, List, Optional

from .cloakmap import CloakMap

logger = logging.getLogger(__name__)


class CloakMapEnhancer:
    """Enhances CloakMap with Presidio metadata for advanced reversibility.
    
    This class provides functionality to add, extract, and manage Presidio
    operator results within CloakMap instances, enabling perfect reversibility
    for Presidio-based anonymization operations.
    
    The enhancer maintains backward compatibility with v1.0 CloakMaps while
    enabling v2.0 features when Presidio metadata is available.
    
    Examples:
        >>> enhancer = CloakMapEnhancer()
        >>> 
        >>> # Add Presidio metadata to existing CloakMap
        >>> operator_results = [
        ...     {
        ...         "entity_type": "PHONE_NUMBER",
        ...         "start": 10, "end": 22,
        ...         "operator": "encrypt",
        ...         "encrypted_value": "...",
        ...         "key_reference": "key_123"
        ...     }
        ... ]
        >>> enhanced_map = enhancer.add_presidio_metadata(
        ...     cloakmap, operator_results, engine_version="2.2.x"
        ... )
        >>>
        >>> # Extract operator results for deanonymization
        >>> results = enhancer.extract_operator_results(enhanced_map)
        >>> print(len(results))  # 1
    """

    def __init__(self) -> None:
        """Initialize the CloakMapEnhancer."""
        pass

    def add_presidio_metadata(
        self,
        cloakmap: CloakMap,
        operator_results: List[dict[str, Any]],
        engine_version: Optional[str] = None,
        reversible_operators: Optional[List[str]] = None,
        batch_id: Optional[str] = None,
    ) -> CloakMap:
        """Add Presidio operator results to CloakMap, creating v2.0 format.
        
        Args:
            cloakmap: Existing CloakMap to enhance
            operator_results: List of Presidio operator result dictionaries
            engine_version: Presidio engine version (optional)
            reversible_operators: List of reversible operator names (optional)
            batch_id: Optional batch tracking identifier
            
        Returns:
            New CloakMap v2.0 with Presidio metadata
            
        Raises:
            ValueError: If operator_results is invalid or empty
        """
        if not operator_results:
            raise ValueError("operator_results cannot be empty")
            
        if not isinstance(operator_results, list):
            raise ValueError("operator_results must be a list")
            
        # Validate each operator result
        for i, result in enumerate(operator_results):
            if not isinstance(result, dict):
                raise ValueError(f"operator_result[{i}] must be a dictionary")
                
            # Check required fields
            required_fields = ["entity_type", "start", "end", "operator"]
            for field in required_fields:
                if field not in result:
                    raise ValueError(
                        f"operator_result[{i}] missing required field: {field}"
                    )
        
        # Auto-detect reversible operators if not provided
        if reversible_operators is None:
            reversible_operators = self._detect_reversible_operators(operator_results)
        
        # Build presidio metadata
        presidio_metadata = {
            "operator_results": operator_results,
            "reversible_operators": reversible_operators,
        }
        
        if engine_version:
            presidio_metadata["engine_version"] = engine_version
            
        if batch_id:
            presidio_metadata["batch_id"] = batch_id
            
        # Create new CloakMap v2.0 with presidio metadata, preserving all original data
        return CloakMap(
            version="2.0",
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            policy_snapshot=cloakmap.policy_snapshot,
            crypto=cloakmap.crypto,
            signature=cloakmap.signature,
            created_at=cloakmap.created_at,  # Preserve original timestamp
            metadata=cloakmap.metadata,
            presidio_metadata=presidio_metadata,
        )

    def extract_operator_results(self, cloakmap: CloakMap) -> List[dict[str, Any]]:
        """Extract Presidio operator results from CloakMap for deanonymization.
        
        Args:
            cloakmap: CloakMap with Presidio metadata
            
        Returns:
            List of operator result dictionaries for deanonymization
            
        Raises:
            ValueError: If CloakMap doesn't contain Presidio metadata
        """
        if not self.is_presidio_enabled(cloakmap):
            raise ValueError(
                "CloakMap does not contain Presidio metadata. "
                "Use add_presidio_metadata() first or check CloakMap version."
            )
            
        operator_results = cloakmap.presidio_metadata.get("operator_results", [])
        
        if not operator_results:
            logger.warning("CloakMap has presidio_metadata but no operator_results")
            
        return operator_results

    def is_presidio_enabled(self, cloakmap: CloakMap) -> bool:
        """Check if CloakMap contains Presidio metadata.
        
        Args:
            cloakmap: CloakMap to check
            
        Returns:
            True if Presidio metadata is present, False otherwise
        """
        return cloakmap.is_presidio_enabled

    def get_reversible_operators(self, cloakmap: CloakMap) -> List[str]:
        """Get list of reversible operators from CloakMap.
        
        Args:
            cloakmap: CloakMap with Presidio metadata
            
        Returns:
            List of reversible operator names
            
        Raises:
            ValueError: If CloakMap doesn't contain Presidio metadata
        """
        if not self.is_presidio_enabled(cloakmap):
            raise ValueError("CloakMap does not contain Presidio metadata")
            
        return cloakmap.presidio_metadata.get("reversible_operators", [])

    def get_engine_version(self, cloakmap: CloakMap) -> Optional[str]:
        """Get Presidio engine version from CloakMap.
        
        Args:
            cloakmap: CloakMap with Presidio metadata
            
        Returns:
            Engine version string or None if not available
            
        Raises:
            ValueError: If CloakMap doesn't contain Presidio metadata
        """
        if not self.is_presidio_enabled(cloakmap):
            raise ValueError("CloakMap does not contain Presidio metadata")
            
        return cloakmap.presidio_metadata.get("engine_version")

    def get_batch_id(self, cloakmap: CloakMap) -> Optional[str]:
        """Get batch ID from CloakMap.
        
        Args:
            cloakmap: CloakMap with Presidio metadata
            
        Returns:
            Batch ID string or None if not available
            
        Raises:
            ValueError: If CloakMap doesn't contain Presidio metadata
        """
        if not self.is_presidio_enabled(cloakmap):
            raise ValueError("CloakMap does not contain Presidio metadata")
            
        return cloakmap.presidio_metadata.get("batch_id")

    def migrate_to_v2(
        self, 
        cloakmap: CloakMap,
        operator_results: List[dict[str, Any]],
        engine_version: Optional[str] = None,
        **kwargs
    ) -> CloakMap:
        """Migrate v1.0 CloakMap to v2.0 with Presidio metadata.
        
        This is an alias for add_presidio_metadata() with clearer intent
        for migration scenarios.
        
        Args:
            cloakmap: v1.0 CloakMap to migrate
            operator_results: Presidio operator results to add
            engine_version: Presidio engine version
            **kwargs: Additional arguments for add_presidio_metadata()
            
        Returns:
            New CloakMap v2.0 with Presidio metadata
        """
        return self.add_presidio_metadata(
            cloakmap, operator_results, engine_version, **kwargs
        )

    def update_presidio_metadata(
        self,
        cloakmap: CloakMap,
        operator_results: Optional[List[dict[str, Any]]] = None,
        engine_version: Optional[str] = None,
        reversible_operators: Optional[List[str]] = None,
        batch_id: Optional[str] = None,
    ) -> CloakMap:
        """Update existing Presidio metadata in CloakMap.
        
        Args:
            cloakmap: CloakMap with existing Presidio metadata
            operator_results: New operator results (optional)
            engine_version: New engine version (optional)
            reversible_operators: New reversible operators list (optional) 
            batch_id: New batch ID (optional)
            
        Returns:
            New CloakMap with updated Presidio metadata
            
        Raises:
            ValueError: If CloakMap doesn't have existing Presidio metadata
        """
        if not self.is_presidio_enabled(cloakmap):
            raise ValueError(
                "CloakMap does not contain Presidio metadata to update. "
                "Use add_presidio_metadata() instead."
            )
        
        # Start with existing metadata
        current_metadata = cloakmap.presidio_metadata.copy()
        
        # Update fields if provided
        if operator_results is not None:
            # Validate operator results
            for i, result in enumerate(operator_results):
                if not isinstance(result, dict):
                    raise ValueError(f"operator_result[{i}] must be a dictionary")
                    
                required_fields = ["entity_type", "start", "end", "operator"]
                for field in required_fields:
                    if field not in result:
                        raise ValueError(
                            f"operator_result[{i}] missing required field: {field}"
                        )
            
            current_metadata["operator_results"] = operator_results
            
        if engine_version is not None:
            current_metadata["engine_version"] = engine_version
            
        if reversible_operators is not None:
            current_metadata["reversible_operators"] = reversible_operators
            
        if batch_id is not None:
            current_metadata["batch_id"] = batch_id
            
        # Create new CloakMap with updated metadata
        return CloakMap(
            version="2.0",
            doc_id=cloakmap.doc_id,
            doc_hash=cloakmap.doc_hash,
            anchors=cloakmap.anchors,
            policy_snapshot=cloakmap.policy_snapshot,
            crypto=cloakmap.crypto,
            signature=cloakmap.signature,
            created_at=cloakmap.created_at,
            metadata=cloakmap.metadata,
            presidio_metadata=current_metadata,
        )

    def _detect_reversible_operators(
        self, operator_results: List[dict[str, Any]]
    ) -> List[str]:
        """Detect which operators are reversible from operator results.
        
        Args:
            operator_results: List of operator result dictionaries
            
        Returns:
            List of unique reversible operator names
        """
        # Define known reversible operators
        reversible_operator_types = {
            "encrypt", 
            "custom",  # Custom operators may be reversible
        }
        
        # Extract unique operators that are reversible
        reversible_found = set()
        for result in operator_results:
            operator = result.get("operator")
            if operator in reversible_operator_types:
                reversible_found.add(operator)
                
        return list(reversible_found)

    def get_statistics(self, cloakmap: CloakMap) -> dict[str, Any]:
        """Get statistics about Presidio metadata in CloakMap.
        
        Args:
            cloakmap: CloakMap to analyze
            
        Returns:
            Dictionary with Presidio metadata statistics
        """
        if not self.is_presidio_enabled(cloakmap):
            return {
                "presidio_enabled": False,
                "version": cloakmap.version,
            }
            
        operator_results = self.extract_operator_results(cloakmap)
        reversible_operators = self.get_reversible_operators(cloakmap)
        
        # Count operators by type
        operator_counts = {}
        for result in operator_results:
            operator = result.get("operator", "unknown")
            operator_counts[operator] = operator_counts.get(operator, 0) + 1
            
        return {
            "presidio_enabled": True,
            "version": cloakmap.version,
            "engine_version": self.get_engine_version(cloakmap),
            "batch_id": self.get_batch_id(cloakmap),
            "total_operator_results": len(operator_results),
            "operator_counts": operator_counts,
            "reversible_operators": reversible_operators,
            "reversible_count": len(reversible_operators),
        }