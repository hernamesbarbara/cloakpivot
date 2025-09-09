"""Migration utilities for legacy to Presidio-based masking transition."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from ..core.cloakmap import CloakMap

logger = logging.getLogger(__name__)


class CloakMapMigrator:
    """Utilities for migrating legacy CloakMaps to Presidio format."""

    def migrate_cloakmap(self, legacy_path: Path,
                          output_path: Optional[Path] = None) -> Path:
        """Migrate single CloakMap from v1.0 to v2.0.
        
        Args:
            legacy_path: Path to the legacy CloakMap file
            output_path: Optional output path for migrated CloakMap
            
        Returns:
            Path to the migrated CloakMap file
            
        Raises:
            FileNotFoundError: If the legacy CloakMap file doesn't exist
            ValueError: If migration fails
        """
        if not legacy_path.exists():
            raise FileNotFoundError(f"CloakMap file not found: {legacy_path}")
            
        # Load legacy CloakMap
        try:
            legacy_cloakmap = CloakMap.load_from_file(legacy_path)
        except Exception as e:
            raise ValueError(f"Failed to load CloakMap from {legacy_path}: {e}") from e
        
        if legacy_cloakmap.version == "2.0":
            logger.info(f"CloakMap {legacy_path} already at v2.0")
            return legacy_path
        
        # Enhance with Presidio metadata (where possible)
        enhanced_cloakmap = self._enhance_with_presidio_metadata(legacy_cloakmap)
        
        # Save enhanced version
        output_path = output_path or legacy_path
        try:
            enhanced_cloakmap.save_to_file(output_path)
        except Exception as e:
            raise ValueError(f"Failed to save migrated CloakMap to {output_path}: {e}") from e
        
        logger.info(f"Migrated {legacy_path} → {output_path}")
        return output_path
    
    def bulk_migrate(self, directory: Path, pattern: str = "*.cloakmap") -> Dict[str, Any]:
        """Migrate all CloakMaps in a directory.
        
        Args:
            directory: Directory containing CloakMap files
            pattern: Glob pattern to match CloakMap files
            
        Returns:
            Dictionary with migration results
        """
        results: Dict[str, List[Dict[str, str]]] = {
            "migrated": [],
            "skipped": [],
            "errors": []
        }
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        cloakmap_files = list(directory.glob(pattern))
        logger.info(f"Found {len(cloakmap_files)} CloakMaps to process")
        
        for cloakmap_file in cloakmap_files:
            try:
                # Check if already v2.0
                cloakmap = CloakMap.load_from_file(cloakmap_file)
                if cloakmap.version == "2.0":
                    logger.info(f"Skipping {cloakmap_file} - already v2.0")
                    results["skipped"].append({
                        "file": str(cloakmap_file),
                        "reason": "Already v2.0"
                    })
                    continue
                
                # Migrate
                output_path = self.migrate_cloakmap(cloakmap_file)
                results["migrated"].append({
                    "source": str(cloakmap_file),
                    "target": str(output_path)
                })
            except Exception as e:
                logger.error(f"Failed to migrate {cloakmap_file}: {e}")
                results["errors"].append({
                    "file": str(cloakmap_file),
                    "error": str(e)
                })
        
        return results
    
    def _enhance_with_presidio_metadata(self, legacy_cloakmap: CloakMap) -> CloakMap:
        """Add Presidio metadata to legacy CloakMap where possible.
        
        Args:
            legacy_cloakmap: The v1.0 CloakMap to enhance
            
        Returns:
            Enhanced CloakMap with v2.0 format
        """
        # Attempt to reconstruct Presidio metadata from anchors
        presidio_metadata = {
            "engine_version": "2.2.0",
            "engine_used": "legacy",  # Store engine_used in presidio_metadata
            "operator_results": [],
            "reversible_operators": [],
            "migration_source": "legacy_v1.0"
        }
        
        # Analyze anchors to infer operator types
        for anchor in legacy_cloakmap.anchors:
            operator_result = self._infer_operator_result(anchor)
            if operator_result:
                presidio_metadata["operator_results"].append(operator_result)
        
        # Create enhanced CloakMap directly with all fields
        return CloakMap(
            version="2.0",
            doc_id=legacy_cloakmap.doc_id,
            doc_hash=legacy_cloakmap.doc_hash,
            anchors=legacy_cloakmap.anchors,
            policy_snapshot=legacy_cloakmap.policy_snapshot if hasattr(legacy_cloakmap, 'policy_snapshot') else {},
            metadata={
                **legacy_cloakmap.metadata,
                "migrated_from": "v1.0",
                "migration_timestamp": legacy_cloakmap.created_at.isoformat() if hasattr(legacy_cloakmap, 'created_at') and legacy_cloakmap.created_at else "",
                "upgraded_from_v1": True
            },
            presidio_metadata=presidio_metadata,
            created_at=legacy_cloakmap.created_at if hasattr(legacy_cloakmap, 'created_at') else None
        )
    
    def _infer_operator_result(self, anchor: Any) -> Optional[Dict[str, Any]]:
        """Infer Presidio operator result from anchor entry.
        
        Args:
            anchor: AnchorEntry to analyze
            
        Returns:
            Operator result dictionary or None if cannot infer
        """
        # Map strategy names to Presidio operators
        strategy_to_operator = {
            "redact": "redact",
            "template": "replace",
            "hash": "hash",
            "surrogate": "replace",
            "encrypt": "encrypt",
            "mask": "mask"
        }
        
        operator = strategy_to_operator.get(anchor.strategy_used, "custom")
        
        return {
            "entity_type": anchor.entity_type,
            "start": anchor.start,
            "end": anchor.end,
            "operator": operator,
            "new_value": anchor.masked_value,
            "text": anchor.original_text if hasattr(anchor, 'original_text') else None
        }


class StrategyMigrator:
    """Migration utilities for strategy configurations."""
    
    def migrate_policy_file(self, policy_path: Path) -> Path:
        """Migrate policy file to use Presidio-optimized strategies.
        
        Args:
            policy_path: Path to the existing policy file
            
        Returns:
            Path to the migrated policy file
            
        Raises:
            FileNotFoundError: If policy file doesn't exist
            ValueError: If migration fails
        """
        if not policy_path.exists():
            raise FileNotFoundError(f"Policy file not found: {policy_path}")
        
        # Load existing policy
        try:
            with open(policy_path) as f:
                policy_data = yaml.safe_load(f)
        except Exception as e:
            raise ValueError(f"Failed to load policy from {policy_path}: {e}") from e
        
        # Migrate strategies to Presidio-optimized versions
        if "strategies" in policy_data:
            for entity_type, strategy_config in policy_data["strategies"].items():
                policy_data["strategies"][entity_type] = self._migrate_strategy_config(
                    strategy_config
                )
        
        # Add Presidio-specific options
        policy_data["presidio"] = {
            "enabled": True,
            "fallback_to_legacy": True,
            "operator_optimizations": True,
            "engine_version": "2.2.0"
        }
        
        # Add migration metadata
        policy_data["migration"] = {
            "migrated_from": str(policy_path),
            "migration_version": "2.0"
        }
        
        # Save migrated policy
        output_path = policy_path.with_suffix('.presidio.yml')
        try:
            with open(output_path, 'w') as f:
                yaml.dump(policy_data, f, default_flow_style=False, sort_keys=False)
        except Exception as e:
            raise ValueError(f"Failed to save migrated policy to {output_path}: {e}") from e
        
        logger.info(f"Migrated policy {policy_path} → {output_path}")
        return output_path
    
    def _migrate_strategy_config(self, strategy_config: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate individual strategy to Presidio-optimized version.
        
        Args:
            strategy_config: Strategy configuration dictionary
            
        Returns:
            Migrated strategy configuration
        """
        if not isinstance(strategy_config, dict):
            return strategy_config
            
        strategy_type = strategy_config.get("kind", "redact")
        
        # Create a copy to avoid modifying original
        migrated_config = strategy_config.copy()
        
        # Add Presidio-specific optimizations
        if strategy_type == "hash":
            # Use Presidio's optimized hash operator
            migrated_config["presidio_optimized"] = True
            migrated_config.setdefault("algorithm", "sha256")
            migrated_config["operator"] = "hash"
        
        elif strategy_type == "surrogate":
            # Use Presidio's faker integration
            migrated_config["use_presidio_faker"] = True
            migrated_config["operator"] = "replace"
            migrated_config["faker_provider"] = "name"  # Default provider
        
        elif strategy_type == "template":
            # Map to Presidio replace operator
            migrated_config["operator"] = "replace"
            migrated_config["presidio_optimized"] = True
        
        elif strategy_type == "redact":
            # Map to Presidio redact operator
            migrated_config["operator"] = "redact"
            migrated_config["presidio_optimized"] = True
        
        elif strategy_type == "encrypt":
            # Use Presidio's encryption operator
            migrated_config["operator"] = "encrypt"
            migrated_config["reversible"] = True
            migrated_config.setdefault("key_id", "default")
        
        return migrated_config
    
    def bulk_migrate_policies(self, directory: Path, pattern: str = "*.yml") -> Dict[str, Any]:
        """Migrate all policy files in a directory.
        
        Args:
            directory: Directory containing policy files
            pattern: Glob pattern to match policy files
            
        Returns:
            Dictionary with migration results
        """
        results: Dict[str, List[Dict[str, str]]] = {
            "migrated": [],
            "skipped": [],
            "errors": []
        }
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        policy_files = list(directory.glob(pattern))
        
        # Skip already migrated files
        policy_files = [f for f in policy_files if not f.name.endswith('.presidio.yml')]
        
        logger.info(f"Found {len(policy_files)} policy files to process")
        
        for policy_file in policy_files:
            try:
                # Check if already has Presidio configuration
                with open(policy_file) as f:
                    data = yaml.safe_load(f)
                    if "presidio" in data:
                        logger.info(f"Skipping {policy_file} - already has Presidio config")
                        results["skipped"].append({
                            "file": str(policy_file),
                            "reason": "Already has Presidio configuration"
                        })
                        continue
                
                # Migrate
                output_path = self.migrate_policy_file(policy_file)
                results["migrated"].append({
                    "source": str(policy_file),
                    "target": str(output_path)
                })
            except Exception as e:
                logger.error(f"Failed to migrate {policy_file}: {e}")
                results["errors"].append({
                    "file": str(policy_file),
                    "error": str(e)
                })
        
        return results