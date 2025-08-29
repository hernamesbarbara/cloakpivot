"""Security utilities for CloakMap cryptographic operations."""

import base64
import hashlib
import hmac
import os
import secrets
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

# Constants for cryptographic operations
DEFAULT_HMAC_ALGORITHM = 'sha256'
DEFAULT_PBKDF2_ITERATIONS = 100000
DEFAULT_SALT_LENGTH = 32
SUPPORTED_HMAC_ALGORITHMS = {'sha256', 'sha512', 'sha384'}


@dataclass(frozen=True)
class SecurityConfig:
    """Configuration for security operations."""
    
    hmac_algorithm: str = DEFAULT_HMAC_ALGORITHM
    pbkdf2_iterations: int = DEFAULT_PBKDF2_ITERATIONS
    salt_length: int = DEFAULT_SALT_LENGTH
    key_derivation_enabled: bool = True
    constant_time_verification: bool = True
    
    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.hmac_algorithm not in SUPPORTED_HMAC_ALGORITHMS:
            raise ValueError(f"Unsupported HMAC algorithm: {self.hmac_algorithm}")
        
        if self.pbkdf2_iterations < 10000:
            raise ValueError("PBKDF2 iterations must be at least 10,000 for security")
        
        if self.salt_length < 16:
            raise ValueError("Salt length must be at least 16 bytes")


class KeyManager(ABC):
    """Abstract base class for key management implementations."""
    
    @abstractmethod
    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """
        Retrieve a key by ID and optional version.
        
        Args:
            key_id: Unique identifier for the key
            version: Optional key version (defaults to latest)
            
        Returns:
            The raw key bytes
            
        Raises:
            KeyError: If key is not found
            ValueError: If key format is invalid
        """
        pass
    
    @abstractmethod
    def list_keys(self) -> list[str]:
        """
        List all available key IDs.
        
        Returns:
            List of key IDs
        """
        pass
    
    def derive_key(self, base_key: bytes, salt: bytes, info: str, 
                   config: SecurityConfig) -> bytes:
        """
        Derive a key using PBKDF2.
        
        Args:
            base_key: Base key material
            salt: Salt for key derivation
            info: Context information for derivation
            config: Security configuration
            
        Returns:
            Derived key bytes (32 bytes)
        """
        if not config.key_derivation_enabled:
            return base_key
        
        # Use PBKDF2 for key derivation
        derived_key = hashlib.pbkdf2_hmac(
            config.hmac_algorithm,
            base_key + info.encode('utf-8'),
            salt,
            config.pbkdf2_iterations,
            dklen=32  # 256-bit key
        )
        
        return derived_key


class EnvironmentKeyManager(KeyManager):
    """Key manager that reads keys from environment variables."""
    
    def __init__(self, prefix: str = "CLOAKPIVOT_KEY_"):
        """
        Initialize environment key manager.
        
        Args:
            prefix: Prefix for environment variable names
        """
        self.prefix = prefix
    
    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Get key from environment variable."""
        env_name = f"{self.prefix}{key_id.upper()}"
        if version:
            env_name += f"_V{version}"
        
        key_value = os.environ.get(env_name)
        if not key_value:
            raise KeyError(f"Key not found in environment: {env_name}")
        
        try:
            # Support both hex and base64 encoded keys
            if key_value.startswith('hex:'):
                return bytes.fromhex(key_value[4:])
            elif key_value.startswith('b64:'):
                return base64.b64decode(key_value[4:])
            else:
                # Default to UTF-8 encoding
                return key_value.encode('utf-8')
        except Exception as e:
            raise ValueError(f"Invalid key format in {env_name}: {e}") from e
    
    def list_keys(self) -> list[str]:
        """List all keys with the configured prefix."""
        keys = []
        for env_name in os.environ:
            if env_name.startswith(self.prefix):
                # Extract key_id from environment name
                key_part = env_name[len(self.prefix):]
                # Remove version suffix if present
                if '_V' in key_part:
                    key_part = key_part.split('_V')[0]
                keys.append(key_part.lower())
        
        return sorted(set(keys))


class FileKeyManager(KeyManager):
    """Key manager that reads keys from files."""
    
    def __init__(self, key_directory: Union[str, Path]):
        """
        Initialize file key manager.
        
        Args:
            key_directory: Directory containing key files
        """
        self.key_directory = Path(key_directory)
        if not self.key_directory.exists():
            raise ValueError(f"Key directory does not exist: {key_directory}")
    
    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Get key from file."""
        filename = f"{key_id}"
        if version:
            filename += f".v{version}"
        filename += ".key"
        
        key_path = self.key_directory / filename
        if not key_path.exists():
            raise KeyError(f"Key file not found: {key_path}")
        
        try:
            key_content = key_path.read_text().strip()
            
            # Support various key formats
            if key_content.startswith('hex:'):
                return bytes.fromhex(key_content[4:])
            elif key_content.startswith('b64:'):
                return base64.b64decode(key_content[4:])
            else:
                return key_content.encode('utf-8')
        except Exception as e:
            raise ValueError(f"Failed to read key from {key_path}: {e}") from e
    
    def list_keys(self) -> list[str]:
        """List all key files in the directory."""
        keys = []
        for key_file in self.key_directory.glob("*.key"):
            key_name = key_file.stem
            # Remove version suffix if present
            if '.v' in key_name:
                key_name = key_name.split('.v')[0]
            keys.append(key_name)
        
        return sorted(set(keys))


class CompositeKeyManager(KeyManager):
    """Key manager that tries multiple sources in order."""
    
    def __init__(self, managers: list[KeyManager]):
        """
        Initialize composite key manager.
        
        Args:
            managers: List of key managers to try in order
        """
        if not managers:
            raise ValueError("At least one key manager is required")
        
        self.managers = managers
    
    def get_key(self, key_id: str, version: Optional[str] = None) -> bytes:
        """Try to get key from managers in order."""
        errors = []
        
        for manager in self.managers:
            try:
                return manager.get_key(key_id, version)
            except (KeyError, ValueError) as e:
                errors.append(f"{manager.__class__.__name__}: {e}")
                continue
        
        raise KeyError(f"Key '{key_id}' not found in any manager: {'; '.join(errors)}")
    
    def list_keys(self) -> list[str]:
        """List keys from all managers."""
        all_keys = set()
        for manager in self.managers:
            try:
                all_keys.update(manager.list_keys())
            except Exception:
                continue  # Skip managers that fail
        
        return sorted(all_keys)


class CryptoUtils:
    """Utility class for cryptographic operations."""
    
    @staticmethod
    def generate_salt(length: int = DEFAULT_SALT_LENGTH) -> bytes:
        """Generate a cryptographically secure random salt."""
        return secrets.token_bytes(length)
    
    @staticmethod
    def compute_salted_checksum(data: str, salt: bytes, 
                              config: SecurityConfig) -> str:
        """
        Compute a salted checksum using PBKDF2.
        
        Args:
            data: Data to checksum
            salt: Salt for the checksum
            config: Security configuration
            
        Returns:
            Hex-encoded checksum
        """
        data_bytes = data.encode('utf-8')
        
        checksum = hashlib.pbkdf2_hmac(
            config.hmac_algorithm,
            data_bytes,
            salt,
            config.pbkdf2_iterations,
            dklen=32
        )
        
        return checksum.hex()
    
    @staticmethod
    def verify_salted_checksum(data: str, salt: bytes, expected_checksum: str,
                             config: SecurityConfig) -> bool:
        """
        Verify a salted checksum.
        
        Args:
            data: Original data
            salt: Salt used for checksum
            expected_checksum: Expected checksum (hex)
            config: Security configuration
            
        Returns:
            True if checksum matches
        """
        computed_checksum = CryptoUtils.compute_salted_checksum(data, salt, config)
        
        if config.constant_time_verification:
            return hmac.compare_digest(computed_checksum, expected_checksum)
        else:
            return computed_checksum == expected_checksum
    
    @staticmethod
    def compute_hmac(data: bytes, key: bytes, algorithm: str = DEFAULT_HMAC_ALGORITHM) -> str:
        """
        Compute HMAC for data.
        
        Args:
            data: Data to sign
            key: HMAC key
            algorithm: Hash algorithm to use
            
        Returns:
            Hex-encoded HMAC
        """
        if algorithm not in SUPPORTED_HMAC_ALGORITHMS:
            raise ValueError(f"Unsupported HMAC algorithm: {algorithm}")
        
        hash_func = getattr(hashlib, algorithm)
        mac = hmac.new(key, data, hash_func)
        return mac.hexdigest()
    
    @staticmethod
    def verify_hmac(data: bytes, key: bytes, expected_mac: str,
                   algorithm: str = DEFAULT_HMAC_ALGORITHM,
                   constant_time: bool = True) -> bool:
        """
        Verify HMAC signature.
        
        Args:
            data: Original data
            key: HMAC key
            expected_mac: Expected MAC (hex)
            algorithm: Hash algorithm used
            constant_time: Use constant-time comparison
            
        Returns:
            True if MAC is valid
        """
        computed_mac = CryptoUtils.compute_hmac(data, key, algorithm)
        
        if constant_time:
            return hmac.compare_digest(computed_mac, expected_mac)
        else:
            return computed_mac == expected_mac


@dataclass(frozen=True)
class SecurityMetadata:
    """Metadata for security operations."""
    
    algorithm: str
    key_id: str
    key_version: Optional[str] = None
    salt: Optional[bytes] = None
    iterations: Optional[int] = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "algorithm": self.algorithm,
            "key_id": self.key_id,
        }
        
        if self.key_version:
            result["key_version"] = self.key_version
        
        if self.salt:
            result["salt"] = base64.b64encode(self.salt).decode('ascii')
        
        if self.iterations:
            result["iterations"] = self.iterations
        
        return result
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SecurityMetadata":
        """Create from dictionary."""
        salt = None
        if data.get("salt"):
            salt = base64.b64decode(data["salt"])
        
        return cls(
            algorithm=data["algorithm"],
            key_id=data["key_id"],
            key_version=data.get("key_version"),
            salt=salt,
            iterations=data.get("iterations")
        )


class SecurityValidator:
    """Comprehensive security validator for CloakMaps."""
    
    def __init__(self, config: Optional[SecurityConfig] = None,
                 key_manager: Optional[KeyManager] = None):
        """
        Initialize security validator.
        
        Args:
            config: Security configuration
            key_manager: Key manager for signature verification
        """
        self.config = config or SecurityConfig()
        self.key_manager = key_manager or create_default_key_manager()
    
    def validate_cloakmap(self, cloakmap: "CloakMap") -> dict[str, Any]:
        """
        Perform comprehensive security validation of a CloakMap.
        
        Args:
            cloakmap: CloakMap to validate
            
        Returns:
            Detailed validation results
        """
        from .cloakmap import CloakMap  # Import here to avoid circular import
        
        results = {
            "valid": True,
            "security_level": "none",
            "errors": [],
            "warnings": [],
            "checks": {
                "structure": False,
                "anchors": False,
                "signature": False,
                "encryption": False,
                "tampering": False,
                "key_availability": False
            },
            "performance": {
                "validation_time_ms": 0,
                "anchor_count": len(cloakmap.anchors),
                "crypto_operations": 0
            }
        }
        
        start_time = self._get_time_ms()
        
        try:
            # Structure validation
            self._validate_structure(cloakmap, results)
            
            # Anchor validation with security checks
            self._validate_anchors(cloakmap, results)
            
            # Signature validation
            self._validate_signature(cloakmap, results)
            
            # Encryption validation
            self._validate_encryption(cloakmap, results)
            
            # Tampering detection
            self._detect_tampering(cloakmap, results)
            
            # Key availability check
            self._check_key_availability(cloakmap, results)
            
            # Determine overall security level
            self._assess_security_level(results)
            
        except Exception as e:
            results["valid"] = False
            results["errors"].append(f"Validation failed: {e}")
        
        results["performance"]["validation_time_ms"] = max(1, int(self._get_time_ms() - start_time))
        
        return results
    
    def _validate_structure(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate CloakMap structure."""
        try:
            # Check version compatibility
            version_parts = cloakmap.version.split('.')
            if len(version_parts) < 2:
                results["errors"].append("Invalid version format")
                return
            
            # Check required fields
            if not cloakmap.doc_id:
                results["errors"].append("Missing document ID")
            
            if not cloakmap.doc_hash:
                results["warnings"].append("Missing document hash")
            
            results["checks"]["structure"] = True
            
        except Exception as e:
            results["errors"].append(f"Structure validation error: {e}")
    
    def _validate_anchors(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate anchors with security considerations."""
        try:
            from .anchors import AnchorIndex
            
            # Basic anchor validation
            anchor_index = AnchorIndex(cloakmap.anchors)
            
            # Check for overlapping anchors (security risk)
            overlaps_found = False
            for i, anchor in enumerate(cloakmap.anchors):
                overlapping = anchor_index.find_overlapping_anchors(anchor)
                if overlapping:
                    overlaps_found = True
                    results["errors"].append(
                        f"Overlapping anchors detected: {anchor.replacement_id}"
                    )
            
            # Check checksum salt presence (for new format)
            missing_salts = 0
            for anchor in cloakmap.anchors:
                if not hasattr(anchor, 'checksum_salt') or not anchor.checksum_salt:
                    missing_salts += 1
            
            if missing_salts > 0:
                results["warnings"].append(
                    f"{missing_salts} anchors missing checksum salts (legacy format)"
                )
            
            if not overlaps_found:
                results["checks"]["anchors"] = True
                
        except Exception as e:
            results["errors"].append(f"Anchor validation error: {e}")
    
    def _validate_signature(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate CloakMap signature."""
        if not cloakmap.is_signed:
            results["warnings"].append("CloakMap is not signed")
            results["checks"]["signature"] = True  # Not an error if not signed
            return
        
        try:
            # Try to verify signature with available keys
            key_id = cloakmap.crypto.get('key_id', 'default') if cloakmap.crypto else 'default'
            
            try:
                signing_key = self.key_manager.get_key(key_id)
                is_valid = cloakmap.verify_signature(self.key_manager, config=self.config)
                
                if is_valid:
                    results["checks"]["signature"] = True
                    results["performance"]["crypto_operations"] += 1
                else:
                    results["errors"].append("Signature verification failed")
                    
            except (KeyError, ValueError) as e:
                results["errors"].append(f"Signature key not available: {e}")
                
        except Exception as e:
            results["errors"].append(f"Signature validation error: {e}")
    
    def _validate_encryption(self, cloakmap: "CloakMap", results: dict) -> None:
        """Validate encryption metadata."""
        if not cloakmap.is_encrypted:
            results["warnings"].append("CloakMap is not encrypted")
            results["checks"]["encryption"] = True  # Not required
            return
        
        try:
            crypto = cloakmap.crypto
            if not crypto or not crypto.get("algorithm"):
                results["errors"].append("Invalid encryption metadata")
                return
            
            # Check for supported algorithms
            algorithm = crypto.get("algorithm", "")
            if "AES" not in algorithm and "ChaCha20" not in algorithm:
                results["warnings"].append(f"Unusual encryption algorithm: {algorithm}")
            
            results["checks"]["encryption"] = True
            
        except Exception as e:
            results["errors"].append(f"Encryption validation error: {e}")
    
    def _detect_tampering(self, cloakmap: "CloakMap", results: dict) -> None:
        """Detect signs of tampering."""
        try:
            # Check for timestamp anomalies
            if cloakmap.created_at:
                from datetime import datetime, timezone
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                
                # Check for future timestamps (suspicious)
                if cloakmap.created_at > now:
                    results["warnings"].append("CloakMap timestamp is in the future")
            
            # Check anchor timestamp consistency
            if cloakmap.anchors and cloakmap.created_at:
                for anchor in cloakmap.anchors:
                    if anchor.timestamp and anchor.timestamp > cloakmap.created_at:
                        results["warnings"].append(
                            f"Anchor {anchor.replacement_id} timestamp is after CloakMap creation"
                        )
            
            # Check for suspicious patterns in replacement IDs
            replacement_ids = [a.replacement_id for a in cloakmap.anchors]
            if len(set(replacement_ids)) != len(replacement_ids):
                results["errors"].append("Duplicate replacement IDs detected")
                return
            
            results["checks"]["tampering"] = True
            
        except Exception as e:
            results["errors"].append(f"Tampering detection error: {e}")
    
    def _check_key_availability(self, cloakmap: "CloakMap", results: dict) -> None:
        """Check if required keys are available."""
        try:
            required_keys = set()
            
            # Check signature key
            if cloakmap.is_signed and cloakmap.crypto:
                key_id = cloakmap.crypto.get('key_id', 'default')
                required_keys.add(key_id)
            
            # Check encryption key
            if cloakmap.is_encrypted and cloakmap.crypto:
                key_id = cloakmap.crypto.get('encryption_key_id')
                if key_id:
                    required_keys.add(key_id)
            
            # Verify key availability
            available_keys = set(self.key_manager.list_keys())
            missing_keys = required_keys - available_keys
            
            if missing_keys:
                results["warnings"].append(f"Missing keys: {', '.join(missing_keys)}")
            else:
                results["checks"]["key_availability"] = True
                
        except Exception as e:
            results["warnings"].append(f"Key availability check failed: {e}")
    
    def _assess_security_level(self, results: dict) -> None:
        """Assess overall security level."""
        if results["errors"]:
            results["security_level"] = "compromised"
        elif results["checks"]["signature"] and results["checks"]["encryption"]:
            results["security_level"] = "high"
        elif results["checks"]["signature"] or results["checks"]["encryption"]:
            results["security_level"] = "medium"
        else:
            results["security_level"] = "low"
    
    def _get_time_ms(self) -> float:
        """Get current time in milliseconds using high precision counter."""
        import time
        return time.perf_counter() * 1000


def create_default_key_manager() -> KeyManager:
    """
    Create a default key manager with common sources.
    
    Returns:
        CompositeKeyManager with environment and file sources
    """
    managers = [EnvironmentKeyManager()]
    
    # Try to add file manager if key directory exists
    possible_key_dirs = [
        Path.home() / ".cloakpivot" / "keys",
        Path("/etc/cloakpivot/keys"),
        Path("./keys")
    ]
    
    for key_dir in possible_key_dirs:
        if key_dir.exists() and key_dir.is_dir():
            managers.append(FileKeyManager(key_dir))
            break
    
    return CompositeKeyManager(managers)