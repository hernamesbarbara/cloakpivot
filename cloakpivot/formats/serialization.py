"""Enhanced serialization module for CloakPivot with masking-aware formatting."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

from docling_core.types import DoclingDocument

from .registry import FormatRegistry, SupportedFormat

logger = logging.getLogger(__name__)


class SerializationError(Exception):
    """Exception raised during serialization operations."""
    
    def __init__(self, message: str, format_name: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.format_name = format_name
        self.context = context or {}


@dataclass
class SerializationResult:
    """Result of a serialization operation."""
    
    content: str
    format_name: str
    size_bytes: int
    metadata: Dict[str, Any]
    
    @property
    def size_kb(self) -> float:
        """Get size in kilobytes."""
        return self.size_bytes / 1024
    
    def save_to_file(self, file_path: Union[str, Path]) -> None:
        """Save serialized content to file.
        
        Args:
            file_path: Path where to save the content
        """
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.content)
        
        logger.info(f"Serialized content saved to {path} ({self.size_kb:.1f} KB)")


class CloakPivotSerializer:
    """Enhanced serializer with CloakPivot-specific features.
    
    This class wraps docpivot's SerializerProvider with additional functionality
    for masking-aware serialization, format detection, and enhanced error handling.
    """
    
    def __init__(self):
        """Initialize the CloakPivot serializer."""
        self._registry = FormatRegistry()
        
        logger.debug("CloakPivotSerializer initialized")
    
    @property
    def supported_formats(self) -> list[str]:
        """Get list of supported formats."""
        return self._registry.list_supported_formats()
    
    def serialize_document(
        self,
        document: DoclingDocument,
        format_name: str,
        **kwargs: Any
    ) -> SerializationResult:
        """Serialize a document to the specified format.
        
        Args:
            document: DoclingDocument to serialize
            format_name: Target format name
            **kwargs: Additional serialization parameters
            
        Returns:
            SerializationResult containing the serialized content
            
        Raises:
            SerializationError: If serialization fails
            
        Examples:
            >>> serializer = CloakPivotSerializer()
            >>> result = serializer.serialize_document(doc, "markdown")
            >>> print(f"Markdown content: {result.content}")
        """
        if not self._registry.is_format_supported(format_name):
            raise SerializationError(
                f"Unsupported format: {format_name}",
                format_name=format_name,
                context={"supported_formats": self.supported_formats}
            )
        
        try:
            # Get the appropriate serializer from docpivot
            serializer = self._registry.get_serializer(format_name, document)
            
            # Perform serialization
            logger.debug(f"Serializing document to {format_name} format")
            serialized_content = serializer.serialize(document)
            
            # Handle different return types from docpivot serializers
            if hasattr(serialized_content, 'text'):
                content = serialized_content.text
                metadata = getattr(serialized_content, 'metadata', {})
            else:
                content = str(serialized_content)
                metadata = {}
            
            # Apply format-specific post-processing
            content = self._apply_format_specific_processing(content, format_name)
            
            # Create result
            result = SerializationResult(
                content=content,
                format_name=format_name,
                size_bytes=len(content.encode('utf-8')),
                metadata={
                    "document_name": document.name,
                    "document_texts": len(document.texts),
                    "document_tables": len(document.tables),
                    "serializer_metadata": metadata,
                    **kwargs
                }
            )
            
            logger.info(f"Successfully serialized document to {format_name} ({result.size_kb:.1f} KB)")
            return result
            
        except Exception as e:
            logger.error(f"Serialization failed for format {format_name}: {e}")
            raise SerializationError(
                f"Failed to serialize document to {format_name}: {e}",
                format_name=format_name,
                context={"document_name": document.name, "error": str(e)}
            ) from e
    
    def _apply_format_specific_processing(self, content: str, format_name: str) -> str:
        """Apply format-specific post-processing to serialized content.
        
        Args:
            content: Raw serialized content
            format_name: Target format name
            
        Returns:
            Post-processed content
        """
        try:
            fmt = SupportedFormat.from_string(format_name)
        except ValueError:
            return content
        
        if fmt in (SupportedFormat.MARKDOWN, SupportedFormat.MD):
            return self._process_markdown_content(content)
        elif fmt == SupportedFormat.HTML:
            return self._process_html_content(content)
        
        return content
    
    def _process_markdown_content(self, content: str) -> str:
        """Apply markdown-specific processing for masked content.
        
        Args:
            content: Raw markdown content
            
        Returns:
            Processed markdown content
        """
        # Ensure masked tokens are properly escaped for markdown
        # Common replacement tokens that might need escaping
        replacements = [
            ("***", "\\*\\*\\*"),  # Escape asterisks that might interfere with markdown
            ("___", "\\_\\_\\_"),  # Escape underscores
            ("###", "\\#\\#\\#"),  # Escape hash symbols
        ]
        
        processed = content
        for original, escaped in replacements:
            # Only escape if it appears to be a replacement token (e.g., surrounded by brackets or standalone)
            if f"[{original}]" in processed:
                processed = processed.replace(f"[{original}]", f"[{escaped}]")
        
        return processed
    
    def _process_html_content(self, content: str) -> str:
        """Apply HTML-specific processing for masked content.
        
        Args:
            content: Raw HTML content
            
        Returns:
            Processed HTML content with masking styles
        """
        # Add CSS classes for masked content styling
        # This is a simple implementation - could be more sophisticated
        masked_patterns = [
            ("[REDACTED]", '<span class="cloak-redacted">[REDACTED]</span>'),
            ("[MASKED]", '<span class="cloak-masked">[MASKED]</span>'),
            ("[***]", '<span class="cloak-replaced">[***]</span>'),
        ]
        
        processed = content
        for pattern, replacement in masked_patterns:
            processed = processed.replace(pattern, replacement)
        
        # Add CSS styles if not already present
        if '<style>' not in processed and '<span class="cloak-' in processed:
            css_styles = """
<style>
.cloak-redacted { 
    background-color: #f0f0f0; 
    color: #666; 
    font-family: monospace; 
    border: 1px solid #ccc; 
    padding: 0 4px; 
    border-radius: 3px; 
}
.cloak-masked { 
    background-color: #fff3cd; 
    color: #856404; 
    font-family: monospace; 
    border: 1px solid #ffeaa7; 
    padding: 0 4px; 
    border-radius: 3px; 
}
.cloak-replaced { 
    background-color: #d4edda; 
    color: #155724; 
    font-family: monospace; 
    border: 1px solid #c3e6cb; 
    padding: 0 4px; 
    border-radius: 3px; 
}
</style>
"""
            # Insert CSS in the head if present, otherwise at the beginning
            if '<head>' in processed:
                processed = processed.replace('<head>', f'<head>{css_styles}')
            else:
                processed = css_styles + processed
        
        return processed
    
    def convert_format(
        self,
        input_path: Union[str, Path],
        output_format: str,
        output_path: Optional[Union[str, Path]] = None,
        **kwargs: Any
    ) -> SerializationResult:
        """Convert a document from one format to another.
        
        Args:
            input_path: Path to input document
            output_format: Target format name
            output_path: Optional output path (auto-generated if not provided)
            **kwargs: Additional conversion parameters
            
        Returns:
            SerializationResult with converted content
            
        Raises:
            SerializationError: If conversion fails
        """
        from ..document.processor import DocumentProcessor
        
        input_path = Path(input_path)
        
        # Detect input format
        input_format = self._registry.detect_format_from_path(input_path)
        if input_format:
            logger.info(f"Detected input format: {input_format.value}")
        else:
            logger.warning(f"Could not detect input format for {input_path}")
        
        # Validate format compatibility
        if input_format and not self._registry.validate_format_compatibility(
            input_format.value, output_format
        ):
            raise SerializationError(
                f"Conversion from {input_format.value} to {output_format} is not supported",
                format_name=output_format,
                context={"input_format": input_format.value}
            )
        
        try:
            # Load document
            processor = DocumentProcessor()
            document = processor.load_document(input_path)
            
            # Serialize to target format
            result = self.serialize_document(document, output_format, **kwargs)
            
            # Save to output file if path provided
            if output_path:
                result.save_to_file(output_path)
            else:
                # Auto-generate output path
                suggested_ext = self._registry.suggest_output_extension(output_format)
                auto_output_path = input_path.with_suffix(f".converted{suggested_ext}")
                result.save_to_file(auto_output_path)
                
                # Update metadata with actual output path
                result.metadata["output_path"] = str(auto_output_path)
            
            return result
            
        except Exception as e:
            raise SerializationError(
                f"Format conversion failed: {e}",
                format_name=output_format,
                context={
                    "input_path": str(input_path),
                    "output_path": str(output_path) if output_path else None
                }
            ) from e
    
    def detect_format(self, file_path: Union[str, Path]) -> Optional[str]:
        """Detect the format of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Detected format name or None
        """
        detected = self._registry.detect_format_from_path(file_path)
        return detected.value if detected else None
    
    def get_format_info(self, format_name: str) -> Dict[str, Any]:
        """Get information about a format.
        
        Args:
            format_name: Name of the format
            
        Returns:
            Dictionary with format information
        """
        if not self._registry.is_format_supported(format_name):
            return {"supported": False, "error": "Format not supported"}
        
        return {
            "supported": True,
            "extensions": list(self._registry.get_format_extensions(format_name)),
            "suggested_extension": self._registry.suggest_output_extension(format_name),
            "is_text_format": format_name in ["markdown", "md", "html"],
            "is_json_format": format_name in ["lexical", "docling"],
        }