"""Registry for managing custom recognizer plugins."""

import logging
from typing import Any, Dict, List, Optional

from .base import BaseRecognizerPlugin, RecognizerPluginResult
from ..exceptions import PluginError, PluginExecutionError

logger = logging.getLogger(__name__)


class RecognizerPluginRegistry:
    """
    Registry for managing custom Presidio recognizer plugins.
    
    This registry provides integration between recognizer plugins and
    the existing AnalyzerEngineWrapper system.
    """
    
    def __init__(self, main_registry: Optional[Any] = None) -> None:
        """
        Initialize the recognizer plugin registry.
        
        Args:
            main_registry: Reference to main plugin registry
        """
        self._main_registry = main_registry
        self._active_recognizers: Dict[str, BaseRecognizerPlugin] = {}
        
    def register_recognizer_plugin(
        self,
        plugin: BaseRecognizerPlugin,
        config: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register and initialize a recognizer plugin.
        
        Args:
            plugin: Recognizer plugin to register
            config: Optional configuration for the plugin
            
        Raises:
            PluginError: If registration fails
        """
        plugin_name = plugin.info.name
        
        try:
            # Initialize plugin with configuration
            if config:
                plugin.update_config(config)
            
            plugin.initialize()
            
            self._active_recognizers[plugin_name] = plugin
            
            logger.info(f"Recognizer plugin {plugin_name} registered and initialized")
            
        except Exception as e:
            logger.error(f"Failed to register recognizer plugin {plugin_name}: {e}")
            raise PluginError(
                f"Failed to register recognizer plugin {plugin_name}: {e}",
                plugin_name=plugin_name
            ) from e
    
    def analyze_text(
        self,
        text: str,
        language: str = "en",
        entity_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> List[RecognizerPluginResult]:
        """
        Analyze text using all active recognizer plugins.
        
        Args:
            text: Text to analyze
            language: Language code for analysis
            entity_types: Specific entity types to look for (None for all)
            context: Optional context information
            
        Returns:
            Combined list of recognition results from all plugins
        """
        if not text or not text.strip():
            return []
        
        all_results = []
        
        for plugin_name, plugin in self._active_recognizers.items():
            try:
                # Check if plugin supports the language
                if not plugin.supports_language(language):
                    logger.debug(f"Plugin {plugin_name} does not support language {language}")
                    continue
                
                # Check if plugin supports requested entity types
                if entity_types:
                    supported_types = plugin.get_supported_entity_types()
                    if not any(plugin.supports_entity_type(et) for et in entity_types):
                        logger.debug(f"Plugin {plugin_name} does not support requested entity types")
                        continue
                
                # Run analysis
                results = plugin.analyze_text_safe(text, language, context)
                
                # Filter by requested entity types if specified
                if entity_types:
                    results = [
                        result for result in results
                        if result.entity_type in entity_types
                    ]
                
                all_results.extend(results)
                
                logger.debug(f"Plugin {plugin_name} found {len(results)} entities")
                
            except Exception as e:
                logger.error(f"Error in recognizer plugin {plugin_name}: {e}")
                # Continue with other plugins
                continue
        
        # Sort results by start position for consistency
        all_results.sort(key=lambda r: (r.start, r.end, -r.confidence))
        
        logger.info(f"Total entities found by recognizer plugins: {len(all_results)}")
        return all_results
    
    def analyze_with_plugin(
        self,
        plugin_name: str,
        text: str,
        language: str = "en",
        context: Optional[Dict[str, Any]] = None
    ) -> List[RecognizerPluginResult]:
        """
        Analyze text using a specific recognizer plugin.
        
        Args:
            plugin_name: Name of recognizer plugin to use
            text: Text to analyze
            language: Language code
            context: Optional context information
            
        Returns:
            List of recognition results
            
        Raises:
            PluginError: If plugin not found or execution fails
        """
        if plugin_name not in self._active_recognizers:
            raise PluginError(
                f"Recognizer plugin {plugin_name} not found or not active",
                plugin_name=plugin_name
            )
        
        plugin = self._active_recognizers[plugin_name]
        
        try:
            return plugin.analyze_text_safe(text, language, context)
            
        except Exception as e:
            logger.error(f"Recognizer plugin {plugin_name} execution failed: {e}")
            raise PluginExecutionError(
                f"Recognizer plugin {plugin_name} execution failed: {e}",
                plugin_name=plugin_name,
                original_exception=e
            ) from e
    
    def get_active_recognizer_plugins(self) -> Dict[str, BaseRecognizerPlugin]:
        """Get all active recognizer plugins."""
        return self._active_recognizers.copy()
    
    def get_recognizer_plugin(self, plugin_name: str) -> Optional[BaseRecognizerPlugin]:
        """Get a specific recognizer plugin by name."""
        return self._active_recognizers.get(plugin_name)
    
    def list_recognizer_plugins(self) -> List[str]:
        """Get list of active recognizer plugin names."""
        return list(self._active_recognizers.keys())
    
    def get_supported_entity_types(self) -> List[str]:
        """
        Get all entity types supported by active recognizer plugins.
        
        Returns:
            List of unique entity types across all plugins
        """
        entity_types = set()
        
        for plugin in self._active_recognizers.values():
            supported_types = plugin.get_supported_entity_types()
            entity_types.update(supported_types)
        
        return sorted(list(entity_types))
    
    def get_supported_languages(self) -> List[str]:
        """
        Get all languages supported by active recognizer plugins.
        
        Returns:
            List of unique language codes across all plugins
        """
        languages = set()
        
        for plugin in self._active_recognizers.values():
            supported_languages = plugin.get_supported_languages()
            languages.update(supported_languages)
        
        return sorted(list(languages))
    
    def supports_entity_type(self, entity_type: str) -> bool:
        """
        Check if any active plugin supports the given entity type.
        
        Args:
            entity_type: Entity type to check
            
        Returns:
            True if at least one plugin supports the entity type
        """
        return any(
            plugin.supports_entity_type(entity_type)
            for plugin in self._active_recognizers.values()
        )
    
    def supports_language(self, language: str) -> bool:
        """
        Check if any active plugin supports the given language.
        
        Args:
            language: Language code to check
            
        Returns:
            True if at least one plugin supports the language
        """
        return any(
            plugin.supports_language(language)
            for plugin in self._active_recognizers.values()
        )
    
    def get_plugins_for_entity_type(self, entity_type: str) -> List[str]:
        """
        Get list of plugin names that support a given entity type.
        
        Args:
            entity_type: Entity type to check
            
        Returns:
            List of plugin names that support the entity type
        """
        return [
            name
            for name, plugin in self._active_recognizers.items()
            if plugin.supports_entity_type(entity_type)
        ]
    
    def get_plugins_for_language(self, language: str) -> List[str]:
        """
        Get list of plugin names that support a given language.
        
        Args:
            language: Language code to check
            
        Returns:
            List of plugin names that support the language
        """
        return [
            name
            for name, plugin in self._active_recognizers.items()
            if plugin.supports_language(language)
        ]
    
    def cleanup_recognizer_plugin(self, plugin_name: str) -> None:
        """
        Clean up a recognizer plugin.
        
        Args:
            plugin_name: Name of plugin to clean up
        """
        if plugin_name in self._active_recognizers:
            plugin = self._active_recognizers[plugin_name]
            try:
                plugin.cleanup()
                del self._active_recognizers[plugin_name]
                logger.info(f"Recognizer plugin {plugin_name} cleaned up")
            except Exception as e:
                logger.error(f"Error cleaning up recognizer plugin {plugin_name}: {e}")
    
    def cleanup_all_recognizers(self) -> None:
        """Clean up all recognizer plugins."""
        for plugin_name in list(self._active_recognizers.keys()):
            self.cleanup_recognizer_plugin(plugin_name)
    
    def get_recognizer_registry_status(self) -> Dict[str, Any]:
        """Get status of recognizer plugin registry."""
        plugin_status = {}
        
        for name, plugin in self._active_recognizers.items():
            plugin_status[name] = {
                "name": plugin.info.name,
                "version": plugin.info.version,
                "initialized": plugin.is_initialized,
                "supported_entities": plugin.get_supported_entity_types(),
                "supported_languages": plugin.get_supported_languages(),
            }
        
        return {
            "active_recognizers": len(self._active_recognizers),
            "supported_entity_types": self.get_supported_entity_types(),
            "supported_languages": self.get_supported_languages(),
            "plugins": plugin_status,
        }