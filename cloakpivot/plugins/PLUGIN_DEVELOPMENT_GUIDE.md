# CloakPivot Plugin Development Guide

This guide explains how to create custom plugins for CloakPivot to extend its masking and recognition capabilities.

## Overview

CloakPivot supports two types of plugins:

1. **Strategy Plugins**: Custom masking strategies that transform detected PII
2. **Recognizer Plugins**: Custom PII detection logic that identifies entities in text

## Plugin Discovery

CloakPivot uses Python entry points for plugin discovery. Plugins can be:

- **Built-in**: Included with CloakPivot installation
- **External**: Installed as separate Python packages
- **Local**: Developed and registered locally

## Strategy Plugin Development

### Basic Strategy Plugin

```python
from cloakpivot.plugins.strategies.base import BaseStrategyPlugin, StrategyPluginResult
from cloakpivot.plugins.base import PluginInfo
from typing import Any, Dict, Optional

class MyStrategyPlugin(BaseStrategyPlugin):
    """Custom strategy plugin example."""
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="my_strategy",
            version="1.0.0",
            description="My custom masking strategy",
            author="Your Name",
            plugin_type="strategy",
            metadata={
                "reversible": False,
                "deterministic": True,
                "preserves_length": False
            }
        )
    
    def apply_strategy(
        self,
        original_text: str,
        entity_type: str,
        confidence: float,
        context: Optional[Dict[str, Any]] = None
    ) -> StrategyPluginResult:
        """Apply custom masking logic."""
        try:
            # Your custom masking logic here
            masked_text = f"CUSTOM_{len(original_text)}"
            
            return StrategyPluginResult(
                masked_text=masked_text,
                execution_time_ms=0.0,  # Will be calculated by framework
                metadata={
                    "original_length": len(original_text),
                    "strategy_type": "custom"
                }
            )
            
        except Exception as e:
            return StrategyPluginResult(
                masked_text=original_text,
                execution_time_ms=0.0,
                metadata={"error": str(e)},
                success=False,
                error_message=str(e)
            )
```

### Advanced Strategy Plugin Features

#### Configuration Support

```python
def _validate_strategy_config(self, config: Dict[str, Any]) -> bool:
    """Validate plugin configuration."""
    required_keys = ["format", "prefix"]
    for key in required_keys:
        if key not in config:
            raise PluginValidationError(f"Missing required config: {key}")
    return True

def apply_strategy(self, original_text: str, entity_type: str, confidence: float, context=None):
    format_type = self.get_config_value("format", "default")
    prefix = self.get_config_value("prefix", "MASKED")
    
    # Use configuration in masking logic
    masked_text = f"{prefix}_{format_type}_{len(original_text)}"
    # ...
```

#### Entity Type Restrictions

```python
def get_supported_entity_types(self) -> Optional[List[str]]:
    """Restrict to specific entity types."""
    return ["PHONE_NUMBER", "EMAIL_ADDRESS"]
```

#### Parameter Schema

```python
def get_strategy_parameters_schema(self) -> Dict[str, Any]:
    """Define JSON schema for parameters."""
    return {
        "type": "object",
        "properties": {
            "format": {
                "type": "string",
                "enum": ["short", "long", "custom"],
                "description": "Output format type"
            },
            "prefix": {
                "type": "string",
                "description": "Prefix for masked values"
            }
        },
        "required": ["format"]
    }
```

## Recognizer Plugin Development

### Basic Recognizer Plugin

```python
from cloakpivot.plugins.recognizers.base import BaseRecognizerPlugin, RecognizerPluginResult
from cloakpivot.plugins.base import PluginInfo
from typing import Any, Dict, List, Optional
import re

class MyRecognizerPlugin(BaseRecognizerPlugin):
    """Custom recognizer plugin example."""
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="my_recognizer",
            version="1.0.0",
            description="My custom PII recognizer",
            author="Your Name",
            plugin_type="recognizer"
        )
    
    def analyze_text(
        self,
        text: str,
        language: str = "en",
        context: Optional[Dict[str, Any]] = None
    ) -> List[RecognizerPluginResult]:
        """Analyze text for custom PII entities."""
        results = []
        
        # Example: Find pattern like "ID-12345"
        pattern = r'\bID-\d{5}\b'
        
        for match in re.finditer(pattern, text):
            results.append(RecognizerPluginResult(
                entity_type="CUSTOM_ID",
                start=match.start(),
                end=match.end(),
                confidence=0.9,
                text=match.group(),
                metadata={
                    "pattern": "ID-NNNNN",
                    "recognizer": "my_recognizer"
                }
            ))
        
        return results
```

### Pattern-Based Recognizer Plugin

For simpler pattern-based recognition, extend `PatternBasedRecognizerPlugin`:

```python
from cloakpivot.plugins.recognizers.base import PatternBasedRecognizerPlugin
import re

class PatternRecognizerPlugin(PatternBasedRecognizerPlugin):
    """Pattern-based recognizer example."""
    
    @property
    def info(self) -> PluginInfo:
        return PluginInfo(
            name="pattern_recognizer",
            version="1.0.0",
            description="Pattern-based PII recognizer",
            author="Your Name",
            plugin_type="recognizer"
        )
    
    def _initialize_recognizer(self) -> None:
        """Set up patterns."""
        default_config = {
            "entity_types": ["EMPLOYEE_ID"],
            "supported_languages": ["en"],
            "min_confidence": 0.8,
            "patterns": {
                "EMPLOYEE_ID": [
                    r'\bEMP-\d{4,6}\b',
                    r'\b[A-Z]{2}\d{4}\b'
                ]
            }
        }
        
        # Merge with user config
        for key, value in default_config.items():
            if key not in self.config:
                self.config[key] = value
        
        super()._initialize_recognizer()
```

## Plugin Configuration

### In Code

```python
from cloakpivot.plugins.policy_extensions import (
    PluginConfiguration, 
    EnhancedMaskingPolicy
)

# Configure strategy plugin
strategy_config = PluginConfiguration(
    plugin_name="my_strategy",
    plugin_type="strategy",
    config={
        "format": "short",
        "prefix": "MASKED"
    },
    enabled=True,
    priority=5
)

# Configure recognizer plugin
recognizer_config = PluginConfiguration(
    plugin_name="my_recognizer", 
    plugin_type="recognizer",
    config={
        "min_confidence": 0.8
    },
    enabled=True
)

# Create enhanced policy
policy = EnhancedMaskingPolicy(
    plugin_configurations={
        "my_strategy": strategy_config,
        "my_recognizer": recognizer_config
    },
    plugin_strategy_mapping={
        "PHONE_NUMBER": "my_strategy"
    },
    enabled_recognizer_plugins=["my_recognizer"]
)
```

### YAML Configuration

```yaml
# policy.yaml
plugin_configurations:
  my_strategy:
    plugin_name: my_strategy
    plugin_type: strategy
    config:
      format: short
      prefix: MASKED
    enabled: true
    priority: 5
    
  my_recognizer:
    plugin_name: my_recognizer
    plugin_type: recognizer
    config:
      min_confidence: 0.8
    enabled: true

plugin_strategy_mapping:
  PHONE_NUMBER: my_strategy
  EMAIL_ADDRESS: my_strategy

enabled_recognizer_plugins:
  - my_recognizer
```

## Plugin Distribution

### Entry Points Setup

To distribute plugins as packages, add entry points to `pyproject.toml`:

```toml
[project.entry-points."cloakpivot.plugins.strategies"]
my_strategy = "my_package.plugins:MyStrategyPlugin"

[project.entry-points."cloakpivot.plugins.recognizers"]  
my_recognizer = "my_package.plugins:MyRecognizerPlugin"
```

### Package Structure

```
my_cloakpivot_plugins/
├── pyproject.toml
├── src/
│   └── my_cloakpivot_plugins/
│       ├── __init__.py
│       ├── strategies.py
│       └── recognizers.py
└── tests/
    ├── test_strategies.py
    └── test_recognizers.py
```

## Best Practices

### Error Handling

```python
def apply_strategy(self, original_text, entity_type, confidence, context=None):
    try:
        # Your logic here
        result = self._perform_masking(original_text)
        
        return StrategyPluginResult(
            masked_text=result,
            execution_time_ms=0.0,
            metadata={"success": True}
        )
        
    except Exception as e:
        self.logger.error(f"Strategy failed: {e}")
        
        # Return failure result
        return StrategyPluginResult(
            masked_text=original_text,  # Fallback to original
            execution_time_ms=0.0,
            metadata={"error": str(e)},
            success=False,
            error_message=str(e)
        )
```

### Logging

```python
import logging

class MyPlugin(BaseStrategyPlugin):
    def __init__(self, config=None):
        super().__init__(config)
        # Logger is automatically set up as self.logger
    
    def apply_strategy(self, original_text, entity_type, confidence, context=None):
        self.logger.debug(f"Processing {entity_type} with confidence {confidence}")
        # ...
```

### Testing

```python
import pytest
from my_cloakpivot_plugins import MyStrategyPlugin

class TestMyStrategyPlugin:
    def test_basic_functionality(self):
        plugin = MyStrategyPlugin()
        plugin.initialize()
        
        result = plugin.apply_strategy_safe("test", "TEST", 0.9)
        
        assert result.success
        assert result.masked_text != "test"
    
    def test_configuration(self):
        config = {"format": "long", "prefix": "CUSTOM"}
        plugin = MyStrategyPlugin(config)
        plugin.initialize()
        
        result = plugin.apply_strategy_safe("test", "TEST", 0.9)
        assert "CUSTOM" in result.masked_text
```

### Performance Considerations

1. **Lazy Initialization**: Initialize expensive resources only when needed
2. **Caching**: Cache compiled patterns or lookup tables
3. **Batch Processing**: Support batch operations when possible
4. **Memory Management**: Clean up resources in the `cleanup()` method

### Security Considerations

1. **Input Validation**: Always validate input parameters
2. **Error Information**: Don't leak sensitive information in error messages
3. **Resource Limits**: Implement timeouts and resource limits
4. **Safe Defaults**: Use secure defaults for configuration

## Plugin Registry Usage

### Manual Registration

```python
from cloakpivot.plugins import get_plugin_registry

registry = get_plugin_registry()

# Register plugin manually
plugin = MyStrategyPlugin({"format": "short"})
registry.register_plugin(plugin)
registry.initialize_plugin("my_strategy")

# Use in masking
from cloakpivot.plugins.strategy_applicator import PluginAwareStrategyApplicator

applicator = PluginAwareStrategyApplicator(strategy_registry=registry.strategy_registry)
```

### Programmatic Discovery

```python
from cloakpivot.plugins import get_plugin_registry

registry = get_plugin_registry()
registry.discover_plugins()  # Find plugins via entry points

# List available plugins
strategy_plugins = registry.get_strategy_plugins()
recognizer_plugins = registry.get_recognizer_plugins()
```

## Examples Repository

See the `cloakpivot.plugins.examples` package for complete working examples:

- `ROT13StrategyPlugin`: Simple reversible text transformation
- `ColorCodeStrategyPlugin`: Creative color-based masking
- `CustomPhoneRecognizerPlugin`: Enhanced phone number detection
- `LicensePlateRecognizerPlugin`: Custom entity type recognition

## Troubleshooting

### Plugin Not Found

- Check entry points configuration
- Verify plugin class inheritance
- Ensure package is installed in same environment

### Validation Errors

- Review plugin configuration schema
- Check required parameters
- Validate configuration data types

### Runtime Errors

- Check plugin logs for details
- Verify plugin initialization
- Test with simple inputs first

For more information, see the API documentation and example implementations.