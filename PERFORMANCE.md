# CloakPivot Performance Optimization Guide

<!-- 
Performance documentation for CloakPivot data masking library.
This guide provides comprehensive instructions for optimizing performance across development,
testing, and production environments. It documents the existing performance infrastructure
including singleton patterns, caching strategies, parallel processing, and profiling tools.

Maintenance Notes:
- Update performance metrics when upgrading dependencies
- Validate code examples when APIs change
- Review configuration defaults quarterly
- Update troubleshooting guide based on common developer issues
-->

[![Performance CI](https://github.com/{GITHUB_ORG}/cloakpivot/workflows/Performance/badge.svg)](https://github.com/{GITHUB_ORG}/cloakpivot/actions)

<!-- Note: Replace {GITHUB_ORG} with your actual GitHub organization name -->

CloakPivot is designed for optimal performance with large documents and batch processing operations. This guide provides comprehensive guidance on performance optimization, monitoring, and troubleshooting using the existing performance infrastructure built into the system.

## Table of Contents

- [Quick Start](#quick-start)
- [Performance Architecture](#performance-architecture)
- [Configuration Options](#configuration-options)
- [Best Practices](#best-practices)
- [Profiling and Monitoring](#profiling-and-monitoring)
- [Troubleshooting](#troubleshooting)
- [Advanced Optimization](#advanced-optimization)
- [Performance Testing](#performance-testing)

## Quick Start

### Environment Configuration for Optimal Performance

```bash
# Development: Fast startup, acceptable accuracy
export MODEL_SIZE=small
export CLOAKPIVOT_USE_SINGLETON=true
export ANALYZER_CACHE_SIZE=8

# Production: Balanced performance and accuracy
export MODEL_SIZE=medium
export ANALYZER_CACHE_SIZE=16
export MAX_WORKERS=4

# High-accuracy: Best results, higher resource usage
export MODEL_SIZE=large
export ANALYZER_CACHE_SIZE=32
export MAX_WORKERS=2  # Fewer workers due to memory usage
```

### Quick Performance Wins

1. **Enable Singleton Pattern**: Set `CLOAKPIVOT_USE_SINGLETON=true` (default)
2. **Use Session Fixtures**: In tests, use session-scoped fixtures
3. **Configure Model Size**: Choose appropriate `MODEL_SIZE` for your needs
4. **Enable Parallel Processing**: Use `ENABLE_PARALLEL=true` for batch operations

### 5-Minute Performance Setup

```python
from cloakpivot import mask_document
from cloakpivot.loaders import get_presidio_analyzer, get_detection_pipeline
from cloakpivot.core.performance import get_profiler

# Enable performance monitoring
profiler = get_profiler()

# Use singleton loaders for efficiency
analyzer = get_presidio_analyzer(language="en")
pipeline = get_detection_pipeline()

# Profile your operations
with profiler.measure_operation("document_masking"):
    result = mask_document("document.pdf", policy="balanced.yaml")

# Get performance insights
stats = profiler.get_operation_stats("document_masking")
print(f"Average masking time: {stats.average_duration_ms:.1f}ms")
```

## Performance Architecture

### Singleton Pattern

CloakPivot uses thread-safe singleton loaders to minimize expensive resource initialization. The singleton behavior is controlled by the `CLOAKPIVOT_USE_SINGLETON` environment variable (default: `true`).

```python
from cloakpivot.loaders import get_presidio_analyzer, get_detection_pipeline

# Efficient: Uses cached instances with LRU caching
analyzer = get_presidio_analyzer(language="en")
pipeline = get_detection_pipeline()

# Less efficient: Creates new instances every time
from cloakpivot.core.analyzer import AnalyzerEngineWrapper
analyzer = AnalyzerEngineWrapper()  # Avoid in production
```

**Benefits of Singleton Pattern:**
- **75-90% reduction** in analyzer initialization time
- **Thread-safe** LRU caching with configurable size
- **Memory efficiency** through instance reuse
- **Zero configuration** - works out of the box

### Caching Strategy

CloakPivot implements multiple levels of caching:

```python
# 1. Analyzer Caching: Thread-safe LRU cache
# Configurable size via ANALYZER_CACHE_SIZE (default: 8)
analyzer = get_presidio_analyzer(language="en")  # Cached by language + config

# 2. Model Caching: spaCy models cached after first load
# Models persist for the lifetime of the process

# 3. Pipeline Caching: Detection pipelines cached by configuration
pipeline = get_detection_pipeline()  # Cached by analyzer + policy hash
```

### Parallel Processing

CloakPivot includes a sophisticated parallel processing engine for batch operations:

```python
from cloakpivot.core.parallel_analysis import ParallelAnalysisEngine

# Automatic worker count based on CPU cores and memory
engine = ParallelAnalysisEngine(max_workers=None)  # Auto-detected

# Process documents in parallel
results = engine.analyze_documents(
    documents=large_document_list,
    policy=policy,
    chunk_size=1000  # Optimal chunk size
)

print(f"Processed {len(results.entities)} entities using {results.threads_used} threads")
```

## Configuration Options

### Model Size Performance Impact

| Model Size | Memory Usage | Load Time | Accuracy | Use Case |
|------------|-------------|-----------|----------|----------|
| `small`    | ~15MB       | ~800ms    | Good     | Development, CI |
| `medium`   | ~50MB       | ~1.5s     | Better   | Testing, balanced prod |
| `large`    | ~150MB      | ~3s       | Best     | Production, high accuracy |

### Environment Variables Reference

| Variable | Default | Description | Performance Impact |
|----------|---------|-------------|-------------------|
| `MODEL_SIZE` | `small` | spaCy model size | Load time, memory, accuracy |
| `CLOAKPIVOT_USE_SINGLETON` | `true` | Enable singleton pattern | **75-90% initialization time savings** |
| `ANALYZER_CACHE_SIZE` | `8` | LRU cache size for analyzers | Memory vs. cache hit rate |
| `ENABLE_PARALLEL` | `true` | Enable parallel processing | CPU utilization |
| `MAX_WORKERS` | `auto` | Worker thread limit | Concurrency vs. memory |
| `PYTEST_FAST_MODE` | `true` | Fast test mode | Test execution speed |
| `PYTEST_WORKERS` | `auto` | Pytest parallel workers | Test parallelization |

### Performance Configuration Examples

```python
# Development configuration
from cloakpivot.core.config import performance_config

print(f"Singleton enabled: {performance_config.use_singleton_analyzers}")
print(f"Cache size: {performance_config.analyzer_cache_size}")
print(f"Parallel enabled: {performance_config.enable_parallel}")

# Runtime configuration with validation
import os

def set_performance_config(cache_size=16, max_workers=6):
    """Set performance configuration with validation."""
    try:
        # Validate cache_size
        if not isinstance(cache_size, int) or cache_size < 1:
            raise ValueError(f"Invalid cache_size: {cache_size}. Must be positive integer.")
        
        # Validate max_workers
        if not isinstance(max_workers, int) or max_workers < 1:
            raise ValueError(f"Invalid max_workers: {max_workers}. Must be positive integer.")
        
        os.environ['ANALYZER_CACHE_SIZE'] = str(cache_size)
        os.environ['MAX_WORKERS'] = str(max_workers)
        
        print(f"Performance configuration set: cache_size={cache_size}, max_workers={max_workers}")
    except Exception as e:
        print(f"Error setting performance configuration: {e}")
        raise

# Example usage
set_performance_config(cache_size=16, max_workers=6)
```

## Best Practices

### Application Development

#### 1. Use Singleton Loaders

```python
# ✅ Recommended: Use singleton loaders
from cloakpivot.loaders import get_presidio_analyzer, get_detection_pipeline

def process_documents(documents, policy):
    # Get cached instances - this is fast after first call
    analyzer = get_presidio_analyzer(language="en")
    pipeline = get_detection_pipeline()
    
    results = []
    for doc in documents:
        result = pipeline.detect_entities(doc, analyzer)
        results.append(result)
    
    return results

# ❌ Avoid: Direct instantiation in loops or frequently called code
def process_documents_inefficient(documents):
    results = []
    for doc in documents:
        # This creates a new analyzer for EVERY document!
        analyzer = AnalyzerEngineWrapper()  # Expensive: ~1-3 seconds each time
        result = analyzer.analyze(doc.text)
        results.append(result)
    
    return results
```

**Performance Impact:** Using singleton loaders reduces initialization time from ~1-3 seconds per call to ~1-10 milliseconds.

#### 2. Optimize Configuration

```python
# ✅ Recommended: Configure once, reuse everywhere
from cloakpivot.core.analyzer import AnalyzerConfig
from cloakpivot.loaders import get_presidio_analyzer_from_config

# Create config once
config = AnalyzerConfig(
    language="en",
    min_confidence=0.8,
    enabled_recognizers=["PERSON", "EMAIL_ADDRESS", "PHONE_NUMBER"],
    nlp_engine_name="spacy"
)

# Reuse cached analyzer
analyzer = get_presidio_analyzer_from_config(config)

# ❌ Avoid: Complex configurations in hot paths
def process_text_inefficient(text):
    # Don't create complex configs in frequently called functions
    config = AnalyzerConfig(
        language="en",
        custom_recognizers={...}  # Complex setup
    )
    analyzer = get_presidio_analyzer_from_config(config)
    return analyzer.analyze(text)
```

#### 3. Leverage Parallel Processing

```python
# ✅ Recommended: Use parallel processing for batches
from cloakpivot.core.parallel_analysis import ParallelAnalysisEngine

def process_large_batch(documents, policy):
    engine = ParallelAnalysisEngine(
        max_workers=4,  # Or None for auto-detection
        chunk_size=1000  # Optimal for most use cases
    )
    
    results = engine.analyze_documents(
        documents=documents,
        policy=policy
    )
    
    return results

# ❌ Avoid: Sequential processing for large batches
def process_large_batch_inefficient(documents, policy):
    results = []
    for doc in documents:  # Sequential: doesn't use multiple cores
        result = analyze_single(doc, policy)
        results.append(result)
    
    return results  # This can be 4-8x slower than parallel processing
```

#### 4. Memory-Efficient Processing

```python
# ✅ Recommended: Process documents in batches
from cloakpivot.core.chunking import ChunkedDocumentProcessor

def process_large_documents(documents):
    processor = ChunkedDocumentProcessor(
        chunk_size=1000,        # Balance memory and overhead
        enable_parallel=True    # Use parallel processing
    )
    
    for batch in processor.process_batch(documents, batch_size=10):
        yield from batch  # Generator pattern for memory efficiency
        
        # Optional: Force garbage collection periodically
        if processor.processed_count % 100 == 0:
            import gc
            gc.collect()

# ❌ Avoid: Loading all documents into memory at once
def process_large_documents_inefficient(documents):
    all_results = []
    for doc in documents:  # All documents stay in memory
        result = expensive_processing(doc)
        all_results.append(result)  # Memory usage grows linearly
    
    return all_results  # Peak memory usage is very high
```

### Test Development

#### 1. Use Session-Scoped Fixtures

```python
# ✅ Recommended: Session scope for expensive resources
import pytest
from cloakpivot.loaders import get_presidio_analyzer, get_detection_pipeline

@pytest.fixture(scope="session")
def shared_analyzer():
    """Analyzer shared across all tests in the session."""
    return get_presidio_analyzer()

@pytest.fixture(scope="session")
def shared_detection_pipeline():
    """Detection pipeline shared across all tests."""
    return get_detection_pipeline()

def test_entity_detection(shared_analyzer):
    # Uses cached analyzer - fast!
    results = shared_analyzer.analyze("John Doe's email is john@email.com")
    assert len(results) > 0

# ❌ Avoid: Function scope for expensive resources
@pytest.fixture  # Function scope - recreated for each test!
def analyzer_per_test():
    # This creates a new analyzer for every single test
    # Total overhead: ~1-3 seconds × number of tests
    return AnalyzerEngineWrapper()  # Expensive initialization
```

**Performance Impact:** Session fixtures can reduce test suite execution time by 50-80%.

#### 2. Use Fast Mode for Development

```bash
# Development testing - runs in ~30 seconds
export PYTEST_FAST_MODE=true
pytest -m "not slow"

# Comprehensive testing - runs in ~5 minutes  
export PYTEST_FAST_MODE=false
pytest

# Parallel testing - 2-4x speedup
pytest -n auto  # Auto-detect worker count
pytest -n 4     # Explicit worker count
```

#### 3. Optimize Test Parallelization

```python
# ✅ Recommended: Worker-aware session fixtures
import pytest

@pytest.fixture(scope="session")
def shared_resource(worker_id):
    """Session fixture that works with pytest-xdist."""
    if worker_id == "master":
        # Main process
        return create_main_resource()
    else:
        # Worker process
        return create_worker_resource(worker_id)

# ✅ Recommended: Use proper test distribution
# In pytest.ini or setup.cfg:
# [tool:pytest]
# addopts = --dist=loadfile  # Better for session fixtures
```

## Profiling and Monitoring

### Using PerformanceProfiler

CloakPivot includes a comprehensive performance profiler for timing and resource monitoring:

#### Basic Profiling

```python
from cloakpivot.core.performance import PerformanceProfiler

# Create profiler instance
profiler = PerformanceProfiler(
    enable_memory_tracking=True,      # Track memory usage deltas
    enable_detailed_logging=False,    # Log every operation
    auto_report_threshold_ms=1000.0   # Auto-log slow operations
)

# Profile a code block
with profiler.measure_operation("entity_detection") as metric:
    results = analyzer.analyze(text)
    # metric object is available during execution
    print(f"Entities found: {len(results)}")

# Get performance statistics
stats = profiler.get_operation_stats("entity_detection")
print(f"Average time: {stats.average_duration_ms:.2f}ms")
print(f"Call count: {stats.total_calls}")
print(f"Success rate: {stats.success_rate:.1%}")
```

#### Decorator-Based Profiling

```python
from cloakpivot.core.performance import profile_method

class DocumentAnalyzer:
    @profile_method("document_analysis", include_args=True)
    def analyze_document(self, document, policy):
        """Method will be automatically profiled."""
        return self._internal_analysis(document, policy)
    
    @profile_method("batch_processing")
    def process_batch(self, documents):
        """Profile batch operations."""
        results = []
        for doc in documents:
            result = self.analyze_document(doc)
            results.append(result)
        return results

# Usage
analyzer = DocumentAnalyzer()
results = analyzer.process_batch(documents)

# Get profiling results
from cloakpivot.core.performance import get_profiler
profiler = get_profiler()
stats = profiler.get_operation_stats()

for operation, stat in stats.items():
    print(f"{operation}: {stat.average_duration_ms:.1f}ms avg, {stat.total_calls} calls")
```

#### Continuous Monitoring

```python
# Enable global profiler for the entire application
from cloakpivot.core.performance import get_profiler

profiler = get_profiler()

# Profile key operations throughout your application
with profiler.measure_operation("batch_processing"):
    process_documents(documents)

with profiler.measure_operation("database_operations"):
    save_results_to_db(results)

# Generate comprehensive performance report
report = profiler.generate_performance_report()

print(f"Total operations: {report['summary']['total_operations']}")
print(f"Overall success rate: {report['summary']['overall_success_rate']:.1%}")

# Export metrics for external monitoring
profiler.export_metrics_to_log()
```

### Performance Benchmarking

#### Running Built-in Benchmarks

```bash
# Run performance benchmarks
pytest tests/performance/ --benchmark-only

# Generate performance report with JSON output
pytest tests/performance/ --benchmark-only --benchmark-json=perf.json

# Compare with baseline performance
python scripts/performance_regression_analysis.py \
    --baseline baseline.json \
    --current perf.json \
    --threshold 0.10  # 10% regression threshold

# Generate performance trends over time
python scripts/generate-performance-trends.py \
    --input-dir ./performance-data/ \
    --output trends.html
```

#### Custom Benchmarks

```python
import pytest
from cloakpivot.core.performance import PerformanceProfiler

def test_analyzer_performance(benchmark, shared_analyzer):
    """Benchmark analyzer performance with pytest-benchmark."""
    text = "John Doe's email is john.doe@email.com and phone is 555-123-4567"
    
    # Use pytest-benchmark for statistical analysis
    result = benchmark(shared_analyzer.analyze, text)
    
    assert len(result) >= 3  # Should find at least name, email, phone

def test_batch_processing_performance():
    """Custom benchmark with PerformanceProfiler."""
    profiler = PerformanceProfiler()
    
    documents = generate_test_documents(count=100)  # Helper function
    
    with profiler.measure_operation("batch_processing"):
        results = process_documents_in_batch(documents)
    
    stats = profiler.get_operation_stats("batch_processing")
    
    # Assert performance requirements
    assert stats.average_duration_ms < 5000, f"Batch processing too slow: {stats.average_duration_ms:.1f}ms"
    assert stats.success_rate > 0.95, f"Success rate too low: {stats.success_rate:.1%}"
```

#### Memory Profiling

```python
from cloakpivot.core.performance import PerformanceProfiler

# Enable memory tracking
profiler = PerformanceProfiler(enable_memory_tracking=True)

with profiler.measure_operation("memory_intensive_operation") as metric:
    # Your memory-intensive code here
    large_results = process_large_dataset(data)

# Check memory usage
if metric.memory_delta_mb and metric.memory_delta_mb > 100:
    print(f"Warning: Operation used {metric.memory_delta_mb:.1f}MB of memory")

# Get recent metrics with memory information
recent_metrics = profiler.get_recent_metrics(limit=10)
for metric in recent_metrics:
    if metric.memory_delta_mb:
        print(f"{metric.operation}: {metric.memory_delta_mb:+.1f}MB memory delta")
```

## Troubleshooting

### Common Performance Issues

#### 1. Slow Analyzer Initialization

**Symptom:** Long delays when creating analyzers (1-3 seconds per call)

**Diagnosis:**
```python
# Check singleton status
import os
singleton_enabled = os.getenv('CLOAKPIVOT_USE_SINGLETON', 'true') == 'true'
print(f"Singleton enabled: {singleton_enabled}")

# Verify cache hits
from cloakpivot.loaders import get_presidio_analyzer
analyzer1 = get_presidio_analyzer()
analyzer2 = get_presidio_analyzer()
print(f"Same instance: {analyzer1 is analyzer2}")  # Should be True

# Check cache statistics
from cloakpivot.loaders import get_cache_info
cache_info = get_cache_info()
print(f"Cache hits: {cache_info['analyzer']['hits']}")
print(f"Cache misses: {cache_info['analyzer']['misses']}")
```

**Solutions:**
1. Enable singleton pattern: `export CLOAKPIVOT_USE_SINGLETON=true`
2. Verify you're using loader functions instead of direct instantiation
3. Check cache hit rates - low hit rates indicate configuration issues

#### 2. High Memory Usage

**Symptom:** Memory usage grows over time or exceeds expectations

**Diagnosis:**
```python
# Check cache size configuration with error handling
import os

try:
    cache_size = int(os.getenv('ANALYZER_CACHE_SIZE', '8'))
    print(f"Cache size: {cache_size}")
except ValueError as e:
    print(f"Error: Invalid ANALYZER_CACHE_SIZE value: {e}")
    cache_size = 8  # Use default
    print(f"Using default cache size: {cache_size}")

# Monitor memory usage
from cloakpivot.core.performance import PerformanceProfiler
profiler = PerformanceProfiler(enable_memory_tracking=True)

with profiler.measure_operation("memory_analysis") as metric:
    # Your code here
    results = analyze_large_document(document)

print(f"Memory delta: {metric.memory_delta_mb:+.1f}MB")

# Check system memory usage
import psutil
process = psutil.Process()
memory_info = process.memory_info()
print(f"RSS: {memory_info.rss / 1024 / 1024:.1f}MB")
print(f"VMS: {memory_info.vms / 1024 / 1024:.1f}MB")
```

**Solutions:**
1. Reduce `ANALYZER_CACHE_SIZE` if memory-constrained
2. Use `MODEL_SIZE=small` to reduce model memory footprint
3. Process documents in batches to limit peak memory usage
4. Enable garbage collection more frequently: `gc.collect()`

#### 3. Slow Test Execution

**Symptom:** Test suite takes too long to execute (>5 minutes)

**Diagnosis:**
```bash
# Profile test execution
pytest --durations=10  # Show 10 slowest tests

# Check fixture scopes
python scripts/audit_fixtures.py

# Analyze test distribution
pytest --collect-only | grep -c "test_"
```

**Solutions:**
1. Convert expensive fixtures to session scope:
```python
@pytest.fixture(scope="session")  # Instead of function scope
def expensive_resource():
    return create_expensive_resource()
```

2. Enable parallel execution:
```bash
pytest -n auto  # Auto-detect workers
pytest -n 4     # Explicit worker count
```

3. Use fast mode for development:
```bash
export PYTEST_FAST_MODE=true
pytest -m "not slow"
```

#### 4. Poor Parallel Performance

**Symptom:** Parallel processing slower than expected or not utilizing multiple cores

**Diagnosis:**
```python
import multiprocessing
import os

print(f"CPU cores: {multiprocessing.cpu_count()}")

# Check worker configuration
max_workers = os.getenv('MAX_WORKERS', 'auto')
print(f"Configured workers: {max_workers}")

# Check parallel processing usage
from cloakpivot.core.parallel_analysis import ParallelAnalysisEngine
engine = ParallelAnalysisEngine()
results = engine.analyze_documents(documents)
print(f"Threads used: {results.threads_used}")
print(f"Total processing time: {results.total_processing_time_ms:.1f}ms")
```

**Solutions:**
1. Adjust `MAX_WORKERS` based on workload type:
   - CPU-bound: `MAX_WORKERS = cpu_count()`
   - Memory-bound: `MAX_WORKERS = cpu_count() // 2`
   - I/O-bound: `MAX_WORKERS = cpu_count() * 2`

2. Use appropriate test distribution:
```bash
pytest -n auto --dist=loadfile  # Better for session fixtures
```

3. Monitor CPU and memory usage during parallel execution:
```bash
htop  # Or similar system monitor
```

### Performance Regression Investigation

#### 1. Identify Performance Changes

```bash
# Compare current vs baseline performance
python scripts/performance_regression_analysis.py \
    --baseline baseline.json \
    --current current.json \
    --threshold 0.10 \
    --output regression_report.json

# Generate detailed performance comparison
python scripts/generate-performance-trends.py \
    --compare baseline.json current.json \
    --output comparison.html
```

#### 2. Profile Specific Operations

```python
# Target specific slow operations for detailed analysis
from cloakpivot.core.performance import PerformanceProfiler

profiler = PerformanceProfiler(
    enable_detailed_logging=True,  # Log every operation
    auto_report_threshold_ms=100.0  # Lower threshold for investigation
)

with profiler.measure_operation("slow_operation") as metric:
    # Code under investigation
    suspicious_function(data)

# Analyze detailed timing and memory usage
stats = profiler.get_operation_stats("slow_operation")
recent_metrics = profiler.get_recent_metrics("slow_operation", limit=1)

print(f"Operation stats:")
print(f"  Average duration: {stats.average_duration_ms:.1f}ms")
print(f"  Min/Max duration: {stats.min_duration_ms:.1f}/{stats.max_duration_ms:.1f}ms")
print(f"  Call count: {stats.total_calls}")
print(f"  Success rate: {stats.success_rate:.1%}")

if recent_metrics and recent_metrics[0].memory_delta_mb:
    print(f"  Memory usage: {recent_metrics[0].memory_delta_mb:+.1f}MB")
```

#### 3. Comparative Analysis

```python
# Compare performance between different configurations
def performance_comparison():
    configurations = [
        {"MODEL_SIZE": "small", "ANALYZER_CACHE_SIZE": "4"},
        {"MODEL_SIZE": "medium", "ANALYZER_CACHE_SIZE": "8"},
        {"MODEL_SIZE": "large", "ANALYZER_CACHE_SIZE": "16"},
    ]
    
    results = {}
    
    for config in configurations:
        # Set environment variables
        for key, value in config.items():
            os.environ[key] = value
        
        # Clear caches to ensure fresh start
        from cloakpivot.loaders import clear_all_caches
        clear_all_caches()
        
        # Run performance test
        profiler = PerformanceProfiler()
        with profiler.measure_operation("config_test"):
            # Your test code here
            run_performance_test()
        
        stats = profiler.get_operation_stats("config_test")
        results[str(config)] = {
            "duration_ms": stats.average_duration_ms,
            "memory_mb": get_memory_usage()
        }
    
    return results

# Analyze results
comparison = performance_comparison()
for config, metrics in comparison.items():
    print(f"{config}: {metrics['duration_ms']:.1f}ms, {metrics['memory_mb']:.1f}MB")
```

## Advanced Optimization

### Custom Caching Strategies

```python
# Custom TTL (Time-To-Live) cache implementation for CloakPivot performance optimization
# Use this pattern when you need temporary caching that automatically expires old data

from functools import lru_cache
import time
from typing import Any, Dict, Optional

class TTLCache:
    """
    Time-To-Live cache that automatically expires entries after a specified time.
    
    This is useful for caching expensive operations that should be refreshed periodically,
    such as model inference results or API responses that may become stale.
    
    Performance characteristics:
    - O(1) get/set operations when cache isn't full
    - O(n) set operation when evicting oldest item (happens rarely)
    - Memory usage: fixed maximum based on maxsize parameter
    """
    
    def __init__(self, maxsize: int = 128, ttl: float = 3600.0):
        self.cache: Dict[Any, tuple] = {}  # Store (value, timestamp) tuples
        self.maxsize = maxsize
        self.ttl = ttl  # Time-to-live in seconds
    
    def get(self, key: Any) -> Optional[Any]:
        """Retrieve cached value if exists and not expired."""
        if key in self.cache:
            value, timestamp = self.cache[key]
            # Check if entry has expired
            if time.time() - timestamp < self.ttl:
                return value  # Cache hit - return valid value
            else:
                del self.cache[key]  # Lazy expiration - remove stale entry
        return None  # Cache miss or expired
    
    def set(self, key: Any, value: Any) -> None:
        """Store value with current timestamp."""
        # Implement LRU eviction when cache is full
        if len(self.cache) >= self.maxsize:
            # Find and remove the oldest entry (least recently stored)
            # Note: This is O(n) but happens infrequently
            oldest_key = min(self.cache.keys(), key=lambda k: self.cache[k][1])
            del self.cache[oldest_key]
        
        # Store value with current timestamp
        self.cache[key] = (value, time.time())

# Example usage: Cache expensive entity analysis results
ttl_cache = TTLCache(maxsize=64, ttl=3600)  # 1 hour TTL, 64 entries max

def expensive_computation(data_hash: str):
    """
    Cache expensive computation results with automatic expiration.
    
    This pattern is ideal for:
    - Model inference results that may change with model updates
    - External API responses that should be refreshed periodically
    - Analysis results where input data might change
    """
    cached_result = ttl_cache.get(data_hash)
    if cached_result is not None:
        return cached_result
    
    # Perform expensive computation
    result = perform_complex_analysis(data_hash)
    ttl_cache.set(data_hash, result)
    return result

# LRU cache for frequently accessed patterns
@lru_cache(maxsize=256)
def get_entity_patterns(entity_type: str, language: str):
    """Cache compiled regex patterns for entity recognition."""
    return compile_entity_patterns(entity_type, language)

# Memory-efficient result caching
class ResultCache:
    def __init__(self, max_memory_mb: float = 100.0):
        self.cache = {}
        self.max_memory = max_memory_mb * 1024 * 1024  # Convert to bytes
        self.current_memory = 0
    
    def cache_result(self, key: str, result: Any) -> None:
        result_size = len(str(result).encode('utf-8'))
        
        # Evict items if necessary
        while self.current_memory + result_size > self.max_memory and self.cache:
            evict_key = next(iter(self.cache))  # Remove first item
            evicted_size = len(str(self.cache[evict_key]).encode('utf-8'))
            del self.cache[evict_key]
            self.current_memory -= evicted_size
        
        self.cache[key] = result
        self.current_memory += result_size
```

### Memory Optimization

```python
import gc
from contextlib import contextmanager
from typing import Generator, Any

@contextmanager
def memory_optimized_processing(force_gc_every: int = 100) -> Generator[None, None, None]:
    """Context manager for memory-optimized batch processing."""
    processed_count = 0
    
    try:
        yield
    finally:
        # Force garbage collection
        gc.collect()

def process_large_batch_optimized(documents, batch_size: int = 50):
    """Memory-efficient batch processing with automatic cleanup."""
    
    def process_batch_chunk(chunk):
        results = []
        with memory_optimized_processing():
            for doc in chunk:
                result = process_single_document(doc)
                results.append(result)
                
                # Clear references to help GC
                doc = None
        
        return results
    
    # Process in chunks
    for i in range(0, len(documents), batch_size):
        chunk = documents[i:i + batch_size]
        
        with memory_optimized_processing():
            chunk_results = process_batch_chunk(chunk)
            yield from chunk_results
        
        # Force garbage collection after each batch
        if i % (batch_size * 10) == 0:
            gc.collect()

# Streaming processing for very large datasets
def stream_process_documents(document_stream):
    """Process documents from a stream to minimize memory usage."""
    
    for batch in batch_documents(document_stream, batch_size=20):
        with memory_optimized_processing():
            results = process_document_batch(batch)
            
            for result in results:
                yield result
                
                # Clear result reference immediately after yielding
                result = None

# Memory monitoring and alerting
class MemoryMonitor:
    def __init__(self, threshold_mb: float = 1000.0):
        self.threshold = threshold_mb * 1024 * 1024  # Convert to bytes
        self.baseline = self._get_memory_usage()
    
    def _get_memory_usage(self) -> int:
        import psutil
        return psutil.Process().memory_info().rss
    
    def check_memory_usage(self) -> tuple[bool, float]:
        current = self._get_memory_usage()
        delta_mb = (current - self.baseline) / 1024 / 1024
        exceeded = (current - self.baseline) > self.threshold
        
        if exceeded:
            print(f"Warning: Memory usage increased by {delta_mb:.1f}MB")
        
        return exceeded, delta_mb

# Usage example
monitor = MemoryMonitor(threshold_mb=500)

for batch in process_large_dataset():
    process_batch(batch)
    
    exceeded, delta = monitor.check_memory_usage()
    if exceeded:
        gc.collect()  # Force cleanup
        print("Forced garbage collection due to high memory usage")
```

### Multi-threading Considerations

```python
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import List, Callable, Any

# Advanced Multi-threading Pattern for CloakPivot
# This example demonstrates how to safely use CloakPivot analyzers in a multi-threaded environment

def worker_function(texts: List[str], worker_id: int) -> List[Any]:
    """
    Worker function that processes a chunk of texts in parallel.
    
    Key implementation details:
    - Each worker thread gets its own analyzer instance via singleton loader
    - Singleton pattern ensures efficient resource sharing without thread conflicts
    - Worker ID helps track which thread processed which results for debugging
    
    Thread Safety: CloakPivot analyzers are thread-safe when obtained via singleton loaders
    """
    # Get analyzer instance - singleton pattern makes this efficient across threads
    # Multiple threads calling this will get the same underlying resources safely
    analyzer = get_presidio_analyzer()
    
    results = []
    for text in texts:
        # Each worker processes its assigned texts independently
        result = analyzer.analyze(text)
        results.append({
            'text': text,
            'entities': result,
            'worker_id': worker_id  # For debugging and result tracking
        })
    
    return results

def parallel_text_analysis(texts: List[str], max_workers: int = 4) -> List[Any]:
    """
    Parallel text analysis using ThreadPoolExecutor with intelligent work distribution.
    
    Performance characteristics:
    - Scales linearly up to CPU core count
    - Memory usage: O(n) where n is number of texts
    - Optimal for CPU-bound entity detection tasks
    
    Best practices:
    - Use max_workers = CPU cores for CPU-bound tasks
    - Use max_workers = 2-4x CPU cores for I/O-bound tasks
    - Monitor memory usage for large text volumes
    """
    
    # Distribute work evenly across workers to maximize CPU utilization
    chunk_size = max(1, len(texts) // max_workers)  # Ensure at least 1 text per chunk
    text_chunks = [texts[i:i + chunk_size] for i in range(0, len(texts), chunk_size)]
    
    all_results = []
    
    # Use ThreadPoolExecutor for automatic thread management and cleanup
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all work upfront - ThreadPoolExecutor handles scheduling
        future_to_chunk = {
            executor.submit(worker_function, chunk, i): i 
            for i, chunk in enumerate(text_chunks) if chunk  # Skip empty chunks
        }
        
        # Process results as they become available (not necessarily in order)
        # This pattern provides better responsiveness than waiting for all threads
        for future in as_completed(future_to_chunk):
            chunk_id = future_to_chunk[future]
            try:
                chunk_results = future.result()
                all_results.extend(chunk_results)
            except Exception as exc:
                # Handle individual chunk failures gracefully
                # Production code should implement retry logic or partial failure handling
                print(f'Chunk {chunk_id} failed with exception: {exc}')
                # Continue processing other chunks even if one fails
    
    return all_results

# Producer-Consumer pattern for streaming processing
class DocumentProcessor:
    def __init__(self, max_workers: int = 4, queue_size: int = 100):
        self.max_workers = max_workers
        self.input_queue = Queue(maxsize=queue_size)
        self.output_queue = Queue()
        self.workers = []
        self.running = False
    
    def worker(self):
        """Worker thread that processes documents from the queue."""
        analyzer = get_presidio_analyzer()  # Each worker gets its own analyzer
        
        while self.running:
            try:
                document = self.input_queue.get(timeout=1)
                if document is None:  # Poison pill to stop worker
                    break
                
                result = analyzer.analyze(document['text'])
                self.output_queue.put({
                    'document_id': document['id'],
                    'entities': result,
                    'processed_by': threading.current_thread().name
                })
                
                self.input_queue.task_done()
                
            except Empty:
                continue  # Timeout, check if still running
    
    def start(self):
        """Start worker threads."""
        self.running = True
        
        for i in range(self.max_workers):
            worker_thread = threading.Thread(target=self.worker, name=f'Worker-{i}')
            worker_thread.start()
            self.workers.append(worker_thread)
    
    def stop(self):
        """Stop worker threads gracefully."""
        self.running = False
        
        # Send poison pills to stop workers
        for _ in range(self.max_workers):
            self.input_queue.put(None)
        
        # Wait for all workers to finish
        for worker in self.workers:
            worker.join()
    
    def process_documents(self, documents):
        """Process documents using the worker pool."""
        self.start()
        
        # Feed documents to workers
        for doc in documents:
            self.input_queue.put(doc)
        
        # Wait for all documents to be processed
        self.input_queue.join()
        
        # Collect results
        results = []
        while not self.output_queue.empty():
            results.append(self.output_queue.get())
        
        self.stop()
        return results

# Usage example
processor = DocumentProcessor(max_workers=4)
documents = [{'id': i, 'text': f'Sample text {i}'} for i in range(100)]
results = processor.process_documents(documents)

print(f"Processed {len(results)} documents using multiple workers")
```

### Performance Monitoring Dashboard

```python
import json
import time
from datetime import datetime, timezone
from typing import Dict, Any, List

class PerformanceMonitor:
    """Comprehensive performance monitoring and alerting system with configurable thresholds."""
    
    def __init__(self, custom_thresholds: Optional[Dict[str, Any]] = None):
        self.metrics_history = []
        self.alerts = []
        
        # Default thresholds - can be overridden via environment variables or constructor
        default_thresholds = {
            'avg_duration_ms': float(os.getenv('PERF_ALERT_DURATION_MS', '5000')),
            'memory_delta_mb': float(os.getenv('PERF_ALERT_MEMORY_MB', '100')), 
            'success_rate': float(os.getenv('PERF_ALERT_SUCCESS_RATE', '0.95')),
            'queue_size': int(os.getenv('PERF_ALERT_QUEUE_SIZE', '1000'))
        }
        
        # Allow custom thresholds to override defaults
        self.thresholds = default_thresholds
        if custom_thresholds:
            self.thresholds.update(custom_thresholds)
    
    def collect_metrics(self, profiler: PerformanceProfiler) -> Dict[str, Any]:
        """Collect current performance metrics."""
        stats = profiler.get_operation_stats()
        recent_metrics = profiler.get_recent_metrics(limit=10)
        
        metrics = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'operations': {},
            'system': self._get_system_metrics(),
            'cache': self._get_cache_metrics()
        }
        
        for operation, stat in stats.items():
            metrics['operations'][operation] = {
                'avg_duration_ms': stat.average_duration_ms,
                'total_calls': stat.total_calls,
                'success_rate': stat.success_rate,
                'min_duration_ms': stat.min_duration_ms,
                'max_duration_ms': stat.max_duration_ms
            }
        
        # Add memory information from recent metrics
        for metric in recent_metrics:
            if metric.memory_delta_mb is not None:
                op_metrics = metrics['operations'].get(metric.operation, {})
                op_metrics['memory_delta_mb'] = metric.memory_delta_mb
        
        self.metrics_history.append(metrics)
        self._check_thresholds(metrics)
        
        return metrics
    
    def _get_system_metrics(self) -> Dict[str, float]:
        """Get system-level performance metrics."""
        import psutil
        
        process = psutil.Process()
        memory_info = process.memory_info()
        cpu_percent = process.cpu_percent()
        
        return {
            'cpu_percent': cpu_percent,
            'memory_rss_mb': memory_info.rss / 1024 / 1024,
            'memory_vms_mb': memory_info.vms / 1024 / 1024,
            'num_threads': process.num_threads(),
            'num_fds': process.num_fds() if hasattr(process, 'num_fds') else 0
        }
    
    def _get_cache_metrics(self) -> Dict[str, Any]:
        """Get cache performance metrics."""
        from cloakpivot.loaders import get_cache_info
        return get_cache_info()
    
    def _check_thresholds(self, metrics: Dict[str, Any]) -> None:
        """Check metrics against thresholds and generate alerts."""
        for operation, op_metrics in metrics['operations'].items():
            # Check duration threshold
            if op_metrics['avg_duration_ms'] > self.thresholds['avg_duration_ms']:
                self._add_alert(
                    f"High average duration for {operation}: {op_metrics['avg_duration_ms']:.1f}ms",
                    severity='warning'
                )
            
            # Check success rate threshold
            if op_metrics['success_rate'] < self.thresholds['success_rate']:
                self._add_alert(
                    f"Low success rate for {operation}: {op_metrics['success_rate']:.1%}",
                    severity='error'
                )
            
            # Check memory usage threshold
            if 'memory_delta_mb' in op_metrics:
                if op_metrics['memory_delta_mb'] > self.thresholds['memory_delta_mb']:
                    self._add_alert(
                        f"High memory usage for {operation}: {op_metrics['memory_delta_mb']:.1f}MB",
                        severity='warning'
                    )
    
    def _add_alert(self, message: str, severity: str = 'info') -> None:
        """Add performance alert."""
        alert = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'message': message,
            'severity': severity
        }
        
        self.alerts.append(alert)
        print(f"[{severity.upper()}] {message}")
    
    def generate_dashboard_data(self) -> Dict[str, Any]:
        """Generate data for performance dashboard."""
        if not self.metrics_history:
            return {}
        
        latest_metrics = self.metrics_history[-1]
        
        dashboard = {
            'summary': {
                'last_updated': latest_metrics['timestamp'],
                'total_operations': len(latest_metrics['operations']),
                'active_alerts': len([a for a in self.alerts[-10:] if a['severity'] in ['warning', 'error']]),
                'system_health': self._calculate_health_score(latest_metrics)
            },
            'operations': self._format_operations_data(latest_metrics['operations']),
            'system': latest_metrics['system'],
            'cache': latest_metrics['cache'],
            'trends': self._calculate_trends(),
            'recent_alerts': self.alerts[-10:],
            'recommendations': self._generate_recommendations()
        }
        
        return dashboard
    
    def _calculate_health_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall system health score (0-100)."""
        score = 100.0
        
        # Deduct points for alerts
        error_count = len([a for a in self.alerts[-10:] if a['severity'] == 'error'])
        warning_count = len([a for a in self.alerts[-10:] if a['severity'] == 'warning'])
        
        score -= error_count * 20  # 20 points per error
        score -= warning_count * 10  # 10 points per warning
        
        # Check cache hit rates
        cache_metrics = metrics.get('cache', {})
        for cache_name, cache_info in cache_metrics.items():
            if cache_info['hits'] + cache_info['misses'] > 0:
                hit_rate = cache_info['hits'] / (cache_info['hits'] + cache_info['misses'])
                if hit_rate < 0.8:  # Less than 80% hit rate
                    score -= 10
        
        return max(0.0, score)
    
    def _format_operations_data(self, operations: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Format operations data for dashboard display."""
        formatted = []
        
        for op_name, metrics in operations.items():
            formatted.append({
                'name': op_name,
                'avg_duration': metrics['avg_duration_ms'],
                'call_count': metrics['total_calls'],
                'success_rate': metrics['success_rate'],
                'status': self._get_operation_status(metrics)
            })
        
        # Sort by average duration (slowest first)
        formatted.sort(key=lambda x: x['avg_duration'], reverse=True)
        return formatted
    
    def _get_operation_status(self, metrics: Dict[str, Any]) -> str:
        """Determine operation status based on metrics."""
        if metrics['success_rate'] < self.thresholds['success_rate']:
            return 'error'
        elif metrics['avg_duration_ms'] > self.thresholds['avg_duration_ms']:
            return 'warning'
        else:
            return 'ok'
    
    def _calculate_trends(self) -> Dict[str, Any]:
        """Calculate performance trends over time."""
        if len(self.metrics_history) < 2:
            return {}
        
        current = self.metrics_history[-1]
        previous = self.metrics_history[-2]
        
        trends = {}
        
        for op_name in current['operations']:
            if op_name in previous['operations']:
                curr_duration = current['operations'][op_name]['avg_duration_ms']
                prev_duration = previous['operations'][op_name]['avg_duration_ms']
                
                if prev_duration > 0:
                    change_percent = ((curr_duration - prev_duration) / prev_duration) * 100
                    trends[op_name] = {
                        'duration_change_percent': change_percent,
                        'trend': 'improving' if change_percent < -5 else 'degrading' if change_percent > 5 else 'stable'
                    }
        
        return trends
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance optimization recommendations."""
        if not self.metrics_history:
            return []
        
        recommendations = []
        latest = self.metrics_history[-1]
        
        # Check cache hit rates
        cache_metrics = latest.get('cache', {})
        for cache_name, cache_info in cache_metrics.items():
            total_requests = cache_info['hits'] + cache_info['misses']
            if total_requests > 0:
                hit_rate = cache_info['hits'] / total_requests
                if hit_rate < 0.8:
                    recommendations.append(
                        f"Consider increasing {cache_name} cache size (current hit rate: {hit_rate:.1%})"
                    )
        
        # Check for slow operations
        for op_name, metrics in latest['operations'].items():
            if metrics['avg_duration_ms'] > self.thresholds['avg_duration_ms']:
                recommendations.append(
                    f"Optimize {op_name} operation (current avg: {metrics['avg_duration_ms']:.1f}ms)"
                )
        
        # Check memory usage
        memory_mb = latest['system']['memory_rss_mb']
        if memory_mb > 1000:  # More than 1GB
            recommendations.append(
                f"Consider memory optimization - current usage: {memory_mb:.1f}MB"
            )
        
        return recommendations

# Usage example
def setup_monitoring():
    """Setup comprehensive performance monitoring."""
    from cloakpivot.core.performance import get_profiler
    
    monitor = PerformanceMonitor()
    profiler = get_profiler()
    
    # Collect metrics periodically
    def collect_metrics_periodically():
        while True:
            metrics = monitor.collect_metrics(profiler)
            
            # Export to external systems if needed
            export_metrics_to_monitoring_system(metrics)
            
            time.sleep(60)  # Collect every minute
    
    # Start monitoring in background thread
    import threading
    monitor_thread = threading.Thread(target=collect_metrics_periodically, daemon=True)
    monitor_thread.start()
    
    return monitor

def export_metrics_to_monitoring_system(metrics: Dict[str, Any]):
    """Export metrics to external monitoring systems."""
    # Example: Send to logging system
    import logging
    logger = logging.getLogger('performance.monitor')
    logger.info("Performance metrics", extra={"metrics": metrics})
    
    # Example: Send to StatsD
    # statsd_client.gauge('cloakpivot.avg_duration', avg_duration)
    
    # Example: Send to Prometheus
    # prometheus_gauge.set(avg_duration)
    
    pass  # Implement based on your monitoring infrastructure
```

## Performance Testing

### CI/CD Integration

CloakPivot includes automated performance monitoring in CI/CD pipelines:

#### GitHub Actions Performance Workflow

The repository includes automated performance monitoring:

```yaml
# .github/workflows/performance-monitoring.yml
name: Performance Monitoring

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  performance:
    runs-on: ubuntu-latest
    
    steps:
      - uses: actions/checkout@v3
      
      - name: Setup Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      
      - name: Install dependencies
        run: |
          pip install -e .[dev]
          pip install pytest-benchmark
      
      - name: Run performance benchmarks
        run: |
          pytest tests/performance/ \
            --benchmark-only \
            --benchmark-json=benchmark.json
      
      - name: Performance regression analysis
        run: |
          python scripts/performance_regression_analysis.py \
            --current benchmark.json \
            --baseline benchmark-baseline.json \
            --threshold 0.15
```

#### Running Performance Tests Locally

```bash
# Basic performance test run
pytest tests/performance/ --benchmark-only

# Generate detailed performance report
pytest tests/performance/ \
  --benchmark-only \
  --benchmark-json=performance.json \
  --benchmark-histogram=histograms

# Compare with baseline
python scripts/performance_regression_analysis.py \
  --current performance.json \
  --baseline baseline.json \
  --threshold 0.10 \
  --output regression-report.html

# Performance trends over time
python scripts/generate-performance-trends.py \
  --input-dir ./performance-history/ \
  --output trends.html
```

### Custom Performance Tests

```python
# tests/performance/test_custom_benchmarks.py
import pytest
from cloakpivot.core.performance import PerformanceProfiler

class TestCustomPerformance:
    
    @pytest.mark.benchmark(group="entity_detection")
    def test_entity_detection_performance(self, benchmark, shared_analyzer):
        """Benchmark entity detection performance."""
        
        text = """
        John Doe (SSN: 123-45-6789) works at Example Corp.
        His email is john.doe@example.com and phone is (555) 123-4567.
        The project manager Sarah Smith can be reached at sarah@example.com.
        """
        
        # Benchmark the analyze operation
        result = benchmark(shared_analyzer.analyze, text)
        
        # Verify results
        assert len(result) >= 4  # Should find at least name, SSN, email, phone
        
        # Performance assertions
        stats = benchmark.stats
        assert stats.mean < 0.1, f"Entity detection too slow: {stats.mean:.3f}s"
    
    @pytest.mark.benchmark(group="batch_processing")
    def test_batch_processing_performance(self, benchmark):
        """Benchmark batch processing performance."""
        
        # Generate test documents
        documents = [
            f"Document {i}: John Smith works at Company {i}. Email: john{i}@company.com"
            for i in range(50)
        ]
        
        def process_batch():
            from cloakpivot.core.batch import BatchProcessor
            processor = BatchProcessor(policy="balanced.yaml")
            return processor.process_documents(documents)
        
        result = benchmark(process_batch)
        
        assert len(result) == 50
        
        # Performance assertions
        stats = benchmark.stats
        assert stats.mean < 5.0, f"Batch processing too slow: {stats.mean:.3f}s"
    
    def test_memory_usage_limits(self):
        """Test that operations stay within memory limits."""
        
        profiler = PerformanceProfiler(enable_memory_tracking=True)
        
        # Large document processing
        large_text = "John Doe's information: john@email.com. " * 10000  # ~400KB
        
        with profiler.measure_operation("large_document") as metric:
            analyzer = get_presidio_analyzer()
            result = analyzer.analyze(large_text)
        
        # Memory usage assertions
        assert metric.memory_delta_mb is not None
        assert metric.memory_delta_mb < 50, f"Memory usage too high: {metric.memory_delta_mb:.1f}MB"
        assert len(result) > 0, "Should detect entities in large document"
    
    def test_concurrent_processing_performance(self):
        """Test performance under concurrent load."""
        
        import concurrent.futures
        import threading
        
        profiler = PerformanceProfiler()
        
        def worker_task(worker_id: int):
            with profiler.measure_operation(f"concurrent_worker_{worker_id}"):
                analyzer = get_presidio_analyzer()  # Should use cached instance
                text = f"Worker {worker_id}: John Doe (john{worker_id}@email.com)"
                return analyzer.analyze(text)
        
        # Run 10 concurrent workers
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(worker_task, i) for i in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert len(results) == 10
        
        # Check that all workers completed successfully
        stats = profiler.get_operation_stats()
        worker_stats = {k: v for k, v in stats.items() if k.startswith("concurrent_worker_")}
        
        for worker_name, worker_stat in worker_stats.items():
            assert worker_stat.success_rate == 1.0, f"{worker_name} had failures"
            assert worker_stat.average_duration_ms < 1000, f"{worker_name} too slow"
    
    @pytest.mark.parametrize("cache_size", [4, 8, 16, 32])
    def test_cache_size_performance(self, cache_size):
        """Test performance impact of different cache sizes."""
        
        import os
        from cloakpivot.loaders import clear_all_caches
        
        # Set cache size
        original_cache_size = os.environ.get('ANALYZER_CACHE_SIZE')
        os.environ['ANALYZER_CACHE_SIZE'] = str(cache_size)
        
        try:
            # Clear existing caches
            clear_all_caches()
            
            profiler = PerformanceProfiler()
            
            # Test with multiple different language/config combinations
            configurations = [
                {"language": "en", "min_confidence": 0.5},
                {"language": "en", "min_confidence": 0.7},
                {"language": "en", "min_confidence": 0.9},
            ]
            
            for config in configurations:
                with profiler.measure_operation(f"cache_test_{cache_size}"):
                    analyzer = get_presidio_analyzer(**config)
                    result = analyzer.analyze("John Doe john@email.com")
                    assert len(result) >= 2
            
            stats = profiler.get_operation_stats(f"cache_test_{cache_size}")
            
            # Record results for comparison
            pytest.cache_performance_results = getattr(pytest, 'cache_performance_results', {})
            pytest.cache_performance_results[cache_size] = stats.average_duration_ms
            
        finally:
            # Restore original cache size
            if original_cache_size:
                os.environ['ANALYZER_CACHE_SIZE'] = original_cache_size
            else:
                os.environ.pop('ANALYZER_CACHE_SIZE', None)
```

## Conclusion

This comprehensive guide provides everything needed to optimize CloakPivot performance:

### Key Takeaways

1. **Enable Singleton Pattern**: 75-90% reduction in initialization time
2. **Choose Appropriate Model Size**: Balance accuracy vs. performance based on use case
3. **Use Session Fixtures in Tests**: 50-80% reduction in test suite execution time
4. **Leverage Parallel Processing**: Better CPU utilization for batch operations
5. **Monitor Continuously**: Catch regressions early with automated performance testing

### Performance Targets

- **Entity Detection**: < 100ms average for typical documents
- **Document Masking**: < 500ms for documents up to 100KB
- **Test Suite**: < 2 minutes in fast mode, < 5 minutes comprehensive
- **Memory Usage**: < 500MB for typical operations
- **Cache Hit Rate**: > 80% for analyzer caches

### Next Steps

1. **Implement Monitoring**: Set up performance dashboards for your environment
2. **Tune Configuration**: Adjust cache sizes and worker counts for your workload
3. **Profile Your Code**: Use the PerformanceProfiler to identify bottlenecks
4. **Automate Testing**: Include performance tests in your CI/CD pipeline

### Additional Resources

- [CloakPivot Performance Tests](tests/performance/) - Comprehensive benchmark suite
- [Performance Scripts](scripts/) - Analysis and monitoring tools
- [Configuration Documentation](cloakpivot/core/config.py) - All configuration options
- [GitHub Performance Monitoring](.github/workflows/performance-monitoring.yml) - CI/CD integration

For additional help with performance optimization, see the troubleshooting section above or check the existing performance test suite for practical examples.

---

**Performance Monitoring**: This documentation is continuously updated based on performance test results and user feedback. Last updated: 2025-08-31