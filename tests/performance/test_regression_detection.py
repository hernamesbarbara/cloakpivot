"""Tests for performance regression detection functionality.

This module tests the performance regression analysis scripts and infrastructure
to ensure reliable detection of performance changes in CI/CD pipelines.
"""

import json
import tempfile
from pathlib import Path
from typing import Dict

import pytest

from scripts.performance_regression_analysis import (
    PerformanceBenchmark,
    PerformanceRegressionAnalyzer,
)


class TestPerformanceBenchmark:
    """Test the PerformanceBenchmark data class."""

    def test_from_pytest_benchmark_valid_data(self):
        """Test creating PerformanceBenchmark from valid pytest-benchmark data."""
        benchmark_data = {
            'name': 'test_benchmark',
            'stats': {
                'mean': 0.12345,
                'stddev': 0.01234,
                'min': 0.11000,
                'max': 0.14000,
                'rounds': 10
            }
        }
        
        benchmark = PerformanceBenchmark.from_pytest_benchmark(benchmark_data)
        
        assert benchmark.name == 'test_benchmark'
        assert benchmark.mean == 0.12345
        assert benchmark.stddev == 0.01234
        assert benchmark.min == 0.11000
        assert benchmark.max == 0.14000
        assert benchmark.rounds == 10

    def test_from_pytest_benchmark_missing_stats(self):
        """Test handling of malformed benchmark data."""
        benchmark_data = {
            'name': 'test_benchmark'
            # Missing 'stats' key
        }
        
        with pytest.raises(KeyError):
            PerformanceBenchmark.from_pytest_benchmark(benchmark_data)


class TestPerformanceRegressionAnalyzer:
    """Test the performance regression analysis functionality."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer with default threshold."""
        return PerformanceRegressionAnalyzer()

    @pytest.fixture
    def sample_baseline_data(self):
        """Sample baseline performance data."""
        return {
            'benchmarks': [
                {
                    'name': 'test_small_document_performance',
                    'stats': {
                        'mean': 0.100,
                        'stddev': 0.010,
                        'min': 0.090,
                        'max': 0.110,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_regression_baseline',
                    'stats': {
                        'mean': 0.200,
                        'stddev': 0.020,
                        'min': 0.180,
                        'max': 0.220,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_batch_processing',
                    'stats': {
                        'mean': 0.500,
                        'stddev': 0.050,
                        'min': 0.450,
                        'max': 0.550,
                        'rounds': 5
                    }
                }
            ]
        }

    @pytest.fixture
    def sample_current_data_stable(self):
        """Sample current data showing stable performance."""
        return {
            'benchmarks': [
                {
                    'name': 'test_small_document_performance',
                    'stats': {
                        'mean': 0.105,  # 5% increase - within threshold
                        'stddev': 0.011,
                        'min': 0.094,
                        'max': 0.116,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_regression_baseline',
                    'stats': {
                        'mean': 0.185,  # 7.5% improvement (exceeds 5% critical threshold)
                        'stddev': 0.018,
                        'min': 0.167,
                        'max': 0.203,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_batch_processing',
                    'stats': {
                        'mean': 0.510,  # 2% increase - within general threshold
                        'stddev': 0.051,
                        'min': 0.459,
                        'max': 0.561,
                        'rounds': 5
                    }
                }
            ]
        }

    @pytest.fixture 
    def sample_current_data_regression(self):
        """Sample current data showing performance regressions."""
        return {
            'benchmarks': [
                {
                    'name': 'test_small_document_performance',
                    'stats': {
                        'mean': 0.125,  # 25% increase - regression
                        'stddev': 0.012,
                        'min': 0.113,
                        'max': 0.137,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_regression_baseline',
                    'stats': {
                        'mean': 0.220,  # 10% increase - regression for critical
                        'stddev': 0.022,
                        'min': 0.198,
                        'max': 0.242,
                        'rounds': 5
                    }
                },
                {
                    'name': 'test_new_benchmark',  # New benchmark
                    'stats': {
                        'mean': 0.300,
                        'stddev': 0.030,
                        'min': 0.270,
                        'max': 0.330,
                        'rounds': 5
                    }
                }
            ]
        }

    def test_adaptive_thresholds(self, analyzer):
        """Test that adaptive thresholds work correctly."""
        # Critical benchmark should use 5% threshold
        critical_threshold = analyzer._get_threshold_for_benchmark('test_regression_baseline')
        assert critical_threshold == 0.05
        
        # Important benchmark should use 10% threshold
        important_threshold = analyzer._get_threshold_for_benchmark('test_round_trip_performance')
        assert important_threshold == 0.10
        
        # General benchmark should use 20% threshold
        general_threshold = analyzer._get_threshold_for_benchmark('test_batch_processing')
        assert general_threshold == 0.20
        
        # Unknown benchmark should use default threshold
        default_threshold = analyzer._get_threshold_for_benchmark('test_unknown_benchmark')
        assert default_threshold == 0.10  # Default threshold

    def test_load_benchmarks_valid_file(self, analyzer, sample_baseline_data):
        """Test loading valid benchmark file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_baseline_data, f)
            temp_file = Path(f.name)
        
        try:
            benchmarks = analyzer.load_benchmarks(temp_file)
            
            assert len(benchmarks) == 3
            assert 'test_small_document_performance' in benchmarks
            assert benchmarks['test_small_document_performance'].mean == 0.100
        finally:
            temp_file.unlink()

    def test_load_benchmarks_missing_file(self, analyzer):
        """Test loading non-existent benchmark file."""
        with pytest.raises(FileNotFoundError):
            analyzer.load_benchmarks(Path('/nonexistent/file.json'))

    def test_load_benchmarks_invalid_format(self, analyzer):
        """Test loading file with invalid format."""
        invalid_data = {'invalid': 'format'}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(invalid_data, f)
            temp_file = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="missing 'benchmarks' key"):
                analyzer.load_benchmarks(temp_file)
        finally:
            temp_file.unlink()

    def test_compare_benchmarks_stable_performance(self, analyzer, sample_baseline_data, sample_current_data_stable):
        """Test comparison showing stable performance."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_baseline_data, f)
            baseline_file = Path(f.name)
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_current_data_stable, f)
            current_file = Path(f.name)

        try:
            baseline = analyzer.load_benchmarks(baseline_file)
            current = analyzer.load_benchmarks(current_file)
            
            results = analyzer.compare_benchmarks(baseline, current)
            
            # All should be stable or improvements
            assert results['test_small_document_performance']['status'] == 'stable'
            assert results['test_regression_baseline']['status'] == 'improvement'
            assert results['test_batch_processing']['status'] == 'stable'
            
        finally:
            baseline_file.unlink()
            current_file.unlink()

    def test_compare_benchmarks_with_regressions(self, analyzer, sample_baseline_data, sample_current_data_regression):
        """Test comparison showing regressions."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_baseline_data, f)
            baseline_file = Path(f.name)
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(sample_current_data_regression, f)
            current_file = Path(f.name)

        try:
            baseline = analyzer.load_benchmarks(baseline_file)
            current = analyzer.load_benchmarks(current_file)
            
            results = analyzer.compare_benchmarks(baseline, current)
            
            # Should detect regressions based on adaptive thresholds
            assert results['test_small_document_performance']['status'] == 'regression'
            assert results['test_regression_baseline']['status'] == 'regression'  # Critical threshold exceeded
            assert results['test_new_benchmark']['status'] == 'new'
            
            # Check missing benchmark
            assert 'test_batch_processing' in results
            assert results['test_batch_processing']['status'] == 'missing'
            
        finally:
            baseline_file.unlink()
            current_file.unlink()

    def test_statistical_significance_calculation(self, analyzer):
        """Test statistical significance calculations."""
        baseline = PerformanceBenchmark('test', 0.100, 0.010, 0.090, 0.110, 5)
        
        # Highly significant change
        current_high_sig = PerformanceBenchmark('test', 0.130, 0.005, 0.125, 0.135, 5)
        sig = analyzer._calculate_significance(baseline, current_high_sig)
        assert sig in ['significant', 'highly_significant']
        
        # Low significance change
        current_low_sig = PerformanceBenchmark('test', 0.102, 0.010, 0.092, 0.112, 5)
        sig = analyzer._calculate_significance(baseline, current_low_sig)
        assert sig in ['uncertain', 'likely']

    def test_generate_report_no_regressions(self, analyzer):
        """Test report generation with no regressions."""
        comparison_results = {
            'benchmark1': {'status': 'stable', 'change_pct': 2.0},
            'benchmark2': {'status': 'improvement', 'change_pct': -5.0, 'baseline_mean': 0.100, 
                          'current_mean': 0.095, 'threshold_used': 10.0, 'significance': 'likely'},
            'benchmark3': {'status': 'stable', 'change_pct': 1.0}
        }
        
        report = analyzer.generate_report(comparison_results)
        
        assert "Performance Summary" in report
        assert "**Regressions**: 0" in report
        assert "**Improvements**: 1" in report
        assert "Great work on optimization" in report

    def test_generate_report_with_regressions(self, analyzer):
        """Test report generation with regressions."""
        comparison_results = {
            'critical_benchmark': {
                'status': 'regression',
                'change_pct': 15.0,
                'baseline_mean': 0.100,
                'current_mean': 0.115,
                'threshold_used': 5.0,
                'significance': 'significant'
            },
            'stable_benchmark': {
                'status': 'stable',
                'change_pct': 2.0
            }
        }
        
        report = analyzer.generate_report(comparison_results)
        
        assert "Performance Summary" in report
        assert "**Regressions**: 1" in report
        assert "🔴 Performance Regressions" in report
        assert "critical_benchmark" in report
        assert "consider optimization" in report

    def test_generate_report_severe_regressions(self, analyzer):
        """Test report generation with severe regressions."""
        comparison_results = {
            'severe_regression': {
                'status': 'regression',
                'change_pct': 75.0,  # Severe regression
                'baseline_mean': 0.100,
                'current_mean': 0.175,
                'threshold_used': 10.0,
                'significance': 'highly_significant'
            }
        }
        
        report = analyzer.generate_report(comparison_results)
        
        assert "🚨" in report  # Severe regression emoji
        assert "severe regression(s)" in report

    def test_missing_and_new_benchmarks_in_report(self, analyzer):
        """Test report handling of missing and new benchmarks."""
        comparison_results = {
            'missing_benchmark': {
                'status': 'missing',
                'baseline_mean': 0.100,
                'current_mean': None,
                'message': 'Benchmark missing from current run'
            },
            'new_benchmark': {
                'status': 'new',
                'current_mean': 0.200,
                'message': 'New benchmark - no baseline comparison available'
            }
        }
        
        report = analyzer.generate_report(comparison_results)
        
        assert "**Missing**: 1" in report
        assert "**New**: 1" in report
        assert "⚠️ Missing Benchmarks" in report
        assert "🆕 New Benchmarks" in report
        assert "missing_benchmark" in report
        assert "new_benchmark" in report


@pytest.mark.integration
class TestRegressionDetectionIntegration:
    """Integration tests for the complete regression detection pipeline."""

    def test_end_to_end_analysis_no_regression(self):
        """Test complete analysis pipeline with no regressions."""
        analyzer = PerformanceRegressionAnalyzer()
        
        # Create sample data files
        baseline_data = {
            'benchmarks': [
                {
                    'name': 'test_performance',
                    'stats': {'mean': 0.100, 'stddev': 0.010, 'min': 0.090, 'max': 0.110, 'rounds': 5}
                }
            ]
        }
        
        current_data = {
            'benchmarks': [
                {
                    'name': 'test_performance', 
                    'stats': {'mean': 0.105, 'stddev': 0.011, 'min': 0.094, 'max': 0.116, 'rounds': 5}
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / 'baseline.json'
            current_file = Path(temp_dir) / 'current.json'
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(current_file, 'w') as f:
                json.dump(current_data, f)
            
            # Load and compare
            baseline_benchmarks = analyzer.load_benchmarks(baseline_file)
            current_benchmarks = analyzer.load_benchmarks(current_file)
            
            results = analyzer.compare_benchmarks(baseline_benchmarks, current_benchmarks)
            report = analyzer.generate_report(results)
            
            assert 'test_performance' in results
            assert results['test_performance']['status'] == 'stable'
            assert "Safe to merge" in report

    def test_end_to_end_analysis_with_regression(self):
        """Test complete analysis pipeline with regression detected."""
        analyzer = PerformanceRegressionAnalyzer()
        
        # Create sample data with regression
        baseline_data = {
            'benchmarks': [
                {
                    'name': 'test_critical_performance',
                    'stats': {'mean': 0.100, 'stddev': 0.010, 'min': 0.090, 'max': 0.110, 'rounds': 5}
                }
            ]
        }
        
        current_data = {
            'benchmarks': [
                {
                    'name': 'test_critical_performance',
                    'stats': {'mean': 0.120, 'stddev': 0.012, 'min': 0.108, 'max': 0.132, 'rounds': 5}  # 20% increase
                }
            ]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            baseline_file = Path(temp_dir) / 'baseline.json'
            current_file = Path(temp_dir) / 'current.json'
            
            with open(baseline_file, 'w') as f:
                json.dump(baseline_data, f)
            with open(current_file, 'w') as f:
                json.dump(current_data, f)
            
            # Load and compare
            baseline_benchmarks = analyzer.load_benchmarks(baseline_file)
            current_benchmarks = analyzer.load_benchmarks(current_file)
            
            results = analyzer.compare_benchmarks(baseline_benchmarks, current_benchmarks)
            report = analyzer.generate_report(results)
            
            assert 'test_critical_performance' in results
            assert results['test_critical_performance']['status'] == 'regression'
            assert "performance regression(s)" in report
            assert "consider optimization" in report


@pytest.mark.slow
class TestPerformanceRegressionDetectionComprehensive:
    """Comprehensive tests for performance regression detection with real benchmark data."""

    @pytest.mark.performance
    def test_realistic_benchmark_thresholds(self):
        """Test with realistic benchmark data to verify thresholds."""
        analyzer = PerformanceRegressionAnalyzer()
        
        # Test various benchmark categories
        test_cases = [
            ('test_analyzer_initialization', 0.04, 'stable'),  # Under critical threshold
            ('test_analyzer_initialization', 0.06, 'regression'),  # Over critical threshold
            ('test_round_trip_performance', 0.09, 'stable'),  # Under important threshold
            ('test_round_trip_performance', 0.11, 'regression'),  # Over important threshold
            ('test_batch_processing', 0.15, 'stable'),  # Under general threshold
            ('test_batch_processing', 0.25, 'regression'),  # Over general threshold
        ]
        
        for bench_name, change_ratio, expected_status in test_cases:
            baseline_mean = 0.100
            current_mean = baseline_mean * (1 + change_ratio)
            
            baseline = PerformanceBenchmark(bench_name, baseline_mean, 0.010, 0.090, 0.110, 5)
            current = PerformanceBenchmark(bench_name, current_mean, 0.010, 0.090, current_mean + 0.020, 5)
            
            baseline_dict = {bench_name: baseline}
            current_dict = {bench_name: current}
            
            results = analyzer.compare_benchmarks(baseline_dict, current_dict)
            
            assert results[bench_name]['status'] == expected_status, (
                f"Benchmark {bench_name} with {change_ratio*100:.0f}% change "
                f"should be {expected_status}, got {results[bench_name]['status']}"
            )