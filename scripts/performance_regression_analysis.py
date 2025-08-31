#!/usr/bin/env python3
"""Performance regression analysis script for CloakPivot CI/CD pipeline.

This script analyzes pytest-benchmark JSON output to detect performance regressions
between baseline and current performance measurements, with configurable thresholds
and statistical significance testing to minimize false positives.
"""

import argparse
import json
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class PerformanceBenchmark:
    """Performance benchmark data point from pytest-benchmark."""
    
    name: str
    mean: float
    stddev: float
    min: float
    max: float
    rounds: int
    
    @classmethod
    def from_pytest_benchmark(cls, benchmark_data: dict) -> 'PerformanceBenchmark':
        """Create from pytest-benchmark JSON format."""
        stats = benchmark_data['stats']
        return cls(
            name=benchmark_data['name'],
            mean=stats['mean'],
            stddev=stats['stddev'],
            min=stats['min'],
            max=stats['max'],
            rounds=stats['rounds']
        )


class PerformanceRegressionAnalyzer:
    """Analyze performance regressions between baseline and current measurements."""
    
    # Performance categories with different thresholds
    PERFORMANCE_CATEGORIES = {
        "critical": {
            "threshold": 0.05,  # 5% threshold
            "patterns": ["analyzer_initialization", "core_entity_detection", "regression_baseline"]
        },
        "important": {
            "threshold": 0.10,  # 10% threshold  
            "patterns": ["pipeline_creation", "document_processing", "masking_engine", "round_trip"]
        },
        "general": {
            "threshold": 0.20,  # 20% threshold
            "patterns": ["batch_processing", "memory_usage", "concurrent", "strategy"]
        }
    }
    
    def __init__(self, regression_threshold: float = 0.10):
        """Initialize analyzer with configurable threshold.
        
        Args:
            regression_threshold: Default threshold for regression detection (default: 10%)
        """
        self.default_threshold = regression_threshold
        
    def _get_threshold_for_benchmark(self, benchmark_name: str) -> float:
        """Get appropriate threshold based on benchmark category."""
        for category, config in self.PERFORMANCE_CATEGORIES.items():
            for pattern in config["patterns"]:
                if pattern in benchmark_name.lower():
                    return config["threshold"]
        return self.default_threshold
        
    def load_benchmarks(self, filepath: Path) -> Dict[str, PerformanceBenchmark]:
        """Load benchmarks from pytest-benchmark JSON."""
        if not filepath.exists():
            raise FileNotFoundError(f"Benchmark file not found: {filepath}")
            
        with open(filepath) as f:
            data = json.load(f)
        
        if 'benchmarks' not in data:
            raise ValueError(f"Invalid benchmark file format: missing 'benchmarks' key")
            
        benchmarks = {}
        for benchmark in data['benchmarks']:
            try:
                perf_bench = PerformanceBenchmark.from_pytest_benchmark(benchmark)
                benchmarks[perf_bench.name] = perf_bench
            except KeyError as e:
                print(f"Warning: Skipping malformed benchmark: {e}", file=sys.stderr)
                continue
                
        return benchmarks
    
    def compare_benchmarks(self, baseline: Dict[str, PerformanceBenchmark], 
                          current: Dict[str, PerformanceBenchmark]) -> Dict[str, dict]:
        """Compare baseline vs current performance with adaptive thresholds."""
        results = {}
        
        for name, current_bench in current.items():
            if name not in baseline:
                results[name] = {
                    'status': 'new',
                    'current_mean': current_bench.mean,
                    'message': 'New benchmark - no baseline comparison available'
                }
                continue
                
            baseline_bench = baseline[name]
            
            # Calculate percentage change
            if baseline_bench.mean > 0:
                pct_change = (current_bench.mean - baseline_bench.mean) / baseline_bench.mean
            else:
                pct_change = float('inf') if current_bench.mean > 0 else 0
            
            # Get adaptive threshold for this benchmark
            threshold = self._get_threshold_for_benchmark(name)
            
            # Determine status based on adaptive threshold
            if abs(pct_change) < threshold:
                status = 'stable'
            elif pct_change > threshold:
                status = 'regression'
            else:
                status = 'improvement'
            
            results[name] = {
                'status': status,
                'baseline_mean': baseline_bench.mean,
                'current_mean': current_bench.mean,
                'change_pct': pct_change * 100,
                'change_abs': current_bench.mean - baseline_bench.mean,
                'threshold_used': threshold * 100,  # Convert to percentage
                'significance': self._calculate_significance(baseline_bench, current_bench)
            }
            
        # Check for missing benchmarks in current run
        for name, baseline_bench in baseline.items():
            if name not in current:
                results[name] = {
                    'status': 'missing',
                    'baseline_mean': baseline_bench.mean,
                    'current_mean': None,
                    'message': 'Benchmark missing from current run'
                }
            
        return results
    
    def _calculate_significance(self, baseline: PerformanceBenchmark, 
                              current: PerformanceBenchmark) -> str:
        """Calculate statistical significance of change using z-score."""
        if baseline.stddev == 0 and current.stddev == 0:
            return 'uncertain'
            
        # Use pooled standard deviation for better significance testing
        pooled_variance = ((baseline.stddev ** 2) + (current.stddev ** 2)) / 2
        pooled_stddev = pooled_variance ** 0.5
        
        if pooled_stddev == 0:
            return 'uncertain'
        
        change_abs = abs(current.mean - baseline.mean)
        z_score = change_abs / pooled_stddev
        
        if z_score > 2.58:  # 99% confidence
            return 'highly_significant'
        elif z_score > 1.96:  # 95% confidence
            return 'significant'
        elif z_score > 1.0:  # Lower confidence
            return 'likely'
        else:
            return 'uncertain'
    
    def generate_report(self, comparison_results: Dict[str, dict]) -> str:
        """Generate comprehensive markdown report of performance analysis."""
        report_lines = []
        
        # Summary statistics
        total_benchmarks = len(comparison_results)
        regressions = sum(1 for r in comparison_results.values() if r['status'] == 'regression')
        improvements = sum(1 for r in comparison_results.values() if r['status'] == 'improvement')
        stable = sum(1 for r in comparison_results.values() if r['status'] == 'stable')
        new = sum(1 for r in comparison_results.values() if r['status'] == 'new')
        missing = sum(1 for r in comparison_results.values() if r['status'] == 'missing')
        
        report_lines.append("### Performance Summary")
        report_lines.append(f"- **Total Benchmarks**: {total_benchmarks}")
        report_lines.append(f"- **Regressions**: {regressions} {'🔴' if regressions > 0 else '⚪'}")
        report_lines.append(f"- **Improvements**: {improvements} {'🟢' if improvements > 0 else '⚪'}") 
        report_lines.append(f"- **Stable**: {stable} ⚪")
        report_lines.append(f"- **New**: {new} {'🆕' if new > 0 else '⚪'}")
        if missing > 0:
            report_lines.append(f"- **Missing**: {missing} ⚠️")
        report_lines.append("")
        
        # Regression details with severity classification
        if regressions > 0:
            report_lines.append("### 🔴 Performance Regressions")
            report_lines.append("| Benchmark | Baseline (s) | Current (s) | Change | Threshold | Significance |")
            report_lines.append("|-----------|--------------|-------------|---------|-----------|--------------|")
            
            # Sort regressions by severity (percentage change)
            regression_items = [
                (name, result) for name, result in comparison_results.items()
                if result['status'] == 'regression'
            ]
            regression_items.sort(key=lambda x: x[1]['change_pct'], reverse=True)
            
            for name, result in regression_items:
                severity_emoji = "🚨" if result['change_pct'] > 50 else "⚠️"
                
                report_lines.append(
                    f"| {severity_emoji} {name} | {result['baseline_mean']:.4f} | "
                    f"{result['current_mean']:.4f} | "
                    f"+{result['change_pct']:.1f}% | "
                    f"{result['threshold_used']:.0f}% | "
                    f"{result['significance'].replace('_', ' ').title()} |"
                )
            report_lines.append("")
        
        # Improvements
        if improvements > 0:
            report_lines.append("### 🟢 Performance Improvements")
            report_lines.append("| Benchmark | Baseline (s) | Current (s) | Change | Threshold | Significance |")
            report_lines.append("|-----------|--------------|-------------|---------|-----------|--------------|")
            
            # Sort improvements by magnitude
            improvement_items = [
                (name, result) for name, result in comparison_results.items()
                if result['status'] == 'improvement'
            ]
            improvement_items.sort(key=lambda x: abs(x[1]['change_pct']), reverse=True)
            
            for name, result in improvement_items:
                report_lines.append(
                    f"| {name} | {result['baseline_mean']:.4f} | "
                    f"{result['current_mean']:.4f} | "
                    f"{result['change_pct']:.1f}% | "
                    f"{result['threshold_used']:.0f}% | "
                    f"{result['significance'].replace('_', ' ').title()} |"
                )
            report_lines.append("")
        
        # Missing benchmarks warning
        if missing > 0:
            report_lines.append("### ⚠️ Missing Benchmarks")
            missing_items = [
                name for name, result in comparison_results.items()
                if result['status'] == 'missing'
            ]
            for name in sorted(missing_items):
                report_lines.append(f"- {name}")
            report_lines.append("")
        
        # New benchmarks info
        if new > 0:
            report_lines.append("### 🆕 New Benchmarks")
            new_items = [
                (name, result) for name, result in comparison_results.items()
                if result['status'] == 'new'
            ]
            for name, result in sorted(new_items):
                report_lines.append(f"- {name}: {result['current_mean']:.4f}s")
            report_lines.append("")
        
        # Overall assessment with specific recommendations
        if regressions > 0:
            severe_regressions = sum(
                1 for r in comparison_results.values() 
                if r['status'] == 'regression' and r['change_pct'] > 50
            )
            
            report_lines.append("### ⚠️ Recommendation")
            if severe_regressions > 0:
                report_lines.append(
                    f"This PR introduces **{regressions} performance regression(s)**, "
                    f"including **{severe_regressions} severe regression(s)** (>50% slowdown). "
                    "Please review the changes carefully and consider optimization before merging."
                )
            else:
                report_lines.append(
                    f"This PR introduces **{regressions} performance regression(s)**. "
                    "Please review the changes and consider optimization before merging."
                )
        elif improvements > 0:
            report_lines.append("### ✅ Recommendation") 
            report_lines.append(
                f"This PR shows **{improvements} performance improvement(s)**! "
                "Great work on optimization. 🚀"
            )
        else:
            report_lines.append("### ✅ Recommendation")
            report_lines.append("No significant performance changes detected. Safe to merge from performance perspective.")
        
        # Add methodology note
        report_lines.append("")
        report_lines.append("### 📊 Analysis Details")
        report_lines.append("- **Thresholds**: Critical (5%), Important (10%), General (20%)")
        report_lines.append("- **Significance**: Statistical analysis using pooled standard deviation")
        report_lines.append("- **Categories**: Adaptive thresholds based on benchmark importance")
        
        return "\n".join(report_lines)


def main():
    """Main entry point for performance regression analysis."""
    parser = argparse.ArgumentParser(
        description="Analyze performance regressions between baseline and current benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --baseline baseline.json --current current.json
  %(prog)s --baseline baseline.json --current current.json --threshold 0.15 --output report.md
  %(prog)s --baseline baseline.json --current current.json --json-output
        """
    )
    parser.add_argument(
        "--baseline", 
        required=True, 
        type=Path,
        help="Baseline performance JSON file from pytest-benchmark"
    )
    parser.add_argument(
        "--current", 
        required=True,
        type=Path,
        help="Current performance JSON file from pytest-benchmark"
    )
    parser.add_argument(
        "--threshold", 
        type=float, 
        default=0.10, 
        help="Default regression threshold (default: 0.10 = 10%%)"
    )
    parser.add_argument(
        "--output", 
        type=Path,
        help="Output report file (default: stdout)"
    )
    parser.add_argument(
        "--json-output",
        action="store_true",
        help="Output results in JSON format instead of markdown"
    )
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with error code if regressions are detected"
    )
    
    args = parser.parse_args()
    
    try:
        analyzer = PerformanceRegressionAnalyzer(regression_threshold=args.threshold)
        
        # Load benchmark data
        print(f"Loading baseline data from {args.baseline}...", file=sys.stderr)
        baseline_benchmarks = analyzer.load_benchmarks(args.baseline)
        print(f"Loaded {len(baseline_benchmarks)} baseline benchmarks", file=sys.stderr)
        
        print(f"Loading current data from {args.current}...", file=sys.stderr)
        current_benchmarks = analyzer.load_benchmarks(args.current)
        print(f"Loaded {len(current_benchmarks)} current benchmarks", file=sys.stderr)
        
        # Perform comparison
        comparison_results = analyzer.compare_benchmarks(baseline_benchmarks, current_benchmarks)
        
        # Generate output
        if args.json_output:
            output = json.dumps(comparison_results, indent=2)
        else:
            output = analyzer.generate_report(comparison_results)
        
        # Write output
        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Report written to {args.output}", file=sys.stderr)
        else:
            print(output)
        
        # Check for regressions and exit with appropriate code
        regressions = sum(1 for r in comparison_results.values() if r['status'] == 'regression')
        if regressions > 0:
            print(f"⚠️ {regressions} performance regression(s) detected!", file=sys.stderr)
            if args.fail_on_regression:
                sys.exit(1)
        else:
            print("✅ No performance regressions detected", file=sys.stderr)
            
        return 0
            
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Skipping performance regression analysis - baseline not available", file=sys.stderr)
        return 0  # Don't fail CI if baseline is missing
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error parsing benchmark data: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n⏹ Analysis cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())