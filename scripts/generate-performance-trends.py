#!/usr/bin/env python3
"""Performance trends generation script for CloakPivot CI/CD pipeline.

This script processes historical performance data to generate trend analysis,
charts, and insights for long-term performance monitoring and optimization.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend for CI
    import matplotlib.pyplot as plt

    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print(
        "Warning: matplotlib not available, chart generation disabled", file=sys.stderr
    )


class PerformanceTrendAnalyzer:
    """Analyze performance trends from historical benchmark data."""

    def __init__(self, output_dir: Path):
        """Initialize analyzer with output directory.

        Args:
            output_dir: Directory to store trend analysis outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_performance_data(self, performance_file: Path) -> dict:
        """Load performance data from pytest-benchmark JSON file."""
        if not performance_file.exists():
            raise FileNotFoundError(
                f"Performance data file not found: {performance_file}"
            )

        with open(performance_file) as f:
            data = json.load(f)

        if "benchmarks" not in data:
            raise ValueError(
                "Invalid performance file format: missing 'benchmarks' key"
            )

        return data

    def extract_benchmark_metrics(self, performance_data: dict) -> dict[str, dict]:
        """Extract key metrics from benchmark data."""
        metrics = {}

        for benchmark in performance_data["benchmarks"]:
            name = benchmark["name"]
            stats = benchmark["stats"]

            metrics[name] = {
                "mean": stats["mean"],
                "stddev": stats["stddev"],
                "min": stats["min"],
                "max": stats["max"],
                "rounds": stats["rounds"],
                "timestamp": datetime.now().isoformat(),
            }

        return metrics

    def load_historical_data(
        self, history_dir: Path
    ) -> list[tuple[datetime, dict[str, dict]]]:
        """Load historical performance data from directory structure.

        Expected structure: history_dir/YYYYMMDD/performance-data.json
        """
        historical_data = []

        if not history_dir.exists():
            print(
                f"Warning: History directory not found: {history_dir}", file=sys.stderr
            )
            return historical_data

        for date_dir in sorted(history_dir.iterdir()):
            if not date_dir.is_dir():
                continue

            try:
                # Parse date from directory name (YYYYMMDD format)
                date_str = date_dir.name
                date = datetime.strptime(date_str, "%Y%m%d")

                # Look for performance data file
                perf_files = list(date_dir.glob("*performance*.json"))
                if not perf_files:
                    continue

                perf_file = perf_files[0]  # Use first matching file
                perf_data = self.load_performance_data(perf_file)
                metrics = self.extract_benchmark_metrics(perf_data)

                historical_data.append((date, metrics))

            except (ValueError, FileNotFoundError, json.JSONDecodeError) as e:
                print(f"Warning: Skipping {date_dir}: {e}", file=sys.stderr)
                continue

        return historical_data

    def analyze_trends(
        self, historical_data: list[tuple[datetime, dict[str, dict]]]
    ) -> dict:
        """Analyze performance trends from historical data."""
        if len(historical_data) < 2:
            return {"error": "Insufficient historical data for trend analysis"}

        trends = {}
        all_benchmarks = set()

        # Collect all benchmark names
        for _, metrics in historical_data:
            all_benchmarks.update(metrics.keys())

        # Analyze each benchmark
        for benchmark_name in all_benchmarks:
            benchmark_trends = {"data_points": [], "mean_values": [], "dates": []}

            # Extract time series for this benchmark
            for date, metrics in historical_data:
                if benchmark_name in metrics:
                    benchmark_data = metrics[benchmark_name]
                    benchmark_trends["dates"].append(date)
                    benchmark_trends["mean_values"].append(benchmark_data["mean"])
                    benchmark_trends["data_points"].append(benchmark_data)

            if len(benchmark_trends["mean_values"]) < 2:
                continue  # Skip benchmarks with insufficient data

            # Calculate trend statistics
            values = benchmark_trends["mean_values"]

            # Simple linear trend (slope calculation)
            n = len(values)
            x_vals = list(range(n))

            # Calculate slope using least squares
            sum_x = sum(x_vals)
            sum_y = sum(values)
            sum_xy = sum(x * y for x, y in zip(x_vals, values))
            sum_x2 = sum(x * x for x in x_vals)

            if n * sum_x2 - sum_x * sum_x != 0:
                slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)
            else:
                slope = 0

            # Trend classification
            recent_mean = sum(values[-3:]) / min(3, len(values))
            early_mean = sum(values[:3]) / min(3, len(values))

            change_pct = (
                ((recent_mean - early_mean) / early_mean * 100) if early_mean > 0 else 0
            )

            if abs(change_pct) < 5:
                trend_status = "stable"
            elif change_pct > 5:
                trend_status = "degrading"
            else:
                trend_status = "improving"

            benchmark_trends.update(
                {
                    "slope": slope,
                    "change_percent": change_pct,
                    "trend_status": trend_status,
                    "recent_mean": recent_mean,
                    "early_mean": early_mean,
                    "volatility": max(values) - min(values) if values else 0,
                }
            )

            trends[benchmark_name] = benchmark_trends

        return trends

    def generate_trend_charts(self, trends: dict) -> list[Path]:
        """Generate trend charts for key performance metrics."""
        if not MATPLOTLIB_AVAILABLE:
            print(
                "Skipping chart generation: matplotlib not available", file=sys.stderr
            )
            return []

        chart_files = []

        # Overall performance overview chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle("CloakPivot Performance Trends", fontsize=16)

        # Select key benchmarks for overview
        key_benchmarks = []
        for name in trends.keys():
            if any(
                keyword in name.lower()
                for keyword in [
                    "regression_baseline",
                    "small_document",
                    "round_trip",
                    "batch",
                ]
            ):
                key_benchmarks.append(name)

        key_benchmarks = key_benchmarks[:4]  # Limit to 4 for overview

        for idx, benchmark_name in enumerate(key_benchmarks):
            if idx >= 4:
                break

            ax = axes[idx // 2, idx % 2]
            trend_data = trends[benchmark_name]

            dates = trend_data["dates"]
            values = trend_data["mean_values"]

            ax.plot(dates, values, "o-", linewidth=2, markersize=4)
            ax.set_title(f"{benchmark_name}", fontsize=10)
            ax.set_ylabel("Time (seconds)")
            ax.tick_params(axis="x", rotation=45)

            # Add trend line
            if len(values) > 1:
                x_numeric = list(range(len(values)))
                z = np.polyfit(x_numeric, values, 1)
                p = np.poly1d(z)
                ax.plot(dates, p(x_numeric), "r--", alpha=0.8)

        plt.tight_layout()
        overview_file = self.output_dir / "performance-overview.png"
        plt.savefig(overview_file, dpi=150, bbox_inches="tight")
        plt.close()
        chart_files.append(overview_file)

        # Individual detailed charts for benchmarks with significant trends
        significant_trends = [
            (name, data)
            for name, data in trends.items()
            if abs(data.get("change_percent", 0)) > 10
            or data.get("trend_status") != "stable"
        ]

        for benchmark_name, trend_data in significant_trends[
            :6
        ]:  # Limit to avoid too many files
            fig, ax = plt.subplots(figsize=(10, 6))

            dates = trend_data["dates"]
            values = trend_data["mean_values"]

            ax.plot(dates, values, "o-", linewidth=2, markersize=6)
            ax.set_title(f"Performance Trend: {benchmark_name}")
            ax.set_xlabel("Date")
            ax.set_ylabel("Time (seconds)")

            # Add trend info
            status = trend_data["trend_status"]
            change_pct = trend_data["change_percent"]

            color = {"improving": "green", "degrading": "red", "stable": "blue"}[status]
            ax.text(
                0.02,
                0.98,
                f"Status: {status.title()}\nChange: {change_pct:+.1f}%",
                transform=ax.transAxes,
                verticalalignment="top",
                bbox={"boxstyle": "round", "facecolor": color, "alpha": 0.1},
            )

            plt.xticks(rotation=45)
            plt.tight_layout()

            chart_file = (
                self.output_dir
                / f'trend-{benchmark_name.replace("/", "_").replace(" ", "_")}.png'
            )
            plt.savefig(chart_file, dpi=150, bbox_inches="tight")
            plt.close()
            chart_files.append(chart_file)

        return chart_files

    def generate_trend_report(
        self, trends: dict, current_data: Optional[dict] = None
    ) -> str:
        """Generate markdown trend report."""
        report_lines = []

        report_lines.append("# Performance Trends Analysis")
        report_lines.append(
            f"*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        )
        report_lines.append("")

        if not trends or "error" in trends:
            report_lines.append("⚠️ **Insufficient historical data for trend analysis**")
            report_lines.append("")
            report_lines.append(
                "Trend analysis requires at least 2 historical data points."
            )
            return "\n".join(report_lines)

        # Summary statistics
        total_benchmarks = len(trends)
        improving = sum(
            1 for t in trends.values() if t.get("trend_status") == "improving"
        )
        degrading = sum(
            1 for t in trends.values() if t.get("trend_status") == "degrading"
        )
        stable = sum(1 for t in trends.values() if t.get("trend_status") == "stable")

        report_lines.append("## Summary")
        report_lines.append(f"- **Total Benchmarks Tracked**: {total_benchmarks}")
        report_lines.append(f"- **Improving Performance**: {improving} 🟢")
        report_lines.append(f"- **Degrading Performance**: {degrading} 🔴")
        report_lines.append(f"- **Stable Performance**: {stable} ⚪")
        report_lines.append("")

        # Degrading benchmarks (highest priority)
        if degrading > 0:
            report_lines.append("## 🔴 Degrading Performance Trends")
            report_lines.append(
                "| Benchmark | Change | Trend | Recent Mean | Volatility |"
            )
            report_lines.append(
                "|-----------|---------|-------|-------------|------------|"
            )

            degrading_items = [
                (name, data)
                for name, data in trends.items()
                if data.get("trend_status") == "degrading"
            ]
            degrading_items.sort(
                key=lambda x: x[1].get("change_percent", 0), reverse=True
            )

            for name, data in degrading_items:
                report_lines.append(
                    f"| {name} | +{data['change_percent']:.1f}% | "
                    f"{'📈' if data['slope'] > 0 else '📉'} | "
                    f"{data['recent_mean']:.4f}s | "
                    f"{data['volatility']:.4f}s |"
                )
            report_lines.append("")

        # Improving benchmarks
        if improving > 0:
            report_lines.append("## 🟢 Improving Performance Trends")
            report_lines.append(
                "| Benchmark | Change | Trend | Recent Mean | Volatility |"
            )
            report_lines.append(
                "|-----------|---------|-------|-------------|------------|"
            )

            improving_items = [
                (name, data)
                for name, data in trends.items()
                if data.get("trend_status") == "improving"
            ]
            improving_items.sort(
                key=lambda x: abs(x[1].get("change_percent", 0)), reverse=True
            )

            for name, data in improving_items:
                report_lines.append(
                    f"| {name} | {data['change_percent']:.1f}% | "
                    f"{'📈' if data['slope'] > 0 else '📉'} | "
                    f"{data['recent_mean']:.4f}s | "
                    f"{data['volatility']:.4f}s |"
                )
            report_lines.append("")

        # Most volatile benchmarks
        volatile_items = [
            (name, data)
            for name, data in trends.items()
            if data.get("volatility", 0) > 0
        ]
        volatile_items.sort(key=lambda x: x[1].get("volatility", 0), reverse=True)

        if volatile_items:
            report_lines.append("## 📊 Most Volatile Performance")
            report_lines.append(
                "*High volatility may indicate inconsistent performance or environmental factors*"
            )
            report_lines.append("")
            report_lines.append("| Benchmark | Volatility | Status | Recent Mean |")
            report_lines.append("|-----------|------------|--------|-------------|")

            for name, data in volatile_items[:5]:  # Top 5 most volatile
                report_lines.append(
                    f"| {name} | {data['volatility']:.4f}s | "
                    f"{data['trend_status'].title()} | "
                    f"{data['recent_mean']:.4f}s |"
                )
            report_lines.append("")

        # Recommendations
        report_lines.append("## 📋 Recommendations")

        if degrading > 0:
            report_lines.append(
                "- **Priority**: Investigate degrading performance trends"
            )
            report_lines.append("- Review recent changes that may impact performance")
            report_lines.append(
                "- Consider profiling slow benchmarks for optimization opportunities"
            )

        if improving > 0:
            report_lines.append("- **Great Work**: Performance improvements detected!")
            report_lines.append(
                "- Document optimization techniques for future reference"
            )

        if len(volatile_items) > 3:
            report_lines.append(
                "- **Stability**: High volatility detected in several benchmarks"
            )
            report_lines.append(
                "- Consider increasing benchmark rounds for more stable measurements"
            )
            report_lines.append(
                "- Check for environmental factors affecting consistency"
            )

        if degrading == 0 and improving == 0:
            report_lines.append(
                "- **Status**: Performance appears stable across all metrics"
            )
            report_lines.append("- Continue monitoring for any emerging trends")

        return "\n".join(report_lines)

    def run_analysis(
        self, performance_data: Path, history_dir: Optional[Path] = None
    ) -> dict:
        """Run complete trend analysis."""
        results = {
            "timestamp": datetime.now().isoformat(),
            "charts_generated": [],
            "reports_generated": [],
        }

        try:
            # Load current performance data
            current_data = self.load_performance_data(performance_data)
            results["current_benchmarks"] = len(current_data["benchmarks"])

            # Load historical data if available
            historical_data = []
            if history_dir:
                historical_data = self.load_historical_data(history_dir)

            results["historical_data_points"] = len(historical_data)

            # Add current data to historical data
            current_metrics = self.extract_benchmark_metrics(current_data)
            historical_data.append((datetime.now(), current_metrics))

            # Analyze trends
            trends = self.analyze_trends(historical_data)
            results["trends"] = trends

            # Generate charts
            if MATPLOTLIB_AVAILABLE:
                chart_files = self.generate_trend_charts(trends)
                results["charts_generated"] = [str(f) for f in chart_files]

            # Generate report
            report_content = self.generate_trend_report(trends, current_data)
            report_file = self.output_dir / "performance-trends-report.md"

            with open(report_file, "w") as f:
                f.write(report_content)

            results["reports_generated"].append(str(report_file))

        except Exception as e:
            results["error"] = str(e)

        return results


# Ensure numpy is available for trend line calculations
try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

    # Provide simple fallback for polyfit
    class np:
        @staticmethod
        def polyfit(x, y, deg):
            return [0, sum(y) / len(y)] if y else [0, 0]

        @staticmethod
        def poly1d(coeffs):
            return lambda x: [coeffs[0] * xi + coeffs[1] for xi in x]


def main():
    """Main entry point for performance trends generation."""
    parser = argparse.ArgumentParser(
        description="Generate performance trends analysis from benchmark data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --performance-data current.json --output-dir trends/
  %(prog)s --performance-data current.json --history-dir history/ --output-dir trends/
        """,
    )
    parser.add_argument(
        "--performance-data",
        required=True,
        type=Path,
        help="Current performance data JSON file from pytest-benchmark",
    )
    parser.add_argument(
        "--history-dir",
        type=Path,
        help="Directory containing historical performance data (YYYYMMDD subdirs)",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Output directory for trend analysis results",
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Output results in JSON format"
    )

    args = parser.parse_args()

    try:
        analyzer = PerformanceTrendAnalyzer(args.output_dir)
        results = analyzer.run_analysis(args.performance_data, args.history_dir)

        if args.json_output:
            print(json.dumps(results, indent=2))
        else:
            print("✅ Trend analysis completed", file=sys.stderr)
            print(f"📊 Output directory: {args.output_dir}", file=sys.stderr)

            if results.get("charts_generated"):
                print(
                    f"📈 Charts generated: {len(results['charts_generated'])}",
                    file=sys.stderr,
                )

            if results.get("reports_generated"):
                print(
                    f"📋 Reports generated: {len(results['reports_generated'])}",
                    file=sys.stderr,
                )

            if "error" in results:
                print(f"⚠️ Error: {results['error']}", file=sys.stderr)
                return 1

        return 0

    except KeyboardInterrupt:
        print("\n⏹ Analysis cancelled by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
