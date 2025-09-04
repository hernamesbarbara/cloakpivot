#!/usr/bin/env python3
"""Analyze GitHub Actions cache performance and provide optimization insights.

This script analyzes recent workflow runs to measure cache hit rates, time savings,
and storage efficiency for model caching optimization.
"""

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run 'pip install requests' first.")
    sys.exit(1)


@dataclass
class AnalysisConfig:
    """Configuration parameters for cache performance analysis."""

    # Baseline timing thresholds (in seconds)
    baseline_model_setup_time: int = 300  # 5 minutes baseline without cache
    high_setup_time_threshold: int = 180  # 3 minutes - when to warn about slow setup

    # Cache hit rate thresholds (percentages)
    spacy_cache_target: float = 80.0
    huggingface_cache_target: float = 80.0
    pip_cache_target: float = 90.0
    overall_cache_target: float = 85.0
    excellent_cache_threshold: float = 95.0

    # API and performance settings
    rate_limit_delay: float = 0.1  # Seconds between API calls
    monthly_multiplier: float = 4.3  # Approximate weeks per month

    def validate(self) -> None:
        """Validate configuration parameters."""
        if self.baseline_model_setup_time <= 0:
            raise ValueError("baseline_model_setup_time must be positive")
        if self.high_setup_time_threshold <= 0:
            raise ValueError("high_setup_time_threshold must be positive")
        if not (0 <= self.spacy_cache_target <= 100):
            raise ValueError("spacy_cache_target must be between 0 and 100")
        if not (0 <= self.huggingface_cache_target <= 100):
            raise ValueError("huggingface_cache_target must be between 0 and 100")
        if not (0 <= self.pip_cache_target <= 100):
            raise ValueError("pip_cache_target must be between 0 and 100")
        if not (0 <= self.overall_cache_target <= 100):
            raise ValueError("overall_cache_target must be between 0 and 100")
        if not (0 <= self.excellent_cache_threshold <= 100):
            raise ValueError("excellent_cache_threshold must be between 0 and 100")
        if self.rate_limit_delay < 0:
            raise ValueError("rate_limit_delay must be non-negative")


class CacheAnalyzer:
    """Analyze GitHub Actions cache performance metrics."""

    def __init__(
        self, github_token: str, repo: str, config: Optional[AnalysisConfig] = None
    ):
        self.github_token = github_token
        self.repo = repo
        self.config = config or AnalysisConfig()
        self.config.validate()
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {github_token}",
            "Accept": "application/vnd.github.v3+json",
        }

    def get_workflow_runs(self, workflow_name: str, days: int = 7) -> list[dict]:
        """Get recent workflow runs for analysis."""
        since = datetime.now() - timedelta(days=days)
        url = (
            f"{self.base_url}/repos/{self.repo}/actions/workflows/{workflow_name}/runs"
        )

        params = {
            "status": "completed",
            "created": f">={since.isoformat()}",
            "per_page": 100,
        }

        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            return response.json().get("workflow_runs", [])
        except Exception as e:
            print(f"Error fetching workflow runs: {e}")
            return []

    def get_job_logs(self, run_id: str) -> Optional[str]:
        """Get logs for a specific workflow run."""
        url = f"{self.base_url}/repos/{self.repo}/actions/runs/{run_id}/logs"

        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.text
        except Exception as e:
            print(f"Error fetching logs for run {run_id}: {e}")
            return None

    def parse_cache_metrics_from_logs(self, logs: str) -> dict:
        """Parse cache performance metrics from workflow logs.

        Analyzes GitHub Actions workflow logs to extract cache performance data
        including cache hit/miss status and timing information.

        Expected Log Format Patterns:
            Cache hits: Lines containing "cache hit" (case-insensitive) along with:
                - "cache-spacy": spaCy model cache
                - "cache-huggingface": HuggingFace transformer cache
                - "pip": pip package cache

            Timing: Lines containing "Total time:" followed by seconds:
                - "Model setup complete - Total time: 45.2s"

        Args:
            logs (str): Raw workflow log text from GitHub Actions API

        Returns:
            dict: Cache metrics with the following structure:
                {
                    "spacy_cache_hit": bool,
                    "huggingface_cache_hit": bool,
                    "pip_cache_hit": bool,
                    "model_setup_time": Optional[float],  # seconds
                    "total_job_time": Optional[float],    # seconds
                    "cache_restore_time": Optional[float], # seconds
                    "model_download_time": Optional[float] # seconds
                }

        Examples:
            >>> logs = "cache hit detected for spacy models\\nTotal time: 45.2s"
            >>> metrics = analyzer.parse_cache_metrics_from_logs(logs)
            >>> metrics["spacy_cache_hit"]
            True
            >>> metrics["model_setup_time"]
            45.2
        """
        metrics = {
            "spacy_cache_hit": False,
            "huggingface_cache_hit": False,
            "pip_cache_hit": False,
            "model_setup_time": None,
            "total_job_time": None,
            "cache_restore_time": None,
            "model_download_time": None,
        }

        lines = logs.split("\n")

        for line in lines:
            # Cache hit detection
            line_lower = line.lower()
            if "cache hit detected for spacy" in line_lower:
                metrics["spacy_cache_hit"] = True
            elif "cache hit detected for huggingface" in line_lower:
                metrics["huggingface_cache_hit"] = True
            elif "pip cache hit detected" in line_lower:
                metrics["pip_cache_hit"] = True

            # Time metrics extraction
            if "Model setup complete" in line:
                # Extract time from setup-models.py output
                if "Total time:" in line:
                    try:
                        time_part = line.split("Total time:")[1].split("s")[0].strip()
                        metrics["model_setup_time"] = float(time_part)
                    except (IndexError, ValueError):
                        pass

            # Extract cache restore times
            if "Restoring cache" in line or "Cache restored" in line:
                # Look for timing information
                pass

        return metrics

    def analyze_runs(self, runs: list[dict]) -> dict:
        """Analyze multiple workflow runs for cache performance."""
        total_runs = len(runs)
        if total_runs == 0:
            return {"error": "No runs found for analysis"}

        cache_hits = {"spacy": 0, "huggingface": 0, "pip": 0}

        setup_times = []
        successful_runs = 0
        time_savings = []

        print(f"Analyzing {total_runs} workflow runs...")

        for i, run in enumerate(runs, 1):
            print(f"  Processing run {i}/{total_runs}: {run['id']}")

            logs = self.get_job_logs(run["id"])
            if not logs:
                continue

            metrics = self.parse_cache_metrics_from_logs(logs)

            if metrics["spacy_cache_hit"]:
                cache_hits["spacy"] += 1
            if metrics["huggingface_cache_hit"]:
                cache_hits["huggingface"] += 1
            if metrics["pip_cache_hit"]:
                cache_hits["pip"] += 1

            if metrics["model_setup_time"]:
                setup_times.append(metrics["model_setup_time"])

                # Estimate time savings using configured baseline
                baseline_time = self.config.baseline_model_setup_time
                if metrics["spacy_cache_hit"]:
                    time_saved = baseline_time - metrics["model_setup_time"]
                    time_savings.append(max(0, time_saved))

            successful_runs += 1

            # Rate limit protection
            time.sleep(self.config.rate_limit_delay)

        # Calculate statistics
        analysis = {
            "analysis_period": {
                "runs_analyzed": successful_runs,
                "total_runs": total_runs,
                "success_rate": (
                    (successful_runs / total_runs) * 100 if total_runs > 0 else 0
                ),
            },
            "cache_performance": {
                "spacy_hit_rate": (
                    (cache_hits["spacy"] / successful_runs) * 100
                    if successful_runs > 0
                    else 0
                ),
                "huggingface_hit_rate": (
                    (cache_hits["huggingface"] / successful_runs) * 100
                    if successful_runs > 0
                    else 0
                ),
                "pip_hit_rate": (
                    (cache_hits["pip"] / successful_runs) * 100
                    if successful_runs > 0
                    else 0
                ),
                "overall_hit_rate": (
                    (sum(cache_hits.values()) / (successful_runs * 3)) * 100
                    if successful_runs > 0
                    else 0
                ),
            },
            "timing_metrics": {},
            "recommendations": [],
        }

        # Timing analysis
        if setup_times:
            analysis["timing_metrics"] = {
                "avg_setup_time": statistics.mean(setup_times),
                "median_setup_time": statistics.median(setup_times),
                "min_setup_time": min(setup_times),
                "max_setup_time": max(setup_times),
                "setup_time_samples": len(setup_times),
            }

        if time_savings:
            total_time_saved = sum(time_savings)
            analysis["timing_metrics"]["total_time_saved"] = total_time_saved
            analysis["timing_metrics"]["avg_time_saved_per_run"] = statistics.mean(
                time_savings
            )
            analysis["timing_metrics"]["estimated_monthly_savings"] = (
                total_time_saved * self.config.monthly_multiplier
            )

        # Generate recommendations
        analysis["recommendations"] = self.generate_recommendations(analysis)

        return analysis

    def generate_recommendations(self, analysis: dict) -> list[str]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        cache_perf = analysis["cache_performance"]

        # Cache hit rate recommendations
        if cache_perf["spacy_hit_rate"] < self.config.spacy_cache_target:
            recommendations.append(
                f"spaCy cache hit rate is {cache_perf['spacy_hit_rate']:.1f}% "
                f"(target: {self.config.spacy_cache_target}%) - consider reviewing cache key strategy"
            )

        if cache_perf["huggingface_hit_rate"] < self.config.huggingface_cache_target:
            recommendations.append(
                f"HuggingFace cache hit rate is {cache_perf['huggingface_hit_rate']:.1f}% "
                f"(target: {self.config.huggingface_cache_target}%) - verify cache paths and keys"
            )

        if cache_perf["pip_hit_rate"] < self.config.pip_cache_target:
            recommendations.append(
                f"Pip cache hit rate is {cache_perf['pip_hit_rate']:.1f}% "
                f"(target: {self.config.pip_cache_target}%) - dependency changes may be frequent"
            )

        # Overall performance recommendations
        if cache_perf["overall_hit_rate"] < self.config.overall_cache_target:
            recommendations.append(
                f"Overall cache performance below target ({self.config.overall_cache_target}%) - consider cache key optimization"
            )
        elif cache_perf["overall_hit_rate"] > self.config.excellent_cache_threshold:
            recommendations.append(
                "Excellent cache performance! Consider documenting current strategy for other projects"
            )

        # Timing recommendations
        timing = analysis.get("timing_metrics", {})
        if timing.get("avg_setup_time", 0) > self.config.high_setup_time_threshold:
            recommendations.append(
                f"Average setup time ({timing['avg_setup_time']:.1f}s) exceeds threshold "
                f"({self.config.high_setup_time_threshold}s) - investigate model download efficiency"
            )

        return recommendations

    def generate_report(self, analysis: dict, output_format: str = "text") -> str:
        """Generate formatted analysis report."""
        if output_format == "json":
            return json.dumps(analysis, indent=2)

        # Text format report
        report = []
        report.append("🔍 GitHub Actions Cache Performance Analysis")
        report.append("=" * 50)

        # Analysis period
        period = analysis["analysis_period"]
        report.append("\n📊 Analysis Period:")
        report.append(f"  Runs analyzed: {period['runs_analyzed']}")
        report.append(f"  Success rate: {period['success_rate']:.1f}%")

        # Cache performance
        cache = analysis["cache_performance"]
        report.append("\n💾 Cache Performance:")
        report.append(
            f"  spaCy models: {cache.get('spacy_hit_rate', 0.0):.1f}% hit rate"
        )
        report.append(
            f"  HuggingFace: {cache.get('huggingface_hit_rate', 0.0):.1f}% hit rate"
        )
        report.append(f"  Pip packages: {cache.get('pip_hit_rate', 0.0):.1f}% hit rate")
        report.append(f"  Overall: {cache.get('overall_hit_rate', 0.0):.1f}% hit rate")

        # Timing metrics
        timing = analysis.get("timing_metrics", {})
        if timing:
            report.append("\n⏱ Timing Metrics:")
            if "avg_setup_time" in timing:
                report.append(f"  Average setup: {timing['avg_setup_time']:.1f}s")
                report.append(f"  Median setup: {timing['median_setup_time']:.1f}s")
                report.append(
                    f"  Setup range: {timing['min_setup_time']:.1f}s - {timing['max_setup_time']:.1f}s"
                )

            if "total_time_saved" in timing:
                report.append(f"  Total time saved: {timing['total_time_saved']:.1f}s")
                report.append(
                    f"  Avg saved per run: {timing['avg_time_saved_per_run']:.1f}s"
                )
                report.append(
                    f"  Est. monthly savings: {timing['estimated_monthly_savings']:.1f}s"
                )

        # Recommendations
        recommendations = analysis.get("recommendations", [])
        if recommendations:
            report.append("\n💡 Recommendations:")
            for rec in recommendations:
                report.append(f"  • {rec}")
        else:
            report.append(
                "\n✅ No specific recommendations - cache performance looks good!"
            )

        return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(
        description="Analyze GitHub Actions cache performance",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--github-token", required=True, help="GitHub personal access token"
    )
    parser.add_argument("--repo", required=True, help="Repository in format owner/repo")
    parser.add_argument(
        "--workflow",
        default="ci.yml",
        help="Workflow file name to analyze (default: ci.yml)",
    )
    parser.add_argument(
        "--days", type=int, default=7, help="Days of history to analyze (default: 7)"
    )
    parser.add_argument(
        "--output-format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text)",
    )
    parser.add_argument("--output-file", help="Output file path (default: stdout)")

    args = parser.parse_args()

    try:
        analyzer = CacheAnalyzer(args.github_token, args.repo)

        print(
            f"🔍 Fetching workflow runs for {args.workflow} (last {args.days} days)..."
        )
        runs = analyzer.get_workflow_runs(args.workflow, args.days)

        if not runs:
            print(f"❌ No workflow runs found for {args.workflow}")
            return 1

        print(f"📊 Analyzing {len(runs)} workflow runs...")
        analysis = analyzer.analyze_runs(runs)

        if "error" in analysis:
            print(f"❌ Analysis failed: {analysis['error']}")
            return 1

        report = analyzer.generate_report(analysis, args.output_format)

        if args.output_file:
            Path(args.output_file).write_text(report)
            print(f"📄 Report written to {args.output_file}")
        else:
            print(f"\n{report}")

        return 0

    except KeyboardInterrupt:
        print("\n⏹ Analysis cancelled by user")
        return 130
    except Exception as e:
        print(f"💥 Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
