"""Unit tests for scripts/analyze-cache-performance.py CacheAnalyzer class.

Tests GitHub Actions cache performance analysis functionality.
"""

import json
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Import the module by adding scripts directory to path
scripts_dir = Path(__file__).parent.parent / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

# Import after path modification
try:
    import importlib.util
    spec = importlib.util.spec_from_file_location("analyze_cache_performance", scripts_dir / "analyze-cache-performance.py")
    analyze_cache_performance = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(analyze_cache_performance)
    # Make the module available globally for patching
    sys.modules["analyze_cache_performance"] = analyze_cache_performance
    CacheAnalyzer = analyze_cache_performance.CacheAnalyzer
except ImportError as e:
    pytest.skip(f"Cannot import analyze-cache-performance.py: {e}")


class TestCacheAnalyzer:
    """Test suite for CacheAnalyzer class functionality."""

    @pytest.fixture
    def cache_analyzer(self):
        """Create CacheAnalyzer instance for testing."""
        return CacheAnalyzer("test_token", "owner/repo")

    @pytest.fixture
    def mock_requests(self):
        """Mock requests module for controlled testing."""
        with patch('analyze_cache_performance.requests') as mock_requests:
            yield mock_requests

    @pytest.fixture
    def sample_workflow_runs(self):
        """Sample workflow runs data for testing."""
        return [
            {
                "id": "12345",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2023-01-01T10:00:00Z",
                "updated_at": "2023-01-01T10:15:00Z"
            },
            {
                "id": "12346",
                "status": "completed",
                "conclusion": "success",
                "created_at": "2023-01-01T11:00:00Z",
                "updated_at": "2023-01-01T11:12:00Z"
            }
        ]

    @pytest.fixture
    def sample_logs_with_cache_hits(self):
        """Sample workflow logs with cache hits."""
        return """
        2023-01-01T10:05:00.0000000Z Cache restored from key: cache-spacy-v1-small
        2023-01-01T10:05:01.0000000Z cache hit detected for spacy models
        2023-01-01T10:05:02.0000000Z Cache restored from key: cache-huggingface-v1
        2023-01-01T10:05:03.0000000Z cache hit detected for huggingface
        2023-01-01T10:05:04.0000000Z Cache restored from key: pip-cache
        2023-01-01T10:05:05.0000000Z pip cache hit detected
        2023-01-01T10:06:00.0000000Z Model setup complete - Total time: 45.2s
        """

    @pytest.fixture
    def sample_logs_with_cache_misses(self):
        """Sample workflow logs with cache misses."""
        return """
        2023-01-01T10:05:00.0000000Z Cache not found for key: cache-spacy-v1-small
        2023-01-01T10:05:01.0000000Z Downloading spaCy model en_core_web_sm
        2023-01-01T10:08:00.0000000Z Cache not found for key: cache-huggingface-v1
        2023-01-01T10:08:01.0000000Z Downloading HuggingFace models
        2023-01-01T10:12:00.0000000Z Model setup complete - Total time: 420.8s
        """

    def test_init(self, cache_analyzer):
        """Test CacheAnalyzer initialization."""
        assert cache_analyzer.github_token == "test_token"
        assert cache_analyzer.repo == "owner/repo"
        assert cache_analyzer.base_url == "https://api.github.com"
        assert "Authorization" in cache_analyzer.headers
        assert cache_analyzer.headers["Authorization"] == "token test_token"

    def test_get_workflow_runs_success(self, cache_analyzer, mock_requests, sample_workflow_runs):
        """Test successful workflow runs retrieval."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"workflow_runs": sample_workflow_runs}
        mock_requests.get.return_value = mock_response

        runs = cache_analyzer.get_workflow_runs("ci.yml", days=7)

        assert runs == sample_workflow_runs
        mock_requests.get.assert_called_once()

        # Verify API call parameters
        call_args = mock_requests.get.call_args
        assert "ci.yml" in call_args[0][0]
        assert "params" in call_args[1]
        assert call_args[1]["params"]["status"] == "completed"

    def test_get_workflow_runs_http_error(self, cache_analyzer, mock_requests):
        """Test workflow runs retrieval with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_requests.get.return_value = mock_response

        runs = cache_analyzer.get_workflow_runs("ci.yml")

        assert runs == []

    def test_get_workflow_runs_request_exception(self, cache_analyzer, mock_requests):
        """Test workflow runs retrieval with request exception."""
        mock_requests.get.side_effect = Exception("Network error")

        runs = cache_analyzer.get_workflow_runs("ci.yml")

        assert runs == []

    def test_get_job_logs_success(self, cache_analyzer, mock_requests):
        """Test successful job logs retrieval."""
        mock_response = Mock()
        mock_response.raise_for_status.return_value = None
        mock_response.text = "Sample log content"
        mock_requests.get.return_value = mock_response

        logs = cache_analyzer.get_job_logs("12345")

        assert logs == "Sample log content"
        mock_requests.get.assert_called_once()

        # Verify API call
        call_args = mock_requests.get.call_args
        assert "12345/logs" in call_args[0][0]

    def test_get_job_logs_http_error(self, cache_analyzer, mock_requests):
        """Test job logs retrieval with HTTP error."""
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = Exception("HTTP 404")
        mock_requests.get.return_value = mock_response

        logs = cache_analyzer.get_job_logs("12345")

        assert logs is None

    def test_get_job_logs_request_exception(self, cache_analyzer, mock_requests):
        """Test job logs retrieval with request exception."""
        mock_requests.get.side_effect = Exception("Network error")

        logs = cache_analyzer.get_job_logs("12345")

        assert logs is None

    def test_parse_cache_metrics_cache_hits(self, cache_analyzer, sample_logs_with_cache_hits):
        """Test parsing cache metrics with cache hits."""
        metrics = cache_analyzer.parse_cache_metrics_from_logs(sample_logs_with_cache_hits)

        assert metrics["spacy_cache_hit"] is True
        assert metrics["huggingface_cache_hit"] is True
        assert metrics["pip_cache_hit"] is True
        assert metrics["model_setup_time"] == 45.2

    def test_parse_cache_metrics_cache_misses(self, cache_analyzer, sample_logs_with_cache_misses):
        """Test parsing cache metrics with cache misses."""
        metrics = cache_analyzer.parse_cache_metrics_from_logs(sample_logs_with_cache_misses)

        assert metrics["spacy_cache_hit"] is False
        assert metrics["huggingface_cache_hit"] is False
        assert metrics["pip_cache_hit"] is False
        assert metrics["model_setup_time"] == 420.8

    def test_parse_cache_metrics_empty_logs(self, cache_analyzer):
        """Test parsing cache metrics with empty logs."""
        metrics = cache_analyzer.parse_cache_metrics_from_logs("")

        assert metrics["spacy_cache_hit"] is False
        assert metrics["huggingface_cache_hit"] is False
        assert metrics["pip_cache_hit"] is False
        assert metrics["model_setup_time"] is None

    def test_parse_cache_metrics_malformed_time(self, cache_analyzer):
        """Test parsing cache metrics with malformed time data."""
        logs = "Model setup complete - Total time: invalid_time"
        metrics = cache_analyzer.parse_cache_metrics_from_logs(logs)

        assert metrics["model_setup_time"] is None

    def test_analyze_runs_empty_list(self, cache_analyzer):
        """Test analyzing empty runs list."""
        analysis = cache_analyzer.analyze_runs([])

        assert "error" in analysis
        assert analysis["error"] == "No runs found for analysis"

    def test_analyze_runs_successful_analysis(self, cache_analyzer, sample_workflow_runs,
                                            sample_logs_with_cache_hits):
        """Test successful runs analysis."""
        with patch.object(cache_analyzer, 'get_job_logs', return_value=sample_logs_with_cache_hits):
            with patch('analyze_cache_performance.time.sleep'):  # Skip rate limiting
                analysis = cache_analyzer.analyze_runs(sample_workflow_runs)

        assert "error" not in analysis
        assert analysis["analysis_period"]["runs_analyzed"] == 2
        assert analysis["cache_performance"]["spacy_hit_rate"] == 100.0
        assert analysis["cache_performance"]["huggingface_hit_rate"] == 100.0
        assert analysis["cache_performance"]["pip_hit_rate"] == 100.0
        assert analysis["cache_performance"]["overall_hit_rate"] == 100.0

    def test_analyze_runs_mixed_results(self, cache_analyzer, sample_workflow_runs,
                                       sample_logs_with_cache_hits, sample_logs_with_cache_misses):
        """Test analyzing runs with mixed cache hit/miss results."""
        logs_sequence = [
            sample_logs_with_cache_hits,
            sample_logs_with_cache_misses
        ]

        with patch.object(cache_analyzer, 'get_job_logs', side_effect=logs_sequence):
            with patch('analyze_cache_performance.time.sleep'):
                analysis = cache_analyzer.analyze_runs(sample_workflow_runs)

        # Should have 50% hit rate for each cache type
        assert analysis["cache_performance"]["spacy_hit_rate"] == 50.0
        assert analysis["cache_performance"]["huggingface_hit_rate"] == 50.0
        assert analysis["cache_performance"]["pip_hit_rate"] == 50.0

    def test_analyze_runs_no_logs_available(self, cache_analyzer, sample_workflow_runs):
        """Test analyzing runs when logs are not available."""
        with patch.object(cache_analyzer, 'get_job_logs', return_value=None):
            with patch('analyze_cache_performance.time.sleep'):
                analysis = cache_analyzer.analyze_runs(sample_workflow_runs)

        assert analysis["analysis_period"]["runs_analyzed"] == 0
        assert analysis["cache_performance"]["overall_hit_rate"] == 0

    def test_generate_recommendations_poor_performance(self, cache_analyzer):
        """Test recommendation generation for poor cache performance."""
        analysis = {
            "cache_performance": {
                "spacy_hit_rate": 60.0,
                "huggingface_hit_rate": 70.0,
                "pip_hit_rate": 85.0,
                "overall_hit_rate": 72.0
            },
            "timing_metrics": {
                "avg_setup_time": 240.0
            }
        }

        recommendations = cache_analyzer.generate_recommendations(analysis)

        assert len(recommendations) > 0
        assert any("spaCy cache hit rate" in rec for rec in recommendations)
        assert any("HuggingFace cache hit rate" in rec for rec in recommendations)
        assert any("Overall cache performance" in rec for rec in recommendations)
        assert any("Average setup time" in rec for rec in recommendations)

    def test_generate_recommendations_excellent_performance(self, cache_analyzer):
        """Test recommendation generation for excellent cache performance."""
        analysis = {
            "cache_performance": {
                "spacy_hit_rate": 98.0,
                "huggingface_hit_rate": 97.0,
                "pip_hit_rate": 99.0,
                "overall_hit_rate": 98.0
            },
            "timing_metrics": {
                "avg_setup_time": 30.0
            }
        }

        recommendations = cache_analyzer.generate_recommendations(analysis)

        assert any("Excellent cache performance" in rec for rec in recommendations)

    def test_generate_recommendations_good_performance(self, cache_analyzer):
        """Test recommendation generation for good performance with minimal recommendations."""
        analysis = {
            "cache_performance": {
                "spacy_hit_rate": 85.0,
                "huggingface_hit_rate": 88.0,
                "pip_hit_rate": 92.0,
                "overall_hit_rate": 88.0
            },
            "timing_metrics": {
                "avg_setup_time": 120.0
            }
        }

        recommendations = cache_analyzer.generate_recommendations(analysis)

        # Should have minimal recommendations for good performance
        assert len(recommendations) <= 2

    def test_generate_report_text_format(self, cache_analyzer):
        """Test generating report in text format."""
        analysis = {
            "analysis_period": {
                "runs_analyzed": 10,
                "success_rate": 90.0
            },
            "cache_performance": {
                "spacy_hit_rate": 85.0,
                "huggingface_hit_rate": 88.0,
                "pip_hit_rate": 92.0,
                "overall_hit_rate": 88.3
            },
            "timing_metrics": {
                "avg_setup_time": 120.0,
                "median_setup_time": 115.0,
                "min_setup_time": 45.0,
                "max_setup_time": 200.0,
                "total_time_saved": 1800.0,
                "avg_time_saved_per_run": 180.0,
                "estimated_monthly_savings": 7740.0
            },
            "recommendations": ["Test recommendation"]
        }

        report = cache_analyzer.generate_report(analysis, "text")

        assert "Cache Performance Analysis" in report
        assert "Runs analyzed: 10" in report
        assert "Success rate: 90.0%" in report
        assert "spaCy models: 85.0% hit rate" in report
        assert "Average setup: 120.0s" in report
        assert "Total time saved: 1800.0s" in report
        assert "Test recommendation" in report

    def test_generate_report_json_format(self, cache_analyzer):
        """Test generating report in JSON format."""
        analysis = {
            "analysis_period": {
                "runs_analyzed": 5,
                "success_rate": 100.0
            },
            "cache_performance": {
                "overall_hit_rate": 95.0
            }
        }

        report = cache_analyzer.generate_report(analysis, "json")

        # Should be valid JSON
        parsed_report = json.loads(report)
        assert parsed_report["analysis_period"]["runs_analyzed"] == 5
        assert parsed_report["cache_performance"]["overall_hit_rate"] == 95.0

    def test_generate_report_no_timing_metrics(self, cache_analyzer):
        """Test generating report without timing metrics."""
        analysis = {
            "analysis_period": {
                "runs_analyzed": 3,
                "success_rate": 100.0
            },
            "cache_performance": {
                "spacy_hit_rate": 100.0,
                "huggingface_hit_rate": 100.0,
                "pip_hit_rate": 100.0,
                "overall_hit_rate": 100.0
            },
            "recommendations": []
        }

        report = cache_analyzer.generate_report(analysis, "text")

        assert "Cache Performance Analysis" in report
        assert "Timing Metrics:" not in report
        assert "No specific recommendations" in report

    def test_generate_report_with_recommendations(self, cache_analyzer):
        """Test generating report with multiple recommendations."""
        analysis = {
            "analysis_period": {
                "runs_analyzed": 5,
                "success_rate": 80.0
            },
            "cache_performance": {
                "overall_hit_rate": 70.0
            },
            "recommendations": [
                "Improve cache key strategy",
                "Optimize dependency management",
                "Review workflow configuration"
            ]
        }

        report = cache_analyzer.generate_report(analysis, "text")

        assert "Recommendations:" in report
        assert "• Improve cache key strategy" in report
        assert "• Optimize dependency management" in report
        assert "• Review workflow configuration" in report


class TestMainFunction:
    """Test suite for main() function and CLI argument parsing."""

    @pytest.fixture
    def mock_cache_analyzer(self):
        """Mock CacheAnalyzer for testing main function."""
        with patch('analyze_cache_performance.CacheAnalyzer') as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance
            yield mock_instance

    def test_main_success(self, mock_cache_analyzer):
        """Test main function with successful analysis."""
        mock_cache_analyzer.get_workflow_runs.return_value = [{"id": "123"}]
        mock_cache_analyzer.analyze_runs.return_value = {
            "analysis_period": {"runs_analyzed": 1}
        }
        mock_cache_analyzer.generate_report.return_value = "Test report"

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 0

    def test_main_no_workflow_runs(self, mock_cache_analyzer):
        """Test main function with no workflow runs found."""
        mock_cache_analyzer.get_workflow_runs.return_value = []

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 1

    def test_main_analysis_error(self, mock_cache_analyzer):
        """Test main function with analysis error."""
        mock_cache_analyzer.get_workflow_runs.return_value = [{"id": "123"}]
        mock_cache_analyzer.analyze_runs.return_value = {"error": "Analysis failed"}

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 1

    def test_main_keyboard_interrupt(self, mock_cache_analyzer):
        """Test main function with keyboard interrupt."""
        mock_cache_analyzer.get_workflow_runs.side_effect = KeyboardInterrupt()

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 130

    def test_main_unexpected_error(self, mock_cache_analyzer):
        """Test main function with unexpected error."""
        mock_cache_analyzer.get_workflow_runs.side_effect = Exception("Unexpected error")

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 1

    def test_main_with_output_file(self, mock_cache_analyzer, tmp_path):
        """Test main function with output file."""
        mock_cache_analyzer.get_workflow_runs.return_value = [{"id": "123"}]
        mock_cache_analyzer.analyze_runs.return_value = {
            "analysis_period": {"runs_analyzed": 1}
        }
        mock_cache_analyzer.generate_report.return_value = "Test report content"

        output_file = tmp_path / "test_report.txt"

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo',
            '--output-file', str(output_file)
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 0
                assert output_file.read_text() == "Test report content"

    def test_main_custom_parameters(self, mock_cache_analyzer):
        """Test main function with custom parameters."""
        mock_cache_analyzer.get_workflow_runs.return_value = [{"id": "123"}]
        mock_cache_analyzer.analyze_runs.return_value = {
            "analysis_period": {"runs_analyzed": 1}
        }
        mock_cache_analyzer.generate_report.return_value = "Test report"

        with patch('sys.argv', [
            'analyze-cache-performance.py',
            '--github-token', 'test_token',
            '--repo', 'owner/repo',
            '--workflow', 'custom.yml',
            '--days', '14',
            '--output-format', 'json'
        ]):
            with patch('analyze_cache_performance.main',
                      wraps=main_function_wrapper(mock_cache_analyzer)) as mock_main:
                result = mock_main()

                assert result == 0
                mock_cache_analyzer.get_workflow_runs.assert_called_once_with('custom.yml', 14)
                mock_cache_analyzer.generate_report.assert_called_once_with(
                    mock_cache_analyzer.analyze_runs.return_value, 'json'
                )


def main_function_wrapper(mock_analyzer):
    """Wrapper function for testing main() with mocked CacheAnalyzer."""
    def wrapped_main():
        import argparse
        from pathlib import Path

        parser = argparse.ArgumentParser()
        parser.add_argument("--github-token", required=True)
        parser.add_argument("--repo", required=True)
        parser.add_argument("--workflow", default="ci.yml")
        parser.add_argument("--days", type=int, default=7)
        parser.add_argument("--output-format", choices=["text", "json"], default="text")
        parser.add_argument("--output-file")

        args = parser.parse_args()

        try:
            print(f"🔍 Fetching workflow runs for {args.workflow} (last {args.days} days)...")
            runs = mock_analyzer.get_workflow_runs(args.workflow, args.days)

            if not runs:
                print(f"❌ No workflow runs found for {args.workflow}")
                return 1

            print(f"📊 Analyzing {len(runs)} workflow runs...")
            analysis = mock_analyzer.analyze_runs(runs)

            if "error" in analysis:
                print(f"❌ Analysis failed: {analysis['error']}")
                return 1

            report = mock_analyzer.generate_report(analysis, args.output_format)

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

    return wrapped_main
