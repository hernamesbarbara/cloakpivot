#!/usr/bin/env python3
"""Comprehensive test runner for CloakPivot testing suite.

This script provides different test execution modes and reporting options
for the comprehensive testing infrastructure.

Parallel Test Execution:
This runner leverages pytest-xdist for parallel test execution to reduce
wall-clock time on multi-core systems. Tests are parallelized using "-n auto"
which automatically detects the number of CPU cores.

Parallelization Strategy:
- Unit Tests: Parallelized (isolated fixtures, no shared state)
- Integration Tests: Parallelized (isolated temp directories via conftest.py)
- E2E Tests: Parallelized (CLI tests use isolated temp workspaces)
- Fast Tests: Parallelized (combination of unit + integration)
- Performance Tests: Not parallelized (benchmarks need dedicated resources)
- Slow Tests: Not parallelized (may have shared resource dependencies)

Test Isolation:
All parallelized test suites use proper fixture isolation patterns:
- Temporary directories via tempfile.TemporaryDirectory()
- No hardcoded ports or shared network resources
- Session-scoped fixtures only for read-only resources (e.g., shared_analyzer)
- Reset of global state via autouse fixtures in conftest.py
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a command and return exit code."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("=" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    if result.returncode != 0:
        print(f"❌ {description} failed with exit code {result.returncode}")
    else:
        print(f"✅ {description} completed successfully")

    return result.returncode


def run_unit_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run unit tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(
        [
            "tests/",
            "-m",
            "unit or not (integration or e2e or golden or performance or slow)",
        ]
    )

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    if coverage:
        import os

        cov_options = [
            "--cov=cloakpivot",
            "--cov-report=term-missing",
            "--cov-fail-under=25",
        ]

        # Only generate HTML coverage if explicitly requested via environment variable
        if os.environ.get("COVERAGE_HTML", "").lower() in ("1", "true"):
            cov_options.append("--cov-report=html:htmlcov")

        cmd.extend(cov_options)

    # Parallel execution with xdist - unit tests are isolated and safe for parallelization
    cmd.extend(["-n", "auto"])

    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose: bool = False) -> int:
    """Run integration tests.

    Integration tests are safe for parallel execution because they:
    - Use isolated temporary directories via pytest fixtures
    - Have no shared state or external service dependencies
    - Use golden file comparisons that don't interfere with each other
    """
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/integration/", "-m", "integration"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    # Longer timeout for integration tests
    cmd.extend(["--timeout=300"])

    # Parallel execution with xdist - integration tests use isolated temp dirs and fixtures
    cmd.extend(["-n", "auto"])

    return run_command(cmd, "Integration Tests")


def run_golden_file_tests(verbose: bool = False) -> int:
    """Run golden file regression tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/integration/test_golden_files.py", "-m", "golden"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    return run_command(cmd, "Golden File Regression Tests")


def run_round_trip_tests(verbose: bool = False) -> int:
    """Run round-trip fidelity tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/integration/test_round_trip.py"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    return run_command(cmd, "Round-Trip Fidelity Tests")


def run_property_based_tests(verbose: bool = False) -> int:
    """Run property-based tests with Hypothesis."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/integration/test_property_based.py"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    # Hypothesis settings
    cmd.extend(["--hypothesis-show-statistics"])

    return run_command(cmd, "Property-Based Tests (Hypothesis)")


def run_e2e_tests(verbose: bool = False) -> int:
    """Run end-to-end CLI tests.

    End-to-end tests are safe for parallel execution because they:
    - Use isolated temporary workspace directories (temp_workspace fixture)
    - Run CLI commands in separate ClickRunner instances
    - Have no shared state or database connections
    - Use independent file system operations with no conflicts
    """
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/e2e/", "-m", "e2e"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    # Longer timeout for E2E tests
    cmd.extend(["--timeout=600"])

    # Parallel execution with xdist - E2E CLI tests use isolated temp workspaces and no shared state
    cmd.extend(["-n", "auto"])

    return run_command(cmd, "End-to-End Tests")


def run_performance_tests(verbose: bool = False) -> int:
    """Run performance benchmark tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/performance/", "-m", "performance"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    # Performance-specific options
    cmd.extend(["--benchmark-only", "--benchmark-sort=mean"])

    return run_command(cmd, "Performance Tests")


def run_slow_tests(verbose: bool = False) -> int:
    """Run slow/stress tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection
    cmd.extend(["tests/", "-m", "slow"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    # Very long timeout for slow tests
    cmd.extend(["--timeout=1800"])  # 30 minutes

    return run_command(cmd, "Slow/Stress Tests")


def run_all_fast_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run all fast tests (unit + integration, excluding slow/performance)."""
    cmd = ["python", "-m", "pytest"]

    # Test selection - exclude slow and performance tests
    cmd.extend(["tests/", "-m", "not (slow or performance)"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    if coverage:
        import os

        cov_options = [
            "--cov=cloakpivot",
            "--cov-report=term-missing",
            "--cov-fail-under=60",  # Slightly lower for comprehensive suite
        ]

        # Only generate HTML coverage if explicitly requested via environment variable
        if os.environ.get("COVERAGE_HTML", "").lower() in ("1", "true"):
            cov_options.append("--cov-report=html:htmlcov")

        cmd.extend(cov_options)

    # Parallel execution with xdist - fast tests combine unit+integration, both safely parallelized
    cmd.extend(["-n", "auto"])

    return run_command(cmd, "All Fast Tests")


def run_comprehensive_tests(verbose: bool = False, coverage: bool = True) -> int:
    """Run comprehensive test suite including slow tests."""
    cmd = ["python", "-m", "pytest"]

    # Test selection - all tests
    cmd.extend(["tests/"])

    # Options
    if verbose:
        cmd.append("-v")
    else:
        cmd.extend(["-q", "--tb=short"])

    if coverage:
        cmd.extend(
            [
                "--cov=cloakpivot",
                "--cov-report=term-missing",
                "--cov-report=html:htmlcov",
                "--cov-report=xml:coverage.xml",  # For CI systems
                "--cov-fail-under=60",  # Lower threshold for full suite
            ]
        )

    # Long timeout for comprehensive suite
    cmd.extend(["--timeout=1800"])

    return run_command(cmd, "Comprehensive Test Suite")


def lint_and_format() -> int:
    """Run linting and formatting checks."""
    print("\n" + "=" * 60)
    print("Running Code Quality Checks")
    print("=" * 60)

    exit_codes = []

    # Black formatting check
    result = subprocess.run(["black", "--check", "--diff", "cloakpivot", "tests"])
    if result.returncode != 0:
        print("❌ Black formatting issues found")
        exit_codes.append(result.returncode)
    else:
        print("✅ Black formatting looks good")

    # Ruff linting
    result = subprocess.run(["ruff", "check", "cloakpivot", "tests"])
    if result.returncode != 0:
        print("❌ Ruff linting issues found")
        exit_codes.append(result.returncode)
    else:
        print("✅ Ruff linting passed")

    # MyPy type checking
    result = subprocess.run(["mypy", "cloakpivot"])
    if result.returncode != 0:
        print("❌ MyPy type checking issues found")
        exit_codes.append(result.returncode)
    else:
        print("✅ MyPy type checking passed")

    return max(exit_codes) if exit_codes else 0


def main() -> None:
    """Main test runner entry point."""
    parser = argparse.ArgumentParser(
        description="CloakPivot Comprehensive Test Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Test Categories:
  unit          - Fast unit tests for individual components
  integration   - Integration tests with golden files and round-trip testing
  e2e           - End-to-end CLI workflow tests
  performance   - Performance benchmarks and regression tests
  property      - Property-based tests using Hypothesis
  slow          - Slow/stress tests and comprehensive scenarios
  fast          - All tests except slow and performance tests
  all           - Complete test suite including slow tests

Examples:
  python run_tests.py unit --coverage
  python run_tests.py integration --verbose
  python run_tests.py fast
  python run_tests.py all --verbose --coverage
        """,
    )

    parser.add_argument(
        "test_type",
        choices=[
            "unit",
            "integration",
            "golden",
            "round-trip",
            "property",
            "e2e",
            "performance",
            "slow",
            "fast",
            "all",
            "lint",
        ],
        help="Type of tests to run",
    )

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--coverage",
        "-c",
        action="store_true",
        default=True,
        help="Include coverage reporting (default: True)",
    )

    parser.add_argument(
        "--no-coverage",
        action="store_false",
        dest="coverage",
        help="Disable coverage reporting",
    )

    args = parser.parse_args()

    # Test type routing
    test_runners = {
        "unit": lambda: run_unit_tests(args.verbose, args.coverage),
        "integration": lambda: run_integration_tests(args.verbose),
        "golden": lambda: run_golden_file_tests(args.verbose),
        "round-trip": lambda: run_round_trip_tests(args.verbose),
        "property": lambda: run_property_based_tests(args.verbose),
        "e2e": lambda: run_e2e_tests(args.verbose),
        "performance": lambda: run_performance_tests(args.verbose),
        "slow": lambda: run_slow_tests(args.verbose),
        "fast": lambda: run_all_fast_tests(args.verbose, args.coverage),
        "all": lambda: run_comprehensive_tests(args.verbose, args.coverage),
        "lint": lambda: lint_and_format(),
    }

    runner = test_runners[args.test_type]
    exit_code = runner()

    if exit_code == 0:
        print(f"\n🎉 {args.test_type.title()} tests completed successfully!")
    else:
        print(f"\n💥 {args.test_type.title()} tests failed with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
