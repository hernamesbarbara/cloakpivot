#!/usr/bin/env python3
"""Test runner script for CloakPivot with performance management.

This script provides different test execution modes with proper marker handling
for performance optimization of property-based tests.
"""

import argparse
import os
import subprocess
import sys


def run_command(cmd: list[str], description: str) -> int:
    """Run a shell command and return the exit code."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)
    print("-" * 60)
    print(f"Exit code: {result.returncode}")
    print()

    return result.returncode


def main() -> int:
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description="Test runner for CloakPivot with performance management",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py unit                    # Run unit tests only
  python run_tests.py property --fast         # Run property tests with fast profile
  python run_tests.py property --ci           # Run property tests with CI profile
  python run_tests.py slow                    # Run slow and performance tests
  python run_tests.py all --skip-slow         # Run all tests except slow ones
        """
    )

    subparsers = parser.add_subparsers(dest='command', help='Test execution mode')

    # Unit tests subcommand
    unit_parser = subparsers.add_parser(
        'unit',
        help='Run unit tests only (excludes property-based tests)'
    )
    unit_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    # Property tests subcommand
    property_parser = subparsers.add_parser(
        'property',
        help='Run property-based tests with performance optimization'
    )
    property_parser.add_argument(
        '--fast',
        action='store_true',
        help='Use fast profile (10 examples, no shrinking)'
    )
    property_parser.add_argument(
        '--ci',
        action='store_true',
        help='Use CI profile (12 examples, limited shrinking)'
    )
    property_parser.add_argument(
        '--thorough',
        action='store_true',
        help='Use thorough profile (100 examples, full shrinking)'
    )
    property_parser.add_argument(
        '--show-stats',
        action='store_true',
        help='Show Hypothesis statistics'
    )

    # Slow tests subcommand
    slow_parser = subparsers.add_parser(
        'slow',
        help='Run slow and performance tests only'
    )
    slow_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    # All tests subcommand
    all_parser = subparsers.add_parser(
        'all',
        help='Run all tests'
    )
    all_parser.add_argument(
        '--skip-slow',
        action='store_true',
        help='Skip slow and performance tests'
    )
    all_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Verbose output'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    # Base pytest command
    base_cmd = ['python', '-m', 'pytest']

    if args.command == 'unit':
        # Run unit tests only, excluding property-based tests
        cmd = base_cmd + [
            '-m', 'not property',
            'tests/',
        ]
        if args.verbose:
            cmd.append('-v')

        return run_command(cmd, "Unit tests (excluding property-based tests)")

    elif args.command == 'property':
        # Set up environment for Hypothesis profile
        env = os.environ.copy()

        if args.fast:
            env['HYPOTHESIS_PROFILE'] = 'fast'
            profile_desc = 'fast profile (10 examples)'
        elif args.ci:
            env['HYPOTHESIS_PROFILE'] = 'ci'
            profile_desc = 'CI profile (12 examples)'
        elif args.thorough:
            env['HYPOTHESIS_PROFILE'] = 'thorough'
            profile_desc = 'thorough profile (100 examples)'
        else:
            # Default to fast for property tests
            env['HYPOTHESIS_PROFILE'] = 'fast'
            profile_desc = 'default fast profile (10 examples)'

        # Run property tests, excluding slow/performance tests
        cmd = base_cmd + [
            '-m', 'property and not slow and not performance',
            'tests/',
        ]

        if args.show_stats:
            cmd.append('--hypothesis-show-statistics')

        print(f"Using Hypothesis {profile_desc}")
        result = subprocess.run(cmd, env=env, capture_output=False)
        print("-" * 60)
        print(f"Exit code: {result.returncode}")

        return result.returncode

    elif args.command == 'slow':
        # Run slow and performance tests only
        cmd = base_cmd + [
            '-m', 'slow or performance',
            'tests/',
            '--hypothesis-show-statistics',  # Always show stats for slow tests
        ]
        if args.verbose:
            cmd.append('-v')

        # Use thorough profile for slow tests
        env = os.environ.copy()
        env['HYPOTHESIS_PROFILE'] = 'thorough'

        print("Using Hypothesis thorough profile for slow/performance tests")
        result = subprocess.run(cmd, env=env, capture_output=False)
        print("-" * 60)
        print(f"Exit code: {result.returncode}")

        return result.returncode

    elif args.command == 'all':
        # Run all tests
        if args.skip_slow:
            cmd = base_cmd + [
                '-m', 'not slow and not performance',
                'tests/',
            ]
            description = "All tests (excluding slow and performance tests)"
        else:
            cmd = base_cmd + [
                'tests/',
                '--hypothesis-show-statistics',
            ]
            description = "All tests (including slow and performance tests)"

            # Use thorough profile when including slow tests
            env = os.environ.copy()
            env['HYPOTHESIS_PROFILE'] = 'thorough'

            if args.verbose:
                cmd.append('-v')

            print("Using Hypothesis thorough profile for comprehensive testing")
            result = subprocess.run(cmd, env=env, capture_output=False)
            print("-" * 60)
            print(f"Exit code: {result.returncode}")

            return result.returncode

        if args.verbose:
            cmd.append('-v')

        return run_command(cmd, description)

    return 0


if __name__ == '__main__':
    sys.exit(main())
