#!/usr/bin/env python3
"""Debug the actual test error more carefully."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run round trip test with more verbose output."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/test_round_trip.py::TestRoundTripFidelity::test_simple_document_round_trip",
        "-vvs", "--tb=long", "--no-header"
    ]
    
    print(f"Running round trip test with verbose output...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent,
            timeout=120,
            text=True,
            capture_output=True
        )
        
        print(f"Exit code: {result.returncode}")
        print("FULL OUTPUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("Test timed out after 120 seconds")
        return 1
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())