#!/usr/bin/env python3
"""Quick test of specific test files."""

import subprocess
import sys
from pathlib import Path

def run_test_file(test_file: str) -> tuple[int, str]:
    """Run a specific test file."""
    cmd = [
        sys.executable, "-m", "pytest", 
        test_file,
        "-v", "--tb=short",
        "-x"  # Stop on first failure
    ]
    
    print(f"Testing {test_file}...")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent,
            timeout=60,
            text=True,
            capture_output=True
        )
        
        if result.returncode == 0:
            passed_count = result.stdout.count(" PASSED")
            print(f"✅ {test_file}: {passed_count} tests PASSED")
        else:
            print(f"🛑 {test_file}: FAILED")
            print("STDOUT:")
            print(result.stdout[:1000] + "..." if len(result.stdout) > 1000 else result.stdout)
        
        return result.returncode, result.stdout
        
    except subprocess.TimeoutExpired:
        print(f"⏰ {test_file}: TIMED OUT")
        return 1, "TIMEOUT"
    except Exception as e:
        print(f"❌ {test_file}: ERROR - {e}")
        return 1, str(e)

def main():
    """Test specific files."""
    test_files = [
        "tests/test_package.py",
        "tests/test_strategies.py", 
        "tests/test_policies.py",
        "tests/test_analyzer.py",
        "tests/test_detection.py"
    ]
    
    results = {}
    
    for test_file in test_files:
        returncode, output = run_test_file(test_file)
        results[test_file] = returncode
        
        # If we find a failure, investigate it
        if returncode != 0:
            print(f"\n=== FAILURE DETAILS for {test_file} ===")
            print(output)
            break  # Focus on one failure at a time
    
    print(f"\n=== SUMMARY ===")
    for test_file, returncode in results.items():
        status = "PASS" if returncode == 0 else "FAIL"
        print(f"{test_file}: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())