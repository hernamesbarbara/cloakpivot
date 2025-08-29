#!/usr/bin/env python3
"""Test runner to systematically check tests."""

import subprocess
import sys
from pathlib import Path

def run_test_category(category: str, timeout: int = 300) -> tuple[int, str, str]:
    """Run a test category and return results."""
    cmd = [
        sys.executable, "-m", "pytest", 
        f"tests/",
        "-v", "--tb=short",
        "-x",  # Stop on first failure
    ]
    
    # Add markers based on category
    if category == "unit":
        cmd.extend(["-m", "unit or not (integration or e2e or golden or performance or slow)"])
    elif category == "integration": 
        cmd.extend(["-m", "integration"])
    elif category == "e2e":
        cmd.extend(["-m", "e2e"])
    elif category == "performance":
        cmd.extend(["-m", "performance"])
    
    print(f"\n{'='*60}")
    print(f"Running {category} tests")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent,
            timeout=timeout,
            text=True,
            capture_output=True
        )
        
        return result.returncode, result.stdout, result.stderr
        
    except subprocess.TimeoutExpired:
        return 1, "", f"Tests timed out after {timeout} seconds"
    except Exception as e:
        return 1, "", f"Error running tests: {e}"

def main():
    """Run tests systematically."""
    categories = ["unit", "integration", "e2e", "performance"]
    
    results = {}
    
    for category in categories:
        print(f"\n🧪 Testing {category} category...")
        
        returncode, stdout, stderr = run_test_category(category, timeout=120)
        results[category] = {
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr
        }
        
        if returncode == 0:
            print(f"✅ {category} tests PASSED")
        else:
            print(f"🛑 {category} tests FAILED")
            print("STDOUT:")
            print(stdout)
            if stderr:
                print("STDERR:")
                print(stderr)
        
        # Stop on first failing category to focus on one issue at a time
        if returncode != 0:
            print(f"\nStopping at first failing category: {category}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print('='*60)
    
    for category, result in results.items():
        status = "PASS" if result["returncode"] == 0 else "FAIL"
        print(f"{category:12}: {status}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())