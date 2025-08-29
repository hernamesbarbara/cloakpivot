#!/usr/bin/env python3
"""Test the lazy initialization test specifically."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run lazy initialization test."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/test_analyzer.py::TestAnalyzerEngineWrapper::test_lazy_initialization",
        "-v", "-s", "--tb=short"
    ]
    
    print(f"Running lazy initialization test...")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd, 
            cwd=Path(__file__).parent,
            timeout=60,
            text=True,
            capture_output=True
        )
        
        print(f"Exit code: {result.returncode}")
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")  
            print(result.stderr)
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("Test timed out after 60 seconds")
        return 1
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())