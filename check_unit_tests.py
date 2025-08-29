#!/usr/bin/env python3
"""Check unit tests specifically."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run unit tests only."""
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/",
        "-v", "--tb=short",
        "-x",  # Stop on first failure 
        "-m", "unit or not (integration or e2e or golden or performance or slow)"
    ]
    
    print(f"Running unit tests...")
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
        print("STDOUT:")
        print(result.stdout)
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
            
        return result.returncode
        
    except subprocess.TimeoutExpired:
        print("Tests timed out after 120 seconds")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())