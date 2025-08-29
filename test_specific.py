#!/usr/bin/env python3
"""Test specific failing test to verify fix."""

import subprocess
import sys
from pathlib import Path

def main():
    """Run specific test."""
    print("Testing property-based test fix...")
    
    cmd = [
        sys.executable, "-m", "pytest", 
        "tests/integration/test_property_based.py::TestPropertyBased::test_masking_preserves_document_structure",
        "-v", "-s", "--tb=short"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
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
        print("Test timed out after 120 seconds")
        return 1
    except Exception as e:
        print(f"Error running test: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())