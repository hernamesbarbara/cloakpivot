#!/usr/bin/env python3
"""Minimal test to debug import issues."""

import sys
import traceback

print("Starting minimal test...")
print(f"Python version: {sys.version}")
print(f"Python path: {sys.path[:3]}...")

try:
    print("Testing basic imports...")
    import os
    print("✓ os imported")
    
    import pathlib
    print("✓ pathlib imported")
    
    import click
    print("✓ click imported")
    
    import pytest
    print("✓ pytest imported")
    
    print("Testing cloakpivot core modules...")
    
    # Try importing one module at a time
    try:
        from cloakpivot.core import strategies
        print("✓ cloakpivot.core.strategies imported")
    except Exception as e:
        print(f"✗ cloakpivot.core.strategies failed: {e}")
        traceback.print_exc()
    
    try:
        from cloakpivot.core import policies
        print("✓ cloakpivot.core.policies imported")
    except Exception as e:
        print(f"✗ cloakpivot.core.policies failed: {e}")
        traceback.print_exc()
    
    try:
        import cloakpivot
        print("✓ main cloakpivot package imported")
    except Exception as e:
        print(f"✗ main cloakpivot package failed: {e}")
        traceback.print_exc()
        
    print("✅ All imports successful")
    
except Exception as e:
    print(f"❌ Import failed: {e}")
    traceback.print_exc()
    sys.exit(1)

print("Minimal test completed successfully")