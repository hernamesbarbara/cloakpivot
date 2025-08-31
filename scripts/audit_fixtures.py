#!/usr/bin/env python3
"""Fixture analysis script for CloakPivot test suite.

Analyzes current fixture scopes and usage patterns to identify
optimization opportunities for better test performance.
"""

import ast
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

# Configuration
FIXTURE_SCOPE_PREFERENCE = {
    # Static data fixtures - safe to convert to session
    "sample_text_with_pii": "session",
    "sample_documents": "session", 
    "detected_entities": "session",
    "mock_analyzer_results": "session",
    "simple_text_segments": "session",
    "complex_text_segments": "session",
    
    # Path fixtures - already session scoped
    "temp_dir": "session",
    "test_files_dir": "session", 
    "golden_files_dir": "session",
    "sample_policies_dir": "session",
    
    # Policy fixtures - static data, safe for session
    "basic_masking_policy": "session",
    "strict_masking_policy": "session",
    "benchmark_policy": "session",
    
    # Document fixtures - expensive to create
    "simple_document": "session",
    "complex_document": "session", 
    "large_document": "session",
    
    # Engine fixtures - expensive to initialize
    "masking_engine": "session",  # Currently module, should be session
    "shared_analyzer": "session",  # Already session
    "shared_document_processor": "session",  # Already session
    "shared_detection_pipeline": "session",  # Already session
    
    # Mock fixtures - function scope needed for isolation
    "mock_presidio_analyzer": "function",
    "mock_presidio_anonymizer": "function",
    
    # Registry fixtures - need function scope for isolation
    "reset_registries": "function"
}

class FixtureInfo:
    """Information about a pytest fixture."""
    
    def __init__(self, name: str, scope: str, file_path: Path):
        self.name = name
        self.scope = scope
        self.file_path = file_path
        self.usages: List[Path] = []
        self.is_expensive = False
        self.is_stateful = False
        self.dependencies: Set[str] = set()
        
    def __repr__(self):
        return f"FixtureInfo({self.name}, {self.scope}, usages={len(self.usages)})"


class FixtureAnalyzer:
    """Analyzes pytest fixtures across the test suite."""
    
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.fixtures: Dict[str, FixtureInfo] = {}
        self.fixture_dependencies: Dict[str, Set[str]] = defaultdict(set)
        
    def analyze_fixtures(self) -> Dict[str, FixtureInfo]:
        """Analyze all fixtures in the test directory."""
        print(f"Analyzing fixtures in {self.test_dir}")
        
        # Find all Python test files
        test_files = list(self.test_dir.glob("**/*.py"))
        
        for test_file in test_files:
            if test_file.name.startswith('test_') or test_file.name == 'conftest.py':
                self._analyze_file(test_file)
                
        # Analyze fixture usage across all test files
        for test_file in test_files:
            self._analyze_fixture_usage(test_file)
            
        return self.fixtures
    
    def _analyze_file(self, file_path: Path):
        """Analyze fixtures defined in a single file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    fixture_info = self._extract_fixture_info(node, file_path, content)
                    if fixture_info:
                        self.fixtures[fixture_info.name] = fixture_info
                        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
    
    def _extract_fixture_info(self, func_node: ast.FunctionDef, file_path: Path, content: str) -> FixtureInfo | None:
        """Extract fixture information from a function node."""
        for decorator in func_node.decorator_list:
            if self._is_fixture_decorator(decorator):
                scope = self._extract_scope(decorator)
                fixture_info = FixtureInfo(func_node.name, scope, file_path)
                
                # Analyze fixture properties
                self._analyze_fixture_properties(func_node, fixture_info, content)
                
                # Extract dependencies from function parameters
                for arg in func_node.args.args:
                    if arg.arg != 'request':  # Skip pytest request parameter
                        fixture_info.dependencies.add(arg.arg)
                        self.fixture_dependencies[func_node.name].add(arg.arg)
                
                return fixture_info
        
        return None
    
    def _is_fixture_decorator(self, decorator: ast.AST) -> bool:
        """Check if a decorator is a pytest.fixture decorator."""
        if isinstance(decorator, ast.Name) and decorator.id == "fixture":
            return True
        elif isinstance(decorator, ast.Attribute) and decorator.attr == "fixture":
            return True
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name) and decorator.func.id == "fixture":
                return True
            elif isinstance(decorator.func, ast.Attribute) and decorator.func.attr == "fixture":
                return True
        return False
    
    def _extract_scope(self, decorator: ast.AST) -> str:
        """Extract scope from fixture decorator."""
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg == "scope" and isinstance(keyword.value, ast.Constant):
                    return keyword.value.value
        return "function"  # Default scope
    
    def _analyze_fixture_properties(self, func_node: ast.FunctionDef, fixture_info: FixtureInfo, content: str):
        """Analyze fixture properties to categorize optimization potential."""
        func_source = ast.get_source_segment(content, func_node) or ""
        docstring = ast.get_docstring(func_node) or ""
        
        # Check for expensive operations
        expensive_patterns = [
            r"AnalyzerEngine\(\)",
            r"DocumentProcessor\(\)",
            r"MaskingEngine\(\)",
            r"load_sample_document",
            r"process_document",
            r"\.analyze\(",
            r"\.process\(",
            r"import.*spacy",
            r"import.*torch",
            r"import.*tensorflow"
        ]
        
        for pattern in expensive_patterns:
            if re.search(pattern, func_source, re.IGNORECASE):
                fixture_info.is_expensive = True
                break
        
        # Check for stateful operations
        stateful_patterns = [
            r"tempfile\.",
            r"temporary",
            r"mock\.",
            r"Mock\(\)",
            r"reset_",
            r"clear_",
            r"registry",
            r"cache"
        ]
        
        for pattern in stateful_patterns:
            if re.search(pattern, func_source, re.IGNORECASE):
                fixture_info.is_stateful = True
                break
                
        # Check docstring for hints
        if any(word in docstring.lower() for word in ["expensive", "slow", "heavy", "initialization"]):
            fixture_info.is_expensive = True
        
        if any(word in docstring.lower() for word in ["isolation", "reset", "clean", "temporary"]):
            fixture_info.is_stateful = True
    
    def _analyze_fixture_usage(self, file_path: Path):
        """Analyze fixture usage in test functions."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Find all test function parameters
            tree = ast.parse(content)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                    for arg in node.args.args:
                        fixture_name = arg.arg
                        if fixture_name in self.fixtures:
                            if file_path not in self.fixtures[fixture_name].usages:
                                self.fixtures[fixture_name].usages.append(file_path)
        
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not analyze usage in {file_path}: {e}")


def generate_optimization_report(fixtures: Dict[str, FixtureInfo]) -> str:
    """Generate a detailed optimization report."""
    
    report = ["# Fixture Scope Optimization Report", ""]
    
    # Current fixture summary
    scope_counts = defaultdict(int)
    for fixture in fixtures.values():
        scope_counts[fixture.scope] += 1
    
    report.extend([
        "## Current Fixture Scopes",
        "",
        f"- Session: {scope_counts['session']} fixtures",
        f"- Module: {scope_counts['module']} fixtures", 
        f"- Function: {scope_counts['function']} fixtures",
        ""
    ])
    
    # Optimization candidates
    session_candidates = []
    module_candidates = []
    keep_function = []
    
    for fixture in fixtures.values():
        current_scope = fixture.scope
        recommended_scope = FIXTURE_SCOPE_PREFERENCE.get(fixture.name)
        
        if recommended_scope and recommended_scope != current_scope:
            if recommended_scope == "session":
                session_candidates.append(fixture)
            elif recommended_scope == "module":
                module_candidates.append(fixture)
        elif fixture.is_expensive and not fixture.is_stateful and current_scope == "function":
            session_candidates.append(fixture)
        elif fixture.is_stateful or "mock" in fixture.name.lower():
            keep_function.append(fixture)
    
    # Session scope candidates
    if session_candidates:
        report.extend([
            "## Convert to Session Scope",
            "",
            "These fixtures can be safely converted to session scope for better performance:",
            ""
        ])
        
        for fixture in session_candidates:
            usage_count = len(fixture.usages)
            report.append(f"- **{fixture.name}** (current: {fixture.scope}, used by {usage_count} files)")
            if fixture.is_expensive:
                report.append(f"  - ✅ Expensive to create")
            if len(fixture.usages) > 1:
                report.append(f"  - ✅ Reused across multiple files")
            if not fixture.is_stateful:
                report.append(f"  - ✅ Stateless, safe for session scope")
            report.append("")
    
    # Module scope candidates  
    if module_candidates:
        report.extend([
            "## Convert to Module Scope",
            "",
            "These fixtures would benefit from module scope:",
            ""
        ])
        
        for fixture in module_candidates:
            usage_count = len(fixture.usages)
            report.append(f"- **{fixture.name}** (current: {fixture.scope}, used by {usage_count} files)")
            report.append("")
    
    # Keep function scope
    if keep_function:
        report.extend([
            "## Keep Function Scope",
            "",
            "These fixtures should remain function-scoped for test isolation:",
            ""
        ])
        
        for fixture in keep_function:
            usage_count = len(fixture.usages)
            reason = "Stateful" if fixture.is_stateful else "Mock/isolation required"
            report.append(f"- **{fixture.name}** (current: {fixture.scope}) - {reason}")
            report.append("")
    
    # Dependency analysis
    report.extend([
        "## Fixture Dependencies",
        "",
        "Key fixture dependency chains that should be optimized together:",
        ""
    ])
    
    # Find fixtures with dependencies
    for fixture_name, fixture in fixtures.items():
        if fixture.dependencies:
            deps = ", ".join(sorted(fixture.dependencies))
            report.append(f"- **{fixture_name}** depends on: {deps}")
    
    report.append("")
    
    # Performance impact estimate
    expensive_fixtures = [f for f in fixtures.values() if f.is_expensive]
    total_usage = sum(len(f.usages) for f in expensive_fixtures)
    
    report.extend([
        "## Expected Performance Impact",
        "",
        f"- {len(expensive_fixtures)} expensive fixtures identified",
        f"- {total_usage} total expensive fixture instantiations across test suite", 
        f"- Converting to session scope could reduce this by ~80-90%",
        ""
    ])
    
    return "\n".join(report)


def main():
    """Run the fixture analysis."""
    test_dir = Path(__file__).parent.parent / "tests"
    
    if not test_dir.exists():
        print(f"Test directory not found: {test_dir}")
        return 1
        
    analyzer = FixtureAnalyzer(test_dir)
    fixtures = analyzer.analyze_fixtures()
    
    print(f"\nFound {len(fixtures)} fixtures")
    
    # Generate report
    report = generate_optimization_report(fixtures)
    
    # Save report
    report_path = Path(__file__).parent.parent / "fixture_optimization_report.md"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"Report saved to: {report_path}")
    
    # Print summary
    print("\n" + "="*50)
    print("FIXTURE OPTIMIZATION SUMMARY")
    print("="*50)
    
    session_candidates = [f for f in fixtures.values() 
                         if f.scope != "session" and 
                         (f.is_expensive or len(f.usages) > 2) and 
                         not f.is_stateful]
    
    print(f"Session scope candidates: {len(session_candidates)}")
    for fixture in session_candidates[:5]:  # Top 5
        print(f"  - {fixture.name} ({fixture.scope} -> session, {len(fixture.usages)} usages)")
    
    if len(session_candidates) > 5:
        print(f"  ... and {len(session_candidates) - 5} more")
    
    return 0


if __name__ == "__main__":
    exit(main())