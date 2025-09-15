# CloakPivot v2.0 Code Review Plan

## 📋 Overview
This document tracks the comprehensive review and update of CloakPivot documentation, tests, and CI/CD pipeline following major source code improvements.

## 🎯 Objectives
1. Rewrite all documentation to match updated source code
2. Rewrite test suite for comprehensive coverage
3. Fix CI/CD pipeline and GitHub Actions
4. Update README with accurate information

---

## 📚 Phase 1: Documentation Rewrite

### Current Status
- ✅ Source code (cloakpivot/) - Updated and working
- ✅ Examples (examples/) - All working correctly
- ❌ Documentation (docs/) - Needs complete rewrite
- ⚠️  README.md - Needs updates

### Tasks

#### 1.1 API Documentation (docs/API.md)
- [ ] Audit current API.md against source code
- [ ] Document CloakEngine class and all methods
- [ ] Document CloakEngineBuilder pattern
- [ ] Document all strategy classes (Hash, Surrogate, etc.)
- [ ] Document configuration options
- [ ] Add code examples for each API method
- [ ] Include type hints and return values

#### 1.2 Migration Guide (docs/MIGRATION.md)
- [ ] Review for v2.0.0 breaking changes
- [ ] Add migration examples from v1.x
- [ ] Document deprecated features
- [ ] Provide upgrade path guidance

#### 1.3 New Documentation Files
- [ ] Create docs/installation.md - Installation and setup guide
- [ ] Create docs/configuration.md - Configuration reference
- [ ] Create docs/strategies.md - Masking strategies guide
- [ ] Create docs/advanced.md - Advanced usage patterns
- [ ] Create docs/troubleshooting.md - Common issues and solutions
- [ ] Create docs/examples.md - Link to working examples with explanations

---

## 🧪 Phase 2: Test Suite Rewrite

### Current Test Files to Review/Rewrite
```
tests/
├── test_analyzer.py
├── test_cloak_engine_builder.py
├── test_cloak_engine_examples.py
├── test_configuration.py
├── test_enhanced_strategies.py
├── test_fixture_scoping.py
├── test_package.py
├── test_policy_loader.py
├── test_property_masking.py
├── test_results.py
├── test_setup_models.py
├── test_strategy_surrogate_integration.py
├── conftest.py
└── core/
    ├── test_cloakmap_enhancement.py
    └── test_presidio_mapper.py
```

### Tasks

#### 2.1 Core Test Updates
- [ ] Review and fix test_analyzer.py
- [ ] Review and fix test_cloak_engine_builder.py
- [ ] Review and fix test_configuration.py
- [ ] Update conftest.py with proper fixtures

#### 2.2 Strategy Tests
- [ ] Fix test_enhanced_strategies.py
- [ ] Fix test_strategy_surrogate_integration.py
- [ ] Add tests for all masking strategies

#### 2.3 Integration Tests
- [ ] Fix test_cloak_engine_examples.py
- [ ] Add end-to-end workflow tests
- [ ] Test all examples from examples/ directory

#### 2.4 Test Infrastructure
- [ ] Ensure all tests use proper fixtures and mocking
- [ ] Remove hardcoded paths and dependencies
- [ ] Add parametrized tests where appropriate
- [ ] Ensure tests can run in CI environment

#### 2.5 Coverage
- [ ] Add missing test coverage for new features
- [ ] Aim for >80% code coverage
- [ ] Add coverage reporting

---

## 🚀 Phase 3: CI/CD Pipeline

### Current CI Files
- .github/workflows/ci.yml
- .github/workflows/release.yml

### Tasks

#### 3.1 CI Workflow Updates
- [ ] Review and update .github/workflows/ci.yml
- [ ] Ensure Python versions are correct (3.8+)
- [ ] Fix any failing CI pipeline tests
- [ ] Add test coverage reporting to CI
- [ ] Add linting and formatting checks
- [ ] Ensure all dependencies are properly installed

#### 3.2 Release Workflow
- [ ] Review .github/workflows/release.yml
- [ ] Ensure PyPI publishing is configured
- [ ] Add automatic changelog generation
- [ ] Add version tagging

---

## 📖 Phase 4: README Updates

### Tasks
- [ ] Update installation instructions
- [ ] Update basic usage examples to match examples/simple_usage.py
- [ ] Update advanced usage to match examples/advanced_usage.py
- [ ] Add troubleshooting section
- [ ] Update badges (version, Python support, license)
- [ ] Add contributing guidelines
- [ ] Add link to full documentation
- [ ] Ensure all code examples are tested and working

---

## 🔄 Execution Strategy

### Session Management
Given the scope of work, we'll need multiple sessions:

1. **Session 1** (Current): Setup and Phase 1 (Documentation)
2. **Session 2**: Phase 2.1-2.3 (Core Tests)
3. **Session 3**: Phase 2.4-2.5 (Test Infrastructure & Coverage)
4. **Session 4**: Phase 3 (CI/CD)
5. **Session 5**: Phase 4 (README) and final review

### Progress Tracking
- Use this document to track completed items
- Mark items with ✅ when complete
- Add notes for any blockers or issues
- Update after each session

### Priority Order
1. **High Priority**: Fix failing tests that block CI
2. **High Priority**: Update critical documentation (API.md)
3. **Medium Priority**: Complete test coverage
4. **Medium Priority**: Update README
5. **Low Priority**: Additional documentation enhancements

---

## 📝 Notes

### Dependencies to Consider
- Docling for document processing
- Presidio for PII detection
- pytest for testing
- GitHub Actions for CI/CD

### Key Files to Reference
- examples/simple_usage.py - Basic API usage
- examples/advanced_usage.py - Advanced features
- examples/pdf_workflow.py - Full workflow example
- cloakpivot/engine.py - Main engine implementation
- cloakpivot/strategies/ - Masking strategies

---

## ✅ Completion Checklist

- [ ] All documentation matches source code
- [ ] All tests pass locally
- [ ] CI pipeline passes all checks
- [ ] README is accurate and complete
- [ ] Examples can be copy-pasted and run
- [ ] Coverage is >80%
- [ ] Ready for v2.0.0 release

---

*Last Updated: 2025-01-15*
*Next Session: Start with Phase 1 - Documentation Rewrite*