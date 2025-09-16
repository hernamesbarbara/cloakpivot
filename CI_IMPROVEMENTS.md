# CI Performance Improvements

## Problem
GitHub Actions CI was taking 20+ minutes, with dependency installation being the main bottleneck.

## Implemented Solutions

### 1. Dependency Caching
- Added pip package caching to avoid re-downloading dependencies
- Added spacy model caching with separate keys for small/large models
- Cache keys based on `pyproject.toml` hash for automatic invalidation

### 2. Small Spacy Model for PRs
- Switched from `en_core_web_lg` (500MB+) to `en_core_web_sm` (12MB) for PR tests
- CloakPivot already supports `MODEL_SIZE` environment variable (defaults to "small")
- Large model only used for full test suite on main branch

### 3. Workflow Split & Parallelization
- **lint**: Fast linting and type checking (runs first, fails fast)
- **test**: Unit and integration tests with small model (runs on PRs)
- **test-full**: Complete test suite with large model (main branch only)
- Added concurrency control to cancel redundant runs
- Reduced Python matrix from [3.11, 3.12] to [3.12] for PRs

## Expected Impact

| Before | After | Savings |
|--------|-------|---------|
| 20+ min | ~5-7 min | ~70% |

### Breakdown:
- Dependency installation: 10+ min → 1-2 min (cached)
- Spacy model download: 2-3 min → 10 sec (small model, cached)
- Test execution: ~8 min → ~3-4 min (parallel jobs, single Python version)

## Configuration

The system respects these environment variables:
- `MODEL_SIZE`: Controls spacy model size (small/medium/large)

## Usage

### For PRs:
- Runs with small model and single Python version
- Skips heavy integration tests

### For main branch:
- Full test suite with large model
- All Python versions tested

### Force full tests on PR:
Add label `full-tests` to the PR to run complete test suite.

## Rollback

If issues arise, the original workflow is preserved in git history and can be restored with:
```bash
git revert <commit-hash>
```