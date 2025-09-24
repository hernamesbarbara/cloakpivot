# Development Guidelines for CloakPivot Python Package

Follow these guidelines precisely when working with this codebase.

## Core Development Rules

1. Package Management

   - Use Python virtual environment in the root of this project: venv/
   - Use `pip`
   - Dependencies in `pyproject.toml`

2. Code Quality

   - Type hints required for all code
   - Public APIs must have docstrings
   - Functions must be focused and small
   - Follow existing patterns exactly
   - Line length: 100 chars maximum

3. Testing Requirements

   - Respect our detailed testing philosophy and guidelines outlined in TESTING.md
   - Testing with pytest via `make test` command
   - Coverage report generated via `make test` command
   - New features require tests
   - Bug fixes require regression tests
   - Prefer testing with sample data vs. generating complex mock data in python code

4. Code Style
   - PEP 8 naming (snake_case for functions/variables)
   - Class names in PascalCase
   - Constants in UPPER_SNAKE_CASE
   - Document with docstrings
   - Use f-strings for formatting

## Development Philosophy

- **Simplicity**: Write simple, straightforward code
- **Readability**: Make code easy to understand
- **Performance**: Consider performance without sacrificing readability
- **Maintainability**: Write code that's easy to update
- **Testability**: Ensure code is testable
- **Reusability**: Create reusable components and functions
- **Less Code = Less Debt**: Minimize code footprint

## Coding Best Practices

- **Early Returns**: Use to avoid nested conditions
- **Descriptive Names**: Use clear variable/function names (prefix handlers with "handle")
- **Constants Over Functions**: Use constants where possible
- **DRY Code**: Don't repeat yourself
- **Functional Style**: Prefer functional, immutable approaches when not verbose
- **Minimal Changes**: Only modify code related to the task at hand
- **Function Ordering**: Define composing functions before their components
- **TODO Comments**: Mark issues in existing code with "TODO:" prefix
- **Simplicity**: Prioritize simplicity and readability over clever solutions
- **Build Iteratively** Start with minimal functionality and verify it works before adding complexity
- **Run Tests**: Test your code frequently with realistic inputs and validate outputs
- **Functional Code**: Use functional and stateless approaches where they improve clarity
- **Clean logic**: Keep core logic clean and push implementation details to the edges
- **File Organsiation**: Balance file organization with simplicity; CloakPivot is meant to be a small python package and used only by a few developers. Do not overcomplicate things.

## System Architecture

CloakPivot is a Python package that brings Presidio PII Anonymization faculties together with Docling's powerful DoclingDocument object.

Devs use CloakPivot for the ability to construct processing pipelines like this: PDF → DoclingDocument → CloakEngine masked PII document → Markdown rendering.

Always keep in mind that CloakPivot is meant to make using Docling and Presidio together easier. CloakPivot should never recreate functionality that is already available in one of those two battletested Python packages. Rather, CloakPivot should make it easy to work with powerful classes, functions, data structures, and processing tools and pipelines that are available already within Docling and Presidio.

## Core Components

- `pyproject.toml`: project configuration, dependencies, testing tools, and other project settings
- `Makefile`: build pipeline, testing pipeline, linting rules, CI pipeline used locally and remotely via github checks
- `.github/workflows/ci.yml`: github actions should match `Makefile`

## Sample Data for Testing, Debugging, and New Feature Dev

- Always remember to use realistic data included in the project in the data/ directory.
- Various formats are available data/json/, data/pdf/, data/md/, etc. 
- Example scripts in examples/ should always use realistic data rather than generate mock data internally.

## Error Resolution

1. CI Failures

- Fix order:
  1. Formatting
  2. Type errors
  3. Linting
- Type errors:
  - Get full line context
  - Check Optional types
  - Add type narrowing
  - Verify function signatures

2. Common Issues

- Line length:
  - Break strings with parentheses
  - Multi-line function calls
  - Split imports
- Types:
  - Add None checks
  - Narrow string types
  - Match existing patterns

3. Best Practices

- Check git status before commits
- Run formatters before type checks
- Keep changes minimal
- Follow existing patterns
- Document public APIs
- Test thoroughly
