# Makefile for CloakPivot Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD  ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile

# Custom targets for CloakPivot documentation

# Clean build directory and rebuild everything
clean-build:
	rm -rf $(BUILDDIR)
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build HTML documentation
html:
	@$(SPHINXBUILD) -M html "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo
	@echo "Build finished. The HTML pages are in $(BUILDDIR)/html."

# Build HTML with all warnings as errors (for CI)
html-strict:
	@$(SPHINXBUILD) -W -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Build and serve documentation locally
serve: html
	@echo "Starting local documentation server..."
	@python -m http.server 8000 --directory $(BUILDDIR)/html
	@echo "Documentation available at http://localhost:8000"

# Check for broken links
linkcheck:
	@$(SPHINXBUILD) -M linkcheck "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Generate API documentation
apidoc:
	@echo "Generating API documentation..."
	@sphinx-apidoc -o api/ ../cloakpivot --force --module-first
	@echo "API documentation generated in api/"

# Build PDF documentation (requires LaTeX)
pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Test notebook execution
test-notebooks:
	@echo "Testing Jupyter notebooks..."
	@jupyter nbconvert --execute --to notebook --inplace notebooks/*.ipynb
	@echo "Notebook tests completed"

# Validate policy examples
validate-policies:
	@echo "Validating policy examples..."
	@if command -v cloakpivot >/dev/null 2>&1; then \
		for policy in ../policies/templates/*.yaml ../policies/industries/*/*.yaml; do \
			echo "Validating $$policy..."; \
			cloakpivot policy validate "$$policy" || echo "Warning: $$policy failed validation"; \
		done; \
	else \
		echo "CloakPivot not installed - skipping policy validation"; \
	fi

# Generate coverage report for documentation
coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo "Coverage report generated in $(BUILDDIR)/coverage/"
	@cat $(BUILDDIR)/coverage/python.txt

# Check spelling (requires sphinxcontrib-spelling)
spelling:
	@$(SPHINXBUILD) -M spelling "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation in different formats
formats: html pdf

# Development build with auto-reload (requires sphinx-autobuild)
dev:
	@echo "Starting development server with auto-reload..."
	@sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port 8000 \
		--open-browser \
		--delay 1 \
		--ignore "*.tmp" \
		--ignore "*~"

# Build documentation for GitHub Pages
github-pages: clean-build
	@echo "Building documentation for GitHub Pages..."
	@touch $(BUILDDIR)/html/.nojekyll
	@echo "GitHub Pages build complete"

# Complete documentation build and validation
full-build: clean-build validate-policies test-notebooks linkcheck coverage
	@echo "Full documentation build and validation complete"

# Quick build for development
quick: html
	@echo "Quick build complete"

# Install documentation dependencies
install-deps:
	@echo "Installing documentation dependencies..."
	@pip install -r requirements-docs.txt
	@echo "Dependencies installed"

# Check for outdated dependencies
check-deps:
	@echo "Checking for outdated documentation dependencies..."
	@pip list --outdated | grep -E "(sphinx|myst|nbsphinx)"

# Generate requirements file for documentation
requirements:
	@echo "Generating requirements-docs.txt..."
	@echo "# Documentation dependencies" > requirements-docs.txt
	@echo "sphinx>=5.0.0" >> requirements-docs.txt
	@echo "sphinx-rtd-theme>=1.2.0" >> requirements-docs.txt
	@echo "myst-parser>=0.18.0" >> requirements-docs.txt
	@echo "nbsphinx>=0.8.0" >> requirements-docs.txt
	@echo "sphinx-click>=4.0.0" >> requirements-docs.txt
	@echo "sphinxcontrib-spelling>=7.0.0" >> requirements-docs.txt
	@echo "sphinx-autobuild>=2021.3.14" >> requirements-docs.txt
	@echo "Requirements file generated"

# Statistics about documentation
stats:
	@echo "Documentation Statistics:"
	@echo "========================="
	@find . -name "*.rst" -type f | wc -l | xargs echo "RST files:"
	@find . -name "*.md" -type f | wc -l | xargs echo "Markdown files:"
	@find . -name "*.ipynb" -type f | wc -l | xargs echo "Jupyter notebooks:"
	@find . -name "*.py" -type f | wc -l | xargs echo "Python files:"
	@find . -name "*.yaml" -path "*/policies/*" -type f | wc -l | xargs echo "Policy files:"
	@echo
	@wc -l $(shell find . -name "*.rst" -o -name "*.md" -type f) | tail -1 | xargs echo "Total lines in docs:"

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)