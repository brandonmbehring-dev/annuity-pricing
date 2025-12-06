# Makefile for Sphinx documentation
#
# Usage:
#   make html      - Build HTML documentation
#   make clean     - Remove build directory
#   make livehtml  - Start live-reload server (requires sphinx-autobuild)
#   make linkcheck - Check external links

SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml

# Clean build directory
clean:
	rm -rf $(BUILDDIR)

# Live reload server (requires: pip install sphinx-autobuild)
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
