PYTHON ?= python3

.PHONY: help install test lint format tables figures animation page report clean

help:
	@echo "install    install the runtime dependencies plus pytest and ruff"
	@echo "test       run the unit test suite"
	@echo "lint       run ruff over the sources"
	@echo "format     apply ruff's automatic fixes"
	@echo "tables     recompute Tables 1-4 and compare them to the report"
	@echo "figures    regenerate every figure, including the solution animation"
	@echo "page       re-derive all 91 numeric cells on the project page"
	@echo "report     build baocao.pdf with latexmk (needs a TeX distribution)"
	@echo "clean      remove build artifacts"

# requirements.txt stays runtime-only, so the report's figures can be rebuilt
# without pulling in a test runner; the dev tools are named here instead.
install:
	$(PYTHON) -m pip install -r requirements.txt pytest ruff

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

format:
	$(PYTHON) -m ruff check --fix .

# Recomputes every number the report quotes and exits non-zero on a mismatch.
tables:
	$(PYTHON) reproduce_tables.py

figures:
	$(PYTHON) make_figs.py
	$(PYTHON) make_figs.py --web

page:
	$(PYTHON) check_page_numbers.py

report:
	latexmk -pdf -interaction=nonstopmode baocao.tex

clean:
	rm -rf .pytest_cache .ruff_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	latexmk -C || true
