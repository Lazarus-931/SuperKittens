SHELL := /bin/bash
PYTHON ?= python3

.PHONY: help build clean install dev test bench pr-list status

help:
	@echo "SuperKittens dev targets:"
	@echo "  make build         — compile metallib + dylib"
	@echo "  make clean         — wipe build/"
	@echo "  make install       — pip install -e . into current venv"
	@echo "  make dev           — install with [dev,hf,tokenizer] extras"
	@echo "  make test          — run pytest (if any tests exist)"
	@echo "  make bench SPEC=… GGUF=…  — SK vs llama.cpp side-by-side"
	@echo "  make pr-list       — show open PRs"
	@echo "  make status        — repo + branch summary"

build:
	./build.sh

clean:
	rm -rf build/

install:
	$(PYTHON) -m pip install -e .

dev:
	$(PYTHON) -m pip install -e .[dev,hf,tokenizer]

test:
	$(PYTHON) -m pytest -x SuperKittens/

bench:
	@if [ -z "$(SPEC)" ] || [ -z "$(GGUF)" ]; then echo "usage: make bench SPEC=<spec> GGUF=<path>"; exit 1; fi
	$(PYTHON) bench.py --spec $(SPEC) --gguf $(GGUF)

pr-list:
	gh pr list

status:
	@echo "branch: $$(git branch --show-current)"
	@echo "ahead-of-main:"
	@for b in $$(git for-each-ref --format='%(refname:short)' refs/heads/); do \
	  n=$$(git log --oneline $$b ^origin/main 2>/dev/null | wc -l | tr -d ' '); \
	  [ $$n -gt 0 ] && echo "  $$b: $$n commits"; \
	done
	@echo "build artifacts:"
	@ls -lh build/libsk.dylib build/libsk.metallib 2>/dev/null || echo "  (none — run make build)"
