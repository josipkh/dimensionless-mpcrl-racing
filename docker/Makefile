SHELL := /usr/bin/env bash

# This Makefile lives in docker/, but it always operates on the repository root.
DOCKER_DIR := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
ROOT := $(abspath $(DOCKER_DIR)/..)

COMPOSE ?= docker compose -f $(DOCKER_DIR)/compose.yaml
SERVICE ?= reproduce
RL_SUITE ?= main

GIT_COMMIT := $(shell git -C $(ROOT) rev-parse HEAD 2>/dev/null || echo unknown)
GIT_DIRTY := $(shell if [ -n "$$(git -C $(ROOT) status --porcelain 2>/dev/null)" ]; then echo dirty; else echo clean; fi)
export GIT_COMMIT GIT_DIRTY

.DEFAULT_GOAL := reproduce
.PHONY: help init metadata build reproduce analysis test rl full shell check-docker

help:
	@echo "Dimensionless MPC-RL reproducibility targets"
	@echo
	@echo "Run these from the repository root, for example 'make -C docker'."
	@echo
	@echo "  make -C docker              Build the image and run tests + analyses"
	@echo "  make -C docker reproduce    Build the image and run tests + all sensitivity analyses"
	@echo "  make -C docker analysis     Alias for 'make -C docker reproduce'"
	@echo "  make -C docker test         Run only the pytest suite in the container"
	@echo "  make -C docker rl           Run the main SAC-ZOP experiments (RL_SUITE=main)"
	@echo "  make -C docker rl RL_SUITE=transfer   Run the transfer experiments"
	@echo "  make -C docker rl RL_SUITE=all        Run baseline, main, and transfer experiments"
	@echo "  make -C docker full         Run tests + analyses, then every RL experiment"
	@echo "  make -C docker shell        Open a shell in the built container"
	@echo
	@echo "Useful environment variables:"
	@echo "  RL_SEEDS='0 1 2 3 4'  Seeds (space-separated)"
	@echo "  RL_JOBS=8             Maximum concurrent RL runs"
	@echo "  RL_MAIN_STEPS=100000  SAC-ZOP training steps per run"
	@echo "  ANALYSIS_STEPS=...    Comma-separated subset of sensitivity analyses"
	@echo
	@echo "Results are written to ./results (analyses) and ./output (RL runs)."

init:
	git -C $(ROOT) submodule update --init --recursive

check-docker:
	@docker compose version >/dev/null 2>&1 || { echo "Docker Compose is not available. Install Docker with the Compose plugin." >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { echo "The Docker daemon is not running. Start Docker and try again." >&2; exit 1; }

metadata:
	@mkdir -p $(ROOT)/results
	@{ \
		echo "# Host repository state captured by make"; \
		echo "date_utc: $$(date -u +%Y-%m-%dT%H:%M:%SZ)"; \
		echo; \
		echo "## commit"; \
		git -C $(ROOT) rev-parse HEAD 2>/dev/null || echo unknown; \
		echo; \
		echo "## status"; \
		git -C $(ROOT) status --short 2>/dev/null || true; \
		echo; \
		echo "## submodules"; \
		git -C $(ROOT) submodule status --recursive 2>/dev/null || true; \
	} > $(ROOT)/results/host-git-info.txt

build: check-docker init metadata
	$(COMPOSE) build

reproduce: build
	$(COMPOSE) run --rm $(SERVICE) analysis

analysis: reproduce

test: build
	$(COMPOSE) run --rm $(SERVICE) test

rl: build
	$(COMPOSE) run --rm $(SERVICE) rl $(RL_SUITE)

full: build
	$(COMPOSE) run --rm $(SERVICE) full

shell: build
	$(COMPOSE) run --rm --entrypoint bash $(SERVICE)
