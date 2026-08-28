#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${TMPDIR:-/tmp}/buy-point-project-uv-cache}"

uv run --no-sync ruff check src tests scripts
uv run --no-sync mypy src scripts
uv run --no-sync yapf --diff --recursive src tests scripts
uv run --no-sync pytest
