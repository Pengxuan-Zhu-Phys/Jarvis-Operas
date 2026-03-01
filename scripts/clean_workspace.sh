#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

find "$ROOT_DIR" -type d -name '__pycache__' -prune -exec rm -rf {} +
rm -rf "$ROOT_DIR/.pytest_cache"
rm -rf "$ROOT_DIR/dist"
rm -rf "$ROOT_DIR"/*.egg-info

echo "Workspace cache/build artifacts cleaned."
