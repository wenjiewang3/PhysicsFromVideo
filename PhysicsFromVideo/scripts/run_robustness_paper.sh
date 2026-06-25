#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../robustness"
python run_robustness.py paper --setting all --seeds 42 "$@"
