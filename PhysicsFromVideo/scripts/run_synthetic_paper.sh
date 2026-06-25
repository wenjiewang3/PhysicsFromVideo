#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../synthetic"
python run_experiment.py paper --dataset all --seeds 0 1 2 3 4 "$@"
