#!/usr/bin/env bash
# Runs the test suite and both experiments with the commands given in README.md.
set -uo pipefail

python -m pytest -m "not market_data"
python -m exp.exp1
python -m exp.exp2
