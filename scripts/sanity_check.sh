#!/usr/bin/env bash
set -euo pipefail
python -m compileall -q -x '(^|/)(\.git|\.venv)(/|$)|(^|/)(outputs|data|models)(/|$)' .
ruff check .
ruff format --check .
pytest -q
