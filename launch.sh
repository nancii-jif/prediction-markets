#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
if [[ -x .venv/bin/python ]]; then
    runner_python=.venv/bin/python
else
    runner_python=python
fi
if [[ $# -eq 0 ]]; then
    set -- --config configs/mvp.yaml
fi
exec "$runner_python" -m prediction_markets run "$@"
