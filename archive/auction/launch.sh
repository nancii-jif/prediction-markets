#!/usr/bin/env bash
# Start a run in two detached tmux sessions.
#
#   ./launch.sh configs/2026-09-04-trial.yaml
#
# pm-stream  loops `stream` on STREAM_INTERVAL_S. The module runs one refresh
#            and exits by design, so the cadence lives here rather than in it.
# pm-runner  runs the round loop, which sleeps between rounds itself. Wrapped
#            in `until` so a crash respawns it — next_round_id() resumes the
#            round that died — while a clean finish of a bounded run exits.
#
#   tmux attach -t pm-runner      watch it
#   Ctrl-b d                      DETACH, leaving it running
#   Ctrl-c                        does NOT detach — it SIGINTs the runner
#   tmux kill-session -t pm-runner / -t pm-stream
set -euo pipefail

CONFIG=${1:?usage: ./launch.sh configs/<name>.yaml}
[ -f "$CONFIG" ] || { echo "no such config: $CONFIG" >&2; exit 1; }

# Pin the interpreter by absolute path. tmux runs `$SHELL -c` non-interactively,
# so zsh reads .zshenv but not .zshrc — a pyenv or venv activation that lives in
# .zshrc is simply absent inside the pane, and a bare `python` there resolves to
# a different install with none of this project's dependencies. That failure is
# silent in the worst way: every LLM call raises, the round records no quotes,
# and the run continues looking healthy.
#
# Prefers this project's own .venv, so the answer does not depend on which venv
# happens to be active in the calling shell — the failure that mode produces is
# a silent one, where every LLM call raises and the round records no quotes.
# Falls back to PATH, and an explicit PYTHON= overrides both:
#     PYTHON=~/.pyenv/versions/3.10.8/bin/python ./launch.sh configs/...
HERE=$(cd "$(dirname "$0")" && pwd)
if [ -z "${PYTHON:-}" ] && [ -x "$HERE/.venv/bin/python" ]; then
  PYTHON="$HERE/.venv/bin/python"
fi
PYTHON=$("${PYTHON:-python}" -c 'import sys; print(sys.executable)')
echo "interpreter: $PYTHON"

# Import what the run needs before spending anything on it. `litellm` is
# imported lazily inside Agent._call, so nothing else — the test suite included
# — notices it is missing until the first real quote.
"$PYTHON" - "$CONFIG" <<'PY'
import importlib.util
import sys
from prediction_markets import config

values = config.load(sys.argv[1])
required = ["litellm"] + (["wandb"] if values["WANDB_PROJECT"] else [])
missing = [m for m in required if not importlib.util.find_spec(m)]
if missing:
    raise SystemExit(
        f"missing modules {missing} for {sys.executable}\n"
        f"install them into THIS interpreter: {sys.executable} -m pip install "
        + " ".join(missing)
    )
print(values["RUN_NAME"], values["STREAM_INTERVAL_S"])
PY

read -r RUN_NAME INTERVAL <<<"$("$PYTHON" - "$CONFIG" <<'PY'
import sys
from prediction_markets import config
values = config.load(sys.argv[1])
print(values["RUN_NAME"], values["STREAM_INTERVAL_S"])
PY
)"

# Fail before spending anything if the budget or the config is wrong.
"$PYTHON" -m prediction_markets.runner --config "$CONFIG" --dry-run

# One stream pass in the foreground: it sweeps and seats the markets, so
# round 1 opens on a populated board instead of racing the first refresh.
"$PYTHON" -m prediction_markets.stream --config "$CONFIG"

# caffeinate wraps the runner rather than watching its pid: `caffeinate <cmd>`
# holds the no-sleep assertion for exactly that command's lifetime, so it covers
# every respawn of the until-loop and needs no pid plumbing (pgrep -f matches
# both the wrapper shell and the python process, which `-w` cannot take).
#
# tmux keeps the process alive when the TERMINAL goes away; caffeinate keeps it
# running when the MACHINE would otherwise sleep. They solve different problems
# and a long run needs both — a sleeping laptop freezes a tmux session just as
# dead as a closed one, it simply thaws later with a hole in the round timings.
CAFFEINATE=""
command -v caffeinate >/dev/null && CAFFEINATE="caffeinate -dimsu"

tmux new-session -d -s pm-stream \
  "while true; do '$PYTHON' -m prediction_markets.stream --config '$CONFIG'; sleep $INTERVAL; done"
tmux new-session -d -s pm-runner \
  "until $CAFFEINATE '$PYTHON' -m prediction_markets.runner --config '$CONFIG'; do echo 'runner died; respawning'; sleep 30; done"

echo
echo "run:    $RUN_NAME"
echo "logs:   runs/$RUN_NAME/{runner,stream}.log"
echo "db:     runs/$RUN_NAME/run.db"
echo "attach: tmux attach -t pm-runner"
