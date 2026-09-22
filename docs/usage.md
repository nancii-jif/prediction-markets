# Detailed usage guide

Independent participant loops share one SQLite exchange and a remote inference
backend: Qwen on vLLM or OpenAI-hosted models. Kalshi supplies public market definitions and
lifecycle updates only. **No orders are sent to Kalshi.**

See the [project overview](../README.md), [package architecture](architecture.md), and [design notes](design.md). Commands below run from the repository root.

## Local setup

Use Python 3.10+; local checks were run with Python 3.10.8. From this directory:

```bash
python -m venv .venv  # only if a virtual environment does not already exist
source .venv/bin/activate
python -m pip install -e '.[analysis,dev]'
python -m pytest -q -p no:cacheprovider
```

The runtime needs neither Docker nor a local GPU, PyTorch, W&B, tmux, or a
separate market listener. The root contains only the current MVP; the complete
previous source tree, auction configs/docs, and old experiment data are under
[archive/](../archive/README.md). The archive is excluded from test discovery and
is not a second supported runtime. `configs/mvp.yaml` selects Qwen;
`configs/openai.yaml` selects OpenAI with the same tools and exchange.

## Tavily news tool

Every participant can call `search_news(query)`, regardless of its model. The
shared tool dispatcher uses [Tavily's Python SDK](https://docs.tavily.com/sdk/python/reference)
with `topic="news"` and `search_depth="advanced"`, returning up to five source
titles, URLs, snippets, relevance scores and publication dates when provided.
Searches have a 30-second deadline, run asynchronously, and are never retried
automatically. Returned news is data, not instructions.

The backend sends Tavily's [`exclude_domains`](https://docs.tavily.com/documentation/api-reference/endpoint/search)
for `kalshi.com`, `polymarket.us`, and `polymarket.com`. It also drops returned
URLs on these domains or any subdomain, including `news.kalshi.com`, before
exposing results to an agent. Both sync and async entry points enforce this
policy. The exclusions are absent from the agent prompt, tool schema and
response metadata; agents cannot override them. A filtered search can return
fewer than five sources, or an ordinary empty result, and still costs one news
call. This blocks those sites, but third-party reporting can still quote their
odds. Restart the runner to apply backend changes to a new experiment; existing
transcripts and already-loaded processes are unaffected.

Put `TAVILY_API_KEY=your-key` in the private root `.env` or export it in your
shell. The runner loads that file. Only the environment-variable name is saved
in configuration; the key is never sent to the model. The SDK is included in
`requirements.txt`.

```yaml
news_tool_minutes: 10  # Elapsed cost per search; no search-count limit.
tavily_api_key_env: TAVILY_API_KEY
```

News searches have no daily or whole-run count cap. Each dispatched search has
an elapsed time cost (`news_tool_minutes`, ten minutes by default), including
returned errors. Search activity is recorded in `agent_state.news_tool_calls`
for analysis, without limiting dispatch. News, orders, and reads are governed by
time costs and the shared trading session. Research is available before any orders.

For other agent frameworks, reuse the plain sync/async Python functions and
JSON function schema from `prediction_markets.news`:

```python
from prediction_markets.integrations.tavily import NEWS_TOOL_SCHEMA, search_news, asearch_news

result = search_news("latest central bank policy news")
# In an async agent loop: result = await asearch_news("latest central bank policy news")
```

These standalone functions read the environment and return JSON-compatible
dictionaries; they do not load YAML or track agent identity. The participant
harness applies the configured elapsed time cost before the next action. There
is no total news-search count limit. Other frameworks can register either entry point directly.

## OpenAI backend: local or Modal CPU

Put `OPENAI_API_KEY` in the private root `.env` or export it, along with
`TAVILY_API_KEY` when news is enabled. Then run the existing launcher:

```bash
./launch.sh --config configs/openai.yaml
```

The example uses `model_provider: openai`, `openai_api: responses`, and
`gpt-5.6-luna` with `openai_reasoning_effort: high`.
Change `model_name` to select another model available to your
account; `model_revision` stays `null` because API snapshots are specified in
the model name. `OPENAI_BASE_URL` is optional and defaults to
`https://api.openai.com/v1`. The Qwen config and default launcher remain available.
See the [Luna model documentation](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

Both providers use the same prompt, tool schemas, private conversation, session timing,
exchange, transcript format, concurrency limit and shutdown behavior. The
OpenAI transport supports both [Responses](https://developers.openai.com/api/docs/guides/reasoning)
and [Chat Completions](https://developers.openai.com/api/reference/resources/chat/subresources/completions/methods/create).
`openai_api` defaults to `chat_completions` for compatibility with older configs;
the Luna example explicitly selects `responses`. Both disable parallel tool
calls and use `store: false`. Responses sends `max_output_tokens`; Chat sends
`max_completion_tokens`. Both omit
vLLM's `top_k`, sampling seed and chat-template options. `temperature` and
`top_p` can be `null` to omit them; choose parameters supported by the selected
model. `enable_thinking` remains false for OpenAI; use the optional
`openai_reasoning_effort` instead. The Luna configuration uses:

```yaml
model_name: gpt-5.6-luna
openai_api: responses
openai_reasoning_effort: high
temperature: null
top_p: null
max_output_tokens: 4096
```

Unlike `null` (omit the option), `openai_reasoning_effort: none` explicitly
requests no reasoning on models that support it. Luna currently rejects
function tools on `/v1/chat/completions` when this setting is omitted or enables
reasoning. Use `openai_api: responses` to enable Luna reasoning with tools;
changing only the effort flag to `low` or `medium` is insufficient. This was
verified against the endpoint while diagnosing an HTTP 400. For reasoning models,
`max_output_tokens` includes both hidden reasoning and visible output; allow
enough room to complete tool arguments. Truncated responses remain rejected.
[Luna's supported reasoning levels](https://developers.openai.com/api/docs/models/gpt-5.6-luna)
are model-specific; they do not all apply to older models.

The Responses adapter preserves ordered output and encrypted reasoning in each
agent's private history and replays it alongside tool results. Function schemas
use explicit `strict: false` to preserve the shared tools' optional arguments.
Incomplete responses never execute tools. Usage includes the native Responses
fields plus the existing prompt/completion token fields for transcript analysis.

OpenAI runs use `tiktoken` for local history-size estimates, including tool
schemas and framing headroom. They download tokenizer vocabulary if uncached,
but no Hugging Face tokenizer or model weights. The API's exact chat template
is not public, so this is an estimate, while API-reported usage is saved in each
response alongside the returned model ID and any system fingerprint. Opaque
reasoning is counted using reported reasoning tokens, rather than treating its
encrypted representation as prompt text. The example
keeps the same 16,384 working-token budget and uses a 32,768 local context guard,
below Luna's actual context limit. OpenAI runs do not use reproducible sampling
seeds or probe/warm a GPU endpoint.

To host the whole experiment on Modal, use `deploy/modal_openai.py`. Install
the deployment-only SDK (`python -m pip install 'modal==1.5.5'`) and authenticate
with `modal setup`. Create a Modal Secret named `prediction-markets-openai`
containing `OPENAI_API_KEY` and `TAVILY_API_KEY` when news is enabled. Then:

```bash
modal run --detach deploy/modal_openai.py --config configs/openai.yaml
```

This runs the existing agent loops, market polling and exchange in **one CPU
container, with no GPU**. Only the runtime Python package and validated config
values are uploaded; credentials come from the Secret. SQLite runs on local
container disk. On completion or a handled failure, the closed database and
logs are copied to the `prediction-markets-runs` Volume and committed. Download
them using the run ID printed in the logs:

```bash
python -m prediction_markets download RUN_ID

# Raw download only (the existing parent directory is essential):
mkdir -p ./runs
modal volume get prediction-markets-runs RUN_ID ./runs --force
```

An abrupt container loss can lose artifacts that have not reached the Volume.
Saved runs can be continued using `--resume-from` (below). The CPU function permits runs up to 23 hours and has no
warm minimum or configured task retries. OpenAI API usage, CPU runtime, storage
and news calls are billed by their providers. An existing Qwen deployment
remains separate: stop `prediction-markets-qwen` when it is no longer needed
to stop its warm GPU charges. See [Modal CPU resources](https://modal.com/docs/guide/resources)
and [persistent Volumes](https://modal.com/docs/guide/volumes).

## Qwen inference deployment — requires authorization and incurs costs

Do not deploy or start the live experiment until GPU spending and live execution
have been approved. The script is configured for one **A100-40GB**, one fixed
replica, eight concurrent inputs and a 20,480-token context. Modal provides the
container; there is no local Docker setup. The explicit GPU name avoids an
automatic 80-GB upgrade. See [Modal GPU options](https://modal.com/docs/guide/gpu).

Pinned serving configuration:

| Component | Pin |
| --- | --- |
| Model and tokenizer | `Qwen/Qwen3.5-9B` |
| Immutable model/tokenizer revision | `c202236235762e1c871ad0ccb60c8ee5ba337b9a` |
| vLLM | `0.17.1` |
| Transformers / tokenizers / Jinja2 | `4.57.6` / `0.22.2` / `3.1.6` locally and remotely |
| Modal SDK used for local import checks | `1.5.5` |
| Server Python / base image | `3.12` / `nvidia/cuda:12.9.0-devel-ubuntu22.04` |
| Tool / reasoning parsers | `qwen3_coder` / `qwen3` |

The tool-parser choice follows the [Qwen model card](https://huggingface.co/Qwen/Qwen3.5-9B).
The server is text-only, BF16, with thinking disabled by default and on every
client request. No quantization, extra replicas or automatic GPU fallback.
The vLLM version is a **candidate pin, not yet GPU-tested here**; see its
[release notes](https://github.com/vllm-project/vllm/releases/tag/v0.17.1).

After approval:

1. Install the deployment-only SDK and authenticate your Modal account:

   ```bash
   python -m pip install 'modal==1.5.5'
   modal setup
   ```

2. In the Modal dashboard, create a secret named `prediction-markets-model`
   with one key, `VLLM_API_KEY`, containing a newly generated strong random token.
   Keep it private. vLLM uses it for bearer authentication on `/v1` requests.

3. Deploy **by this file path**, not as a Python package:

   ```bash
   modal deploy deploy/modal_qwen.py
   ```

   Modal includes this standalone serving file and installs/downloads public
   dependencies and model weights remotely. No database, `.env`, repository
   directory, or Kalshi credential is mounted. Inspect startup logs for the
   actual GPU, loaded revision, parser initialization and usable context/cache.
   Do not assume eight long requests fit or meet the 120-second deadline until
   the workload has run successfully on that GPU.

4. Put the endpoint URL (with `/v1` appended) and the same API key in a private
   repository-root `.env` file, or export them in your shell:

   ```dotenv
   MODEL_BASE_URL=https://YOUR-MODAL-ENDPOINT.modal.run/v1
   MODEL_API_KEY=YOUR-PRIVATE-TOKEN
   ```

   `.env`, `data/`, and `runs/` are gitignored. Never commit credentials. The
   saved settings contain environment-variable names, not secret values.

5. Before the live experiment, verify `/v1/models` rejects a request without a
   token and accepts an authenticated request. The model ID must match the pin
   above. A responding endpoint alone does not establish tool-calling or memory
   capacity; the bounded experiment is the remaining acceptance test.

**Stop the deployed Modal app when finished** (including failed tests):

```bash
modal app stop prediction-markets-qwen
```

`min_containers=1` keeps the replica warm and billable. Stopping the local runner
does **not** stop the deployment. A redeploy should not overlap an experiment.

## Run an experiment

```bash
python -m prediction_markets run --config configs/mvp.yaml
```

`./launch.sh` forwards to the same command using `.venv/bin/python` when present.
It does not restart failed runs. Keep the computer awake for the experiment.

For Qwen, the runner loads only the pinned tokenizer artifacts into `data/huggingface/`,
creates a fresh `runs/<UTC-time>-<unique-id>/run.db`, then selects ten qualifying
markets from distinct events. Selection uses the **earlier of expected and
latest expiration** within the configured hour window (currently 12–72 hours
from startup), with at least 500 contracts of 24-hour volume. If only one
expiration is provided, use it; if neither is provided, exclude the market.
Malformed supplied timestamps are still rejected. Keep both original fields
unchanged in storage and agent observations; the earlier value is only a
selection rule, not a settlement trigger. Each market must already be tradable:
`status == active` and `open_time <= now < close_time`. Missing or invalid opening
times are excluded. The config keys remain `expected_expiration_min_hours` and
`expected_expiration_max_hours`; old `_days` keys are not accepted.
If fewer than ten qualify, startup fails without inference; the filters are
not relaxed. Complete market rules and tool schemas must fit the input
bound or startup fails.
Older complete private conversation groups are trimmed later as needed.

Initial selection also applies an uncertainty price range:

```yaml
uncertainty_price_range_cents: [10, 90]  # [lo, hi], inclusive YES cents
```

The filter uses `last_price_dollars` (the latest YES trade price) from the first
startup discovery response. A market must satisfy `lo <= 100 * price <= hi`
before sector rotation and volume ranking. Fractional-cent prices are compared
exactly; missing or invalid prices are excluded, with no bid/ask fallback.
Bounds must be two integers with `0 <= lo <= hi <= 100`; `[0, 100] allows the
full price range while still requiring a valid price. The filter is never
relaxed to fill the cohort. Later price changes do not remove markets or admit
replacements. Bounds are saved with the run configuration; source prices stay
out of market rows and agent briefs.

The input bound is **measured, not fixed**. The system prompt and the ten tool
schemas are byte-identical for every agent and every turn, so the server caches that prefix with
`--enable-prefix-caching` rather than re-prefilling it. Each agent is then given
`working_token_budget` (16,384) on top of the measured prefix. The
brief and the conversation history share that budget; oldest complete groups are
dropped to stay inside it. A longer system prompt therefore raises the limit
instead of squeezing out history. Note that only the prefix is cacheable: market
rules live in the brief, which is rebuilt every turn and sits at the end.

Startup selection is limited to Elections, Politics, Culture, Commodities,
Climate, Economics, Mentions, Finance, and Tech & Sciences. The
`allowed_market_categories` list in `configs/mvp.yaml` uses Kalshi's API labels:
Culture = `Entertainment`, Climate = `Climate and Weather`, Finance = `Financials`,
and Tech & Sciences = `Science and Technology`.

**That list's order is significant.** Eligible markets are queued by their
primary series category, and the cohort is filled one sector at a time in
configuration order, repeating from the top until ten are chosen, so no single
high-volume sector can claim every slot. Within a sector, markets are still
ranked by 24-hour volume then ticker; empty sectors are skipped, and an allowed
cross-listing takes only its primary sector's turn. Distinct events are still
required, and too few eligible markets still fails startup.

Discovery fetches
[series categories](https://docs.kalshi.com/api-reference/market/get-series-list)
once and joins them to [events with nested markets](https://docs.kalshi.com/api-reference/events/get-events).
Every reported series category must be allowed: Sports, Crypto, unknown/missing
categories, and cross-listings outside the allowlist are excluded. Event labels
and ticker/title guesses are not used as fallbacks. This affects new runs only.

After selection, polling and eight participant tasks run together. Polling is
every 300 seconds; market observations older than 600 seconds block new orders.
Known close times are enforced locally between polls. Settled markets are never
replaced. Kalshi quotes/prices are not included in agent observations.

Every market starts with an empty order book and no trade history. Only
participants are funded; all orders and liquidity come from their decisions.
Agents can research news and observe the internal book before choosing whether
to trade. They form and update their own beliefs from rules, research and
observed agent activity. Every order is one-sided: buy or sell, market or limit,
quantity, and a limit price when applicable. There is no `submit_quote` agent tool.
Orders and trades are optional. Market orders execute only against available resting orders
and cancel any unfilled remainder. The startup uncertainty filter uses external
prices solely to select markets; it supplies neither local prices nor liquidity.

Each agent has at most one pending inference request and dispatches **at most
one tool per response**, so it reads each result before choosing its next call;
any later tool in the same response is returned explicitly not executed. A call
rejected before dispatch — unknown tool, malformed arguments — does not consume
that response's turn.

All agents share a daily **08:00–16:00 America/New_York** session, including
weekends. This uses Eastern local time with daylight saving. Each agent has its
own action deadline; a hold never advances other agents' clocks.

```yaml
trading_timezone: America/New_York
trading_day_start: "08:00"
trading_day_end: "16:00"
order_tool_minutes: 2       # submit_order and cancel_order
read_tool_minutes: 5        # book, recent trades, own orders, account
news_tool_minutes: 10
hold_for_now_minutes: 60    # Tunable; the agent cannot override the duration.
inference_retry_seconds: 60
run_duration_seconds: 82800 # 23 real hours, including closed hours.
```

A tool's cost starts at dispatch. Its result returns as soon as available, and
the next inference can run during that cost. The next tool executes at the later
of the previous cost deadline and inference completion, within an open session.
For example, an order at 10:00 followed by 30 seconds of inference permits the
next action at 10:02; three minutes of inference permits it at 10:03. Network
latency consumes the same cost interval. Dispatched business errors cost time;
a wholly rejected response gets a retry delay. There is no inference-count quota.

`hold_for_now(memory_note=None)` consumes the configured elapsed minutes.
A response without tools implies this hold. `hold_until_next_day(memory_note=None)`
relinquishes the rest of the session until the next day's opening. Both can update
the agent's private note. Resting orders, funds, positions, and memory persist.

If any tool's cost or actual execution reaches/passes 16:00, finish it completely,
retain its result as end-of-day data, and begin a fresh decision at the first
opening after completion. A 60-minute hold at 15:30 finishes at 16:30, followed by
an overnight wait until 08:00; closed hours do not pause or reset holds. Longer
configured holds finish fully even if they span another opening. No new
inference starts outside the session or while relinquishing the day. An inference
that finishes after its session closes is logged, but its proposed action is
not executed. The next session uses a fresh brief. Market polling and settlement
continue overnight; agent session hours never override a market's own cutoff.

Both example YAMLs now use a 23-hour run limit to fit the existing Modal CPU
runner's 24-hour timeout with cleanup headroom. Start before 08:00 to cover a full
session. Shorter smoke tests can override `run_duration_seconds`; a run entirely
outside session hours legitimately makes no model calls. Local multi-day runs
can use a larger duration; the Modal CPU launcher currently caps it at 23 hours.

Before any participant starts, the runner polls `/v1/models` until the endpoint
answers with the pinned model, for up to 900 seconds — matching the Modal
`startup_timeout`. Modal holds requests while its container boots, so a cold
endpoint looks like a hang rather than an error; without this wait all eight
agents burn their 120-second deadline against a server that is still loading.
Progress is logged each attempt. A rejected API key or a different served model
is a configuration failure and stops the run at once rather than retrying.

The timer starts when participant/polling tasks start, after setup. Each
inference attempt has a 120-second total harness deadline (including any local
concurrency queue), with at most 1,024 generated tokens. Transport/timeouts,
HTTP 408/429 and server failures trigger an ordinary 60-second agent wait.
Permanent endpoint/auth/configuration failures stop the entire run. No
state-changing tool is retried automatically.

At the configured duration, all-markets-settled, Ctrl-C, SIGTERM, or a task failure, the runner
cancels pending tasks, cancels open orders, releases reservations, and closes
SQLite. Unresolved positions and agent transcripts remain in the artifact;
there is no invented payout or liquidation.

## Continue a saved run

Resume uses the parent's **entire saved configuration**, including model,
reasoning/sampling settings, agent IDs/count, trading hours and timezone,
tool costs, and run duration. Do not pass `--config` with `--resume-from`;
editing the YAML does not change a continuation. Credentials still come from
the same configured environment-variable names.

Locally, pass a stopped run ID, directory, or database path:

```bash
./launch.sh --resume-from 20260920T170052Z-116113cc
```

On Modal, the parent must already be saved in `prediction-markets-runs`.
Its configuration and database are read from that Volume:

```bash
modal run --detach deploy/modal_openai.py --resume-from 20260920T170052Z-116113cc
```

Each continuation writes a **new run ID** using SQLite backup, leaving its
parent intact. It inherits balances, positions, orders, fills, book events,
raw initial market API objects, settlement records and transcripts. Previously
canceled orders stay canceled: normal shutdown cancels all resting orders,
so agents must choose whether to quote again. Settlement payouts remain
idempotent; newly finalized markets pay once before agents start.

Only original tickers are refreshed. Resume never discovers/replaces markets
or reapplies initial volume, horizon or uncertainty filters. Agents see the
currently tradable original markets; closed, settled, missing and stale markets
are excluded from that list. Unresolved holdings stay in accounts and closed
markets remain tracked for settlement. Missing markets can become tradable
again after a successful ordinary lifecycle poll. If no original market is
tradable after the initial refresh, the child saves its state and exits without
model inference. A failed refresh aborts startup rather than trusting old data.

Agents retain their private notes and complete conversation groups, including
saved Responses reasoning items. New runs checkpoint their retained context
and system prompt; older runs reconstruct context from their transcripts using
the current prompt. The usual context budget still trims the oldest groups.
Completed tools are never replayed. Interrupted decisions are discarded or
quoted as historical data, with missing results explicitly marked unknown;
agents make a fresh decision from current observations.

Time remains real wall time. Downtime counts toward an outstanding hold or
tool cost; any remaining cost finishes before a fresh resumed decision. Agents
wait for the saved session opening if started outside trading hours. Daily
activity counters reset only when entering a new session. Each child gets the
same configured duration, starting after setup (including overnight waiting).

The database's `run_segments` table records parent/root lineage, start/stop
times, and the first new transcript/trade IDs. Transcript and order/trade IDs
continue without reuse. Existing trace exports and price plots include the
inherited history; time filters can isolate a continuation. Run summaries
report cumulative `trade_count` and segment-only `new_trade_count`. A child can
itself be resumed. Active runs are locked against copying; legacy runs require
their shutdown `run.log` alongside `run.db`.

## Inspect and verify

To read an agent's reasoning and actions, newest run by default:

```bash
python -m prediction_markets trace                    # every agent
python -m prediction_markets trace --agent agent-03   # one agent
python -m prediction_markets trace --follow           # watch a live run
```

It opens the database read-only and is safe to run against an experiment in
progress. `message.content` contains the model's visible response; OpenAI's
hidden reasoning is not exposed. Qwen runs with thinking disabled by default.
`--full` stops truncating tool results.

Plot agent exchange prices against historical Kalshi prices:

```bash
pip install -r requirements-analysis.txt
python -m prediction_markets plot --run-id 20260920T170052Z-116113cc --combined
python -m prediction_markets plot --run-id 20260920T170052Z-116113cc --ticker KXTRUMPSAY-26SEP21-BIBI
python -m prediction_markets plot --run-id 20260920T170052Z-116113cc --agent-id agent-03 --combined
```

`--run-id` accepts an ID, run directory, or database path; omitting it selects
the newest local run. Repeat `--ticker` to select markets. `--agent-id` filters
fills where that agent was either buyer or seller (an empty value means all).
`--combined` draws all selected markets on one plot; without it, each market
gets its own plot. Each market keeps its marker and color across plots, with
lines, arrows, and numbered points in chronological fill order. Repeated-price
fills remain observations and coincident points may overlap.

The x-axis is the most recent Kalshi YES trade at or before each internal fill;
the y-axis is the internal YES execution price, both in cents. This uses public
[trade history](https://docs.kalshi.com/api-reference/market/get-trades), including
the [historical archive](https://docs.kalshi.com/getting_started/historical_data)
when appropriate, not today's market snapshot or a future candle close. The
internal timestamps have one-second resolution. `y = x` indicates price
agreement with Kalshi, not statistical calibration against eventual outcomes.
Unfilled quotes are not execution prices: markets without selected fills are
listed but have no plotted points. Self-trades are included and flagged in the
data; use `--exclude-self-trades` to omit them.

Outputs default to `runs/<id>/price_comparison/` (selection/agent subdirectories
when filtered): PNG and SVG plots, `points.jsonl` with both prices, timestamps,
human-price age, buyer/seller and matching status, and `metadata.json`.
`--out DIR` overrides the output directory; rerunning replaces these plot
outputs. Raw API history is cached as compressed JSON in
`runs/<id>/kalshi_history_cache/`, reusable across agent filters. `--offline`
uses that cache without network access; `--refresh` refetches it. The default
maximum human-trade age/lookback is 24 hours, tunable with
`--max-human-age-hours`. Missing/stale matches are explicitly reported and
leave gaps in the plot. This analysis reads the run database without modifying
it and never supplies external prices to agents.

Export actions and full order-book history as standard JSON Lines, one object
per line (newest run by default):

```bash
python -m prediction_markets export
python -m prediction_markets export runs/<id>/run.db --out exports/my-run
python -m prediction_markets export runs/<id> --format json --out exports/my-run-arrays
python -m prediction_markets export runs/<id> --agent agent-03 --tool submit_order \
  --start '2026-09-19T08:00:00-04:00' --end '2026-09-19T16:00:00-04:00' \
  --out exports/agent-03-orders
```

The output directory contains `actions.jsonl`, `orderbooks.jsonl`, `markets.json`, and
`metadata.json`. `--format json` instead writes JSON arrays in `actions.json`
and `orderbooks.json`. Existing exports are never overwritten. The database is
opened read-only using a consistent snapshot, including when a run is active.
JSONL streams without materializing the full history; no extra packages are
required. To get ordinary Python lists of dictionaries:

```python
from prediction_markets.analysis.export import TraceExport, Filters

with TraceExport("runs/<id>/run.db") as run:
    filters = Filters(agent="agent-03", start="2026-09-19T12:00:00Z")
    actions = list(run.actions(filters))
    orderbooks = list(run.orderbooks(filters))
    markets = list(run.markets(filters))
    metadata = run.metadata()
```

Each action has `time`, `agent_id`, and `tool_query: {name, arguments}`, plus
`action_id`, `sequence`, and `time_source` for alignment and provenance. Actions
are ordered by dispatch time and sequence. Fired calls that return errors and
calls still awaiting a result are included; unexecuted proposals are excluded.

Each book object has `time`, `ticker`, `bids`, and `asks`, plus the initiating
`agent_id`, `tool`, `action_id`, `event_id`, and `time_source`. Bids and asks
contain every resting order at every price level, with `order_id`, `agent_id`,
`price_cents`, and remaining `quantity`, sorted by price and then FIFO. A snapshot
is emitted after each committed book change, including fills and cancellations.
An incoming market order's unfilled remainder never rests. System changes such
as shutdown cancellations have a null initiating agent/action. New databases
store compact deltas transactionally in `book_events`; full snapshots are only
assembled during export.

`markets.json` is always a JSON array of **unmodified Kalshi market objects**,
including all original fields and nested values. New runs archive the first
discovery response for each selected market in `market_api_responses`, separate
from the normalized fields used in agent briefs and tools. Lifecycle polling
does not replace this initial snapshot. By default all markets in the fixed
cohort are included; agent/tool/time filters restrict this to markets explicitly
referenced by matching actions or book changes. `--market` restricts it by ticker.
Capture times and source endpoints are stored separately under
`metadata.json` → `market_payloads.sources`, keeping the API objects unchanged.

Older runs did not retain raw responses. Their missing tickers are reported in
metadata, without substituting normalized rows. To fetch **current** Kalshi
payloads for missing markets, explicitly enable backfill:

```bash
python -m prediction_markets export runs/<id> --fetch-missing-markets --out exports/backfilled
```

Backfilled payloads are labeled `fetched_at_export` with their retrieval time;
they are not the original run-time observations. Saved first-pull responses
are labeled `run_first_pull`. Python callers use `run.markets(fetch_missing=True)`.
Without this option exporting never makes an HTTP request.

Times are UTC ISO 8601 at the database's one-second resolution. Agent-caused
book changes use the same **tool dispatch time** as their action, regardless of
inference completion, tool latency, or cooldown. Multiple changes in one second
remain separate, ordered by `event_id`. Time filters are **[start, end)** and
accept Unix seconds or ISO timestamps with an explicit timezone. Repeat
`--agent`, `--tool`, or `--market` to select multiple values; distinct filter
types are combined with AND. For book history, agent/tool filters select the
cause of the change. Every earlier event and other agent's orders still
contribute to the complete snapshot.

Older databases are reconstructed from orders, fills, and transcripts; the
sibling `run.log` (or `--log PATH`) supplies shutdown timing when available.
Some older action times can only be estimated, and some automatic cancellation
times are missing. Inspect `time_source` and `metadata.json` warnings and
`book_history_complete`; missing timestamps are not invented. Read metadata
after consuming the book iterator so final-state validation is included.

`runs/<id>/run.log` records cohort selection, tool outcomes, orders, fills,
shutdown reason, trade count and final account totals. `run.db` stores frozen
settings, the fixed cohort, orders/trades, settlement markers, private notes,
order/read/news activity counters and action/wake deadlines.
`agent_state.status` distinguishes thinking, tool execution, cooldown, holding,
and overnight waiting. `transcript_entries` includes original model responses,
tool results, and timing events; the trace viewer displays all three. Treat these artifacts as private experiment data.
Final equity uses last internal trade marks for unresolved holdings; it is not
a settled payout or realized profit. Zero trades is a valid, explicitly logged
outcome.

Offline verification covers HTTP request shape/authentication, errors and
cancellation, eight concurrent loops, scripted fills, private-memory isolation,
shutdown, elapsed tool costs, inference overlap, daily boundaries, DST,
overnight holds, research beyond the former count cap, and both settlement outcomes. The real pinned Qwen
tokenizer/chat template was also checked locally, including rejected historical
tool arguments. Those tokenizer tests use the ignored cache and never download
during pytest; without that cache they are explicitly skipped. To populate it
without hosting a model:

```bash
python -c 'from prediction_markets import config; from prediction_markets.runner import load_tokenizer; load_tokenizer(config.load("configs/mvp.yaml"))'
python -m pytest -q -p no:cacheprovider
```

Still unverified: Modal image build/startup, actual A100-40GB capacity/latency,
live vLLM tool parsing, and the bounded eight-agent/ten-market run. Live
settlement is also untested in a live run with the new session schedule.
After authorization, check that every agent completes a valid tool loop and
inspect the real responses, order outcomes, trade count (including zero) and
final balances. Report any untested settlement path rather than marking it
passed based on offline tests.
