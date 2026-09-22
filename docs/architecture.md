# Architecture

`cli.py` routes commands to one implementation per operation. The runtime and
offline analysis share storage and data contracts; neither maintains a second
copy of matching or continuation logic.

```text
prediction_markets/
├── __main__.py, cli.py, config.py
├── runtime/
│   ├── runner.py
│   └── schedule.py
├── agents/
│   ├── participant.py
│   ├── prompt.py
│   ├── history.py
│   └── tools.py
├── exchange/
│   ├── engine.py
│   └── bookkeeping.py
├── markets/
│   ├── selection.py
│   └── update.py
├── integrations/
│   ├── kalshi.py
│   ├── tavily.py
│   ├── model_client.py
│   ├── model_protocol.py
│   ├── openai_responses.py
│   └── tokenization.py
├── storage/
│   ├── schema.py
│   ├── database.py
│   ├── runs.py
│   └── download.py
└── analysis/
    ├── dataset.py
    ├── legacy.py
    ├── trace.py
    ├── export.py
    └── prices.py
```

## Runtime ownership

The runner receives frozen settings, inference and tokenization implementations,
and an optional output directory. It creates a connection owned by that run and
passes it to the exchange and agents. Modal supplies a temporary directory
explicitly, then copies closed artifacts to the volume. It does not change
global configuration to redirect outputs.

Fresh runs initialize a cohort; resumed runs copy and refresh the saved cohort.
Both enter the same orchestration loop and use the same shutdown path. Market
updates call exchange methods rather than accessing its private connection.
Matching, balances, position changes and settlement remain atomic transactions
inside the exchange engine. `bookkeeping.py` contains the pure financial math.

`schedule.py` handles daily openings, closings and timezone rules. Participants
combine that calendar with elapsed tool costs, inference completion and holds.
`update.py` polls status and settlement for the fixed cohort. It never replaces
stale markets or injects human prices into agent observations.

## Model and history boundary

`model_client.py` performs HTTP requests. `openai_responses.py` adapts the
Responses wire format. `model_protocol.py` owns shared errors and completion
validation, removing the former dependency from integrations back into agents.
`tokenization.py` loads the Qwen tokenizer or supplies the OpenAI context estimate.

`prompt.py` renders instructions. `history.py` formats and restores private
conversation groups, including opaque provider state. Interrupted tool calls
are quoted as history and never replayed during restoration.

## Storage and analysis

`storage/runs.py` resolves identifiers for every command. A stopped run is
identified by an available run lock or a legacy shutdown log marker. Stopped
databases without pending WAL frames can be read as immutable. Pending frames
are recovered in a private copy. Active or unverified databases use ordinary
read-only SQLite connections; live following starts a fresh read each poll.

`analysis/dataset.py` shares filters, timestamp parsing, lineage boundaries,
transcript reads and fill queries. Its `RunDataset` reconstructs complete books
before filtering output so an agent or time filter does not discard other agents'
resting orders. Dispatch timestamps align action and book records. Historical
formats without book events use the explicitly isolated `legacy.py` fallback.

`--segment current` uses the latest saved transcript/trade boundaries. Book
changes without an initiating action use the segment start time; legacy records
have one-second time resolution. The default includes the full inherited history.

Trace rendering, JSON/JSONL serialization and plotting stay separate. Kalshi
history fetching belongs to the integration, never to runtime market polling.

## Compatibility

The old top-level modules are forwarding imports or command wrappers, with no
parallel implementation. `agents` and `exchange` retain their public exports as
packages. Legacy singleton database helpers and remembered config loading remain
available for callers that use them; production runs use explicit settings and
connections instead. New imports should use the responsibility packages.

Schema version 1, saved configuration, lineage/checkpoint tables, the historical
shutdown log marker, and the per-run artifact layout are preserved. Existing
run databases require no migration. Source parents are never modified on resume.
