"""Versioned exchange tables and additive continuation metadata."""

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE run_settings (
    singleton_id INTEGER PRIMARY KEY CHECK (singleton_id = 1),
    schema_version INTEGER NOT NULL
        CHECK (typeof(schema_version) = 'integer' AND schema_version = 1),
    created_at INTEGER NOT NULL
        CHECK (typeof(created_at) = 'integer' AND created_at > 0),
    config_json TEXT NOT NULL CHECK (json_valid(config_json))
);

CREATE TABLE markets (
    ticker TEXT PRIMARY KEY CHECK (length(trim(ticker)) > 0),
    event_ticker TEXT NOT NULL UNIQUE
        CHECK (length(trim(event_ticker)) > 0),
    cohort_index INTEGER NOT NULL UNIQUE
        CHECK (typeof(cohort_index) = 'integer' AND cohort_index > 0),
    market_type TEXT NOT NULL CHECK (market_type = 'binary'),
    title TEXT NOT NULL CHECK (length(trim(title)) > 0),
    rules TEXT NOT NULL CHECK (length(trim(rules)) > 0),
    status TEXT NOT NULL CHECK (
        status IN (
            'initialized', 'inactive', 'active', 'closed', 'determined',
            'disputed', 'amended', 'finalized'
        )
    ),
    close_time INTEGER NOT NULL
        CHECK (typeof(close_time) = 'integer' AND close_time > 0),
    expected_expiration_time INTEGER
        CHECK (
            expected_expiration_time IS NULL OR (
                typeof(expected_expiration_time) = 'integer'
                AND expected_expiration_time > 0
            )
        ),
    latest_expiration_time INTEGER
        CHECK (
            latest_expiration_time IS NULL
            OR (
                typeof(latest_expiration_time) = 'integer'
                AND latest_expiration_time > 0
            )
        ),
    last_successful_poll INTEGER NOT NULL
        CHECK (
            typeof(last_successful_poll) = 'integer'
            AND last_successful_poll > 0
        ),
    payout_cents INTEGER
        CHECK (
            payout_cents IS NULL
            OR (
                typeof(payout_cents) = 'integer'
                AND payout_cents BETWEEN 0 AND 100
            )
        )
);

CREATE TABLE accounts (
    agent_id TEXT PRIMARY KEY CHECK (length(trim(agent_id)) > 0),
    balance_cents INTEGER NOT NULL
        CHECK (typeof(balance_cents) = 'integer' AND balance_cents >= 0)
);

-- Immutable first-pull API market objects for analysis/export only. Participant
-- briefs and tools read the normalized markets table, never this archive.
CREATE TABLE market_api_responses (
    ticker TEXT PRIMARY KEY REFERENCES markets(ticker) ON DELETE RESTRICT,
    captured_at INTEGER NOT NULL CHECK (typeof(captured_at) = 'integer' AND captured_at > 0),
    source_url TEXT NOT NULL,
    payload_json TEXT NOT NULL CHECK (json_valid(payload_json))
);

CREATE TABLE agent_state (
    agent_id TEXT PRIMARY KEY
        REFERENCES accounts(agent_id) ON DELETE CASCADE,
    private_note TEXT NOT NULL DEFAULT '',
    activity_day_started_at INTEGER
        CHECK (
            activity_day_started_at IS NULL
            OR (
                typeof(activity_day_started_at) = 'integer'
                AND activity_day_started_at > 0
            )
        ),
    shared_tool_calls INTEGER NOT NULL DEFAULT 0
        CHECK (
            typeof(shared_tool_calls) = 'integer'
            AND shared_tool_calls >= 0
        ),
    news_tool_calls INTEGER NOT NULL DEFAULT 0
        CHECK (
            typeof(news_tool_calls) = 'integer'
            AND news_tool_calls >= 0
        ),
    status TEXT NOT NULL DEFAULT 'ready'
        CHECK (status IN ('ready', 'thinking', 'tool_running', 'cooldown', 'holding', 'overnight')),
    next_action_at INTEGER CHECK (next_action_at IS NULL OR next_action_at > 0),
    scheduled_wake_at INTEGER
        CHECK (
            scheduled_wake_at IS NULL
            OR (
                typeof(scheduled_wake_at) = 'integer'
                AND scheduled_wake_at > 0
            )
        )
);

-- Order-tool activity per agent per market. Rows appear on first use, because
-- the cohort is stored after the accounts are initialized; a missing row is
-- simply a used count of zero.
CREATE TABLE agent_market_calls (
    agent_id TEXT NOT NULL REFERENCES accounts(agent_id) ON DELETE CASCADE,
    ticker TEXT NOT NULL REFERENCES markets(ticker) ON DELETE RESTRICT,
    order_tool_calls INTEGER NOT NULL DEFAULT 0
        CHECK (
            typeof(order_tool_calls) = 'integer'
            AND order_tool_calls >= 0
        ),
    PRIMARY KEY (agent_id, ticker)
);

CREATE TABLE positions (
    agent_id TEXT NOT NULL REFERENCES accounts(agent_id) ON DELETE RESTRICT,
    ticker TEXT NOT NULL REFERENCES markets(ticker) ON DELETE RESTRICT,
    signed_qty INTEGER NOT NULL CHECK (typeof(signed_qty) = 'integer'),
    PRIMARY KEY (agent_id, ticker)
);

CREATE TABLE orders (
    order_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES accounts(agent_id) ON DELETE RESTRICT,
    ticker TEXT NOT NULL REFERENCES markets(ticker) ON DELETE RESTRICT,
    action TEXT NOT NULL CHECK (action IN ('buy', 'sell')),
    order_type TEXT NOT NULL CHECK (order_type IN ('limit', 'market')),
    price_cents INTEGER,
    original_quantity INTEGER NOT NULL
        CHECK (
            typeof(original_quantity) = 'integer'
            AND original_quantity > 0
        ),
    filled_quantity INTEGER NOT NULL DEFAULT 0
        CHECK (
            typeof(filled_quantity) = 'integer'
            AND filled_quantity >= 0
        ),
    remaining_quantity INTEGER NOT NULL
        CHECK (
            typeof(remaining_quantity) = 'integer'
            AND remaining_quantity >= 0
        ),
    status TEXT NOT NULL CHECK (status IN ('open', 'filled', 'canceled')),
    cancel_reason TEXT CHECK (
        cancel_reason IS NULL OR cancel_reason IN (
            'user', 'quote_rollback', 'market_remainder',
            'market_closed', 'market_settled', 'run_shutdown'
        )
    ),
    created_at INTEGER NOT NULL
        CHECK (typeof(created_at) = 'integer' AND created_at > 0),
    CHECK (
        (order_type = 'limit'
            AND typeof(price_cents) = 'integer'
            AND price_cents BETWEEN 1 AND 99)
        OR (order_type = 'market' AND price_cents IS NULL)
    ),
    CHECK (filled_quantity + remaining_quantity <= original_quantity),
    CHECK (
        (status = 'open'
            AND remaining_quantity > 0
            AND filled_quantity + remaining_quantity = original_quantity
            AND cancel_reason IS NULL)
        OR (status = 'filled'
            AND remaining_quantity = 0
            AND filled_quantity = original_quantity
            AND cancel_reason IS NULL)
        OR (status = 'canceled'
            AND remaining_quantity = 0
            AND cancel_reason IS NOT NULL)
    )
);

CREATE TABLE trades (
    trade_id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL REFERENCES markets(ticker) ON DELETE RESTRICT,
    maker_order_id INTEGER NOT NULL
        REFERENCES orders(order_id) ON DELETE RESTRICT,
    taker_order_id INTEGER NOT NULL
        REFERENCES orders(order_id) ON DELETE RESTRICT,
    price_cents INTEGER NOT NULL
        CHECK (
            typeof(price_cents) = 'integer'
            AND price_cents BETWEEN 1 AND 99
        ),
    quantity INTEGER NOT NULL
        CHECK (typeof(quantity) = 'integer' AND quantity > 0),
    executed_at INTEGER NOT NULL
        CHECK (typeof(executed_at) = 'integer' AND executed_at > 0),
    CHECK (maker_order_id != taker_order_id)
);

CREATE TABLE market_settlements (
    ticker TEXT PRIMARY KEY REFERENCES markets(ticker) ON DELETE RESTRICT,
    payout_cents INTEGER NOT NULL
        CHECK (
            typeof(payout_cents) = 'integer'
            AND payout_cents BETWEEN 0 AND 100
        ),
    settled_at INTEGER NOT NULL
        CHECK (typeof(settled_at) = 'integer' AND settled_at > 0)
);

CREATE TABLE transcript_entries (
    entry_id INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id TEXT NOT NULL REFERENCES accounts(agent_id) ON DELETE RESTRICT,
    sequence_number INTEGER NOT NULL
        CHECK (
            typeof(sequence_number) = 'integer'
            AND sequence_number > 0
        ),
    entry_type TEXT NOT NULL
        CHECK (entry_type IN ('model_response', 'tool_result', 'timing')),
    content_json TEXT NOT NULL CHECK (json_valid(content_json)),
    created_at INTEGER NOT NULL
        CHECK (typeof(created_at) = 'integer' AND created_at > 0),
    UNIQUE (agent_id, sequence_number)
);

-- One compact delta per committed book mutation. Full snapshots are exported
-- on demand rather than duplicating the entire book in the live database.
CREATE TABLE book_events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    fired_at INTEGER NOT NULL CHECK (fired_at > 0),
    action_entry_id INTEGER REFERENCES transcript_entries(entry_id),
    agent_id TEXT REFERENCES accounts(agent_id),
    tool TEXT NOT NULL,
    changes_json TEXT NOT NULL CHECK (json_valid(changes_json))
);

CREATE INDEX idx_book_events_time ON book_events(fired_at, event_id);

CREATE INDEX idx_orders_book
    ON orders(ticker, status, action, price_cents, order_id);
CREATE INDEX idx_orders_agent
    ON orders(agent_id, order_id DESC);
CREATE INDEX idx_trades_market
    ON trades(ticker, trade_id DESC);
"""

# Additive continuation metadata; older schema-1 runs can be copied without
# altering their accounting tables or the original database.
CONTINUATION_SCHEMA = """
CREATE TABLE IF NOT EXISTS run_segments (
    run_id TEXT PRIMARY KEY,
    parent_run_id TEXT,
    root_run_id TEXT NOT NULL,
    started_at INTEGER NOT NULL,
    stopped_at INTEGER,
    stop_reason TEXT,
    first_transcript_entry_id INTEGER NOT NULL,
    first_trade_id INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS agent_checkpoints (
    agent_id TEXT PRIMARY KEY REFERENCES accounts(agent_id),
    sequence_number INTEGER NOT NULL,
    state_json TEXT NOT NULL CHECK (json_valid(state_json))
);
"""


