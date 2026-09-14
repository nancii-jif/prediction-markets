"""SQLite schema and all reads/writes.

One connection per process. WAL mode lets the cron stream process write while
the runner reads without either blocking.

Two separate worlds live in here and must not be confused:

  - Kalshi-side (`markets`, `live_markets`, `snapshots`). Written by
    `stream.py`, read by `live_markets.py`. The price columns of `snapshots`
    are *never* rendered into an agent-visible surface.
  - The internal exchange (`cash`, `positions`, `quotes`, `auctions`, `fills`,
    `settlements`). Integer cents, integer contracts, prices on the 1-99 grid,
    always expressed in YES space regardless of the run's quote frame.
"""

import sqlite3
from pathlib import Path

from . import config

SCHEMA = """
-- --- Kalshi side ---------------------------------------------------------

CREATE TABLE IF NOT EXISTS markets (
    ticker       TEXT PRIMARY KEY,
    event_ticker TEXT,
    title        TEXT,
    rules        TEXT,
    sector       TEXT,
    open_ts      INTEGER,
    close_ts     INTEGER,
    first_seen   INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS live_markets (
    ticker     TEXT PRIMARY KEY REFERENCES markets(ticker),
    sector     TEXT    NOT NULL,
    added_at   INTEGER NOT NULL,
    retired_at INTEGER
);

CREATE TABLE IF NOT EXISTS snapshots (
    ticker           TEXT    NOT NULL,
    captured_at      INTEGER NOT NULL,
    yes_bid          REAL,
    yes_ask          REAL,
    bid_size         REAL,
    ask_size         REAL,
    last             REAL,
    volume_24h       REAL,
    status           TEXT,
    result           TEXT,
    settlement_value REAL,
    PRIMARY KEY (ticker, captured_at)
);

-- --- Internal exchange ---------------------------------------------------

CREATE TABLE IF NOT EXISTS cash (
    agent         TEXT PRIMARY KEY,
    balance_cents INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    agent            TEXT    NOT NULL,
    ticker           TEXT    NOT NULL,
    signed_qty       INTEGER NOT NULL,
    cost_basis_cents INTEGER NOT NULL,
    PRIMARY KEY (agent, ticker)
);

CREATE TABLE IF NOT EXISTS quotes (
    round_id      INTEGER NOT NULL,
    agent         TEXT    NOT NULL,
    ticker        TEXT    NOT NULL,
    bid_cents     INTEGER,
    bid_size      INTEGER,
    ask_cents     INTEGER,
    ask_size      INTEGER,
    status        TEXT    NOT NULL,
    reject_reason TEXT,
    PRIMARY KEY (round_id, agent, ticker)
);

-- clear_cents NULL means the auction produced no print. indicative_cents is
-- then the microprice of the uncrossed book: an estimate, never a trade, and
-- deliberately a separate column so last_clear_cents cannot return one.
CREATE TABLE IF NOT EXISTS auctions (
    ticker           TEXT    NOT NULL,
    round_id         INTEGER NOT NULL,
    ts               INTEGER NOT NULL,
    clear_cents      INTEGER,
    cleared_qty      INTEGER NOT NULL,
    tie_lo           INTEGER,
    tie_hi           INTEGER,
    indicative_cents INTEGER,
    PRIMARY KEY (ticker, round_id)
);

CREATE TABLE IF NOT EXISTS fills (
    id          INTEGER PRIMARY KEY,
    round_id    INTEGER NOT NULL,
    agent       TEXT    NOT NULL,
    ticker      TEXT    NOT NULL,
    signed_qty  INTEGER NOT NULL,
    price_cents INTEGER NOT NULL
);

-- void = 1 marks a position closed out at the last internal clear because
-- its market was retired on the clock without ever producing a result, as
-- opposed to a real payout. Kept separable so analysis can exclude them.
CREATE TABLE IF NOT EXISTS settlements (
    agent        TEXT    NOT NULL,
    ticker       TEXT    NOT NULL,
    round_id     INTEGER NOT NULL,
    qty          INTEGER NOT NULL,
    payout_cents INTEGER NOT NULL,
    scalar       INTEGER NOT NULL DEFAULT 0,
    void         INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (agent, ticker, round_id)
);

-- --- Provenance ----------------------------------------------------------

CREATE TABLE IF NOT EXISTS calls (
    agent        TEXT    NOT NULL,
    ticker       TEXT    NOT NULL,
    round_id     INTEGER NOT NULL,
    ts           INTEGER NOT NULL,
    brief        TEXT,
    raw_response TEXT,
    parse_ok     INTEGER NOT NULL,
    PRIMARY KEY (agent, ticker, round_id)
);

CREATE TABLE IF NOT EXISTS run_config (
    run_id      TEXT NOT NULL,
    ts          INTEGER NOT NULL,
    config_json TEXT NOT NULL,
    PRIMARY KEY (run_id, ts)
);

"""

# Applied after _migrate, so an index can name a column that migration adds.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_live_active   ON live_markets(retired_at);
CREATE INDEX IF NOT EXISTS idx_positions_tkr ON positions(ticker);
CREATE INDEX IF NOT EXISTS idx_fills_agent   ON fills(agent, ticker);
CREATE INDEX IF NOT EXISTS idx_auctions_tkr  ON auctions(ticker, round_id);
CREATE INDEX IF NOT EXISTS idx_markets_sect  ON markets(sector);
CREATE INDEX IF NOT EXISTS idx_markets_event ON markets(event_ticker);
"""

SNAPSHOT_COLUMNS = (
    "ticker", "captured_at", "yes_bid", "yes_ask", "bid_size",
    "ask_size", "last", "volume_24h", "status", "result", "settlement_value",
)

MARKET_COLUMNS = (
    "ticker", "event_ticker", "title", "rules", "sector", "open_ts",
    "close_ts", "first_seen",
)

# SQLite caps host parameters per statement; chunk IN clauses below this.
_PARAM_CHUNK = 900

_conn: sqlite3.Connection | None = None


def connect(path=None) -> sqlite3.Connection:
    """Open a new connection with the pragmas this project depends on.

    An explicit path is self-contained: it creates its own parent and never
    consults the run config, so a helper process can open a database without
    loading one.
    """
    path = Path(path) if path is not None else config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after a database was first created."""
    existing = {r["name"] for r in conn.execute("PRAGMA table_info(markets)")}
    if "event_ticker" not in existing:
        conn.execute("ALTER TABLE markets ADD COLUMN event_ticker TEXT")
        # Backfill from the ticker: Kalshi's convention is EVENT-STRIKE, so
        # the event is everything before the last dash. That is a convention
        # rather than a guarantee, which is why new rows store the payload's
        # event_ticker verbatim — the next sweep overwrites these derived
        # values with the real ones.
        rows = conn.execute("SELECT ticker FROM markets").fetchall()
        conn.executemany(
            "UPDATE markets SET event_ticker = ? WHERE ticker = ?",
            [(r["ticker"].rsplit("-", 1)[0], r["ticker"]) for r in rows],
        )
        conn.commit()

    existing = {r["name"] for r in conn.execute("PRAGMA table_info(auctions)")}
    if "indicative_cents" not in existing:
        conn.execute("ALTER TABLE auctions ADD COLUMN indicative_cents INTEGER")
        conn.commit()

    existing = {r["name"] for r in conn.execute("PRAGMA table_info(settlements)")}
    if "void" not in existing:
        conn.execute(
            "ALTER TABLE settlements ADD COLUMN void INTEGER NOT NULL DEFAULT 0"
        )
        conn.commit()


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()
    _migrate(conn)
    conn.executescript(INDEXES)
    conn.commit()


def db() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _conn = connect()
        _create_schema(_conn)
    return _conn


def init(path=None) -> sqlite3.Connection:
    """Create the schema, migrating an older database if needed. Safe to call
    repeatedly."""
    global _conn
    if path is not None:
        close()
        _conn = connect(path)
    conn = db()
    _create_schema(conn)
    return conn


def close() -> None:
    global _conn
    if _conn is not None:
        _conn.close()
        _conn = None


# --- markets / snapshots (Kalshi side) -------------------------------------


def upsert_markets(rows) -> int:
    """Write market metadata verbatim. This is the only entry point for text
    that agents will eventually read, so nothing here is rewritten or trimmed.

    first_seen is preserved on conflict: it records when we first observed the
    market, not when we last refreshed it.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn = db()
    columns = ", ".join(MARKET_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in MARKET_COLUMNS)
    with conn:
        cursor = conn.executemany(
            f"INSERT INTO markets ({columns}) VALUES ({placeholders}) "
            "ON CONFLICT(ticker) DO UPDATE SET "
            "event_ticker = excluded.event_ticker, title = excluded.title, "
            "rules = excluded.rules, sector = excluded.sector, "
            "open_ts = excluded.open_ts, close_ts = excluded.close_ts",
            rows,
        )
    return cursor.rowcount


def update_market_times(rows) -> int:
    """Refresh close/open times on markets already tracked.

    `rows` is a sequence of (close_ts, open_ts, ticker). Routine targeted
    fetches carry these fields, and clock-based retirement reads close_ts, so
    it has to track Kalshi's amendments — markets can close early or be
    extended. Text fields (title, rules) still refresh only on sweeps.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn = db()
    with conn:
        cursor = conn.executemany(
            "UPDATE markets SET close_ts = ?, open_ts = ? WHERE ticker = ?",
            rows,
        )
    return cursor.rowcount


def market(ticker: str) -> sqlite3.Row | None:
    return db().execute(
        "SELECT * FROM markets WHERE ticker = ?", (ticker,)
    ).fetchone()


def markets_in_sector(sector: str) -> list[sqlite3.Row]:
    return db().execute(
        "SELECT * FROM markets WHERE sector = ? ORDER BY ticker", (sector,)
    ).fetchall()


def write_snapshots(rows) -> int:
    """Append snapshot rows, keyed by (ticker, captured_at).

    REPLACE rather than IGNORE on conflict: timestamps have one-second
    resolution, so a sweep and a targeted fetch landing in the same second
    would otherwise see the second row silently dropped — and if that dropped
    row is the one carrying a newly non-empty `result`, the resolution is lost
    until the market next moves. Within a single second the later observation
    is the truer one, and reruns stay idempotent either way.
    """
    rows = list(rows)
    if not rows:
        return 0
    conn = db()
    columns = ", ".join(SNAPSHOT_COLUMNS)
    placeholders = ", ".join(f":{c}" for c in SNAPSHOT_COLUMNS)
    with conn:
        cursor = conn.executemany(
            f"INSERT OR REPLACE INTO snapshots ({columns}) VALUES ({placeholders})",
            rows,
        )
    return cursor.rowcount


def latest_snapshots(tickers=None) -> dict[str, sqlite3.Row]:
    """Most recent snapshot per ticker, keyed by ticker.

    Relies on SQLite's documented bare-column behaviour: with MAX(captured_at)
    the other selected columns come from that same row.
    """
    conn = db()
    select = (
        "SELECT ticker, MAX(captured_at) AS captured_at, yes_bid, yes_ask, "
        "bid_size, ask_size, last, volume_24h, status, result, settlement_value "
        "FROM snapshots"
    )
    if tickers is None:
        rows = conn.execute(f"{select} GROUP BY ticker").fetchall()
        return {r["ticker"]: r for r in rows}

    tickers = list(tickers)
    out: dict[str, sqlite3.Row] = {}
    for start in range(0, len(tickers), _PARAM_CHUNK):
        chunk = tickers[start : start + _PARAM_CHUNK]
        marks = ", ".join("?" * len(chunk))
        rows = conn.execute(
            f"{select} WHERE ticker IN ({marks}) GROUP BY ticker", chunk
        ).fetchall()
        out.update({r["ticker"]: r for r in rows})
    return out


def latest_snapshot(ticker: str) -> sqlite3.Row | None:
    return db().execute(
        "SELECT * FROM snapshots WHERE ticker = ? ORDER BY captured_at DESC LIMIT 1",
        (ticker,),
    ).fetchone()


def snapshot_count() -> int:
    return db().execute("SELECT COUNT(*) AS n FROM snapshots").fetchone()["n"]


# --- live market membership ------------------------------------------------


def active_live_markets() -> list[sqlite3.Row]:
    """Active members, with the close_ts the clock-based retirement reads."""
    return db().execute(
        "SELECT l.ticker, l.sector, l.added_at, m.close_ts "
        "FROM live_markets l JOIN markets m ON m.ticker = l.ticker "
        "WHERE l.retired_at IS NULL ORDER BY l.ticker"
    ).fetchall()


def active_live_tickers() -> list[str]:
    return [r["ticker"] for r in active_live_markets()]


def active_in_sector(sector: str) -> list[str]:
    rows = db().execute(
        "SELECT ticker FROM live_markets "
        "WHERE sector = ? AND retired_at IS NULL ORDER BY ticker",
        (sector,),
    ).fetchall()
    return [r["ticker"] for r in rows]


def live_event_tickers() -> set[str]:
    """Events that currently hold a seat.

    Membership is stratified by event as well as sector: two markets from one
    event are two strikes on the same underlying question, often logically
    nested, so seating both spends two seats on one question and lets the
    exchange print an internally incoherent pair.
    """
    rows = db().execute(
        "SELECT DISTINCT m.event_ticker FROM live_markets l "
        "JOIN markets m ON m.ticker = l.ticker "
        "WHERE l.retired_at IS NULL AND m.event_ticker IS NOT NULL"
    ).fetchall()
    return {r["event_ticker"] for r in rows}


def ever_admitted() -> set[str]:
    """Every ticker ever admitted, retired or not. Refills must never reuse
    one: a returning market would carry internal history agents already saw."""
    rows = db().execute("SELECT ticker FROM live_markets").fetchall()
    return {r["ticker"] for r in rows}


def add_live_market(ticker: str, sector: str, added_at: int) -> bool:
    conn = db()
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO live_markets (ticker, sector, added_at) "
            "VALUES (?, ?, ?)",
            (ticker, sector, added_at),
        )
    return cursor.rowcount > 0


def retire_live_market(ticker: str, retired_at: int) -> bool:
    conn = db()
    with conn:
        cursor = conn.execute(
            "UPDATE live_markets SET retired_at = ? "
            "WHERE ticker = ? AND retired_at IS NULL",
            (retired_at, ticker),
        )
    return cursor.rowcount > 0


def held_tickers() -> list[str]:
    """Tickers any agent holds. These stay fetched after retirement so open
    positions can still be settled."""
    rows = db().execute(
        "SELECT DISTINCT ticker FROM positions WHERE signed_qty != 0"
    ).fetchall()
    return [r["ticker"] for r in rows]


# --- ledger ----------------------------------------------------------------


def register_agent(agent: str, balance_cents: int) -> bool:
    """Open an account. Returns False if the agent already exists."""
    conn = db()
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO cash (agent, balance_cents) VALUES (?, ?)",
            (agent, balance_cents),
        )
    return cursor.rowcount > 0


def cash_cents(agent: str) -> int:
    row = db().execute(
        "SELECT balance_cents FROM cash WHERE agent = ?", (agent,)
    ).fetchone()
    return row["balance_cents"] if row else 0


def set_cash_cents(agent: str, balance_cents: int) -> None:
    conn = db()
    with conn:
        conn.execute(
            "INSERT INTO cash (agent, balance_cents) VALUES (?, ?) "
            "ON CONFLICT(agent) DO UPDATE SET balance_cents = excluded.balance_cents",
            (agent, balance_cents),
        )


def position(agent: str, ticker: str) -> tuple[int, int]:
    """(signed_qty, cost_basis_cents) — zeros if never traded."""
    row = db().execute(
        "SELECT signed_qty, cost_basis_cents FROM positions "
        "WHERE agent = ? AND ticker = ?",
        (agent, ticker),
    ).fetchone()
    return (row["signed_qty"], row["cost_basis_cents"]) if row else (0, 0)


def open_positions(agent: str) -> list[sqlite3.Row]:
    return db().execute(
        "SELECT ticker, signed_qty, cost_basis_cents FROM positions "
        "WHERE agent = ? AND signed_qty != 0 ORDER BY ticker",
        (agent,),
    ).fetchall()


def positions_in(ticker: str) -> list[sqlite3.Row]:
    return db().execute(
        "SELECT agent, signed_qty, cost_basis_cents FROM positions "
        "WHERE ticker = ? AND signed_qty != 0 ORDER BY agent",
        (ticker,),
    ).fetchall()


def set_position(agent: str, ticker: str, signed_qty: int, cost_basis_cents: int) -> None:
    conn = db()
    with conn:
        conn.execute(
            "INSERT INTO positions (agent, ticker, signed_qty, cost_basis_cents) "
            "VALUES (?, ?, ?, ?) ON CONFLICT(agent, ticker) DO UPDATE SET "
            "signed_qty = excluded.signed_qty, "
            "cost_basis_cents = excluded.cost_basis_cents",
            (agent, ticker, signed_qty, cost_basis_cents),
        )


def insert_settlement(agent, ticker, round_id, qty, payout_cents, scalar=0,
                      void=0) -> bool:
    """Returns False if this settlement was already recorded (rerun of a round)."""
    conn = db()
    with conn:
        cursor = conn.execute(
            "INSERT OR IGNORE INTO settlements "
            "(agent, ticker, round_id, qty, payout_cents, scalar, void) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (agent, ticker, round_id, qty, payout_cents, scalar, void),
        )
    return cursor.rowcount > 0


def settlements_for(agent: str) -> list[sqlite3.Row]:
    return db().execute(
        "SELECT * FROM settlements WHERE agent = ? ORDER BY round_id, ticker",
        (agent,),
    ).fetchall()


def fills_for(agent: str, ticker: str | None = None) -> list[sqlite3.Row]:
    if ticker is None:
        return db().execute(
            "SELECT * FROM fills WHERE agent = ? ORDER BY round_id, id", (agent,)
        ).fetchall()
    return db().execute(
        "SELECT * FROM fills WHERE agent = ? AND ticker = ? ORDER BY round_id, id",
        (agent, ticker),
    ).fetchall()


# --- rounds ----------------------------------------------------------------


def insert_quotes(rows) -> int:
    """Write quote rows for a round. Rejected sets are recorded too, one row
    per market in the set, so a rejection is visible per market."""
    rows = list(rows)
    if not rows:
        return 0
    conn = db()
    with conn:
        cursor = conn.executemany(
            "INSERT OR REPLACE INTO quotes (round_id, agent, ticker, bid_cents, "
            "bid_size, ask_cents, ask_size, status, reject_reason) "
            "VALUES (:round_id, :agent, :ticker, :bid_cents, :bid_size, "
            ":ask_cents, :ask_size, :status, :reject_reason)",
            rows,
        )
    return cursor.rowcount


def quotes_for(agent: str, ticker: str) -> list[sqlite3.Row]:
    return db().execute(
        "SELECT * FROM quotes WHERE agent = ? AND ticker = ? ORDER BY round_id",
        (agent, ticker),
    ).fetchall()


def accepted_quotes_by_round(ticker: str) -> dict[int, list[dict]]:
    """The book each round of a market presented, keyed by round_id.

    Returned as the plain dicts `auction` works on, so the book a brief reports
    is read by the same functions that cleared it. Derived from the quote rows
    rather than stored alongside the auction: the rows already are the book, and
    deriving keeps this correct for rounds run before anything reported it.

    Rejected sets never entered the auction and are excluded.
    """
    rows = db().execute(
        "SELECT round_id, agent, bid_cents, bid_size, ask_cents, ask_size "
        "FROM quotes WHERE ticker = ? AND status = 'accepted' "
        "ORDER BY round_id, agent",
        (ticker,),
    ).fetchall()

    books: dict[int, list[dict]] = {}
    for row in rows:
        books.setdefault(row["round_id"], []).append({
            "agent": row["agent"],
            "bid_cents": row["bid_cents"], "bid_size": row["bid_size"],
            "ask_cents": row["ask_cents"], "ask_size": row["ask_size"],
        })
    return books


def auction_exists(ticker: str, round_id: int) -> bool:
    return db().execute(
        "SELECT 1 FROM auctions WHERE ticker = ? AND round_id = ?",
        (ticker, round_id),
    ).fetchone() is not None


def record_auction(auction_row: dict, fill_rows, cash_rows=None, position_rows=None) -> bool:
    """Write one auction result, its fills, and the ledger effects of those
    fills, in a single transaction.

    The (ticker, round_id) primary key is what makes a crashed round safe to
    rerun: if the auction row is already there the whole unit is skipped, so
    fills can never be written twice — and because the money moves in the same
    transaction, a printed auction whose cash was never moved is not a
    reachable state.
    """
    conn = db()
    try:
        with conn:
            conn.execute(
                "INSERT INTO auctions (ticker, round_id, ts, clear_cents, "
                "cleared_qty, tie_lo, tie_hi, indicative_cents) "
                "VALUES (:ticker, :round_id, :ts, :clear_cents, :cleared_qty, "
                ":tie_lo, :tie_hi, :indicative_cents)",
                {"indicative_cents": None, **auction_row},
            )
            fill_rows = list(fill_rows)
            if fill_rows:
                conn.executemany(
                    "INSERT INTO fills (round_id, agent, ticker, signed_qty, "
                    "price_cents) VALUES (:round_id, :agent, :ticker, "
                    ":signed_qty, :price_cents)",
                    fill_rows,
                )
            for agent, balance in (cash_rows or {}).items():
                conn.execute(
                    "INSERT INTO cash (agent, balance_cents) VALUES (?, ?) "
                    "ON CONFLICT(agent) DO UPDATE SET "
                    "balance_cents = excluded.balance_cents",
                    (agent, balance),
                )
            for (agent, ticker), (qty, basis) in (position_rows or {}).items():
                conn.execute(
                    "INSERT INTO positions (agent, ticker, signed_qty, "
                    "cost_basis_cents) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(agent, ticker) DO UPDATE SET "
                    "signed_qty = excluded.signed_qty, "
                    "cost_basis_cents = excluded.cost_basis_cents",
                    (agent, ticker, qty, basis),
                )
    except sqlite3.IntegrityError:
        return False
    return True


def auction_tickers_in_round(round_id: int) -> set[str]:
    rows = db().execute(
        "SELECT ticker FROM auctions WHERE round_id = ?", (round_id,)
    ).fetchall()
    return {r["ticker"] for r in rows}


def auctions_for(ticker: str) -> list[sqlite3.Row]:
    """Full public print history for a market, oldest first. This is the
    market history agents inherit — no Kalshi data is involved."""
    return db().execute(
        "SELECT * FROM auctions WHERE ticker = ? ORDER BY round_id", (ticker,)
    ).fetchall()


def last_clear_cents(ticker: str) -> int | None:
    row = db().execute(
        "SELECT clear_cents FROM auctions WHERE ticker = ? AND clear_cents IS NOT NULL "
        "ORDER BY round_id DESC LIMIT 1",
        (ticker,),
    ).fetchone()
    return row["clear_cents"] if row else None


def last_round_id() -> int | None:
    row = db().execute("SELECT MAX(round_id) AS r FROM auctions").fetchone()
    return row["r"] if row and row["r"] is not None else None


def record_call(agent, ticker, round_id, ts, brief, raw_response, parse_ok) -> None:
    conn = db()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO calls (agent, ticker, round_id, ts, brief, "
            "raw_response, parse_ok) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (agent, ticker, round_id, ts, brief, raw_response, int(parse_ok)),
        )


def log_run_config(run_id: str, ts: int, config_json: str | None = None) -> None:
    conn = db()
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO run_config (run_id, ts, config_json) "
            "VALUES (?, ?, ?)",
            (run_id, ts, config_json if config_json is not None else config.as_json()),
        )


def run_configs() -> list[sqlite3.Row]:
    return db().execute(
        "SELECT * FROM run_config ORDER BY ts"
    ).fetchall()
