"""Continuation filtering preserves inherited book state and dispatch timing."""

import asyncio
from pathlib import Path

from prediction_markets.analysis.dataset import RunDataset, RunReader, Filters
from prediction_markets.storage.runs import begin_segment


def test_current_segment_filters_actions_fills_and_trace_but_replays_prior_book(participant_rig):
    rig = participant_rig
    path = Path(rig.conn.execute("PRAGMA database_list").fetchone()[2])
    agent, _ = rig.make()
    seller, _ = rig.make(agent_id="agent-02")
    def submit(participant, side, price, qty):
        asyncio.run(participant._handle_response(rig.response(rig.call("submit_order", {
            "ticker": "MKT-01", "action": side, "order_type": "limit",
            "quantity": qty, "price_cents": price,
        }))))
    submit(agent, "buy", 40, 10)
    # A new segment with an inherited resting bid exercises replay before filtering.
    begin_segment(rig.conn, "child", path)
    boundary = rig.conn.execute("SELECT first_transcript_entry_id FROM run_segments").fetchone()[0]
    rig.advance(120)
    submit(seller, "sell", 30, 4)
    with RunDataset(path, segment="current") as run:
        actions = list(run.actions())
        assert len(actions) == 1 and actions[0]["agent_id"] == "agent-02"
        books = list(run.orderbooks())
        assert len(books) == 1
        assert books[0]["bids"][0]["quantity"] == 6
        assert books[0]["bids"][0]["price_cents"] == 40
        assert books[0]["time"] == actions[0]["time"]
        assert [fill["quantity"] for fill in run.prices(agent_id="agent-01")[1]] == [4]
        assert run.prices(agent_id="agent-03")[1] == []
        assert all(row["entry_id"] >= boundary for row in run.transcript())
    with RunReader(path) as run:
        rows = list(run.transcript(filters=Filters(agent="agent-01")))
        assert rows and all(row["agent_id"] == "agent-01" for row in rows)
