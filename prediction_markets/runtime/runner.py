"""One process: fixed cohort, lifecycle polling, and independent agent loops."""

from __future__ import annotations

import argparse
import asyncio
import fcntl
import json
import logging
import os
import signal
from datetime import datetime, timezone
from functools import partial
from uuid import uuid4

import httpx

from .. import config
from ..storage import database, runs as resume
from ..integrations import model_client
from ..integrations.tokenization import load_tokenizer
from ..agents import Participant, PermanentInferenceError
from ..exchange import Exchange
from ..markets.update import MarketFeed

# Keep CLI execution (python -m, where __name__ is __main__) under the project
# logger so startup, failure and shutdown diagnostics reach each run.log.
log = logging.getLogger("prediction_markets.runner")

# Modal holds requests while its container boots, so a cold endpoint looks like
# a hang rather than an error: every participant burns its whole request
# deadline against a server that is still loading. Wait for readiness once here
# instead. The budget matches startup_timeout in deploy/modal_qwen.py.
ENDPOINT_READY_SECONDS = 900
ENDPOINT_PROBE_SECONDS = 30
ENDPOINT_RETRY_SECONDS = 5


async def wait_for_endpoint(
    client, base_url: str, api_key: str, model: str, *,
    budget: float = ENDPOINT_READY_SECONDS,
    probe_timeout: float = ENDPOINT_PROBE_SECONDS,
    retry_after: float = ENDPOINT_RETRY_SECONDS,
) -> float:
    """Block until the endpoint serves the pinned model; never start agents blind.

    Authentication and model-identity failures are configuration errors and
    stop the run immediately. Transport failures, timeouts and other statuses
    are ordinary cold-start symptoms and are retried until the budget expires.
    """
    loop = asyncio.get_running_loop()
    started = loop.time()
    deadline = started + budget
    attempts = 0
    while True:
        attempts += 1
        try:
            response = await client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=probe_timeout,
            )
        except httpx.TransportError as exc:
            detail = type(exc).__name__
        else:
            if response.status_code in {401, 403}:
                raise PermanentInferenceError(
                    "model endpoint rejected the API key; check the environment "
                    "variable and the Modal secret"
                )
            if response.status_code == 200:
                try:
                    served = {entry["id"] for entry in response.json()["data"]}
                except (ValueError, KeyError, TypeError):
                    raise PermanentInferenceError(
                        "model endpoint returned a malformed model list"
                    ) from None
                if model not in served:
                    raise PermanentInferenceError(
                        f"model endpoint does not serve the pinned model {model}"
                    )
                waited = loop.time() - started
                log.info("endpoint ready after %.0fs and %d probe(s)", waited, attempts)
                return waited
            detail = f"HTTP {response.status_code}"
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise PermanentInferenceError(
                f"model endpoint was not ready within {budget:.0f}s ({detail}); "
                "check the serving logs before starting a run"
            )
        log.info("waiting for the model endpoint (%s); %.0fs of budget left", detail, remaining)
        await asyncio.sleep(min(retry_after, remaining))


async def run(settings: config.Settings, *, infer, tokenizer, run_id: str | None = None,
              resume_from: str | None = None, runs_dir=None) -> dict:
    """Execute a fresh experiment or a continuation with frozen settings.

    SQLite and exchange access stay on this event-loop thread. A task failure
    stops the entire run. Continuations copy a stopped parent to a new run ID.
    """
    source = resume.resolve_source(resume_from) if resume_from is not None else None
    if source is not None and resume.load_settings(source) != settings:
        raise ValueError("resume must use the parent's entire saved configuration; overrides are not supported")
    run_id = run_id or f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{uuid4().hex[:8]}"
    run_dir = resume.create_run_directory(run_id, runs_dir=runs_dir)
    run_lock = (run_dir / "run.lock").open("x")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    handler = logging.FileHandler(run_dir / "run.log", mode="x", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    project_log = logging.getLogger("prediction_markets")
    previous_level = project_log.level
    project_log.setLevel(logging.INFO)
    project_log.addHandler(handler)
    conn = exchange = None
    tasks = []
    participants = []
    lineage = None
    stop_reason = "startup_failed"
    summary = {}
    try:
        log.info("run directory: %s", run_dir)
        log.info("model: %s provider=%s revision=%s", settings.model_name,
                 settings.model_provider, settings.model_revision)
        conn = (database.init(run_dir / "run.db", run_settings=settings, register=False) if source is None else
                database.fork(run_dir / "run.db", source, run_settings=settings, register=False))
        lineage = resume.begin_segment(conn, run_id, source)
        exchange = Exchange(conn)
        feed = MarketFeed(exchange, settings)
        if source is None:
            await feed.initialize()
        else:
            log.info("resuming parent=%s root=%s with frozen settings", lineage["parent_run_id"], lineage["root_run_id"])
            await feed.resume()

        semaphore = asyncio.Semaphore(settings.max_concurrent_inference)

        async def limited_infer(**request):
            async with semaphore:
                return await infer(**request)

        participants = [
            Participant(
                agent_id, settings.model_name, exchange=exchange, conn=conn,
                settings=settings, infer=limited_infer, tokenizer=tokenizer,
            )
            for agent_id in settings.agent_ids
        ]
        # Fail before any hosted request if unabridged mandatory input cannot fit.
        for participant in participants:
            if source is not None:
                participant.restore(lineage["parent_run_id"])
            participant.build_messages()
        stop_reason = "task_failed"
        tradable = [market["ticker"] for market in exchange.assigned_markets(settings.agent_ids[0]) if market["tradable"]]
        if source is not None:
            log.info("resumed tradable markets (%d): %s", len(tradable), tradable)
        if source is None or tradable:
            tasks = [asyncio.create_task(feed.run(), name="market-feed")]
            tasks.extend(asyncio.create_task(p.run(), name=p.agent_id) for p in participants)
        log.info("started %d participants; duration=%ds; inference concurrency=%d",
                 len(participants) if tasks else 0, settings.run_duration_seconds, settings.max_concurrent_inference)
        log.info("daily sessions: %s–%s %s; elapsed costs order=%dm read=%dm news=%dm hold=%dm",
                 settings.trading_day_start, settings.trading_day_end, settings.trading_timezone,
                 settings.order_tool_minutes, settings.read_tool_minutes,
                 settings.news_tool_minutes, settings.hold_for_now_minutes)
        done, _ = await asyncio.wait(
            tasks, timeout=settings.run_duration_seconds, return_when=asyncio.FIRST_COMPLETED,
        ) if tasks else (set(), set())
        for task in done:
            # Propagate even when another task reports settlement simultaneously.
            task.result()
        if exchange.all_markets_settled():
            stop_reason = "all_markets_settled"
        elif not tasks:
            stop_reason = "no_tradable_markets"
        elif done:
            raise RuntimeError("runtime task exited before all markets settled")
        else:
            stop_reason = "duration_elapsed"
    except asyncio.CancelledError:
        stop_reason = "canceled"
        raise
    except BaseException as exc:
        if isinstance(exc, PermanentInferenceError):
            log.error("run stopped by %s: %s", type(exc).__name__, exc)
        else:
            log.error("run stopped by %s", type(exc).__name__)
        raise
    finally:
        try:
            # Stop every producer before touching the final book or closing SQLite.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if exchange is not None:
                canceled = exchange.cancel_all_orders()
                trades = conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]
                settled = conn.execute("SELECT COUNT(*) FROM market_settlements").fetchone()[0]
                accounts = [exchange.get_account(agent_id) for agent_id in settings.agent_ids]
                summary = {
                    "run_directory": str(run_dir), "stop_reason": stop_reason,
                    "trade_count": trades, "settled_market_count": settled,
                    "unsettled_markets": exchange.unsettled_tickers(),
                    "canceled_order_count": canceled, "accounts": accounts,
                    "parent_run_id": lineage["parent_run_id"], "root_run_id": lineage["root_run_id"],
                    "new_trade_count": conn.execute("SELECT COUNT(*) FROM trades WHERE trade_id>=?",
                                                    (lineage["first_trade_id"],)).fetchone()[0],
                }
                log.info("stopped: %s; trades=%d; settled=%d; shutdown cancellations=%d",
                         stop_reason, trades, settled, canceled)
                if source is not None:
                    log.info("continuation trades=%d; cumulative trades=%d", summary["new_trade_count"], trades)
                if trades == 0:
                    log.info("zero trades: no fills occurred in this run")
                for account in accounts:
                    log.info("final account (positions remain unresolved; equity uses last internal trade): %s",
                             json.dumps(account, sort_keys=True))
                log.info("unsettled markets retained without payout: %s", summary["unsettled_markets"])
        finally:
            if conn is not None:
                if lineage is not None:
                    resume.finish_segment(conn, run_id, stop_reason)
                conn.close()
            project_log.removeHandler(handler)
            handler.close()
            project_log.setLevel(previous_level)
            run_lock.close()
    return summary


async def run_live(settings: config.Settings, *, run_id: str | None = None,
                   resume_from: str | None = None, runs_dir=None) -> dict:
    if resume_from is not None and resume.load_settings(resume_from) != settings:
        raise ValueError("resume must use the parent's entire saved configuration; overrides are not supported")
    config.load_credentials()
    default_url = "https://api.openai.com/v1" if settings.model_provider == "openai" else ""
    base_url = os.environ.get(settings.model_base_url_env, default_url)
    api_key = os.environ.get(settings.model_api_key_env, "")
    base_url = model_client.validate_endpoint(base_url, api_key)
    log.info("loading %s token counter (no model weights)", settings.model_provider)
    tokenizer = await asyncio.to_thread(load_tokenizer, settings)
    async with httpx.AsyncClient(
        limits=httpx.Limits(max_connections=settings.max_concurrent_inference),
        follow_redirects=False,
    ) as client:
        if settings.model_provider == "vllm":
            log.info("waiting for the model endpoint before starting participants")
            await wait_for_endpoint(client, base_url, api_key, settings.model_name)
        infer = partial(
            model_client.infer, client=client, base_url=base_url, api_key=api_key,
            timeout_seconds=settings.request_timeout_seconds,
            provider=settings.model_provider, reasoning_effort=settings.openai_reasoning_effort,
            openai_api=settings.openai_api,
        )
        options = {"resume_from": resume_from} if resume_from is not None else {}
        return await run(settings, infer=infer, tokenizer=tokenizer, run_id=run_id, runs_dir=runs_dir, **options)


async def _run_with_signals(settings: config.Settings, *, resume_from: str | None = None) -> dict:
    """First Ctrl-C/SIGTERM cancels awaits and lets run() persist its final state."""
    loop = asyncio.get_running_loop()
    task = asyncio.current_task()
    installed = []
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(sig, task.cancel)
            installed.append(sig)
        return await run_live(settings, **({"resume_from": resume_from} if resume_from is not None else {}))
    finally:
        for sig in installed:
            loop.remove_signal_handler(sig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="prediction-markets run", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", help="Path to the complete MVP YAML configuration for a fresh run")
    mode.add_argument("--resume-from", help="Stopped run ID, directory, or run.db; inherits the entire saved config")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        source = resume.resolve_source(args.resume_from) if args.resume_from else None
        settings = resume.load_settings(source) if source else config.load(args.config, remember=False)
        asyncio.run(_run_with_signals(settings, **({"resume_from": str(source)} if source else {})))
    except (asyncio.CancelledError, KeyboardInterrupt):
        log.info("interrupted; saved artifacts can be continued with --resume-from")
        return 130
    except Exception as exc:
        # HTTP client errors are sanitized; no response bodies or env values.
        log.error("run failed (%s): %s", type(exc).__name__, exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
