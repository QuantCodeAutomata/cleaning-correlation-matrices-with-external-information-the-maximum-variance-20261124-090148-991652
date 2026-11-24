"""Daily return series for the 47-futures universe of Table 2 of the paper.

Equity-index, currency and most bond legs are proxied by U.S.-listed ETFs and
Massive-direct forex pairs loaded through the QuantX Dataloader; five U.S.
Treasury maturities are rebuilt from Massive's direct constant-maturity
Treasury-yield series (duration-approximated bond-future returns), since the
Dataloader/Massive path does not serve the exchange-traded index and
government-bond futures contracts of Table 2 themselves. See
``src/config.py`` for the exact ticker map and ``results/RESULTS.md``,
"Deviations from the paper", for the realised per-series start dates.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

from src import config

DATALOADER_API_ROOT = "https://quantx-api.limex.pro/apis/dataloader"
MASSIVE_API_ROOT = "https://api.massive.com"
BATCH_SIZE = 15
CACHE_DIR = Path(__file__).resolve().parent / "cache"


class DataLoaderError(RuntimeError):
    """A failed Dataloader/Massive request or a load where no ticker loaded."""


def _dataloader_headers(include_massive: bool = True) -> dict[str, str]:
    load_dotenv()
    dataloader_key = os.environ.get("DATALOADER_API_KEY")
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not dataloader_key:
        raise DataLoaderError("DATALOADER_API_KEY must be set")
    headers = {"Authorization": f"Bearer {dataloader_key}"}
    if include_massive:
        if not massive_key:
            raise DataLoaderError("MASSIVE_API_KEY must be set")
        headers["X-Massive-Api-Key"] = massive_key
    return headers


def _massive_headers() -> dict[str, str]:
    load_dotenv()
    massive_key = os.environ.get("MASSIVE_API_KEY")
    if not massive_key:
        raise DataLoaderError("MASSIVE_API_KEY must be set")
    return {"Authorization": f"Bearer {massive_key}"}


def _post_fetch_ohlcvs(tickers: list[str], start_date: str, end_date: str) -> dict:
    # Market data: QuantX Dataloader — https://quantx.limex.pro/
    #
    # Price history in this project is loaded through the QuantX Dataloader, a
    # research-ready market-data service available through a REST API and as an
    # MCP server for AI agents. It is built on market data from Massive
    # (https://massive.com/) and prepares every stock history for backtesting:
    #   - prices adjusted for dividends, ready for total-return research;
    #   - one continuous history when a company changes its ticker;
    #   - clean separation when a ticker has belonged to different companies;
    #   - careful corporate-action handling and fine-grained data polishing.
    #
    # Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
    # MASSIVE_API_KEY at https://massive.com/.
    body = {
        "tickers": tickers,
        "timeframe": "1d",
        "start_date": start_date,
        "end_date": end_date,
        "source": "massive",
        "adjust_dividends": True,
        "yahoo_backstop": True,
    }
    for attempt in range(3):
        response = requests.post(
            DATALOADER_API_ROOT + "/fetch-ohlcvs",
            json=body,
            headers=_dataloader_headers(),
            timeout=300,
        )
        if response.status_code not in (429, 503, 504) or attempt == 2:
            break
        time.sleep(5 * 2 ** attempt)
    if response.status_code != 200:
        try:
            err = response.json()
        except ValueError:
            err = {"error": response.text[:300]}
        raise DataLoaderError(f"fetch-ohlcvs: HTTP {response.status_code}: {err}")
    return response.json()


def fetch_close_prices(tickers: list[str], start_date: str, end_date: str) -> tuple[pd.DataFrame, list[dict]]:
    """Daily close prices for ``tickers`` through the QuantX Dataloader.

    Returns a wide DataFrame indexed by date (one column per ticker) and the
    list of ``ticker_details`` entries reported by the Dataloader.
    """
    tickers = list(dict.fromkeys(tickers))
    batches = [tickers[i:i + BATCH_SIZE] for i in range(0, len(tickers), BATCH_SIZE)]
    closes: dict[str, pd.Series] = {}
    all_details: list[dict] = []
    for batch in batches:
        answer = _post_fetch_ohlcvs(batch, start_date, end_date)
        result = answer["result"]
        details = result.get("ticker_details") or []
        all_details.extend(details)
        if result.get("data_stored"):
            raise DataLoaderError(
                "fetch-ohlcvs returned a stored (large) result; "
                "narrow the batch further to keep the answer inline"
            )
        rows = result["ohlcv_df"]["rows"]
        if not rows:
            continue
        wide = pd.DataFrame(rows)
        stamps = pd.to_datetime(wide["timestamp"], utc=True).dt.tz_convert("America/New_York")
        dates = stamps.dt.normalize().dt.tz_localize(None)
        for detail in details:
            if detail.get("error"):
                continue
            ticker, alias = detail["ticker"], detail.get("alias") or ""
            for name in (f"{ticker}_close", f"{alias}_close", "close"):
                if name in wide.columns:
                    closes[ticker] = pd.Series(wide[name].to_numpy(), index=dates, name=ticker)
                    break
    failed = {d["ticker"]: d["error"] for d in all_details if d.get("error")}
    if not closes:
        raise DataLoaderError(f"No ticker loaded: {failed}")
    prices = pd.DataFrame(closes).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices, all_details


def fetch_treasury_yields(start_date: str, end_date: str) -> pd.DataFrame:
    """Daily constant-maturity U.S. Treasury yields from Massive's Economy data.

    Direct Massive REST call (``GET /fed/v1/treasury-yields``): the Dataloader
    serves OHLCV bars only, not this reference series.
    """
    response = requests.get(
        MASSIVE_API_ROOT + "/fed/v1/treasury-yields",
        headers=_massive_headers(),
        params={"date.gte": start_date, "date.lte": end_date, "limit": 5000},
        timeout=120,
    )
    if response.status_code != 200:
        raise DataLoaderError(f"treasury-yields: HTTP {response.status_code}: {response.text[:300]}")
    payload = response.json()
    rows = payload.get("results") or payload.get("values") or []
    frames = list(rows)
    next_url = payload.get("next_url")
    while next_url:
        response = requests.get(next_url, headers=_massive_headers(), timeout=120)
        if response.status_code != 200:
            break
        payload = response.json()
        new_rows = payload.get("results") or payload.get("values") or []
        if not new_rows:
            break
        frames.extend(new_rows)
        next_url = payload.get("next_url")
    if not frames:
        raise DataLoaderError("treasury-yields: empty response")
    df = pd.DataFrame(frames)
    date_col = "date" if "date" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[date_col])
    df = df.set_index("date").sort_index()
    return df


def _bond_returns_from_yields(yields: pd.DataFrame) -> pd.DataFrame:
    """Duration-approximated returns: r_t = -duration * d(yield_t) / 100.

    A parallel shift of ``duration * d(yield)`` percentage points in a
    constant-maturity yield implies an approximate price return of that sign
    and magnitude divided by 100 (yields are quoted in percent); this is the
    standard modified-duration approximation used when a bond future itself
    is not directly observable (paper Table 2 bond-future legs rebuilt from
    Massive's constant-maturity Treasury-yield series).
    """
    out = {}
    for name, (field, duration) in config.US_TREASURY_MATURITIES.items():
        if field not in yields.columns:
            raise DataLoaderError(f"treasury-yields: missing field {field!r}")
        yield_change = yields[field].astype(float).diff()
        out[name] = -duration * yield_change / 100.0
    return pd.DataFrame(out)


def load_universe(
    start_date: str,
    end_date: str,
    use_cache: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Daily returns for every instrument of ``src.config.ALL_INSTRUMENTS``.

    Returns
    -------
    returns : DataFrame (T, 47)
        Daily returns indexed by calendar date, one column per instrument
        name (``src.config.ALL_INSTRUMENTS``). Non-trading days for a given
        instrument (market holiday in that instrument's home market) are
        filled with a zero return; days on which no instrument has data are
        dropped.
    meta : dict
        Load metadata: per-ticker Dataloader ``ticker_details``, failed
        tickers, non-Massive data sources, and the realised per-instrument
        first/last observation dates.
    """
    key = hashlib.sha256(f"{start_date}|{end_date}|v2".encode()).hexdigest()[:16]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_returns, cache_meta = CACHE_DIR / f"universe_{key}.csv.gz", CACHE_DIR / f"universe_{key}.json"
    if use_cache and cache_returns.exists() and cache_meta.exists():
        returns = pd.read_csv(cache_returns, index_col=0, parse_dates=True)
        return returns, json.loads(cache_meta.read_text())

    equity_fx_tickers = {**config.STOCK_TICKERS, **config.NON_US_BOND_TICKERS, **config.CURRENCY_TICKERS}
    name_of_ticker = {v: k for k, v in equity_fx_tickers.items()}
    prices, details = fetch_close_prices(list(equity_fx_tickers.values()), start_date, end_date)
    prices = prices.rename(columns=name_of_ticker)
    equity_fx_returns = prices.pct_change()

    yields = fetch_treasury_yields(start_date, end_date)
    bond_returns = _bond_returns_from_yields(yields)

    returns = equity_fx_returns.join(bond_returns, how="outer")
    returns = returns[[c for c in config.ALL_INSTRUMENTS if c in returns.columns]]
    missing = [c for c in config.ALL_INSTRUMENTS if c not in returns.columns]
    if missing:
        raise DataLoaderError(f"No data loaded for instruments: {missing}")

    # Each column's history can start later than the requested window (a young
    # ETF proxy, paper Sec. "Data-availability reduction"). The common sample
    # used for estimation starts only once every instrument has data, so a
    # young proxy's pre-listing period is excluded rather than filled with a
    # fabricated zero return.
    first_valid = {c: returns[c].first_valid_index() for c in returns.columns}
    common_start = max(first_valid.values())
    returns = returns.loc[common_start:]
    returns = returns.dropna(how="all")
    returns = returns.fillna(0.0)  # isolated single-market holidays within the common window

    first_obs = {c: str(first_valid[c]) for c in config.ALL_INSTRUMENTS}
    meta = {
        "request": {"start_date": start_date, "end_date": end_date},
        "ticker_details": details,
        "failed": {d["ticker"]: d["error"] for d in details if d.get("error")},
        "non_massive_source": {
            d["ticker"]: d["data_source"] for d in details
            if d.get("data_source") not in (None, "massive")
        },
        "first_observation": first_obs,
        "row_count": int(len(returns)),
        "date_range": [str(returns.index.min()), str(returns.index.max())],
    }
    returns.to_csv(cache_returns)
    cache_meta.write_text(json.dumps(meta, indent=2, default=str))
    return returns, meta
