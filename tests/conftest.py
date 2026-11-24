"""Shared pytest fixtures: a cached, reduced real-data slice for market_data tests."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from data import data_loader


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "market_data: tests that load a real market-data slice (skipped without API keys)"
    )


@pytest.fixture(scope="session")
def market_data_returns(request: pytest.FixtureRequest) -> pd.DataFrame:
    """A reduced real-data slice: a few instruments, two years, through data_loader.

    Cached across test sessions with pytest's cache so repeated runs do not
    re-download it.
    """
    if not (os.environ.get("DATALOADER_API_KEY") and os.environ.get("MASSIVE_API_KEY")):
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cache_key = "market_data/reduced_slice_v1"
    cached = request.config.cache.get(cache_key, None)
    if cached is not None:
        return pd.DataFrame(cached["data"]).set_index(
            pd.to_datetime(cached["index"])
        )
    tickers = {"SP500": "SPY", "Russell2000": "IWM", "EuroStoxx50": "FEZ"}
    prices, _ = data_loader.fetch_close_prices(
        list(tickers.values()), "2016-01-01", "2017-12-31"
    )
    prices = prices.rename(columns={v: k for k, v in tickers.items()})
    returns = prices.pct_change().dropna(how="all")
    request.config.cache.set(
        cache_key,
        {"data": returns.reset_index(drop=True).to_dict(orient="list"), "index": returns.index.astype(str).tolist()},
    )
    return returns
