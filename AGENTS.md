# Repository guide

This repository replicates two experiments from J. Bun, S. Ciliberti,
E. Sérié, "Cleaning correlation matrices with external information: the
Maximum Variance estimator" (Capital Fund Management working paper): a Monte
Carlo recovery test of the Maximum Variance (MaxVar) correlation estimator
against the Rotationally Invariant Estimator (RIE) and the empirical
correlation matrix (paper Sec. 3), and an out-of-sample risk-parity backtest
on 47 futures contracts, 1993-2018 (paper Sec. 4).

## Layout

- `src/` — flat library modules: `config.py` (every paper-derived parameter,
  as module-level variables with a source comment), `ema.py` (EMA volatility
  and correlation estimation), `maxvar.py` (MaxVar estimator), `rie.py`
  (rotationally invariant estimator), `portfolio.py` (risk-parity / maximum
  diversification weights), `metrics.py` (Frobenius distance, Sharpe ratio),
  `backtest.py` (the shared weekly backtest loop used by `exp/exp2.py`).
- `exp/` — `exp1.py` (Monte Carlo recovery), `exp2.py` (risk-parity backtest).
  Both call only the `src/` functions and write their outputs to `results/`.
- `data/` — `data_loader.py`, the only place that loads market data, through
  the QuantX Dataloader REST API. No synthetic data generator is used:
  both experiments run on real daily OHLCV data.
- `tests/` — unit tests on synthetic/small fixtures, and market-data tests
  (`tests/test_market_data.py`, marked `market_data`) that load a reduced
  slice of the real universe through `data/data_loader.py`.
- `results/` — tables (CSV), plots (PNG), JSON summaries, and `RESULTS.md`,
  all written by `exp/exp1.py` and `exp/exp2.py`. Re-running an experiment
  overwrites its files here; the numbers and verdicts in `RESULTS.md` must be
  checked against the new files and updated by hand after a rerun.

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, needs the API keys

python -m exp.exp1                      # Monte Carlo recovery test
python -m exp.exp2                      # risk-parity backtest, 1993-2018
```

`reproduce.sh` runs the same commands in sequence.

## Conventions

- Every experiment parameter lives in `src/config.py` as a module-level
  UPPER_CASE variable with a comment citing the paper section it comes from,
  or stating that the paper does not specify it. No parameter is hardcoded
  inside `src/`, `exp/`, or `data/`.
- Market data is loaded only through `data/data_loader.py`
  (QuantX Dataloader REST API, `https://quantx-api.limex.pro/apis/dataloader`).
  `src/` and `exp/` never call a data API directly.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` file (see `.env.example`); they are never written to source,
  configuration, logs, or results.
- Unit tests run on synthetic or small fixed inputs and do not require
  network access; tests marked `market_data` load a small real slice of the
  universe and are skipped automatically when `DATALOADER_API_KEY` is unset.
- Deviations from the paper (data proxies, inferred conventions, numerical
  gaps and their cause) are recorded only in `results/RESULTS.md`, never as
  silent substitutions in the code.
