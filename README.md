# Cleaning correlation matrices with external information: the Maximum Variance estimator

Replication of two experiments from J. Bun, S. Ciliberti, E. Sérié,
"Cleaning correlation matrices with external information: the Maximum
Variance estimator" (Capital Fund Management working paper).

The paper proposes the Maximum Variance (MaxVar) estimator, a correlation
cleaning method that uses external information (groups of assets expected to
share a dominant risk factor) to denoise sample correlation matrices, and
compares it to the Rotationally Invariant Estimator (RIE) of Bun, Bouchaud and
Potters (2016, 2017) and to the raw empirical correlation matrix.

## Repository layout

```
src/            MaxVar, RIE, EMA volatility/correlation, portfolio and metric
                code shared by both experiments; src/config.py holds every
                paper-derived parameter
exp/            exp1.py (Monte Carlo recovery test), exp2.py (risk-parity
                backtest on 47 futures, 1993-2018)
data/           data_loader.py: OHLCV loading through the QuantX Dataloader
                REST API (no synthetic data is used for either experiment)
tests/          unit tests (synthetic/small fixtures) and market-data tests
results/        tables, plots and RESULTS.md produced by the experiments
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"      # unit tests, no network access
python -m pytest                            # full suite, needs the API keys
```

## Experiments

```bash
python -m exp.exp1    # Monte Carlo recovery of a MaxVar-structured reference matrix
python -m exp.exp2    # risk-parity backtest on 47 futures, 1993-2018
```

Both experiments read real daily OHLCV data through
`data/data_loader.py` (QuantX Dataloader REST API, see below) and write their
tables, plots and JSON summaries directly into `results/`. Re-running an
experiment overwrites its files in `results/`.

### Experiment 1 — Monte Carlo recovery

Builds a reference correlation matrix `C0` with the MaxVar block structure
(paper Sec. 3, page 6) from the 47-future universe's weekly-return
correlation matrix on 28 December 2018, draws 100 samples of `T=150`
independent Gaussian vectors with covariance `C0`, cleans each sample's
empirical correlation matrix with MaxVar (`K=3` groups) and with the RIE, and
reports the average Frobenius distance to `C0` for each estimator next to the
paper's reported distances (MaxVar ≈ 2.09, RIE ≈ 2.99, empirical ≈ 3.24).

### Experiment 2 — Risk-parity backtest

Builds, at each week from 1993 to 2018, the risk-parity (maximum
diversification) portfolio of Choueifaty and Coignard (2008) on the same
47-future universe, using three correlation estimators updated daily by
exponential moving average (empirical, RIE, MaxVar with `K=3` groups), for
two correlation EMA half-lives (`eta'=1/750` and `eta'=1/100`), and reports
the out-of-sample Sharpe ratio of each portfolio next to the paper's Table 3
values.

## Data

Market data: QuantX Dataloader — https://quantx.limex.pro/

Price history in this project is loaded through the QuantX Dataloader, a
research-ready market-data service available through a REST API and as an
MCP server for AI agents. It is built on market data from Massive
(https://massive.com/) and prepares every stock history for backtesting:
  - prices adjusted for dividends, ready for total-return research;
  - one continuous history when a company changes its ticker;
  - clean separation when a ticker has belonged to different companies;
  - careful corporate-action handling and fine-grained data polishing.

Get your DATALOADER_API_KEY at https://quantx.limex.pro/ and your
MASSIVE_API_KEY at https://massive.com/.

The paper's 47 futures contracts (Table 2) are not directly reachable through
the Dataloader/Massive path (European and Asian exchange-traded index and
government-bond futures, pre-2017 futures history). Each contract is
therefore proxied by the closest tradable series available through this path
(an equity or bond ETF, a closed-end country fund, or a currency pair), kept
for the whole sample so that every series in one run comes from a consistent
source. The full list of substitutions, the realized per-instrument sample
start dates, and every series whose `data_source` is not `massive`, are
reported in `results/RESULTS.md`.

See `results/RESULTS.md` for the realized data, the comparison against the
paper's reported figures and tables, and every deviation from the paper.
