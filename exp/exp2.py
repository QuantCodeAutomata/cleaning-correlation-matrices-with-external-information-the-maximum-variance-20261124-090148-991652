"""Experiment 2: risk-parity backtest on the 47-futures universe (1993-2018).

Reproduces paper Sec. 4 ("Application to a universe of 47 futures
contracts"): a weekly-rebalanced maximum-diversification portfolio
(Choueifaty and Coignard, 2008) is built from the empirical, RIE-cleaned and
MaxVar-cleaned correlation matrices, for both the primary EMA decay
eta'=1/750 and the shorter robustness decay eta'=1/100, and the out-of-sample
Sharpe ratio of each is reported over 1 January 1993 to 31 December 2018.

Run as ``python -m exp.exp2`` (reduced data window) or
``python -m exp.exp2 --full`` (paper's data window from 8 May 1985).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from data import data_loader
from src import config
from src.backtest import run_risk_parity_backtest
from src.maxvar import groups_from_sizes
from src.metrics import annualized_sharpe_ratio

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

ETA_PRIME_CONFIGS = [
    ("primary", config.CORR_EMA_ETA_PRIMARY, config.CORR_EMA_EFFECTIVE_T_PRIMARY),
    ("robustness", config.CORR_EMA_ETA_ROBUSTNESS, config.CORR_EMA_EFFECTIVE_T_ROBUSTNESS),
]
METHODS = ["empirical", "rie", "maxvar"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full data window starting 8 May 1985 instead of the reduced window",
    )
    args = parser.parse_args()

    start_date = config.DATA_START_DATE_PAPER if args.full else config.DATA_START_DATE_RUN
    returns, meta = data_loader.load_universe(start_date, config.DATA_END_DATE)
    assert list(returns.columns) == config.ALL_INSTRUMENTS, "instrument order must match src.config"

    groups = groups_from_sizes([len(config.GROUPS[g]) for g in ("stocks", "bonds", "currencies")])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sharpe_table = {}
    all_returns: dict[str, pd.Series] = {}
    for eta_label, eta_prime, effective_t in ETA_PRIME_CONFIGS:
        sharpe_table[eta_label] = {}
        for method in METHODS:
            series = run_risk_parity_backtest(
                returns, groups, method,
                vol_eta=config.VOL_EMA_ETA,
                corr_eta=eta_prime,
                effective_t=effective_t,
                burn_in_weeks=config.BACKTEST_WARM_UP_WEEKS,
            )
            windowed = series.loc[config.BACKTEST_START_DATE:config.BACKTEST_END_DATE]
            key = f"{eta_label}_{method}"
            all_returns[key] = windowed
            if len(windowed.dropna()) > 1 and windowed.std(ddof=1) > 0:
                sharpe = annualized_sharpe_ratio(windowed, config.WEEKS_PER_YEAR)
            else:
                sharpe = None
            sharpe_table[eta_label][method] = {
                "sharpe_ratio": sharpe,
                "n_weeks": int(len(windowed)),
                "n_skipped_weeks": int(len(series.attrs.get("skipped", []))),
            }

    returns_df = pd.DataFrame(all_returns)
    returns_df.to_csv(RESULTS_DIR / "exp2_weekly_portfolio_returns.csv")

    payload = {
        "config": {
            "universe_size": len(config.ALL_INSTRUMENTS),
            "data_start_date_used": start_date,
            "data_end_date": config.DATA_END_DATE,
            "backtest_start_date": config.BACKTEST_START_DATE,
            "backtest_end_date": config.BACKTEST_END_DATE,
            "vol_ema_eta": config.VOL_EMA_ETA,
            "burn_in_weeks": config.BACKTEST_WARM_UP_WEEKS,
            "eta_prime_configs": {label: eta for label, eta, _ in ETA_PRIME_CONFIGS},
        },
        "data_meta": {
            "row_count": meta.get("row_count"),
            "date_range": meta.get("date_range"),
            "first_observation": meta.get("first_observation"),
            "failed_tickers": meta.get("failed"),
            "non_massive_source": meta.get("non_massive_source"),
        },
        "sharpe_ratios": sharpe_table,
        "expected_from_paper": {
            "primary": {"empirical": 1.17, "rie": 1.24, "maxvar": 1.28},
            "robustness": {"empirical": None, "rie": 0.93, "maxvar": 1.05},
        },
    }
    with open(RESULTS_DIR / "exp2_results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, (eta_label, _, _) in zip(axes, ETA_PRIME_CONFIGS):
        reproduced = [sharpe_table[eta_label][m]["sharpe_ratio"] or 0.0 for m in METHODS]
        expected = [payload["expected_from_paper"][eta_label][m] or 0.0 for m in METHODS]
        x = range(len(METHODS))
        width = 0.35
        ax.bar([i - width / 2 for i in x], reproduced, width, label="Reproduced", color="steelblue")
        ax.bar([i + width / 2 for i in x], expected, width, label="Paper (Table 3)", color="lightgray")
        ax.set_xticks(list(x))
        ax.set_xticklabels(["Empirical", "RIE", "MaxVar"])
        ax.set_title(f"eta' = {dict((l, e) for l, e, _ in ETA_PRIME_CONFIGS)[eta_label]:.4f} ({eta_label})")
        ax.set_ylabel("Out-of-sample Sharpe ratio (1993-2018)")
        ax.legend()
    fig.suptitle("Risk-parity portfolio Sharpe ratio by correlation estimator (paper Sec. 4)")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_sharpe_ratios.png", dpi=150)
    plt.close(fig)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
