"""Experiment 1: Monte Carlo test of MaxVar, RIE and the empirical estimator.

Reproduces paper Sec. 3 ("Numerical tests"), page 6-7: a reference
correlation matrix C0 with the MaxVar block structure is built from the
MaxVar-cleaned correlation matrix of the 47-futures universe on 28 December
2018; 100 samples of T=150 independent Gaussian vectors with covariance C0
are drawn, and the empirical, RIE- and MaxVar-cleaned correlation matrices of
each sample are compared to C0 by their Frobenius distance.

Run as ``python -m exp.exp1`` (reduced data window) or
``python -m exp.exp1 --full`` (paper's data window from 8 May 1985).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from data import data_loader
from src import config
from src.ema import (
    daily_to_weekly_returns,
    ema_correlation_sequence,
    ema_volatility,
    rescale_by_volatility,
)
from src.maxvar import groups_from_sizes, maxvar_clean
from src.metrics import frobenius_distance
from src.rie import rie_clean

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def build_reference_matrix(start_date: str) -> tuple[np.ndarray, list[np.ndarray], dict]:
    """Reference matrix C0: the MaxVar-cleaned correlation matrix of the
    47-futures universe on ``config.MC_REFERENCE_DATE`` (paper Sec. 3)."""
    returns, meta = data_loader.load_universe(start_date, config.MC_REFERENCE_DATE)
    volatility = ema_volatility(returns, config.VOL_EMA_ETA)
    rescaled = rescale_by_volatility(returns, volatility)
    weekly = daily_to_weekly_returns(rescaled)
    corr_sequence = ema_correlation_sequence(weekly, config.CORR_EMA_ETA_PRIMARY)
    empirical_at_reference = corr_sequence[-1]

    groups = groups_from_sizes([len(config.GROUPS[g]) for g in ("stocks", "bonds", "currencies")])
    c0 = maxvar_clean(empirical_at_reference, groups)
    meta["weekly_observations"] = int(len(weekly))
    meta["reference_date_used"] = str(weekly.index[-1])
    return c0, groups, meta


def run_monte_carlo(
    c0: np.ndarray, groups: list[np.ndarray], num_samples: int, sample_length: int, seed: int
) -> dict[str, np.ndarray]:
    """Per-sample Frobenius distances of each estimator to ``c0`` (paper Sec. 3)."""
    rng = np.random.default_rng(seed)
    n = c0.shape[0]
    distances: dict[str, list[float]] = {"empirical": [], "rie": [], "maxvar": []}
    for _ in range(num_samples):
        sample = rng.multivariate_normal(mean=np.zeros(n), cov=c0, size=sample_length)
        empirical_cov = np.cov(sample, rowvar=False)
        scale = np.sqrt(np.diag(empirical_cov))
        empirical_corr = empirical_cov / np.outer(scale, scale)
        np.fill_diagonal(empirical_corr, 1.0)

        maxvar_estimate = maxvar_clean(empirical_corr, groups)
        rie_estimate = rie_clean(empirical_corr, n_obs=sample_length)

        distances["empirical"].append(frobenius_distance(empirical_corr, c0))
        distances["maxvar"].append(frobenius_distance(maxvar_estimate, c0))
        distances["rie"].append(frobenius_distance(rie_estimate, c0))
    return {key: np.array(value) for key, value in distances.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="use the paper's full data window starting 8 May 1985 instead of the reduced window",
    )
    args = parser.parse_args()

    start_date = config.DATA_START_DATE_PAPER if args.full else config.DATA_START_DATE_RUN
    c0, groups, meta = build_reference_matrix(start_date)
    assert c0.shape == (len(config.ALL_INSTRUMENTS), len(config.ALL_INSTRUMENTS))
    assert np.allclose(np.diag(c0), 1.0), "C0 must be a unit-diagonal correlation matrix"

    distances = run_monte_carlo(
        c0, groups, config.MC_NUM_SAMPLES, config.MC_SAMPLE_LENGTH_T, config.MC_RANDOM_SEED
    )

    summary = {
        method: {"mean": float(values.mean()), "std": float(values.std(ddof=1))}
        for method, values in distances.items()
    }
    empirical_mean = summary["empirical"]["mean"]
    summary["maxvar_reduction_vs_empirical"] = 1.0 - summary["maxvar"]["mean"] / empirical_mean
    summary["rie_reduction_vs_empirical"] = 1.0 - summary["rie"]["mean"] / empirical_mean

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(RESULTS_DIR / "exp1_reference_matrix_c0.csv", c0, delimiter=",")
    for method, values in distances.items():
        np.savetxt(RESULTS_DIR / f"exp1_distances_{method}.csv", values, delimiter=",")

    payload = {
        "config": {
            "num_samples": config.MC_NUM_SAMPLES,
            "sample_length_T": config.MC_SAMPLE_LENGTH_T,
            "random_seed": config.MC_RANDOM_SEED,
            "k_groups": config.K_GROUPS,
            "universe_size": len(config.ALL_INSTRUMENTS),
            "data_start_date_used": start_date,
            "reference_date_requested": config.MC_REFERENCE_DATE,
        },
        "data_meta": {
            "row_count": meta.get("row_count"),
            "date_range": meta.get("date_range"),
            "weekly_observations": meta.get("weekly_observations"),
            "reference_date_used": meta.get("reference_date_used"),
            "failed_tickers": meta.get("failed"),
            "non_massive_source": meta.get("non_massive_source"),
        },
        "frobenius_distance": summary,
        "expected_from_paper": {"maxvar": 2.09, "rie": 2.99, "empirical": 3.24},
    }
    with open(RESULTS_DIR / "exp1_results.json", "w") as fh:
        json.dump(payload, fh, indent=2, default=str)

    fig, ax = plt.subplots(figsize=(6, 4))
    methods = ["empirical", "rie", "maxvar"]
    means = [summary[m]["mean"] for m in methods]
    stds = [summary[m]["std"] for m in methods]
    expected = [payload["expected_from_paper"][m] for m in methods]
    x = np.arange(len(methods))
    width = 0.35
    ax.bar(x - width / 2, means, width, yerr=stds, label="Reproduced (mean over 100 samples)", color="steelblue")
    ax.bar(x + width / 2, expected, width, label="Paper (Sec. 3)", color="lightgray")
    ax.set_xticks(x)
    ax.set_xticklabels(["Empirical", "RIE", "MaxVar"])
    ax.set_ylabel("Frobenius distance to C0")
    ax.set_title("Monte Carlo recovery of the MaxVar reference matrix (paper Sec. 3)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp1_frobenius_distance.png", dpi=150)
    plt.close(fig)

    print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
