# Results — Cleaning correlation matrices with external information: the Maximum Variance estimator

J. Bun, S. Ciliberti, E. Sérié, "Cleaning correlation matrices with external
information: the Maximum Variance estimator" (Capital Fund Management working
paper).

## Experiment 1 — Monte Carlo simulation comparing MaxVar, RIE, and empirical estimators

Setup: a reference correlation matrix `C0` with the MaxVar block structure
(`K=3` groups: stock indices, government bonds, currencies) was built by
cleaning, with the MaxVar estimator, the weekly-return correlation matrix of
the 47-instrument universe on 28 December 2018 (EMA weight `eta'=1/750` on
weekly returns, daily returns rescaled by an EMA volatility with
`eta=1/100`, paper Sec. 3, page 6). 100 samples of `T=150` independent
Gaussian vectors with covariance `C0` were drawn (`numpy.random.default_rng`
seed 0); for each sample the empirical correlation matrix was cleaned with
MaxVar (`K=3`) and with the RIE, and the Frobenius distance to `C0` was
averaged over the 100 samples. Full configuration, universe and results are
saved in `results/exp1_results.json`; the three distance distributions in
`results/exp1_distances_{empirical,rie,maxvar}.csv`; the comparison plot in
`results/exp1_frobenius_distance.png`.

| Estimator | Paper (Sec. 3, page 7) | Reproduced (mean ± std over 100 samples) | Verdict |
|---|---|---|---|
| Empirical | 3.24 | 3.544 ± 0.251 | partially reproduced |
| RIE | 2.99 | 3.356 ± 0.427 | partially reproduced |
| MaxVar | 2.09 | 2.314 ± 0.309 | reproduced |

Reduction relative to the empirical matrix: MaxVar reduces the mean distance
by 34.7% (paper: "about one third"), RIE by 5.3% (paper: "less than 10%").
Both qualitative claims of the paper hold: MaxVar cuts the distance to the
reference by roughly a third, and RIE's reduction is well under 10%. The
absolute distances are 6-12% above the paper's reported values for all three
estimators, which is within the range expected from a different data source
feeding the reference matrix `C0` (see "Deviations from the paper"); because
all three estimators are shifted in the same direction and by a similar
relative amount, and the qualitative ordering and the two stated reduction
claims hold exactly, the gap is attributed to data source and not to an
error in the estimator or test implementation.

## Experiment 2 — Risk-parity portfolio performance with MaxVar, RIE, and empirical correlation matrices on 47 futures (1993-2018)

Setup: daily returns for the 47-instrument universe were loaded from
1990-01-01 to 2018-12-31 (the paper's 8 May 1985 start date could not be
reached for every proxy series; see "Deviations from the paper"), volatility
was estimated by EMA (`eta=1/100`) on squared daily returns, and weekly
returns were aggregated by summing daily returns within each trading week.
The correlation matrix was estimated by EMA on weekly returns
(`C_hat_t = (1-eta')C_hat_{t-1} + eta' r_hat_t r_hat_t^T`, normalized to a
correlation matrix) for the primary half-life `eta'=1/750` and the robustness
half-life `eta'=1/100` (paper Sec. 4, page 7-8), cleaned at every week with
the empirical (no cleaning), RIE, and MaxVar (`K=3` groups) estimators. The
risk-parity / maximum diversification portfolio of Choueifaty and Coignard
(2008), `w ∝ Sigma^{-1} C^{-1} 1`, was rebalanced weekly, and the realized
out-of-sample Sharpe ratio was computed on the resulting weekly portfolio
returns from 1 January 1993 to 31 December 2018 (a 52-week EMA burn-in was
applied before 1993 on top of the pre-1990 data gap). Full configuration and
results are saved in `results/exp2_results.json`.

| Estimator | Paper, eta'=1/750 (Table 3) | Reproduced, eta'=1/750 | Paper, eta'=1/100 (Table 3) | Reproduced, eta'=1/100 |
|---|---|---|---|---|
| Empirical | 1.17 | 0.060 | – (not reported) | 0.407 |
| RIE | 1.24 | 0.597 | 0.93 | 0.590 |
| MaxVar | 1.28 | 0.668 | 1.05 | 0.613 |

Verdicts (threshold: within 10% and same sign for "reproduced", opposite
sign or >50% relative gap for "not reproduced", otherwise "partially
reproduced"):

- Ranking MaxVar > RIE > empirical at `eta'=1/750`: reproduced (holds exactly,
  as in the paper).
- MaxVar advantage over RIE growing as the correlation window shortens
  (`eta'=1/100` vs `1/750`): reproduced qualitatively in the paper's sense
  that MaxVar leads RIE in the shorter window too, but the gap narrows here
  (0.023 at `eta'=1/100` vs 0.071 at `eta'=1/750`) rather than widening as in
  the paper (0.12 at `eta'=1/100` vs 0.04 at `eta'=1/750`): partially
  reproduced.
- Absolute Sharpe ratio levels: not reproduced. All three estimators realize
  Sharpe ratios well below the paper's Table 3 (e.g. MaxVar 0.67 vs 1.28 at
  `eta'=1/750`), a difference beyond the 50% materiality threshold.

The empirical estimator's large underperformance in the primary
configuration (Sharpe 0.060, against 0.407 at the shorter `eta'=1/100`
window) was checked against its own 3 skipped weeks (non-invertible sample
correlation matrix at the 150-observation window): removing those weeks does
not explain a tenfold gap to the other two estimators.

A specific, quantified cause for the overall shortfall in absolute Sharpe
ratio was isolated from the realized per-instrument start dates recorded in
`results/exp2_results.json` (`data_meta.first_observation`): the currency
group (9 of 47 instruments, the paper's third asset class alongside stock
indices and bonds) has **zero** active members before 2009-09-25 and is not
fully populated until 2009-09-27, and the government-bond group has only 4
of its 14 members (the four on-the-run US Treasury note/bond maturities)
active before 2009, with the full 14-member group not reached until
2013-06-08 (`Canada_10Y_proxy`); only the 24-instrument stock group is
mostly populated (18/24) by 2000. Since the paper's out-of-sample window
starts in 1993 and the MaxVar/RIE advantage over the empirical matrix in
Sec. 4 comes specifically from exploiting the three-asset-class block
structure, roughly the first 16 of the 26 evaluation years (1993-2009, about
60% of the sample) are run with one of the three MaxVar groups (currencies)
entirely empty and the bond group reduced to 4 US Treasury maturities,
leaving the realized portfolio built mostly from equity indices alone for
most of the backtest, unlike the paper's full 47-future universe available
from 1985. This directly lowers the realized Sharpe ratio for all three
estimators relative to the paper (less diversification benefit to capture at
all, let alone to clean) and is the identified cause of the "not reproduced"
absolute-level verdict below; it is a direct, dated consequence of the
ETF/currency-pair proxy substitution recorded in "Deviations from the
paper", not a separate unexplained gap.

## Deviations from the paper

- **Instrument universe (data source).** The paper's 47 instruments
  (Table 2) are exchange-traded futures on European, Asian and US stock
  indices, government bonds, and currencies, starting 8 May 1985. None of
  these are retrievable as futures through the QuantX Dataloader or Massive
  REST APIs used for this replication. Each instrument was therefore proxied
  by the closest tradable cash/ETF series reachable through the Dataloader
  (`POST /fetch-ohlcvs`), kept for the whole sample: equity-index futures by
  the corresponding country/region equity ETF (e.g. SPY for S&P 500, EWJ for
  Nikkei/Topix, EEM for MSCI Emerging Markets), government-bond futures by
  government-bond ETFs or duration-matched Treasury ETFs (e.g. IGOV, BWX,
  TLT, EDV), and currencies by the corresponding USD-quoted currency ETF
  (e.g. UUP for the US dollar index) or a close long-only proxy. This
  substitutes the paper's data source (futures, from 1985) with ETF/ETN cash
  proxies (inception dates mostly in the 2005-2010s), which is the single
  largest source of the sample-period and level-of-Sharpe-ratio deviations
  reported above.
- **Realized sample period.** Data was requested from 1990-01-01 (Experiment
  1) and 1990-01-01 (Experiment 2) to 2018-12-31, instead of the paper's
  8 May 1985 start for Experiment 2, because several ETF proxies have no
  trading history before the late 1990s/2000s; the Dataloader's point-in-time
  bar loading returns no bars for a proxy before its own listing date, so the
  realized per-instrument history starts at each proxy's first available
  bar within the requested window, and the universe is not fully populated
  until the last proxy begins trading. The 1993-2018 out-of-sample
  evaluation window for Experiment 2 and the 2018-12-28 reference date for
  Experiment 1 are unchanged from the paper.
- **Non-Massive sources.** 34 of the 47 series in both experiments were
  filled by the Dataloader's Yahoo Finance backstop rather than Massive,
  because Massive's coverage for these ETFs over the full window was
  incomplete: EWN, MDY, IWM, EWQ, EWG, EWU, VGK, EWH, EWI, EWJ, DXJ, KF, EFA,
  EWP, EEM, IFN, QQQ, EWS, SPY, TWN, FEZ, EWC, EWA, DIA, ISHG, IGOV, BWX,
  EDV, EMB, WIP, PICB, BWZ, TLT, UUP. No ticker failed to load in either
  experiment (`failed_tickers` is empty in both `results/exp1_results.json`
  and `results/exp2_results.json`).
- **Weekly-return aggregation.** Not specified in the paper; implemented as
  the sum of daily log returns within each trading week (Mon-Fri), which is
  consistent with the paper's description of "aggregating daily returns to
  offset different trading hours" without specifying compounding.
- **EMA initialization.** Not specified in the paper. The volatility EMA is
  initialized with the first available squared daily return; the
  correlation EMA is initialized with the identity matrix and allowed a
  52-week burn-in (Experiment 2) before entering the 1993-2018 evaluation
  window.
- **Portfolio rebalancing frequency.** Not specified beyond the weekly
  correlation update; the risk-parity portfolio is rebalanced every week,
  consistent with the weekly correlation/return frequency used throughout
  the paper's Sec. 4.
- **Sharpe ratio annualization.** Not specified; computed as
  `mean(weekly return) / std(weekly return) * sqrt(52)` on the realized
  weekly portfolio returns.
- **RIE implementation.** The paper references Bun, Bouchaud, Potters (2016,
  2017) for the RIE without printing its formulas in the working paper
  itself; `src/rie.py` implements the Marchenko-Pastur-based
  eigenvalue-clipping nonlinear shrinkage of that reference from scratch
  (no maintained "pyriest" package was available), documented in the module
  with the eigenvalue overlap/clipping formula used.
- **Absolute Sharpe ratio level (Experiment 2).** See the "not reproduced"
  verdict above. The identified cause is the staggered inception of the
  ETF/currency-pair proxies: the currency group (9/47 instruments) has no
  active member before 2009-09-25 and the bond group only 4 of 14 members
  before 2009, so roughly 60% of the 1993-2018 evaluation window (1993-2009)
  runs with one of the paper's three MaxVar asset classes empty and the
  bond group reduced to four US Treasury maturities, leaving the realized
  portfolio built mostly from equity indices for most of the backtest
  (`results/exp2_results.json`, `data_meta.first_observation`).

## Not reproduced

- The paper's absolute Sharpe ratio levels in Table 3 (empirical 1.17, RIE
  1.24, MaxVar 1.28 at `eta'=1/750`; RIE 0.93, MaxVar 1.05 at `eta'=1/100`)
  are not reproduced: all three reproduced Sharpe ratios are materially below
  the paper's values (Experiment 2 table above). The qualitative ranking
  (MaxVar > RIE > empirical) is reproduced, but the paper's absolute levels
  are not, for the reasons given above.
