"""Parameters for both experiments, taken from the paper where stated.

Section references are to "Cleaning correlation matrices with external
information: the Maximum Variance estimator" (J. Bun, S. Ciliberti,
E. Sérié, Capital Fund Management working paper).
"""

# ---------------------------------------------------------------------------
# Universe (paper Table 2, page 6): 47 futures contracts in three asset
# classes. The QuantX Dataloader and Massive APIs serve equities/ETFs listed
# on U.S. and other exchanges, Massive-direct forex pairs and U.S. Treasury
# constant-maturity yields, but not the specific European and Asian
# exchange-traded index and government-bond futures contracts of Table 2, nor
# futures history before 2017. Each instrument below is therefore proxied by
# the closest tradable series reachable through the Dataloader/Massive path,
# preferring the longest available history; see data/data_loader.py and
# results/RESULTS.md, "Deviations from the paper", for the realised
# per-series start dates and the exact substitution made.

# Stock index group (24 contracts, paper Table 2) -> equity/closed-end-fund
# ticker loaded through the Dataloader (source="massive", Yahoo backstop).
STOCK_TICKERS = {
    "AEX": "EWN",  # iShares MSCI Netherlands ETF, proxy for the AEX index
    "SP_Midcap400": "MDY",
    "Russell2000": "IWM",
    "CAC40": "EWQ",  # iShares MSCI France ETF, proxy for the CAC 40
    "DAX": "EWG",  # iShares MSCI Germany ETF, proxy for the DAX
    "FTSE100": "EWU",  # iShares MSCI United Kingdom ETF, proxy for the FTSE 100
    "STOXX_Europe_600": "VGK",  # Vanguard FTSE Europe ETF, broad-Europe proxy
    "HangSeng": "EWH",  # iShares MSCI Hong Kong ETF, proxy for the Hang Seng
    "FTSE_MIB": "EWI",  # iShares MSCI Italy ETF, proxy for the FTSE MIB
    "Nikkei225": "EWJ",  # iShares MSCI Japan ETF, proxy for the Nikkei 225
    "Topix": "DXJ",  # WisdomTree Japan Hedged Equity Fund, distinct Japan proxy for the Topix
    "Kospi200": "KF",  # The Korea Fund (closed-end, since 1984), proxy for the Kospi 200
    "MSCI_EAFE": "EFA",  # iShares MSCI EAFE ETF
    "IBEX35": "EWP",  # iShares MSCI Spain ETF, proxy for the IBEX 35
    "MSCI_EM": "EEM",  # iShares MSCI Emerging Markets ETF
    "Nifty": "IFN",  # The India Fund (closed-end, since 1994), proxy for the Nifty
    "Nasdaq100": "QQQ",
    "MSCI_Singapore": "EWS",  # iShares MSCI Singapore ETF
    "SP500": "SPY",
    "MSCI_Taiwan": "TWN",  # The Taiwan Fund (closed-end, since 1986), proxy for MSCI Taiwan
    "EuroStoxx50": "FEZ",  # SPDR EURO STOXX 50 ETF
    "SP_Canada60": "EWC",  # iShares MSCI Canada ETF, proxy for the S&P/TSX 60
    "SPI200": "EWA",  # iShares MSCI Australia ETF, proxy for the SPI 200
    "DowJones": "DIA",
}

# Government bond group (14 contracts, paper Table 2). Five U.S.-Treasury
# maturities are rebuilt from Massive's direct Treasury-yield series (yield
# changes converted to approximate bond-future returns by modified duration);
# the nine non-U.S. legs are proxied by U.S.-exchange-listed international or
# emerging-market sovereign/government bond ETFs, loaded through the
# Dataloader the same way as the stock-index proxies.
US_TREASURY_MATURITIES = {
    # bond future -> (Massive treasury-yields field, duration in years for the
    # return approximation). Massive's constant-maturity series has no 20-year
    # point, so the Ultra T-bond and the 30-year T-bond legs both read the
    # 30-year yield, scaled by their own duration.
    "US_TNote_2Y": ("yield_2_year", 2.0),
    "US_TNote_5Y": ("yield_5_year", 5.0),
    "US_TNote_10Y": ("yield_10_year", 10.0),
    "US_Ultra_TBond": ("yield_30_year", 20.0),
    "US_TBond_30Y": ("yield_30_year", 25.0),  # long-bond effective duration below stated maturity
}
NON_US_BOND_TICKERS = {
    # All nine legs are proxied by U.S.-exchange-listed international or
    # emerging-market sovereign bond ETFs (Massive/Yahoo do not serve the
    # Xetra-, LSE-, TSX- or TSE-listed bond-future contracts of Table 2).
    "Schatz_proxy": "ISHG",  # iShares 1-3Y International Treasury Bond ETF, short-duration proxy
    "Bobl_proxy": "IGOV",  # iShares International Treasury Bond ETF, medium-duration proxy
    "Bund_proxy": "BWX",  # SPDR Bloomberg International Treasury Bond ETF, long-duration proxy
    "Buxl_proxy": "EMLC",  # VanEck J.P. Morgan EM Local Currency Bond ETF, long-duration/high-vol proxy
    "BTP_proxy": "EMB",  # iShares J.P. Morgan USD Emerging Markets Bond ETF, sovereign-spread proxy
    "OAT_proxy": "WIP",  # SPDR FTSE International Government Inflation-Protected Bond ETF
    "Gilt_proxy": "PICB",  # Invesco International Corporate Bond ETF, international duration proxy
    "Canada_10Y_proxy": "BNDX",  # Vanguard Total International Bond ETF, broad international duration proxy
    "JGB_proxy": "BWZ",  # SPDR Blmbg Short Term International Treasury Bond ETF, low-yield/duration proxy
}
BOND_TICKERS = {**{k: v[0] for k, v in US_TREASURY_MATURITIES.items()}, **NON_US_BOND_TICKERS}

# Currency group (9 contracts, paper Table 2), quoted against the U.S. dollar
# and loaded through the Dataloader as Massive-direct forex pairs, except the
# U.S. dollar index which has no forex-pair form and is proxied by an ETF.
CURRENCY_TICKERS = {
    "AUD": "C:AUDUSD",
    "GBP": "C:GBPUSD",
    "CAD": "C:USDCAD",
    "EUR": "C:EURUSD",
    "USD_index": "UUP",  # Invesco DB US Dollar Index Bullish Fund, proxy for the dollar index
    "JPY": "C:USDJPY",
    "MXN": "C:USDMXN",
    "NZD": "C:NZDUSD",
    "CHF": "C:USDCHF",
}

GROUPS = {
    "stocks": list(STOCK_TICKERS.keys()),
    "bonds": list(BOND_TICKERS.keys()),
    "currencies": list(CURRENCY_TICKERS.keys()),
}
ALL_INSTRUMENTS = GROUPS["stocks"] + GROUPS["bonds"] + GROUPS["currencies"]
K_GROUPS = 3  # number of groups used by the MaxVar estimator (paper Sec. 3)

# ---------------------------------------------------------------------------
# EMA parameters (paper Sec. 2 / page 7)
VOL_EMA_ETA = 1.0 / 100.0  # eta for the daily volatility EMA
CORR_EMA_ETA_PRIMARY = 1.0 / 750.0  # eta' for the primary (long-window) correlation EMA
CORR_EMA_ETA_ROBUSTNESS = 1.0 / 100.0  # eta' for the robustness (short-window) correlation EMA

# ---------------------------------------------------------------------------
# Experiment 1: Monte Carlo simulation (paper Sec. 3, page 6-7)
MC_REFERENCE_DATE = "2018-12-28"  # date on which the reference matrix C0 is estimated
MC_NUM_SAMPLES = 100  # number of independent Monte Carlo samples
MC_SAMPLE_LENGTH_T = 150  # length T of each simulated i.i.d. Gaussian sample
MC_RANDOM_SEED = 0  # not specified in the paper; fixed here for reproducibility

# ---------------------------------------------------------------------------
# Experiment 2: risk-parity backtest (paper Sec. 4, page 7-9)
DATA_START_DATE_PAPER = "1985-05-08"  # paper's data start date (warm-up period)
DATA_END_DATE = "2018-12-31"  # paper's data end date
BACKTEST_START_DATE = "1993-01-01"  # start of the out-of-sample performance window
BACKTEST_END_DATE = "2018-12-31"  # end of the out-of-sample performance window
REBALANCE_FREQUENCY = "W"  # weekly rebalancing; not stated explicitly in the paper,
# inferred from the weekly correlation-EMA update described in Sec. 2
TRADING_DAYS_PER_YEAR = 252  # annualisation constant for the Sharpe ratio

# Data-availability reduction: history for several equity/ETF and bond-fund
# proxies does not reach back to 1985; the realised sample start of every
# instrument is logged by data/data_loader.py and reported in
# results/RESULTS.md rather than assumed.
DATA_START_DATE_RUN = "1990-01-01"  # practical request start date; the common
# sample actually used is the latest per-instrument start date found at load
# time (see results/RESULTS.md for the realised per-instrument start dates).
