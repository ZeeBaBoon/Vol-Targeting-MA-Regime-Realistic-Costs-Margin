"""Download once from Yahoo Finance, then reuse the local cache (data/ is not committed)."""

import pandas as pd

from . import config

CACHE = config.DATA / "prices.csv"


def download() -> pd.DataFrame:
    import yfinance as yf

    tick = list(config.TICKERS) + [config.RATE_TICKER]
    df = yf.download(tick, start=config.DATA_START, end=config.DATA_END, auto_adjust=True,
                     progress=False, group_by="column")["Close"]
    df = df[tick].rename(columns={config.RATE_TICKER: "tbill_pct"})
    return df


def load(refresh: bool = False) -> pd.DataFrame:
    """Columns: SPY, TLT (total-return closes), tbill_pct. Trading days where both ETFs trade."""
    if refresh or not CACHE.exists():
        config.DATA.mkdir(exist_ok=True)
        download().to_csv(CACHE)
    df = pd.read_csv(CACHE, index_col=0, parse_dates=True)
    df = df.dropna(subset=list(config.TICKERS))
    df["tbill_pct"] = df.tbill_pct.ffill()
    return df
