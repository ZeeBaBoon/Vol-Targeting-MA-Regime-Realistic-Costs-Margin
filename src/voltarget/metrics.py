import numpy as np
import pandas as pd

from .config import PERIODS


def summarize(sim: pd.DataFrame, tbill_pct: pd.Series, start=None, end=None) -> dict:
    """Metrics over [start, end]. Sharpe is on returns in excess of the T-bill rate."""
    s = sim.loc[start:end]
    eq = s.equity
    prev = sim.equity.shift(1).loc[start:end]
    r = (eq / prev - 1).dropna()
    rf = (tbill_pct.shift(1).reindex(r.index) / 100 / PERIODS).fillna(0)
    ex = r - rf
    years = len(r) / PERIODS
    first = prev.dropna().iloc[0] if prev.notna().any() else eq.iloc[0]
    return {
        "CAGR": (eq.iloc[-1] / first) ** (1 / years) - 1 if years > 0 else np.nan,
        "Vol": r.std(ddof=0) * np.sqrt(PERIODS),
        "Sharpe": ex.mean() / ex.std(ddof=0) * np.sqrt(PERIODS) if ex.std() > 0 else np.nan,
        "MaxDD": (eq / eq.cummax() - 1).min(),
        "AvgLev": s.lev.mean(),
        "Turnover": (s.traded / prev).sum() / years,              # x equity per year
        "TradeCost": (s.trade_cost / prev).sum() / years,         # fraction of equity per year
        "NetInterest": (s.interest / prev).sum() / years,
        "ForcedDays": int(s.forced.sum()),
    }
