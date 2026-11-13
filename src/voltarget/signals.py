"""Target leverage per day. Value at date t uses closes up to and including t only."""

import numpy as np
import pandas as pd

from .config import PERIODS, Strategy


def base_returns(rets: pd.DataFrame, weights) -> pd.Series:
    return (rets * np.asarray(weights)).sum(axis=1)


def vol_target(port_ret: pd.Series, s: Strategy) -> pd.Series:
    vol = port_ret.rolling(s.vol_lookback, min_periods=s.vol_lookback).std(ddof=0) * np.sqrt(PERIODS)
    return (s.target_vol / vol).clip(0.0, s.max_leverage)


def gate_simple(px: pd.Series, ma: int) -> pd.Series:
    m = px.rolling(ma, min_periods=ma).mean()
    return (px >= m).astype(float).where(m.notna())


def gate_hyst(px: pd.Series, ma: int, buffer: float) -> pd.Series:
    """Risk-on after closing > MA*(1+buffer), risk-off after closing < MA*(1-buffer)."""
    m = px.rolling(ma, min_periods=ma).mean().to_numpy()
    p = px.to_numpy()
    out = np.full(len(p), np.nan)
    state = 0.0
    for i in range(len(p)):
        if np.isnan(m[i]):
            continue
        if state == 0 and p[i] > m[i] * (1 + buffer):
            state = 1.0
        elif state == 1 and p[i] < m[i] * (1 - buffer):
            state = 0.0
        out[i] = state
    return pd.Series(out, index=px.index)


def target_leverage(px: pd.DataFrame, rets: pd.DataFrame, s: Strategy) -> pd.Series:
    lev = vol_target(base_returns(rets, s.weights), s)
    if s.ma is None:
        return lev
    driver = px.iloc[:, 0].reindex(lev.index)          # first ticker drives the regime
    g = gate_simple(driver, s.ma) if s.gate == "simple" else gate_hyst(driver, s.ma, s.hyst_buffer)
    return lev.where(g == 1, np.minimum(lev, 1.0)).where(g.notna())   # risk-off: no borrowing
