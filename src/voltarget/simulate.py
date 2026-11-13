"""Daily account simulator: holdings + cash, with trading costs, financing and margin.

Timeline for day t (all at the close):
  1. holdings earn day t's return (close t-1 -> close t)
  2. cash earns / pays one day of interest at the rate known at close t-1
  3. maintenance check: if gross/equity > maint_cap, forced delever to maint_cap (costs charged)
  4. scheduled trade toward the target computed at close t-1  (=> one full day of execution lag:
     a signal from close t-1 first earns returns on day t+1)
     - first trading day of the month: reset to base weights at target leverage
     - other days: rescale only if |actual leverage - target| > band
"""

import numpy as np
import pandas as pd

from .config import PERIODS, Frictions


def daily_rates(tbill_pct: pd.Series, f: Frictions) -> tuple[np.ndarray, np.ndarray]:
    y = tbill_pct.to_numpy() / 100
    borrow = (y + f.borrow_spread) / PERIODS
    cash = np.maximum(y - f.cash_haircut, 0) / PERIODS
    return borrow, cash


def month_starts(index: pd.DatetimeIndex) -> np.ndarray:
    m = index.to_period("M")
    out = np.r_[True, m[1:] != m[:-1]]
    return out


def simulate(rets: pd.DataFrame, target: pd.Series, tbill_pct: pd.Series, weights,
             f: Frictions, band: float = 0.0, charge_interest: bool = True) -> pd.DataFrame:
    idx = rets.index
    R = rets.to_numpy()
    L = target.reindex(idx).to_numpy()
    w = np.asarray(weights, dtype=float)
    rb, rc = daily_rates(tbill_pct.reindex(idx), f)
    reb = month_starts(idx)
    cost = f.trading_bps / 1e4
    init_cap, maint_cap = f.init_cap, f.maint_cap

    n = len(idx)
    cols = {k: np.zeros(n) for k in ("equity", "gross", "cash", "traded", "trade_cost", "interest", "forced")}
    h = np.zeros(len(w))
    cash = 1.0
    for t in range(n):
        h = h * (1 + R[t])
        if charge_interest and t > 0:
            interest = cash * (rb[t - 1] if cash < 0 else rc[t - 1])
            cash += interest
            cols["interest"][t] = interest
        eq = h.sum() + cash
        if eq <= 0:                                    # account wiped out
            cols["equity"][t:] = 0.0
            break
        traded = 0.0
        gross = np.abs(h).sum()
        if gross > maint_cap * eq:
            new = h * (maint_cap * eq / gross)
            traded += np.abs(new - h).sum()
            cash -= (new - h).sum() + cost * np.abs(new - h).sum()
            h = new
            cols["forced"][t] = 1
        if t > 0 and np.isfinite(L[t - 1]):
            lev = min(L[t - 1], init_cap)
            eq = h.sum() + cash
            gross = np.abs(h).sum()
            if reb[t] or gross == 0:
                new = lev * eq * w
            elif abs(gross / eq - lev) > band:
                new = h * (lev * eq / gross)
            else:
                new = h
            d = new - h
            tn = np.abs(d).sum()
            if tn > 0:
                traded += tn
                cash -= d.sum() + cost * tn
                h = new
        cols["traded"][t] = traded
        cols["trade_cost"][t] = cost * traded
        cols["equity"][t] = h.sum() + cash
        cols["gross"][t] = np.abs(h).sum()
        cols["cash"][t] = cash
    out = pd.DataFrame(cols, index=idx)
    out["lev"] = out.gross / out.equity.where(out.equity > 0)
    return out
