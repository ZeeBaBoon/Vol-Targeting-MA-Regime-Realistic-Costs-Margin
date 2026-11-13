import numpy as np
import pandas as pd
import pytest

from voltarget.config import NO_FRICTIONS, Frictions, Strategy
from voltarget.signals import gate_hyst, target_leverage
from voltarget.simulate import simulate

IDX = pd.bdate_range("2020-01-01", periods=300)


def rets(seed=0, n=300):
    r = np.random.default_rng(seed).normal(0.0003, 0.01, (n, 2))
    return pd.DataFrame(r, index=IDX[:n], columns=["A", "B"])


def flat_rate(v=0.0, n=300):
    return pd.Series(v, index=IDX[:n])


def test_signal_uses_only_past_closes():
    R = rets()
    px = (1 + R).cumprod()
    s = Strategy(ma=50, vol_lookback=20)
    a = target_leverage(px, R, s)
    R2 = R.copy(); R2.iloc[200] = 0.2                       # shock on day 200
    b = target_leverage((1 + R2).cumprod(), R2, s)
    pd.testing.assert_series_equal(a.iloc[:200], b.iloc[:200])


def test_one_full_day_execution_lag():
    """A shock on day k changes the target at close k; exposure may change only from day k+2."""
    R = rets()
    tgt = pd.Series(1.0, index=R.index)
    tgt2 = tgt.copy(); tgt2.iloc[100:] = 2.0                # signal jumps at close of day 100
    a = simulate(R, tgt, flat_rate(), (0.5, 0.5), NO_FRICTIONS, charge_interest=False)
    b = simulate(R, tgt2, flat_rate(), (0.5, 0.5), NO_FRICTIONS, charge_interest=False)
    # gross at close t = exposure that earns day t+1. Trade happens at close 101 -> earns day 102.
    np.testing.assert_allclose(a.gross.iloc[:101], b.gross.iloc[:101])
    assert b.lev.iloc[101] == pytest.approx(2.0)
    np.testing.assert_allclose(a.equity.iloc[:102], b.equity.iloc[:102])


def test_costs_and_interest_accounting():
    R = pd.DataFrame(0.0, index=IDX[:10], columns=["A", "B"])        # prices never move
    f = Frictions(trading_bps=10.0, borrow_spread=0.0, cash_haircut=0.0, margin=False)
    sim = simulate(R, pd.Series(2.0, index=R.index), flat_rate(5.0, 10), (0.5, 0.5), f)
    r = 0.05 / 252
    # day 1: idle cash earns one day of interest, then buy 2x equity -> pay 10 bp on that notional
    e1 = 1 + r
    assert sim.trade_cost.iloc[1] == pytest.approx(0.001 * 2 * e1)
    assert sim.cash.iloc[1] == pytest.approx(e1 - 2 * e1 - 0.001 * 2 * e1)
    # days 2..9: at 2x, borrowed = equity -> lose r x equity per day (re-leveraging costs ~1e-7)
    expected = (e1 - 0.001 * 2 * e1) * (1 - r) ** 8
    assert sim.equity.iloc[-1] == pytest.approx(expected, rel=1e-5)


def test_band_suppresses_small_releveraging():
    R = rets()
    tgt = pd.Series(1.5, index=R.index)
    tight = simulate(R, tgt, flat_rate(), (0.5, 0.5), NO_FRICTIONS, band=0.0, charge_interest=False)
    wide = simulate(R, tgt, flat_rate(), (0.5, 0.5), NO_FRICTIONS, band=0.3, charge_interest=False)
    assert (wide.traded > 0).sum() < (tight.traded > 0).sum() / 5


def test_maintenance_margin_forces_delever():
    R = pd.DataFrame(0.0, index=IDX[:5], columns=["A", "B"])
    R.iloc[3] = -0.25                                        # 25% crash on day 3 at 2x leverage
    f = Frictions(trading_bps=0.0, borrow_spread=0.0, cash_haircut=0.0)
    sim = simulate(R, pd.Series(2.0, index=R.index), flat_rate(0.0, 5), (0.5, 0.5), f)
    assert sim.forced.iloc[3] == 1
    assert sim.lev.iloc[3] <= f.maint_cap + 1e-9 or sim.lev.iloc[3] == pytest.approx(f.init_cap)


def test_hysteresis_needs_buffer_to_switch():
    px = pd.Series([100.0] * 10 + [100.5] * 5 + [102.0] * 5)
    g = gate_hyst(px, 5, 0.01)
    assert g.iloc[12] == 0            # 0.5% above MA: not enough to switch on
    assert g.iloc[-1] == 1            # clearly above: on
