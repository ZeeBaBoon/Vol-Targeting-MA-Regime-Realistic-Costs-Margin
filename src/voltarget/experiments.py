"""Runs every experiment; writes reports/tables/*.csv. Pre-registered selection rule:

  Among the 96 regime-cap configs (MA x gate x band), pick the one with the highest
  TRAIN (2005-2015) Sharpe that had no forced-liquidation day in train. Report it on TEST
  (2016 - Oct 2026) next to: Base 50/50, static leverage (fixed on train), vol-target only.
"""

import json
from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from . import config
from .config import BANDS, GATES, MAS, NO_FRICTIONS, Frictions, Strategy
from .data import load
from .metrics import summarize
from .signals import base_returns, target_leverage
from .simulate import simulate

TABLES = config.REPORTS / "tables"


class Setup:
    def __init__(self):
        px = load()
        rets = px[list(config.TICKERS)].pct_change().iloc[1:]
        self.px, self.full_rets = px, rets
        self.start = rets.index[config.WARMUP]
        self.rets = rets.loc[self.start:]
        self.tbill = px.tbill_pct.reindex(self.rets.index)
        self.periods = {"train": (self.start, config.TRAIN_END), "test": (config.TEST_START, None),
                        "full": (self.start, None)}

    def target(self, s: Strategy) -> pd.Series:
        return target_leverage(self.px[list(config.TICKERS)], self.full_rets, s).loc[self.start:]

    def run(self, target, f=Frictions(), band=0.0, charge_interest=True, weights=(0.5, 0.5)):
        return simulate(self.rets, target, self.tbill, weights, f, band, charge_interest)

    def stats(self, sim) -> dict:
        return {p: summarize(sim, self.tbill, a, b) for p, (a, b) in self.periods.items()}


def flat(name: str, st: dict) -> list[dict]:
    return [{"strategy": name, "period": p, **m} for p, m in st.items()]


def run() -> dict:
    S = Setup()
    TABLES.mkdir(parents=True, exist_ok=True)
    const = lambda v: pd.Series(v, index=S.rets.index)

    # --- sweep ---
    rows, sims = [], {}
    for ma, gate, band in product(MAS, GATES, BANDS):
        s = Strategy(ma=ma, gate=gate, band=band)
        sim = S.run(S.target(s), band=band)
        for p, m in S.stats(sim).items():
            rows.append({"ma": ma, "gate": gate, "band": band, "period": p, **m})
        sims[(ma, gate, band)] = sim
    sweep = pd.DataFrame(rows)
    sweep.to_csv(TABLES / "sweep.csv", index=False)

    tr = sweep[(sweep.period == "train") & (sweep.ForcedDays == 0)]
    best = tr.loc[tr.Sharpe.idxmax()]
    key = (int(best.ma), best.gate, float(best.band))
    sel = Strategy(ma=key[0], gate=key[1], band=key[2])

    # --- benchmarks ---
    train_vol = base_returns(S.rets.loc[:config.TRAIN_END], (0.5, 0.5)).std(ddof=0) * np.sqrt(config.PERIODS)
    static_lev = min(Strategy().target_vol / train_vol, Frictions().init_cap)
    bench = {
        "Base 50/50": S.run(const(1.0)),
        f"Static leverage {static_lev:.2f}x": S.run(const(static_lev)),
        "Vol-target only": S.run(S.target(Strategy())),
        f"Selected: MA{key[0]} {key[1]} band {key[2]:.2f}": sims[key],
    }
    comp = pd.DataFrame([r for n, sim in bench.items() for r in flat(n, S.stats(sim))])
    comp.to_csv(TABLES / "comparison.csv", index=False)
    pd.DataFrame({n: sim.equity for n, sim in bench.items()}).to_csv(TABLES / "equity_curves.csv")
    pd.DataFrame({"lev": sims[key].lev, "target_lev": S.target(sel), "tbill_pct": S.tbill}).to_csv(
        TABLES / "selected_leverage.csv")

    # --- realism ladder (vol-target only) ---
    vt = S.target(Strategy())
    ladder_steps = [
        ("A  no frictions", NO_FRICTIONS, False),
        ("B  + trading costs", replace(Frictions(), margin=False), False),
        ("C  + financing (T-bill based)", replace(Frictions(), margin=False), True),
        ("D  + margin rules", Frictions(), True),
    ]
    ladder = pd.DataFrame([r for n, f, ci in ladder_steps for r in flat(n, S.stats(S.run(vt, f, 0.0, ci)))])
    ladder.to_csv(TABLES / "ladder.csv", index=False)

    # --- cost sensitivity, test period ---
    sens = []
    for spread, bps in product((0.005, 0.015, 0.03), (2.0, 10.0)):
        f = replace(Frictions(), borrow_spread=spread, trading_bps=bps)
        for n, tgt, band in (("Base 50/50", const(1.0), 0.0), ("Selected", S.target(sel), sel.band),
                             ("Static leverage", const(static_lev), 0.0)):
            m = summarize(S.run(tgt, f, band), S.tbill, config.TEST_START, None)
            sens.append({"borrow_spread": spread, "trading_bps": bps, "strategy": n, **m})
    pd.DataFrame(sens).to_csv(TABLES / "cost_sensitivity.csv", index=False)

    # --- ablation (post hoc, not used for selection): band alone, no MA cap ---
    abl = []
    for band in BANDS:
        sim = S.run(vt, band=band)
        abl += [{"band": band, **r} for r in flat("Vol-target only", S.stats(sim))]
    pd.DataFrame(abl).to_csv(TABLES / "ablation_band_no_ma.csv", index=False)

    # --- is the test difference vs. Base real? daily return difference, test period ---
    def diff_t(a, b):
        ra = (a.equity / a.equity.shift(1) - 1).loc[config.TEST_START:]
        rb = (b.equity / b.equity.shift(1) - 1).loc[config.TEST_START:]
        d = (ra - rb).dropna()
        return d.mean() / (d.std(ddof=1) / np.sqrt(len(d))), d.mean() * config.PERIODS

    base_sim = bench["Base 50/50"]
    t_sel, ann_sel = diff_t(sims[key], base_sim)

    # --- overfitting diagnostics ---
    piv = sweep.pivot_table(index=["ma", "gate", "band"], columns="period", values="Sharpe")
    base_test = comp[(comp.strategy == "Base 50/50") & (comp.period == "test")].iloc[0]
    sel_rank = int((piv.test > piv.loc[key, "test"]).sum()) + 1
    summary = {
        "test_selected_minus_base_annual_return": round(ann_sel, 4),
        "test_selected_minus_base_t_stat": round(t_sel, 2),
        "sim_start": str(S.start.date()), "data_end": str(S.rets.index[-1].date()),
        "selected": {"ma": key[0], "gate": key[1], "band": key[2]},
        "static_leverage": round(static_lev, 3),
        "train_test_sharpe_rank_corr": round(piv.train.rank().corr(piv.test.rank()), 3),
        "selected_test_rank": f"{sel_rank} of {len(piv)}",
        "configs_beating_base_test_sharpe": int((piv.test > base_test.Sharpe).sum()),
        "n_configs": len(piv),
    }
    (TABLES / "summary.json").write_text(json.dumps(summary, indent=2))
    return {"summary": summary, "comparison": comp, "ladder": ladder}


def report_text(res: dict) -> str:
    c = res["comparison"].set_index(["strategy", "period"])
    cols = ["CAGR", "Vol", "Sharpe", "MaxDD", "AvgLev", "Turnover", "TradeCost", "NetInterest", "ForcedDays"]
    out = [json.dumps(res["summary"], indent=2), "", c[cols].round(3).to_string(), "",
           res["ladder"].set_index(["strategy", "period"])[cols].round(3).to_string()]
    return "\n".join(out)
