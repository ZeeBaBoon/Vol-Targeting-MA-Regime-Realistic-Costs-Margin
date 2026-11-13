"""Charts for REPORT.md -> reports/figures/*.png (reads reports/tables/, run experiments first)."""

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

T = config.REPORTS / "tables"
OUT = config.REPORTS / "figures"
SURFACE, INK, INK2, GRID, GRAY = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de", "#9a9994"
CAT = ["#2a78d6", "#eb6834", "#1baf7a"]            # fixed categorical order

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "font.size": 10,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "savefig.dpi": 200,
})


def _title(ax, title, sub):
    ax.set_title(title, pad=24)
    ax.text(0, 1.02, sub, transform=ax.transAxes, color=INK2, fontsize=9.5, va="bottom")


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", OUT / name)


def _end_label(ax, x, y, text, color):
    ax.plot(x, y, "o", ms=7, color=color, mec=SURFACE, mew=2)
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center", color=INK2, fontsize=9)


def _names(eq):
    base = [c for c in eq.columns if c.startswith("Base")][0]
    static = [c for c in eq.columns if c.startswith("Static")][0]
    sel = [c for c in eq.columns if c.startswith("Selected")][0]
    return base, static, "Vol-target only", sel


def _label_ends(ax, items, min_gap):
    """items: [(y, text, color)] -> text right of the last point, nudged apart so labels never touch."""
    items = sorted(items)
    ys = [y for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_gap)
    x = ax.get_xlim()[1]
    for (y, text, col), yl in zip(items, ys):
        ax.text(x, yl, "  " + text, color=col, fontsize=9.5, va="center", fontweight="bold")


def fig_equity(eq):
    e = eq.loc[config.TEST_START:]
    e = (e / e.iloc[0]).resample("ME").last()            # month-end: readable, no daily fuzz
    base, _, _, sel = _names(eq)
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.plot(e.index, e[base], color=GRAY, lw=1.5)
    ax.plot(e.index, e[sel], color=CAT[0], lw=1.8)
    ax.set_xlim(e.index[0], e.index[-1])
    _label_ends(ax, [(e[base].iloc[-1], f"plain 50/50  {e[base].iloc[-1]:.2f}x", INK2),
                     (e[sel].iloc[-1], f"strategy  {e[sel].iloc[-1]:.2f}x", CAT[0])], 0.07)
    ax.set_ylabel("value of 1 invested in Jan 2016")
    _title(ax, "The strategy ends slightly ahead - within the range of luck",
           "out-of-sample period 2016-2026, month-end values, after all costs")
    _save(fig, "1_equity_test.png")


def fig_drawdown(eq):
    e = eq.loc[config.TEST_START:]
    base, _, _, sel = _names(eq)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    db = (e[base] / e[base].cummax() - 1).resample("W").min() * 100
    ds = (e[sel] / e[sel].cummax() - 1).resample("W").min() * 100
    ax.fill_between(db.index, db, 0, color=GRAY, alpha=0.25, lw=0)
    ax.plot(db.index, db, color=GRAY, lw=1)
    ax.plot(ds.index, ds, color=CAT[0], lw=1.4)
    for d, col in ((db, GRAY), (ds, CAT[0])):
        ax.plot(d.idxmin(), d.min(), "o", ms=6, color=col, mec=SURFACE, mew=1.5, zorder=4)
    x0 = db.index[int(len(db) * 0.03)]
    ax.text(x0, -21, f"plain 50/50: worst {db.min():.0f}%", color=INK2, fontsize=9.5, fontweight="bold")
    ax.text(x0, -24.5, f"strategy: worst {ds.min():.0f}%", color=CAT[0], fontsize=9.5, fontweight="bold")
    ax.set_xlim(db.index[0], db.index[-1])
    ax.set_ylabel("loss from previous peak (%)")
    _title(ax, "The benefit is smaller losses, mostly in 2022",
           "out-of-sample period; grey area = plain 50/50, blue line = strategy")
    _save(fig, "2_drawdown_test.png")


def fig_ladder(ladder, comp):
    lad = ladder.pivot(index="strategy", columns="period", values="Sharpe")["test"]
    base = comp[(comp.strategy.str.startswith("Base")) & (comp.period == "test")].Sharpe.iloc[0]
    steps = [s.split("  ", 1)[1].replace(" (", "\n(") for s in lad.index]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(steps))
    ax.bar(x, lad.values, width=0.42, color=CAT[0], zorder=2)
    for xx, v in zip(x, lad.values):
        ax.text(xx, v - 0.025, f"{v:.2f}", ha="center", va="top", fontsize=10.5, color="white", fontweight="bold", zorder=4)
    ax.axhline(base, color=INK2, lw=1, zorder=3)
    ax.set_xlim(-0.5, len(steps) - 0.5 + 0.9)
    ax.text(len(steps) - 0.5 + 0.05, base, f"plain 50/50\n{base:.2f}", ha="left", va="center", fontsize=9.5, color=INK2)
    ax.set_xticks(x, steps)
    ax.set_ylim(0, max(lad.max(), base) * 1.18)
    ax.set_ylabel("Sharpe ratio, 2016-2026")
    _title(ax, "Costs eat the whole vol-targeting advantage",
           "vol targeting, adding one real-world cost at a time (left to right)")
    _save(fig, "3_realism_ladder.png")


def fig_overfit(sweep, comp, summary):
    p = sweep.pivot_table(index=["ma", "gate", "band"], columns="period", values="Sharpe").reset_index()
    base_test = comp[(comp.strategy.str.startswith("Base")) & (comp.period == "test")].Sharpe.iloc[0]
    s = summary["selected"]
    sel = p[(p.ma == s["ma"]) & (p.gate == s["gate"]) & (p.band == s["band"])]
    fig, ax = plt.subplots(figsize=(6.6, 5.4))
    ax.axhline(base_test, color=INK2, lw=1)
    ax.text(p.train.min(), base_test + 0.003, "plain 50/50 on test", color=INK2, fontsize=9, va="bottom")
    ax.scatter(p.train, p.test, s=26, color=GRAY, alpha=0.6, lw=0, label=f"all {len(p)} settings")
    ax.scatter(sel.train, sel.test, s=70, color=CAT[0], edgecolor=SURFACE, lw=2, zorder=3,
               label=f"picked on train: MA{s['ma']} {s['gate']} band {s['band']:.2f}")
    ax.set_xlabel("train 2005-2015 Sharpe")
    ax.set_ylabel("test 2016-2026 Sharpe")
    ax.legend(loc="lower right")
    _title(ax, "Train ranking barely predicts test ranking",
           f"rank correlation {summary['train_test_sharpe_rank_corr']:.2f}; "
           f"{summary['configs_beating_base_test_sharpe']} of {len(p)} beat 50/50 on test")
    _save(fig, "4_train_vs_test.png")


def fig_band(abl, sweep, comp):
    a = abl[abl.period == "test"].set_index("band").Sharpe
    m = sweep[sweep.period == "test"].groupby(["band"]).Sharpe.mean()
    base_test = comp[(comp.strategy.str.startswith("Base")) & (comp.period == "test")].Sharpe.iloc[0]
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.axhline(base_test, color=INK2, lw=1)
    ax.text(0.3, base_test + 0.002, "plain 50/50", color=INK2, fontsize=9, va="bottom", ha="right")
    ax.plot(a.index, a, color=CAT[0], marker="o", ms=4, lw=1.4, label="without trend filter")
    ax.plot(m.index, m, color=CAT[1], marker="o", ms=4, lw=1.4, label="with trend filter (average of 6 variants)")
    ax.set_xlabel("no-trade band (trade only if leverage drifts this far from target)")
    ax.set_ylabel("test Sharpe")
    ax.legend(loc="upper left")
    _title(ax, "Trading less helps a little; the trend filter doesn't help",
           "test period, post hoc ablation (not used to pick the strategy); the zigzag is noise")
    _save(fig, "5_band_ablation.png")


def run():
    eq = pd.read_csv(T / "equity_curves.csv", index_col=0, parse_dates=True)
    comp = pd.read_csv(T / "comparison.csv")
    sweep = pd.read_csv(T / "sweep.csv")
    summary = json.loads((T / "summary.json").read_text())
    fig_equity(eq)
    fig_drawdown(eq)
    fig_ladder(pd.read_csv(T / "ladder.csv"), comp)
    fig_overfit(sweep, comp, summary)
    fig_band(pd.read_csv(T / "ablation_band_no_ma.csv"), sweep, comp)
