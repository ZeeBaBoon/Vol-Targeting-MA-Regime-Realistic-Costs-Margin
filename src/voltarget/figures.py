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
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "font.size": 10,
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


def fig_equity(eq):
    e = eq.loc[config.TEST_START:]
    e = e / e.iloc[0]
    base, static, vt, sel = _names(eq)
    fig, ax = plt.subplots(figsize=(8.5, 4.4))
    for name, col in ((base, GRAY), (static, CAT[1]), (vt, CAT[2]), (sel, CAT[0])):
        ax.plot(e.index, e[name], color=col, label=f"{name}  ({e[name].iloc[-1]:.2f}x)",
                lw=2.4 if name == sel else 1.6)
    ax.set_ylabel("growth of 1 (start of 2016 = 1)")
    ax.legend(loc="upper left")
    _title(ax, "Out of sample, the overlay ends roughly where 50/50 does",
           "test period Jan 2016 - Oct 2026; all frictions on (trading costs, T-bill financing, margin)")
    _save(fig, "1_equity_test.png")


def fig_drawdown(eq):
    e = eq.loc[config.TEST_START:]
    base, _, _, sel = _names(eq)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    for name, col in ((base, GRAY), (sel, CAT[0])):
        dd = e[name] / e[name].cummax() - 1
        ax.plot(dd.index, dd * 100, color=col, label=name)
        i = dd.idxmin()
        ax.annotate(f"{dd.min():.0%}", (i, dd.min() * 100), xytext=(6, -2), textcoords="offset points",
                    color=INK2, fontsize=9, va="top")
    ax.set_ylabel("drawdown from peak (%)")
    ax.legend(loc="lower left")
    _title(ax, "What it does buy: shallower drawdowns",
           "test period; 2022 was the stock-bond crash where 50/50 had no hiding place")
    _save(fig, "2_drawdown_test.png")


def fig_ladder(ladder, comp):
    lad = ladder.pivot(index="strategy", columns="period", values="Sharpe")
    base = comp[comp.strategy.str.startswith("Base")].set_index("period").Sharpe
    steps = list(lad.index)
    fig, ax = plt.subplots(figsize=(8.5, 4))
    x = np.arange(len(steps))
    w = 0.34
    for i, (p, col) in enumerate((("train", CAT[0]), ("test", CAT[1]))):
        xs = x + (i - 0.5) * w
        ax.bar(xs, lad[p], width=w - 0.04, color=col, label=f"vol-target, {p}", zorder=2)
        for xx, v in zip(xs, lad[p]):
            ax.text(xx, v + 0.015, f"{v:.2f}", ha="center", fontsize=8.5, color=INK2)
        ax.axhline(base[p], color=col, lw=1, ls=(0, (1, 0)), alpha=0.6, zorder=1)
        ax.text(len(steps) - 0.45, base[p], f"50/50 {p}: {base[p]:.2f}", va="center", fontsize=8.5, color=INK2)
    ax.set_xticks(x, [s.split("  ", 1)[1].replace(" (", "\n(") for s in steps])
    ax.set_ylabel("Sharpe ratio (excess of T-bills)")
    ax.set_xlim(-0.6, len(steps) + 0.5)
    ax.legend(loc="upper right", bbox_to_anchor=(0.86, 1))
    _title(ax, "Frictions eat the whole vol-targeting advantage",
           "each bar adds one friction to the previous one; thin lines = plain 50/50 for comparison")
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
    ax.plot(a.index, a, color=CAT[0], marker="o", ms=5, mec=SURFACE, mew=1.5, label="vol-target, no MA cap")
    ax.plot(m.index, m, color=CAT[1], marker="o", ms=5, mec=SURFACE, mew=1.5,
            label="with MA cap (average of 6 MA/gate variants)")
    ax.set_xlabel("no-trade band (trade only if leverage drifts this far from target)")
    ax.set_ylabel("test Sharpe")
    ax.legend(loc="lower right")
    _title(ax, "Trading less helps a little; the MA regime cap doesn't help",
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
