"""All parameters in one place. Selection rule and split are fixed BEFORE the first run."""

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

TICKERS = ("SPY", "TLT")
RATE_TICKER = "^IRX"                     # 13-week T-bill yield, % p.a.
DATA_START = "2004-01-01"
DATA_END = "2026-11-01"                  # exclusive; pinned so runs are reproducible
TRAIN_END = pd.Timestamp("2015-12-31")   # choose parameters on data up to here...
TEST_START = pd.Timestamp("2016-01-01")  # ...and judge them only on data from here
PERIODS = 252


@dataclass(frozen=True)
class Strategy:
    weights: tuple = (0.5, 0.5)
    target_vol: float = 0.10
    vol_lookback: int = 20
    max_leverage: float = 3.0
    ma: int | None = None                # None = no regime cap
    gate: str = "simple"                 # "simple" | "hyst"
    hyst_buffer: float = 0.01
    band: float = 0.0                    # trade only if |actual lev - target| > band


@dataclass(frozen=True)
class Frictions:
    trading_bps: float = 10.0            # per unit of notional traded
    borrow_spread: float = 0.015         # borrow at T-bill + spread
    cash_haircut: float = 0.005          # idle cash earns T-bill - haircut (floored at 0)
    margin: bool = True
    m_init: float = 0.50                 # Reg-T style: max 2x at trade time
    m_maint: float = 0.30
    maint_buffer: float = 0.05           # forced delever back to 1/(0.30+0.05) = 2.86x

    @property
    def init_cap(self) -> float:
        return 1 / self.m_init if self.margin else float("inf")

    @property
    def maint_cap(self) -> float:
        return 1 / (self.m_maint + self.maint_buffer) if self.margin else float("inf")


NO_FRICTIONS = Frictions(trading_bps=0.0, borrow_spread=0.0, cash_haircut=0.0, margin=False)

# Sweep (same grid as the original version of this project)
MAS = (200, 300, 400)
GATES = ("simple", "hyst")
BANDS = tuple(round(0.02 * i, 2) for i in range(16))      # 0.00 .. 0.30
WARMUP = max(MAS) + 1                                     # all signals defined from here on
