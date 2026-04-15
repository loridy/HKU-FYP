from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


def generate_feature_set_combinations(feature_groups: List[str]) -> Dict[str, List[str]]:
    """Generate all non-empty combinations of feature groups.

    Example groups:
      ["momentum", "reversal", "volatility", "liquidity", "cross_sectional", "macro", "fundamental"]

    Returns a dict mapping a stable name -> list of groups (in the same order as `feature_groups`).

    Note: This produces 2^N - 1 combinations (N=7 => 127 feature sets).
    """
    combos: Dict[str, List[str]] = {}
    n = len(feature_groups)
    # bitmask from 1..(2^n - 1)
    for mask in range(1, 2**n):
        groups = [feature_groups[i] for i in range(n) if (mask >> i) & 1]
        # stable, readable name
        name = "FS_" + "_".join(groups)
        combos[name] = groups
    return combos


@dataclass
class AppConfig:
    project_root: Path = Path(__file__).resolve().parents[1]
    data_dir: Path = field(init=False)
    output_dir: Path = field(init=False)

    random_seed: int = 42
    tickers: List[str] = field(default_factory=lambda: [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
        "UNH", "HD", "LLY", "PG", "AVGO", "COST", "MRK", "ABBV",
        "PEP", "KO", "AMD", "BAC"
    ])
    benchmark_ticker: str = "SPY"
    macro_tickers: Dict[str, str] = field(default_factory=lambda: {
        "vix": "^VIX",
        "tnx": "^TNX",
        "qqq": "QQQ",
    })
    start_date: str = "2016-01-01"
    end_date: str = "2026-01-01"

    horizons: List[int] = field(default_factory=lambda: [1, 3])
    transaction_cost_bps: float = 10.0
    top_k: int = 5
    n_deciles: int = 5

    train_frac: float = 0.70
    val_frac: float = 0.15
    test_frac: float = 0.15

    primary_model: str = "logistic_regression"
    use_xgboost: bool = False

    # Canonical feature group list (kept for reference/documentation).
    # Note: fundamentals are excluded from the default experiments due to data quality concerns.
    feature_group_universe: List[str] = field(default_factory=lambda: [
        "momentum",
        "reversal",
        "volatility",
        "liquidity",
        "cross_sectional",
        "macro",
        "fundamental",
    ])

    # Default feature sets: cumulative ladder (F1–F5) WITHOUT fundamentals in F5.
    feature_sets: Dict[str, List[str]] = field(default_factory=lambda: {
        "F1_momentum": ["momentum"],
        "F2_momentum_reversal": ["momentum", "reversal"],
        "F3_plus_risk_liquidity": ["momentum", "reversal", "volatility", "liquidity"],
        "F4_plus_cross_sectional": ["momentum", "reversal", "volatility", "liquidity", "cross_sectional"],
        "F5_full_finance_no_fundamental": ["momentum", "reversal", "volatility", "liquidity", "cross_sectional", "macro"],
    })

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        for sub in ["figures", "metrics", "tables", "metadata", "curves"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload
