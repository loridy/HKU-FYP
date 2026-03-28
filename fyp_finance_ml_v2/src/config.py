from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List


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

    feature_sets: Dict[str, List[str]] = field(default_factory=lambda: {
        "F1_momentum": ["momentum"],
        "F2_momentum_reversal": ["momentum", "reversal"],
        "F3_plus_risk_liquidity": ["momentum", "reversal", "volatility", "liquidity"],
        "F4_plus_cross_sectional": ["momentum", "reversal", "volatility", "liquidity", "cross_sectional"],
        "F5_full_finance": ["momentum", "reversal", "volatility", "liquidity", "cross_sectional", "macro", "fundamental"],
    })

    def __post_init__(self) -> None:
        self.data_dir = self.project_root / "data"
        self.output_dir = self.project_root / "outputs"
        for sub in ["figures", "metrics", "tables", "metadata"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload
