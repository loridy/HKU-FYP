from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List

import json

import pandas as pd


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
    universe: str = "SPY100+QQQ100"
    universe_cache: bool = True
    universe_allow_download: bool = True
    tickers: List[str] = field(default_factory=list)
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
        self.data_dir.mkdir(parents=True, exist_ok=True)
        for sub in ["figures", "metrics", "tables", "metadata", "curves"]:
            (self.output_dir / sub).mkdir(parents=True, exist_ok=True)

        if not self.tickers:
            self.tickers = _load_universe_tickers(
                project_root=self.project_root,
                universe=self.universe,
                use_cache=self.universe_cache,
                allow_download=self.universe_allow_download,
            )

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["project_root"] = str(self.project_root)
        payload["data_dir"] = str(self.data_dir)
        payload["output_dir"] = str(self.output_dir)
        return payload


def _normalize_yfinance_symbol(symbol: str) -> str:
    s = str(symbol).strip().upper()
    # Common ETF constituent formatting: "BRK.B" -> yfinance expects "BRK-B"
    s = s.replace(".", "-")
    s = s.replace(" ", "")
    return s


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    return list(dict.fromkeys(items))


def _load_wikipedia_symbols(url: str, symbol_columns: List[str]) -> List[str]:
    tables = pd.read_html(url)
    for t in tables:
        cols = list(t.columns)
        # Flatten MultiIndex columns if present
        if any(isinstance(c, tuple) for c in cols):
            t = t.copy()
            t.columns = [" ".join([str(x) for x in c if str(x) != "nan"]).strip() for c in cols]
        for col in symbol_columns:
            if col in t.columns:
                syms = [_normalize_yfinance_symbol(x) for x in t[col].tolist()]
                syms = [s for s in syms if s and s != "NAN"]
                return syms
    raise ValueError(f"Could not find symbol column {symbol_columns} in {url}")


def _load_universe_tickers(project_root: Path, universe: str, use_cache: bool, allow_download: bool) -> List[str]:
    uni = universe.strip().upper().replace("_", "").replace(" ", "")
    cache_dir = project_root / "data" / "universe"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{uni.lower()}.json"

    if use_cache and cache_path.exists():
        try:
            payload = json.loads(cache_path.read_text(encoding="utf-8"))
            tickers = payload.get("tickers", [])
            if isinstance(tickers, list) and tickers:
                return [_normalize_yfinance_symbol(t) for t in tickers]
        except Exception:
            pass

    if not allow_download:
        return [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "JPM", "XOM",
            "UNH", "HD", "LLY", "PG", "AVGO", "COST", "MRK", "ABBV",
            "PEP", "KO", "AMD", "BAC",
        ]

    # Constituents sourced from Wikipedia tables (best-effort).
    # - Nasdaq-100: https://en.wikipedia.org/wiki/Nasdaq-100
    # - S&P 100: https://en.wikipedia.org/wiki/S%26P_100
    try:
        spy100 = _load_wikipedia_symbols("https://en.wikipedia.org/wiki/S%26P_100", ["Symbol", "Ticker", "Ticker symbol"])
        qqq100 = _load_wikipedia_symbols("https://en.wikipedia.org/wiki/Nasdaq-100", ["Ticker", "Symbol", "Ticker symbol"])
    except Exception as e:
        # Keep synthetic/offline mode usable even if constituents cannot be fetched.
        print(
            "Warning: failed to download universe constituents; falling back to a built-in liquid US universe. "
            f"Set `universe_allow_download=False` to suppress this. Error: {e}"
        )
        spy100 = _FALLBACK_LIQUID_US_UNIVERSE
        qqq100 = []

    if uni in {"SPY100"}:
        tickers = spy100
    elif uni in {"QQQ100"}:
        tickers = qqq100
    elif uni in {"SPY100+QQQ100", "SPY100QQQ100"}:
        tickers = _dedupe_preserve_order(spy100 + qqq100)
    else:
        raise ValueError(f"Unknown universe '{universe}'. Use SPY100, QQQ100, or SPY100+QQQ100.")

    tickers = _dedupe_preserve_order([_normalize_yfinance_symbol(t) for t in tickers])

    if use_cache and tickers:
        cache_path.write_text(json.dumps({"tickers": tickers}, indent=2), encoding="utf-8")

    return tickers


# Best-effort offline fallback: a liquid US large-cap universe that works with yfinance symbol conventions.
_FALLBACK_LIQUID_US_UNIVERSE: List[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "GOOG", "META", "TSLA", "AVGO", "BRK-B",
    "JPM", "JNJ", "XOM", "V", "MA", "PG", "UNH", "HD", "CVX", "MRK", "LLY", "ABBV",
    "PEP", "KO", "COST", "WMT", "MCD", "NKE", "CRM", "ADBE", "NFLX", "ORCL", "CSCO",
    "INTC", "AMD", "QCOM", "TXN", "AMAT", "LRCX", "KLAC", "MU", "ASML", "SNPS", "CDNS",
    "NOW", "INTU", "ISRG", "AMGN", "GILD", "REGN", "VRTX", "BKNG", "ABNB", "PANW",
    "CRWD", "ZS", "FTNT", "MDB", "TEAM", "DDOG", "SNOW", "PYPL", "SQ", "SHOP", "UBER",
    "LYFT", "SPOT", "RBLX", "EA", "TTWO", "CMCSA", "TMUS", "T", "VZ", "DIS", "CHTR",
    "TMO", "DHR", "ABT", "MDT", "BSX", "SYK", "ZTS", "PFE", "BMY", "AMT", "PLD", "EQIX",
    "O", "NEE", "DUK", "SO", "AEP", "COP", "SLB", "EOG", "PXD", "OXY", "PSX", "MPC",
    "BA", "RTX", "LMT", "NOC", "GE", "CAT", "DE", "MMM", "HON", "IBM", "ACN", "ADP",
    "LOW", "SBUX", "MDLZ", "PM", "MO", "CL", "TGT", "C", "BAC", "WFC", "GS", "MS",
    "BLK", "SCHW", "AXP", "SPGI", "ICE", "CME", "CB", "PGR", "TRV", "AON", "MMC",
    "LIN", "APD", "SHW", "NEM", "FDX", "UPS", "UNP", "CSX", "NSC", "GM", "F", "LULU",
    "COST", "BK", "USB", "PNC", "TFC", "CI", "HUM", "CVS", "VRTX", "ADI", "NXPI",
    "MRVL", "WDAY", "CSGP", "MELI", "PDD", "JD", "BIDU", "NTES",
]
