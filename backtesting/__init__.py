"""
QuantBot Backtesting Module
Provides backtesting engine and performance metrics calculation.
"""

from .engine import BacktestEngine, BacktestResult
from .metrics import (
    sharpe_ratio,
    sortino_ratio,
    max_drawdown,
    calculate_all_metrics,
)

__all__ = [
    "BacktestEngine",
    "BacktestResult",
    "sharpe_ratio",
    "sortino_ratio",
    "max_drawdown",
    "calculate_all_metrics",
]
