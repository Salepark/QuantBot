"""
Backtesting Performance Metrics
Functions for computing standard quantitative finance performance metrics.
"""

import numpy as np
import pandas as pd
from typing import Union


def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualised Sharpe ratio.

    Args:
        returns:          Daily returns series (e.g. 0.01 = 1%).
        risk_free_rate:   Annual risk-free rate (default 5%).
        periods_per_year: Trading days per year (default 252).

    Returns:
        Annualised Sharpe ratio as a float. Returns 0.0 if std is zero.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    std = np.std(excess_returns, ddof=1)
    if std == 0.0:
        return 0.0

    return float(np.mean(excess_returns) / std * np.sqrt(periods_per_year))


def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.05,
    periods_per_year: int = 252,
) -> float:
    """
    Compute the annualised Sortino ratio (penalises only downside volatility).

    Args:
        returns:          Daily returns series.
        risk_free_rate:   Annual risk-free rate (default 5%).
        periods_per_year: Trading days per year.

    Returns:
        Annualised Sortino ratio as a float. Returns 0.0 if downside std is zero.
    """
    returns = np.asarray(returns, dtype=float)
    if len(returns) == 0:
        return 0.0

    daily_rf = risk_free_rate / periods_per_year
    excess_returns = returns - daily_rf

    # Downside deviation: only negative excess returns contribute
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0:
        return float("inf")  # No losing days → infinite Sortino

    downside_std = np.std(downside, ddof=1)
    if downside_std == 0.0:
        return 0.0

    return float(np.mean(excess_returns) / downside_std * np.sqrt(periods_per_year))


def max_drawdown(
    portfolio_values: Union[pd.Series, np.ndarray],
) -> float:
    """
    Compute the maximum drawdown as a negative percentage.

    Args:
        portfolio_values: Series of portfolio values over time.

    Returns:
        Maximum drawdown as a float ≤ 0 (e.g. -0.15 means -15%).
        Returns 0.0 if the input is empty or has only one element.
    """
    values = np.asarray(portfolio_values, dtype=float)
    if len(values) < 2:
        return 0.0

    peak = np.maximum.accumulate(values)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        drawdown = np.where(peak > 0, (values - peak) / peak, 0.0)

    return float(np.min(drawdown))


def calculate_all_metrics(
    portfolio_values: Union[pd.Series, np.ndarray],
    returns: Union[pd.Series, np.ndarray],
    risk_free_rate: float = 0.05,
) -> dict:
    """
    Calculate all standard performance metrics in one call.

    Args:
        portfolio_values: Series of daily portfolio values.
        returns:          Series of daily returns.
        risk_free_rate:   Annual risk-free rate.

    Returns:
        Dict with keys:
            sharpe (float),
            sortino (float),
            max_drawdown (float),
            total_return (float),
            annualised_return (float),
            volatility (float),
            win_rate (float) – fraction of positive-return days.
    """
    portfolio_values = np.asarray(portfolio_values, dtype=float)
    returns = np.asarray(returns, dtype=float)

    total_return = 0.0
    if len(portfolio_values) >= 2 and portfolio_values[0] > 0:
        total_return = float((portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0])

    n_days = len(returns)
    annualised_return = 0.0
    if n_days > 0 and (1 + total_return) > 0:
        annualised_return = float((1 + total_return) ** (252 / max(n_days, 1)) - 1)

    volatility = float(np.std(returns, ddof=1) * np.sqrt(252)) if len(returns) > 1 else 0.0

    win_rate = 0.0
    if len(returns) > 0:
        win_rate = float(np.sum(returns > 0) / len(returns))

    return {
        "sharpe": round(sharpe_ratio(returns, risk_free_rate), 4),
        "sortino": round(sortino_ratio(returns, risk_free_rate), 4),
        "max_drawdown": round(max_drawdown(portfolio_values), 4),
        "total_return": round(total_return, 4),
        "annualised_return": round(annualised_return, 4),
        "volatility": round(volatility, 4),
        "win_rate": round(win_rate, 4),
    }
