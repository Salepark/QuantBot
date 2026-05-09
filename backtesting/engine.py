"""
Backtesting Engine
Simulates trading a strategy on historical price data.
Commission: 0.1% per trade (configurable).
"""

import logging
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, SignalAction
from backtesting.metrics import calculate_all_metrics
from config.settings import BACKTEST_COMMISSION_RATE, DEFAULT_INITIAL_CAPITAL

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of a single executed trade."""
    date: pd.Timestamp
    action: str          # "BUY" or "SELL"
    price: float
    quantity: float
    value: float
    commission: float
    reasoning: str = ""


@dataclass
class BacktestResult:
    """
    Results of a completed backtest run.

    Attributes:
        symbol:          Ticker symbol tested.
        strategy_name:   Name of the strategy.
        start_date:      First date in the backtest window.
        end_date:        Last date in the backtest window.
        initial_capital: Starting capital in USD.
        final_value:     Portfolio value at end of backtest.
        total_return:    Total return as a decimal fraction.
        sharpe:          Annualised Sharpe ratio.
        sortino:         Annualised Sortino ratio.
        max_drawdown:    Maximum drawdown (negative percentage).
        annualised_return: Annualised return.
        volatility:      Annualised volatility.
        win_rate:        Fraction of winning trades.
        trade_count:     Total number of trades executed.
        portfolio_values: Series of daily portfolio values.
        trades:          List of TradeRecord objects.
    """
    symbol: str
    strategy_name: str
    start_date: str
    end_date: str
    initial_capital: float
    final_value: float
    total_return: float
    sharpe: float
    sortino: float
    max_drawdown: float
    annualised_return: float
    volatility: float
    win_rate: float
    trade_count: int
    portfolio_values: pd.Series = field(default_factory=pd.Series)
    trades: list = field(default_factory=list)


class BacktestEngine:
    """
    Event-driven backtesting engine.

    Iterates day-by-day, calling `strategy.generate_signal` on the data
    available up to that day, then simulates the resulting trade.

    - Entry: buy at the next day's open price.
    - Exit:  sell at the next day's open price.
    - Only one position at a time (fully in or fully out).
    - Commission: 0.1% of trade value on each buy and sell.
    """

    def __init__(self, commission_rate: float = BACKTEST_COMMISSION_RATE) -> None:
        self.commission_rate = commission_rate

    # ─── Public Methods ───────────────────────────────────────────────────────

    def run(
        self,
        strategy: BaseStrategy,
        price_data: pd.DataFrame,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run a backtest for a strategy on historical price data.

        Args:
            strategy:        A BaseStrategy instance.
            price_data:      DataFrame with OHLCV columns and DatetimeIndex.
            initial_capital: Starting capital in USD.
            start_date:      ISO date string to start backtest (inclusive).
            end_date:        ISO date string to end backtest (inclusive).

        Returns:
            A BacktestResult with all performance metrics.
        """
        df = price_data.copy()

        # Filter date range
        if start_date:
            df = df[df.index >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df.index <= pd.to_datetime(end_date)]

        if len(df) < 60:
            logger.warning(
                "Backtest data too short (%d rows). Results may be unreliable.", len(df)
            )

        # Ensure lowercase columns
        df.columns = [c.lower() for c in df.columns]
        symbol = getattr(strategy, "symbol", "UNKNOWN")

        cash = initial_capital
        shares_held = 0.0
        portfolio_values: list[float] = []
        trades: list[TradeRecord] = []
        in_position = False

        for i in range(1, len(df)):
            history = df.iloc[:i]
            today_open = float(df["open"].iloc[i])

            # Generate signal on data up to and including yesterday
            try:
                signal = strategy.generate_signal(history, symbol="BACKTEST")
            except Exception as exc:
                logger.warning("Strategy error at index %d: %s", i, exc)
                portfolio_value = cash + shares_held * float(df["close"].iloc[i - 1])
                portfolio_values.append(portfolio_value)
                continue

            current_price = float(df["close"].iloc[i - 1])
            portfolio_value = cash + shares_held * current_price

            # ── Execute BUY at today's open ────────────────────────────────
            if signal.action == SignalAction.BUY and not in_position:
                shares_to_buy = cash / today_open
                commission = shares_to_buy * today_open * self.commission_rate
                if commission < cash:
                    shares_held = shares_to_buy * (1 - self.commission_rate)
                    cash = 0.0
                    in_position = True
                    trades.append(TradeRecord(
                        date=df.index[i],
                        action="BUY",
                        price=today_open,
                        quantity=shares_held,
                        value=shares_held * today_open,
                        commission=commission,
                        reasoning=signal.reasoning,
                    ))

            # ── Execute SELL at today's open ───────────────────────────────
            elif signal.action == SignalAction.SELL and in_position:
                sale_value = shares_held * today_open
                commission = sale_value * self.commission_rate
                cash = sale_value - commission
                shares_held = 0.0
                in_position = False
                trades.append(TradeRecord(
                    date=df.index[i],
                    action="SELL",
                    price=today_open,
                    quantity=shares_held,
                    value=sale_value,
                    commission=commission,
                    reasoning=signal.reasoning,
                ))

            # Track portfolio value
            portfolio_values.append(cash + shares_held * current_price)

        # Close any open position at last available close
        if in_position and len(df) > 0:
            last_price = float(df["close"].iloc[-1])
            sale_value = shares_held * last_price
            commission = sale_value * self.commission_rate
            cash = sale_value - commission
            shares_held = 0.0

        final_value = cash
        pv_series = pd.Series(
            portfolio_values,
            index=df.index[1 : len(portfolio_values) + 1],
        )

        # Compute returns
        if len(pv_series) > 1:
            daily_returns = pv_series.pct_change().dropna()
        else:
            daily_returns = pd.Series(dtype=float)

        metrics = calculate_all_metrics(
            portfolio_values=pv_series.values,
            returns=daily_returns.values,
        )

        # Win rate based on completed trades (buy→sell pairs)
        buy_prices = [t.price for t in trades if t.action == "BUY"]
        sell_prices = [t.price for t in trades if t.action == "SELL"]
        pairs = list(zip(buy_prices, sell_prices))
        win_rate = (
            sum(1 for b, s in pairs if s > b) / len(pairs)
            if pairs else metrics["win_rate"]
        )

        actual_start = str(df.index[0].date()) if len(df) > 0 else (start_date or "N/A")
        actual_end = str(df.index[-1].date()) if len(df) > 0 else (end_date or "N/A")

        return BacktestResult(
            symbol=symbol,
            strategy_name=strategy.name,
            start_date=actual_start,
            end_date=actual_end,
            initial_capital=initial_capital,
            final_value=round(final_value, 2),
            total_return=metrics["total_return"],
            sharpe=metrics["sharpe"],
            sortino=metrics["sortino"],
            max_drawdown=metrics["max_drawdown"],
            annualised_return=metrics["annualised_return"],
            volatility=metrics["volatility"],
            win_rate=round(win_rate, 4),
            trade_count=len(trades),
            portfolio_values=pv_series,
            trades=trades,
        )

    def print_report(self, result: BacktestResult) -> None:
        """Pretty-print a BacktestResult to the console."""
        separator = "=" * 58
        print(f"\n{separator}")
        print(f"  BACKTEST REPORT — {result.symbol} ({result.strategy_name})")
        print(separator)
        print(f"  Period        : {result.start_date} → {result.end_date}")
        print(f"  Initial Capital: ${result.initial_capital:>12,.2f}")
        print(f"  Final Value    : ${result.final_value:>12,.2f}")
        print(separator)
        print(f"  Total Return   : {result.total_return * 100:>+8.2f}%")
        print(f"  Annualised Rtn : {result.annualised_return * 100:>+8.2f}%")
        print(f"  Volatility     : {result.volatility * 100:>8.2f}%")
        print(f"  Sharpe Ratio   : {result.sharpe:>8.4f}")
        print(f"  Sortino Ratio  : {result.sortino:>8.4f}")
        print(f"  Max Drawdown   : {result.max_drawdown * 100:>8.2f}%")
        print(separator)
        print(f"  Trades         : {result.trade_count:>8d}")
        print(f"  Win Rate       : {result.win_rate * 100:>8.2f}%")
        print(f"{separator}\n")
