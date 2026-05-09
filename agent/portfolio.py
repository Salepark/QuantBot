"""
Portfolio Manager
Tracks positions, cash, P&L, and drawdown.
Operates in paper-trading mode by default (no real order execution).
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
import pandas as pd

from config.settings import PAPER_TRADING, DEFAULT_INITIAL_CAPITAL

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Represents a single open position."""
    asset: str
    quantity: float
    avg_cost: float          # average cost basis per share/unit
    current_price: float
    opened_at: datetime = field(default_factory=datetime.now)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealised_pnl(self) -> float:
        return (self.current_price - self.avg_cost) * self.quantity

    @property
    def unrealised_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost


@dataclass
class TradeLog:
    """Log entry for an executed trade."""
    timestamp: datetime
    asset: str
    action: str
    quantity: float
    price: float
    value: float
    realised_pnl: float = 0.0
    note: str = ""


class Portfolio:
    """
    Portfolio manager for paper trading.

    Tracks:
        - Cash balance
        - Open positions (asset → Position)
        - Trade history
        - Peak portfolio value (for drawdown calculation)
    """

    def __init__(
        self,
        initial_capital: float = DEFAULT_INITIAL_CAPITAL,
        paper_trading: bool = PAPER_TRADING,
    ) -> None:
        self.initial_capital = initial_capital
        self.paper_trading = paper_trading
        self.cash: float = initial_capital
        self.positions: dict[str, Position] = {}
        self.trade_log: list[TradeLog] = []
        self._peak_value: float = initial_capital

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def total_value(self) -> float:
        """Current total portfolio value (cash + market value of all positions)."""
        return self.cash + sum(p.market_value for p in self.positions.values())

    @property
    def invested_value(self) -> float:
        """Total market value of all open positions."""
        return sum(p.market_value for p in self.positions.values())

    # ─── Public Methods ───────────────────────────────────────────────────────

    def add_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        note: str = "",
    ) -> bool:
        """
        Open or add to a position (BUY).

        Args:
            asset:    Ticker symbol.
            quantity: Number of shares/units to buy.
            price:    Price per share/unit.
            note:     Optional note for the trade log.

        Returns:
            True if the trade was executed, False if insufficient cash.
        """
        cost = quantity * price
        if cost > self.cash:
            logger.warning(
                "Insufficient cash to buy %s: need $%.2f, have $%.2f.",
                asset, cost, self.cash,
            )
            return False

        self.cash -= cost

        if asset in self.positions:
            pos = self.positions[asset]
            total_qty = pos.quantity + quantity
            pos.avg_cost = (pos.avg_cost * pos.quantity + price * quantity) / total_qty
            pos.quantity = total_qty
            pos.current_price = price
        else:
            self.positions[asset] = Position(
                asset=asset,
                quantity=quantity,
                avg_cost=price,
                current_price=price,
            )

        self._record_trade(asset, "BUY", quantity, price, 0.0, note)
        self._update_peak()
        logger.info(
            "[%s] BUY %.4f x %s @ $%.2f  (cash remaining: $%.2f)",
            "PAPER" if self.paper_trading else "LIVE",
            quantity, asset, price, self.cash,
        )
        return True

    def remove_position(
        self,
        asset: str,
        quantity: float,
        price: float,
        note: str = "",
    ) -> bool:
        """
        Close or reduce a position (SELL).

        Args:
            asset:    Ticker symbol.
            quantity: Number of shares/units to sell.
            price:    Price per share/unit.
            note:     Optional note for the trade log.

        Returns:
            True if successful, False if no position exists or insufficient quantity.
        """
        if asset not in self.positions:
            logger.warning("Cannot sell %s – no open position.", asset)
            return False

        pos = self.positions[asset]
        sell_qty = min(quantity, pos.quantity)  # cap at held quantity
        proceeds = sell_qty * price
        realised_pnl = (price - pos.avg_cost) * sell_qty

        self.cash += proceeds

        if sell_qty >= pos.quantity:
            del self.positions[asset]
        else:
            pos.quantity -= sell_qty

        self._record_trade(asset, "SELL", sell_qty, price, realised_pnl, note)
        self._update_peak()
        logger.info(
            "[%s] SELL %.4f x %s @ $%.2f  P&L: $%.2f  (cash: $%.2f)",
            "PAPER" if self.paper_trading else "LIVE",
            sell_qty, asset, price, realised_pnl, self.cash,
        )
        return True

    def update_prices(self, price_updates: dict[str, float]) -> None:
        """
        Update current prices for held positions.

        Args:
            price_updates: Dict of {asset: current_price}.
        """
        for asset, price in price_updates.items():
            if asset in self.positions:
                self.positions[asset].current_price = price

        self._update_peak()

    def get_allocation(self) -> dict[str, float]:
        """
        Return allocation as percentage of total portfolio value.

        Returns:
            Dict of {asset: allocation_pct} including 'CASH'.
        """
        total = self.total_value
        if total == 0:
            return {"CASH": 100.0}

        allocation: dict[str, float] = {
            asset: round(pos.market_value / total * 100, 2)
            for asset, pos in self.positions.items()
        }
        allocation["CASH"] = round(self.cash / total * 100, 2)
        return allocation

    def calculate_pnl(self) -> dict:
        """
        Calculate P&L summary.

        Returns:
            Dict with keys: unrealised_pnl, realised_pnl, total_pnl,
            total_return_pct.
        """
        unrealised = sum(p.unrealised_pnl for p in self.positions.values())
        realised = sum(t.realised_pnl for t in self.trade_log if t.action == "SELL")
        total_pnl = unrealised + realised
        total_return_pct = (
            (self.total_value - self.initial_capital) / self.initial_capital
            if self.initial_capital > 0 else 0.0
        )
        return {
            "unrealised_pnl": round(unrealised, 2),
            "realised_pnl": round(realised, 2),
            "total_pnl": round(total_pnl, 2),
            "total_return_pct": round(total_return_pct * 100, 4),
        }

    def get_current_drawdown(self) -> float:
        """
        Compute current drawdown from peak portfolio value.

        Returns:
            Drawdown as a negative float (e.g. -0.05 means -5%).
        """
        if self._peak_value <= 0:
            return 0.0
        return (self.total_value - self._peak_value) / self._peak_value

    def get_summary(self) -> dict:
        """Return a comprehensive portfolio summary dict."""
        pnl = self.calculate_pnl()
        return {
            "total_value": round(self.total_value, 2),
            "cash": round(self.cash, 2),
            "invested_value": round(self.invested_value, 2),
            "initial_capital": self.initial_capital,
            "allocation": self.get_allocation(),
            "current_drawdown_pct": round(self.get_current_drawdown() * 100, 4),
            "peak_value": round(self._peak_value, 2),
            "paper_trading": self.paper_trading,
            **pnl,
        }

    def get_positions_df(self) -> pd.DataFrame:
        """Return open positions as a DataFrame."""
        if not self.positions:
            return pd.DataFrame(
                columns=["asset", "quantity", "avg_cost", "current_price",
                         "market_value", "unrealised_pnl", "pnl_pct"]
            )
        rows = []
        for asset, pos in self.positions.items():
            rows.append({
                "asset": asset,
                "quantity": round(pos.quantity, 4),
                "avg_cost": round(pos.avg_cost, 2),
                "current_price": round(pos.current_price, 2),
                "market_value": round(pos.market_value, 2),
                "unrealised_pnl": round(pos.unrealised_pnl, 2),
                "pnl_pct": round(pos.unrealised_pnl_pct * 100, 2),
            })
        return pd.DataFrame(rows)

    def get_trade_log_df(self) -> pd.DataFrame:
        """Return the trade log as a DataFrame."""
        if not self.trade_log:
            return pd.DataFrame(
                columns=["timestamp", "asset", "action", "quantity",
                         "price", "value", "realised_pnl", "note"]
            )
        rows = [
            {
                "timestamp": t.timestamp,
                "asset": t.asset,
                "action": t.action,
                "quantity": round(t.quantity, 4),
                "price": round(t.price, 2),
                "value": round(t.value, 2),
                "realised_pnl": round(t.realised_pnl, 2),
                "note": t.note,
            }
            for t in self.trade_log
        ]
        return pd.DataFrame(rows)

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _record_trade(
        self,
        asset: str,
        action: str,
        quantity: float,
        price: float,
        realised_pnl: float,
        note: str,
    ) -> None:
        self.trade_log.append(TradeLog(
            timestamp=datetime.now(),
            asset=asset,
            action=action,
            quantity=quantity,
            price=price,
            value=quantity * price,
            realised_pnl=realised_pnl,
            note=note,
        ))

    def _update_peak(self) -> None:
        """Update the peak portfolio value for drawdown tracking."""
        current = self.total_value
        if current > self._peak_value:
            self._peak_value = current
