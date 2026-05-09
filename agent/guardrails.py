"""
Guardrails
Hard-coded safety checks that CANNOT be overridden by the AI agent.
All trades must pass guardrail validation before execution.
"""

import logging
from dataclasses import dataclass
from typing import Optional

from config.settings import (
    ALLOWED_ASSETS,
    MAX_POSITION_SIZE,
    MAX_DRAWDOWN_LIMIT,
)

logger = logging.getLogger(__name__)


@dataclass
class TradeProposal:
    """
    A proposed trade from the AI agent.

    Attributes:
        asset:     Ticker or crypto symbol.
        action:    "BUY" or "SELL".
        quantity:  Number of shares/units to trade.
        price:     Estimated execution price.
        reasoning: AI-generated rationale.
    """
    asset: str
    action: str
    quantity: float
    price: float
    reasoning: str = ""

    @property
    def trade_value(self) -> float:
        """Total USD value of this trade."""
        return self.quantity * self.price


class Guardrails:
    """
    Hard-coded risk management guardrails.

    These limits are enforced at the system level and cannot be changed
    by the AI agent's instructions or tool calls at runtime.
    """

    def __init__(
        self,
        allowed_assets: Optional[list[str]] = None,
        max_position_size: float = MAX_POSITION_SIZE,
        max_drawdown_limit: float = MAX_DRAWDOWN_LIMIT,
    ) -> None:
        # Make a frozen copy of limits at construction time
        self._allowed_assets: frozenset[str] = frozenset(
            a.upper() for a in (allowed_assets or ALLOWED_ASSETS)
        )
        self._max_position_size: float = max_position_size      # fraction of total capital
        self._max_drawdown_limit: float = max_drawdown_limit    # absolute value (e.g. 0.15)

    # ─── Individual Checks ────────────────────────────────────────────────────

    def check_asset_whitelist(self, asset: str) -> bool:
        """
        Return True if the asset is in the allowed whitelist.

        Args:
            asset: Ticker or crypto symbol (case-insensitive).
        """
        result = asset.upper() in self._allowed_assets
        if not result:
            logger.warning(
                "GUARDRAIL BLOCK: Asset '%s' is not in the whitelist. "
                "Allowed: %s",
                asset,
                sorted(self._allowed_assets),
            )
        return result

    def check_position_size(
        self,
        proposed_size: float,
        total_capital: float,
    ) -> bool:
        """
        Return True if the proposed trade size is within the allowed limit.

        Args:
            proposed_size: USD value of the proposed trade.
            total_capital: Total portfolio value in USD.
        """
        if total_capital <= 0:
            logger.warning("GUARDRAIL: total_capital is ≤ 0 – rejecting trade.")
            return False

        fraction = proposed_size / total_capital
        result = fraction <= self._max_position_size
        if not result:
            logger.warning(
                "GUARDRAIL BLOCK: Position size %.2f%% exceeds limit %.2f%%.",
                fraction * 100,
                self._max_position_size * 100,
            )
        return result

    def check_drawdown_limit(self, current_drawdown: float) -> bool:
        """
        Return True if the current drawdown is within the allowed limit.

        Args:
            current_drawdown: Current drawdown as a negative float (e.g. -0.12).
        """
        # Normalise: accept both negative and positive representation
        drawdown_abs = abs(current_drawdown)
        result = drawdown_abs <= self._max_drawdown_limit
        if not result:
            logger.warning(
                "GUARDRAIL BLOCK: Current drawdown %.2f%% exceeds limit %.2f%%. "
                "Trading is HALTED until drawdown recovers.",
                drawdown_abs * 100,
                self._max_drawdown_limit * 100,
            )
        return result

    def validate_trade(
        self,
        trade_proposal: TradeProposal,
        portfolio: "Portfolio",   # forward reference to avoid circular import
    ) -> tuple[bool, str]:
        """
        Run all guardrail checks against a trade proposal.

        Args:
            trade_proposal: The proposed trade.
            portfolio:      Current portfolio state.

        Returns:
            (approved: bool, rejection_reason: str)
            rejection_reason is empty string if approved.
        """
        # 1. Asset whitelist
        if not self.check_asset_whitelist(trade_proposal.asset):
            return False, (
                f"Asset '{trade_proposal.asset}' is not in the approved whitelist. "
                f"Allowed assets: {sorted(self._allowed_assets)}"
            )

        # 2. Drawdown limit (only block new BUY orders, not SELLs)
        if trade_proposal.action.upper() == "BUY":
            current_dd = portfolio.get_current_drawdown()
            if not self.check_drawdown_limit(current_dd):
                return False, (
                    f"Current portfolio drawdown ({abs(current_dd) * 100:.2f}%) "
                    f"exceeds the maximum allowed limit "
                    f"({self._max_drawdown_limit * 100:.2f}%). "
                    "New BUY orders are blocked until drawdown recovers."
                )

        # 3. Position size
        total_capital = portfolio.total_value
        if not self.check_position_size(trade_proposal.trade_value, total_capital):
            return False, (
                f"Trade value ${trade_proposal.trade_value:,.2f} "
                f"({trade_proposal.trade_value / max(total_capital, 1) * 100:.1f}% of portfolio) "
                f"exceeds the maximum position size of "
                f"{self._max_position_size * 100:.0f}% "
                f"(${total_capital * self._max_position_size:,.2f})."
            )

        # 4. Paper trading sanity: quantity must be positive
        if trade_proposal.quantity <= 0:
            return False, f"Trade quantity must be positive, got {trade_proposal.quantity}."

        # 5. Price sanity check
        if trade_proposal.price <= 0:
            return False, f"Trade price must be positive, got {trade_proposal.price}."

        logger.info(
            "GUARDRAIL APPROVED: %s %s x %.4f @ $%.2f",
            trade_proposal.action,
            trade_proposal.asset,
            trade_proposal.quantity,
            trade_proposal.price,
        )
        return True, ""

    @property
    def limits(self) -> dict:
        """Return the current guardrail limits as a dict."""
        return {
            "allowed_assets": sorted(self._allowed_assets),
            "max_position_size_pct": self._max_position_size * 100,
            "max_drawdown_limit_pct": self._max_drawdown_limit * 100,
        }
