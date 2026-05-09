"""
Telegram Notifier
Sends trading alerts and portfolio reports via Telegram Bot API.
Gracefully handles missing credentials without crashing.
"""

import asyncio
import logging
from typing import Optional

from config.settings import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """
    Sends notifications to a Telegram chat via the python-telegram-bot library.
    Logs a warning and does nothing if TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID
    are not configured.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        chat_id: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or TELEGRAM_BOT_TOKEN
        self.chat_id = chat_id or TELEGRAM_CHAT_ID
        self._bot = None

        if not self.bot_token or not self.chat_id:
            logger.warning(
                "Telegram credentials not configured. "
                "Set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env to enable notifications."
            )
        else:
            self._init_bot()

    def _init_bot(self) -> None:
        """Attempt to initialise the Telegram bot."""
        try:
            from telegram import Bot
            self._bot = Bot(token=self.bot_token)
            logger.info("Telegram bot initialized successfully.")
        except ImportError:
            logger.warning("python-telegram-bot not installed. Telegram notifications disabled.")
        except Exception as exc:
            logger.warning("Failed to initialize Telegram bot: %s", exc)

    # ─── Public Methods ───────────────────────────────────────────────────────

    async def send_message(self, text: str) -> bool:
        """
        Send a text message to the configured Telegram chat.

        Args:
            text: Message text (supports Markdown).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._bot or not self.chat_id:
            logger.debug("Telegram: skipping send (not configured).")
            return False

        try:
            await self._bot.send_message(
                chat_id=self.chat_id,
                text=text,
                parse_mode="Markdown",
            )
            logger.debug("Telegram message sent successfully.")
            return True
        except Exception as exc:
            logger.warning("Telegram send_message failed: %s", exc)
            return False

    def send_message_sync(self, text: str) -> bool:
        """Synchronous wrapper for send_message."""
        try:
            return asyncio.run(self.send_message(text))
        except RuntimeError:
            # Already in an event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(self.send_message(text))

    async def send_trade_alert(self, trade: dict) -> bool:
        """
        Send a formatted trade alert.

        Args:
            trade: Trade dict with keys: action, asset, quantity, price,
                   trade_value, reasoning (optional).

        Returns:
            True if sent successfully.
        """
        action = trade.get("action", "UNKNOWN")
        asset = trade.get("asset", "UNKNOWN")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)
        trade_value = trade.get("trade_value", quantity * price)
        reasoning = trade.get("reasoning", "")
        status = trade.get("status", "executed")

        emoji = "🟢" if action == "BUY" else "🔴"
        status_emoji = "✅" if status == "executed" else "❌"

        message = (
            f"{emoji} *QuantBot 거래 알림*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"{status_emoji} 상태: {status.upper()}\n"
            f"📌 종목: `{asset}`\n"
            f"📊 방향: *{action}*\n"
            f"💰 수량: {quantity:.4f}\n"
            f"💲 가격: ${price:,.2f}\n"
            f"💵 거래금액: ${trade_value:,.2f}\n"
        )

        if reasoning:
            message += f"\n📝 근거: _{reasoning[:200]}_"

        return await self.send_message(message)

    async def send_portfolio_report(self, portfolio_summary: dict) -> bool:
        """
        Send a formatted portfolio status report.

        Args:
            portfolio_summary: Portfolio summary dict from Portfolio.get_summary().

        Returns:
            True if sent successfully.
        """
        total_value = portfolio_summary.get("total_value", 0)
        cash = portfolio_summary.get("cash", 0)
        invested = portfolio_summary.get("invested_value", 0)
        total_pnl = portfolio_summary.get("total_pnl", 0)
        total_return_pct = portfolio_summary.get("total_return_pct", 0)
        drawdown_pct = portfolio_summary.get("current_drawdown_pct", 0)
        allocation = portfolio_summary.get("allocation", {})
        paper_trading = portfolio_summary.get("paper_trading", True)

        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        pnl_sign = "+" if total_pnl >= 0 else ""
        mode_tag = "📝 [PAPER]" if paper_trading else "🔴 [LIVE]"

        positions_text = ""
        for asset, pct in allocation.items():
            if asset != "CASH":
                positions_text += f"  • `{asset}`: {pct:.1f}%\n"

        message = (
            f"📊 *QuantBot 포트폴리오 리포트*\n"
            f"{mode_tag}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💰 총 자산: ${total_value:,.2f}\n"
            f"💵 현금: ${cash:,.2f} ({allocation.get('CASH', 0):.1f}%)\n"
            f"📦 투자금: ${invested:,.2f}\n"
            f"{pnl_emoji} 손익: `{pnl_sign}${total_pnl:,.2f}` ({pnl_sign}{total_return_pct:.2f}%)\n"
            f"⚠️ 현재 낙폭: {drawdown_pct:.2f}%\n"
        )

        if positions_text:
            message += f"\n*보유 포지션:*\n{positions_text}"

        return await self.send_message(message)
