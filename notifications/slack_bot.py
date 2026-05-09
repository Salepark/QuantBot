"""
Slack Notifier
Sends trading alerts and portfolio reports via Slack Web API.
Gracefully handles missing credentials without crashing.
"""

import logging
from typing import Optional

from config.settings import SLACK_BOT_TOKEN, SLACK_CHANNEL

logger = logging.getLogger(__name__)


class SlackNotifier:
    """
    Sends notifications to a Slack channel via the slack_sdk WebClient.
    Logs a warning and does nothing if SLACK_BOT_TOKEN is not configured.
    """

    def __init__(
        self,
        bot_token: Optional[str] = None,
        channel: Optional[str] = None,
    ) -> None:
        self.bot_token = bot_token or SLACK_BOT_TOKEN
        self.channel = channel or SLACK_CHANNEL or "#trading-alerts"
        self._client = None

        if not self.bot_token or self.bot_token.startswith("xoxb-your_"):
            logger.warning(
                "Slack credentials not configured. "
                "Set SLACK_BOT_TOKEN in .env to enable notifications."
            )
        else:
            self._init_client()

    def _init_client(self) -> None:
        """Attempt to initialize the Slack WebClient."""
        try:
            from slack_sdk import WebClient
            self._client = WebClient(token=self.bot_token)
            logger.info("Slack WebClient initialized successfully.")
        except ImportError:
            logger.warning("slack-sdk not installed. Slack notifications disabled.")
        except Exception as exc:
            logger.warning("Failed to initialize Slack client: %s", exc)

    # ─── Public Methods ───────────────────────────────────────────────────────

    def send_message(self, text: str) -> bool:
        """
        Send a text message to the configured Slack channel.

        Args:
            text: Message text (supports Slack mrkdwn formatting).

        Returns:
            True if sent successfully, False otherwise.
        """
        if not self._client:
            logger.debug("Slack: skipping send (not configured).")
            return False

        try:
            response = self._client.chat_postMessage(
                channel=self.channel,
                text=text,
                mrkdwn=True,
            )
            if response["ok"]:
                logger.debug("Slack message sent successfully.")
                return True
            else:
                logger.warning("Slack API returned error: %s", response.get("error"))
                return False

        except Exception as exc:
            logger.warning("Slack send_message failed: %s", exc)
            return False

    def send_trade_alert(self, trade: dict) -> bool:
        """
        Send a formatted trade alert to Slack.

        Args:
            trade: Trade dict with keys: action, asset, quantity, price,
                   trade_value, reasoning (optional), status.

        Returns:
            True if sent successfully.
        """
        if not self._client:
            return False

        action = trade.get("action", "UNKNOWN")
        asset = trade.get("asset", "UNKNOWN")
        quantity = trade.get("quantity", 0)
        price = trade.get("price", 0)
        trade_value = trade.get("trade_value", quantity * price)
        reasoning = trade.get("reasoning", "")
        status = trade.get("status", "executed")

        emoji = ":large_green_circle:" if action == "BUY" else ":red_circle:"
        status_emoji = ":white_check_mark:" if status == "executed" else ":x:"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": f"{('📈' if action == 'BUY' else '📉')} QuantBot Trade Alert",
                    "emoji": True,
                },
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Status:*\n{status_emoji} {status.upper()}"},
                    {"type": "mrkdwn", "text": f"*Direction:*\n{emoji} *{action}*"},
                    {"type": "mrkdwn", "text": f"*Symbol:*\n`{asset}`"},
                    {"type": "mrkdwn", "text": f"*Quantity:*\n{quantity:.4f}"},
                    {"type": "mrkdwn", "text": f"*Price:*\n${price:,.2f}"},
                    {"type": "mrkdwn", "text": f"*Trade Value:*\n${trade_value:,.2f}"},
                ],
            },
        ]

        if reasoning:
            blocks.append({
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Reasoning:*\n_{reasoning[:300]}_"},
            })

        try:
            response = self._client.chat_postMessage(
                channel=self.channel,
                blocks=blocks,
                text=f"QuantBot {action} alert: {asset}",
            )
            return response["ok"]
        except Exception as exc:
            logger.warning("Slack send_trade_alert failed: %s", exc)
            # Fallback to simple text
            return self.send_message(
                f"*QuantBot Trade Alert*\n"
                f"{action} {quantity:.4f} x {asset} @ ${price:,.2f}"
            )

    def send_portfolio_report(self, portfolio_summary: dict) -> bool:
        """
        Send a formatted portfolio status report to Slack.

        Args:
            portfolio_summary: Portfolio summary dict from Portfolio.get_summary().

        Returns:
            True if sent successfully.
        """
        if not self._client:
            return False

        total_value = portfolio_summary.get("total_value", 0)
        cash = portfolio_summary.get("cash", 0)
        invested = portfolio_summary.get("invested_value", 0)
        total_pnl = portfolio_summary.get("total_pnl", 0)
        total_return_pct = portfolio_summary.get("total_return_pct", 0)
        drawdown_pct = portfolio_summary.get("current_drawdown_pct", 0)
        allocation = portfolio_summary.get("allocation", {})
        paper_trading = portfolio_summary.get("paper_trading", True)

        pnl_sign = "+" if total_pnl >= 0 else ""
        mode_tag = "📝 PAPER" if paper_trading else "🔴 LIVE"

        # Build positions string
        positions = [
            f"`{asset}`: {pct:.1f}%"
            for asset, pct in allocation.items()
            if asset != "CASH"
        ]
        positions_str = " | ".join(positions) if positions else "No open positions"

        blocks = [
            {
                "type": "header",
                "text": {
                    "type": "plain_text",
                    "text": "📊 QuantBot Portfolio Report",
                    "emoji": True,
                },
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": f"Mode: *{mode_tag}*"}],
            },
            {
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*Total Value:*\n${total_value:,.2f}"},
                    {"type": "mrkdwn", "text": f"*Cash:*\n${cash:,.2f} ({allocation.get('CASH', 0):.1f}%)"},
                    {"type": "mrkdwn", "text": f"*Invested:*\n${invested:,.2f}"},
                    {
                        "type": "mrkdwn",
                        "text": f"*Total P&L:*\n{'📈' if total_pnl >= 0 else '📉'} {pnl_sign}${total_pnl:,.2f} ({pnl_sign}{total_return_pct:.2f}%)",
                    },
                    {"type": "mrkdwn", "text": f"*Drawdown:*\n⚠️ {drawdown_pct:.2f}%"},
                ],
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"*Positions:*\n{positions_str}"},
            },
        ]

        try:
            response = self._client.chat_postMessage(
                channel=self.channel,
                blocks=blocks,
                text="QuantBot Portfolio Report",
            )
            return response["ok"]
        except Exception as exc:
            logger.warning("Slack send_portfolio_report failed: %s", exc)
            return self.send_message(
                f"*QuantBot Portfolio Report*\n"
                f"Total: ${total_value:,.2f} | P&L: {pnl_sign}${total_pnl:,.2f}"
            )
