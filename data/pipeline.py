"""
Data Pipeline
Orchestrates all data collectors and provides a unified MarketSnapshot.
Results are cached for DATA_CACHE_TTL_SECONDS to avoid redundant API calls.
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd

from config.settings import DATA_CACHE_TTL_SECONDS
from data.collectors.yahoo_finance import YahooFinanceCollector
from data.collectors.binance_collector import BinanceCollector
from data.collectors.fred_collector import FREDCollector
from data.collectors.news_collector import NewsCollector

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """
    Unified snapshot of market data collected from all sources.

    Attributes:
        stocks:    Dict of stock symbol → {price, volume, change_pct}.
        crypto:    Dict of crypto symbol → current price (float).
        macro:     Dict of macro series_id → {value, date, description}.
        sentiment: Dict with overall_sentiment, article_count, sample_headlines.
        timestamp: Unix timestamp when snapshot was taken.
    """
    stocks: dict = field(default_factory=dict)
    crypto: dict = field(default_factory=dict)
    macro: dict = field(default_factory=dict)
    sentiment: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def age_seconds(self) -> float:
        """Return how many seconds old this snapshot is."""
        return time.time() - self.timestamp

    def is_fresh(self, ttl: int = DATA_CACHE_TTL_SECONDS) -> bool:
        """Return True if the snapshot is younger than ttl seconds."""
        return self.age_seconds() < ttl

    def to_summary(self) -> dict:
        """Return a human-readable summary dict."""
        return {
            "stocks": {k: v.get("price", 0) for k, v in self.stocks.items()},
            "crypto": self.crypto,
            "macro": {k: v.get("value", 0) for k, v in self.macro.items()},
            "sentiment": self.sentiment.get("overall_sentiment", 0),
            "age_seconds": round(self.age_seconds(), 1),
        }


class DataPipeline:
    """
    Orchestrates all data collectors and assembles a unified MarketSnapshot.

    Caches the last snapshot and skips re-fetching within the TTL window.
    """

    STOCK_SYMBOLS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
        "META", "TSLA", "BRK-B", "JPM", "V",
    ]

    def __init__(self) -> None:
        self._yf = YahooFinanceCollector()
        self._binance = BinanceCollector()
        self._fred = FREDCollector()
        self._news = NewsCollector()
        self._cache: Optional[MarketSnapshot] = None

    # ─── Public Methods ───────────────────────────────────────────────────────

    def collect_all(self, force: bool = False) -> MarketSnapshot:
        """
        Collect data from all sources and return a MarketSnapshot.

        Args:
            force: If True, bypass the cache and re-fetch all data.

        Returns:
            A fresh (or cached) MarketSnapshot.
        """
        if not force and self._cache is not None and self._cache.is_fresh():
            logger.debug(
                "Returning cached snapshot (age=%.1fs).", self._cache.age_seconds()
            )
            return self._cache

        logger.info("Collecting market data from all sources…")

        stocks = self._collect_stocks()
        crypto = self._collect_crypto()
        macro = self._collect_macro()
        sentiment = self._collect_sentiment()

        snapshot = MarketSnapshot(
            stocks=stocks,
            crypto=crypto,
            macro=macro,
            sentiment=sentiment,
        )

        self.validate_data(snapshot)
        self._cache = snapshot
        logger.info("Market snapshot collected successfully.")
        return snapshot

    def validate_data(self, snapshot: MarketSnapshot) -> None:
        """
        Validate that critical data is present in the snapshot.

        Raises:
            ValueError: If critical stock or crypto data is completely absent.
        """
        if not snapshot.stocks:
            raise ValueError(
                "DataPipeline: Stock data is completely empty in snapshot."
            )
        if not snapshot.crypto:
            raise ValueError(
                "DataPipeline: Crypto data is completely empty in snapshot."
            )

        # Warn (not raise) on missing macro/sentiment – these are non-critical
        if not snapshot.macro:
            logger.warning("Snapshot macro data is empty.")
        if not snapshot.sentiment:
            logger.warning("Snapshot sentiment data is empty.")

    def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Convenience method to fetch price history for a symbol.
        Automatically routes crypto symbols to BinanceCollector.

        Args:
            symbol:   Ticker or crypto symbol.
            period:   Lookback period (used for stocks).
            interval: Bar interval.

        Returns:
            DataFrame with OHLCV columns.
        """
        if symbol.upper().endswith("USDT"):
            # Map period string to limit for Binance
            period_map = {
                "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
                "6mo": 180, "1y": 365, "2y": 730,
            }
            limit = period_map.get(period, 365)
            return self._binance.get_klines(symbol, interval=interval, limit=limit)
        else:
            return self._yf.get_price_history(symbol, period=period, interval=interval)

    # ─── Private Collectors ───────────────────────────────────────────────────

    def _collect_stocks(self) -> dict:
        """Fetch current stock prices and metadata."""
        try:
            data = self._yf.get_market_data(self.STOCK_SYMBOLS)
            # Also include macro ETFs
            macro_data = self._yf.get_macro_indicators()
            data.update(macro_data)
            return data
        except Exception as exc:
            logger.error("Failed to collect stock data: %s", exc)
            return {}

    def _collect_crypto(self) -> dict:
        """Fetch current crypto prices."""
        try:
            return self._binance.get_top_cryptos()
        except Exception as exc:
            logger.error("Failed to collect crypto data: %s", exc)
            return {}

    def _collect_macro(self) -> dict:
        """Fetch macroeconomic indicators from FRED."""
        try:
            return self._fred.get_key_indicators()
        except Exception as exc:
            logger.error("Failed to collect macro data: %s", exc)
            return {}

    def _collect_sentiment(self) -> dict:
        """Fetch news and compute sentiment."""
        try:
            articles = self._news.get_financial_news(
                query="stock market finance trading economy"
            )
            return self._news.analyze_sentiment(articles)
        except Exception as exc:
            logger.error("Failed to collect sentiment data: %s", exc)
            return {
                "overall_sentiment": 0.0,
                "article_count": 0,
                "sample_headlines": [],
                "bullish_count": 0,
                "bearish_count": 0,
            }
