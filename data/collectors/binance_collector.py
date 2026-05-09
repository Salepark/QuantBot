"""
Binance Public REST API Data Collector
Fetches cryptocurrency market data using Binance's public endpoints.
No authentication required for market data.
"""

import logging
import time
from typing import Optional
import pandas as pd
import numpy as np
import requests

from config.settings import (
    BINANCE_BASE_URL,
    BINANCE_MAX_RETRIES,
    BINANCE_RETRY_DELAY,
)

logger = logging.getLogger(__name__)

# Column names for Binance kline/candlestick response
KLINE_COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "trades", "taker_buy_base",
    "taker_buy_quote", "ignore",
]


class BinanceCollector:
    """
    Collects market data from Binance's public REST API.
    Uses https://api.binance.com/api/v3/ — no API key required for market data.
    Falls back to mock data if the API is unreachable.
    """

    TOP_CRYPTOS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "XRPUSDT"]

    def __init__(self) -> None:
        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    # ─── Public Methods ───────────────────────────────────────────────────────

    def get_klines(
        self,
        symbol: str,
        interval: str = "1d",
        limit: int = 500,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candlestick data for a crypto symbol.

        Args:
            symbol:   Binance symbol (e.g. "BTCUSDT").
            interval: Kline interval (e.g. "1d", "4h", "1h", "15m").
            limit:    Number of candles to return (max 1000).

        Returns:
            DataFrame with columns: open_time, open, high, low, close, volume.
        """
        params = {"symbol": symbol.upper(), "interval": interval, "limit": min(limit, 1000)}
        raw = self._request("GET", "/klines", params=params)

        if raw is None:
            logger.warning("Binance klines unavailable for %s – using mock data.", symbol)
            return self._mock_klines(symbol, limit)

        df = pd.DataFrame(raw, columns=KLINE_COLUMNS)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")

        numeric_cols = ["open", "high", "low", "close", "volume", "quote_volume"]
        df[numeric_cols] = df[numeric_cols].astype(float)
        df.set_index("open_time", inplace=True)
        df.index.name = "Date"

        return df[["open", "high", "low", "close", "volume"]]

    def get_ticker_price(self, symbol: str) -> float:
        """
        Fetch the latest price for a symbol.

        Returns:
            Current price as float, or a mock value if unavailable.
        """
        params = {"symbol": symbol.upper()}
        raw = self._request("GET", "/ticker/price", params=params)

        if raw is None or "price" not in raw:
            logger.warning("Could not fetch price for %s – using mock.", symbol)
            return self._mock_price(symbol)

        return float(raw["price"])

    def get_top_cryptos(self) -> dict[str, float]:
        """
        Fetch current prices for the top 5 cryptocurrencies.

        Returns:
            Dict of {symbol: price}.
        """
        result: dict[str, float] = {}
        for sym in self.TOP_CRYPTOS:
            result[sym] = self.get_ticker_price(sym)
        return result

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        retries: int = BINANCE_MAX_RETRIES,
    ) -> Optional[dict | list]:
        """
        Make a request to the Binance REST API with retry logic.

        Returns the parsed JSON response or None on failure.
        """
        url = BINANCE_BASE_URL + endpoint
        for attempt in range(1, retries + 1):
            try:
                response = self._session.request(
                    method, url, params=params, timeout=10
                )
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", BINANCE_RETRY_DELAY))
                    logger.warning(
                        "Binance rate limit hit – waiting %.1fs (attempt %d/%d).",
                        retry_after, attempt, retries,
                    )
                    time.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response.json()

            except requests.exceptions.Timeout:
                logger.warning("Binance request timed out (attempt %d/%d).", attempt, retries)
            except requests.exceptions.ConnectionError:
                logger.warning(
                    "Binance connection error (attempt %d/%d).", attempt, retries
                )
            except requests.exceptions.HTTPError as exc:
                logger.warning("Binance HTTP error: %s", exc)
                return None
            except Exception as exc:
                logger.warning("Binance unexpected error: %s", exc)
                return None

            if attempt < retries:
                time.sleep(BINANCE_RETRY_DELAY * attempt)

        logger.error("All %d Binance request attempts failed for %s.", retries, endpoint)
        return None

    # ─── Mock Data Generators ─────────────────────────────────────────────────

    def _mock_price(self, symbol: str) -> float:
        """Return a plausible mock price for a crypto symbol."""
        mock_prices = {
            "BTCUSDT": 67_500.0,
            "ETHUSDT": 3_550.0,
            "BNBUSDT": 595.0,
            "SOLUSDT": 175.0,
            "XRPUSDT": 0.62,
        }
        return mock_prices.get(symbol.upper(), 1.0)

    def _mock_klines(self, symbol: str, limit: int = 500) -> pd.DataFrame:
        """Generate realistic mock OHLCV data for a crypto symbol."""
        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        base_price = self._mock_price(symbol)

        # Geometric Brownian Motion (higher vol for crypto)
        daily_return = rng.normal(0.001, 0.03, limit)
        prices = base_price * np.exp(np.cumsum(daily_return))

        dates = pd.date_range(end=pd.Timestamp.utcnow(), periods=limit, freq="D")

        opens = prices * (1 + rng.normal(0, 0.005, limit))
        highs = np.maximum(prices, opens) * (1 + rng.uniform(0.002, 0.025, limit))
        lows = np.minimum(prices, opens) * (1 - rng.uniform(0.002, 0.025, limit))
        volumes = rng.uniform(10_000, 100_000, limit)

        df = pd.DataFrame(
            {
                "open": opens,
                "high": highs,
                "low": lows,
                "close": prices,
                "volume": volumes,
            },
            index=dates,
        )
        df.index.name = "Date"
        return df
