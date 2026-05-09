"""
Yahoo Finance Data Collector
Fetches historical and real-time stock market data using yfinance.
"""

import logging
from typing import Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class YahooFinanceCollector:
    """
    Collects market data from Yahoo Finance via the yfinance library.
    Falls back to mock data if yfinance is unavailable or the API fails.
    """

    def __init__(self) -> None:
        self._yf_available = self._check_yfinance()

    def _check_yfinance(self) -> bool:
        """Check whether yfinance is importable and functional."""
        try:
            import yfinance  # noqa: F401
            return True
        except ImportError:
            logger.warning("yfinance not installed – using mock data for Yahoo Finance.")
            return False

    # ─── Public Methods ───────────────────────────────────────────────────────

    def get_price_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch OHLCV price history for a symbol.

        Args:
            symbol:   Ticker symbol (e.g. "AAPL").
            period:   Lookback period string (e.g. "1y", "6mo", "3mo").
            interval: Bar interval (e.g. "1d", "1h", "15m").

        Returns:
            DataFrame with normalised columns: open, high, low, close, volume.
        """
        if self._yf_available:
            try:
                import yfinance as yf

                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)

                if df.empty:
                    logger.warning("No data returned for %s – using mock data.", symbol)
                    return self._mock_price_history(symbol, period)

                df = self._normalize_columns(df)
                df = self._validate_prices(df, symbol)
                return df

            except Exception as exc:
                logger.warning(
                    "yfinance error for %s (%s) – falling back to mock data.", symbol, exc
                )

        return self._mock_price_history(symbol, period)

    def get_market_data(self, symbols: list[str]) -> dict[str, dict]:
        """
        Fetch current prices and volumes for a list of symbols.

        Returns:
            Dict keyed by symbol with keys: price, volume, change_pct.
        """
        result: dict[str, dict] = {}

        if self._yf_available:
            try:
                import yfinance as yf

                tickers = yf.Tickers(" ".join(symbols))
                for sym in symbols:
                    try:
                        info = tickers.tickers[sym].fast_info
                        result[sym] = {
                            "price": float(getattr(info, "last_price", 0) or 0),
                            "volume": float(getattr(info, "three_month_average_volume", 0) or 0),
                            "change_pct": float(
                                getattr(info, "regular_market_change_percent", 0) or 0
                            ),
                        }
                    except Exception as e:
                        logger.warning("Could not fetch fast_info for %s: %s", sym, e)
                        result[sym] = self._mock_current_price(sym)

                return result

            except Exception as exc:
                logger.warning("get_market_data failed (%s) – using mock data.", exc)

        for sym in symbols:
            result[sym] = self._mock_current_price(sym)
        return result

    def get_macro_indicators(self) -> dict[str, dict]:
        """
        Fetch macro ETF proxies: SPY, QQQ, GLD, TLT, UUP (DXY proxy).

        Returns:
            Dict keyed by ETF symbol with price/change_pct/description.
        """
        macro_symbols = {
            "SPY": "S&P 500 ETF",
            "QQQ": "Nasdaq 100 ETF",
            "GLD": "Gold ETF",
            "TLT": "20+ Year Treasury ETF",
            "UUP": "USD Bull ETF (DXY proxy)",
        }

        prices = self.get_market_data(list(macro_symbols.keys()))
        result: dict[str, dict] = {}
        for sym, desc in macro_symbols.items():
            entry = prices.get(sym, self._mock_current_price(sym))
            entry["description"] = desc
            result[sym] = entry
        return result

    # ─── Private Helpers ──────────────────────────────────────────────────────

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercase all column names and rename yfinance defaults."""
        df = df.copy()
        df.columns = [c.lower().replace(" ", "_") for c in df.columns]

        rename_map = {
            "stock_splits": "splits",
            "capital_gains": "cap_gains",
        }
        df.rename(columns=rename_map, inplace=True)

        # Ensure the standard OHLCV columns exist
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                df[col] = np.nan

        return df[required + [c for c in df.columns if c not in required]]

    def _validate_prices(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Drop rows with NaN in critical price columns and warn if many dropped."""
        price_cols = ["open", "high", "low", "close"]
        before = len(df)
        df = df.dropna(subset=price_cols)
        after = len(df)
        if before != after:
            logger.warning(
                "Dropped %d rows with NaN prices for %s.", before - after, symbol
            )
        return df

    # ─── Mock Data Generators ─────────────────────────────────────────────────

    def _mock_price_history(self, symbol: str, period: str = "1y") -> pd.DataFrame:
        """Generate realistic-looking mock OHLCV data for testing/fallback."""
        period_days_map = {
            "1d": 1, "5d": 5, "1mo": 30, "3mo": 90,
            "6mo": 180, "1y": 252, "2y": 504, "5y": 1260,
        }
        n_days = period_days_map.get(period, 252)

        # Seed from symbol name for reproducibility
        seed = sum(ord(c) for c in symbol)
        rng = np.random.default_rng(seed)

        base_prices = {
            "AAPL": 180.0, "MSFT": 380.0, "GOOGL": 170.0, "AMZN": 185.0,
            "NVDA": 850.0, "META": 500.0, "TSLA": 200.0, "BRK-B": 360.0,
            "JPM": 195.0, "V": 275.0,
            "SPY": 510.0, "QQQ": 440.0, "GLD": 195.0, "TLT": 95.0, "UUP": 28.0,
        }
        start_price = base_prices.get(symbol, 100.0)

        # Geometric Brownian Motion
        daily_return = rng.normal(0.0003, 0.015, n_days)
        prices = start_price * np.exp(np.cumsum(daily_return))

        dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days)

        opens = prices * (1 + rng.normal(0, 0.003, n_days))
        highs = np.maximum(prices, opens) * (1 + rng.uniform(0.001, 0.012, n_days))
        lows = np.minimum(prices, opens) * (1 - rng.uniform(0.001, 0.012, n_days))
        volumes = rng.integers(5_000_000, 80_000_000, n_days).astype(float)

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

    def _mock_current_price(self, symbol: str) -> dict:
        """Return a plausible mock current price for a symbol."""
        mock_prices = {
            "AAPL": 182.5, "MSFT": 383.0, "GOOGL": 172.3, "AMZN": 187.4,
            "NVDA": 855.0, "META": 505.2, "TSLA": 198.7, "BRK-B": 362.1,
            "JPM": 197.3, "V": 277.8,
            "SPY": 512.4, "QQQ": 442.1, "GLD": 197.3, "TLT": 94.6, "UUP": 27.9,
        }
        price = mock_prices.get(symbol, 100.0)
        return {
            "price": price,
            "volume": 35_000_000.0,
            "change_pct": 0.45,
        }
