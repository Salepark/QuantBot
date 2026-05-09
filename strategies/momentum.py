"""
Momentum Strategy
Uses RSI, MACD, and moving averages to generate trend-following signals.
All technical indicators are implemented in pure pandas/numpy (no TA-lib).
"""

import logging
import pandas as pd
import numpy as np
from typing import Optional

from strategies.base import BaseStrategy, Signal, SignalAction
from config.settings import (
    MOMENTUM_RSI_PERIOD,
    MOMENTUM_MACD_FAST,
    MOMENTUM_MACD_SLOW,
    MOMENTUM_MACD_SIGNAL,
    MOMENTUM_MA_SHORT,
    MOMENTUM_MA_LONG,
)

logger = logging.getLogger(__name__)


class MomentumStrategy(BaseStrategy):
    """
    Momentum strategy using RSI, MACD, and moving average crossovers.

    Entry rules:
        BUY  → RSI < rsi_buy_threshold AND price > MA50 (uptrend confirmed)
        SELL → RSI > rsi_sell_threshold (overbought)
        HOLD → everything else

    Confidence is scaled by how extreme the RSI reading is.
    """

    def __init__(
        self,
        rsi_period: int = MOMENTUM_RSI_PERIOD,
        macd_fast: int = MOMENTUM_MACD_FAST,
        macd_slow: int = MOMENTUM_MACD_SLOW,
        macd_signal: int = MOMENTUM_MACD_SIGNAL,
        ma_short: int = MOMENTUM_MA_SHORT,
        ma_long: int = MOMENTUM_MA_LONG,
        rsi_buy_threshold: float = 35.0,
        rsi_sell_threshold: float = 70.0,
    ) -> None:
        super().__init__("MomentumStrategy")
        self.rsi_period = rsi_period
        self.macd_fast = macd_fast
        self.macd_slow = macd_slow
        self.macd_signal_period = macd_signal
        self.ma_short = ma_short
        self.ma_long = ma_long
        self.rsi_buy_threshold = rsi_buy_threshold
        self.rsi_sell_threshold = rsi_sell_threshold

    # ─── Public Methods ───────────────────────────────────────────────────────

    def generate_signal(self, price_df: pd.DataFrame, symbol: str) -> Signal:
        """
        Generate a BUY/SELL/HOLD signal based on momentum indicators.

        Args:
            price_df: DataFrame with at least a 'close' column.
            symbol:   Ticker symbol.

        Returns:
            A Signal instance.
        """
        df = self._ensure_columns(price_df, ["close"])
        if df is None or len(df) < self.ma_long + 5:
            return self._hold_signal(
                symbol,
                f"Insufficient data: need at least {self.ma_long + 5} rows, "
                f"got {len(price_df) if price_df is not None else 0}.",
            )

        close = df["close"].astype(float)

        rsi = self._compute_rsi(close, self.rsi_period)
        macd_line, macd_sig, macd_hist = self._compute_macd(
            close, self.macd_fast, self.macd_slow, self.macd_signal_period
        )
        ma_short_series = close.rolling(window=self.ma_short).mean()
        ma_long_series = close.rolling(window=self.ma_long).mean()

        # Use most recent values
        latest_rsi = float(rsi.iloc[-1])
        latest_macd_hist = float(macd_hist.iloc[-1])
        latest_close = float(close.iloc[-1])
        latest_ma_short = float(ma_short_series.iloc[-1])
        latest_ma_long = float(ma_long_series.iloc[-1])

        metadata = {
            "rsi": round(latest_rsi, 2),
            "macd_histogram": round(latest_macd_hist, 4),
            "ma_short": round(latest_ma_short, 2),
            "ma_long": round(latest_ma_long, 2),
            "close": round(latest_close, 2),
        }

        # ── BUY logic ──────────────────────────────────────────────────────
        if latest_rsi < self.rsi_buy_threshold and latest_close > latest_ma_long:
            # Scale confidence: the more oversold, the higher the confidence
            rsi_extreme = max(0.0, self.rsi_buy_threshold - latest_rsi)
            confidence = min(1.0, 0.4 + (rsi_extreme / self.rsi_buy_threshold) * 0.6)

            # Bonus if MACD histogram is turning up
            if latest_macd_hist > 0:
                confidence = min(1.0, confidence + 0.1)

            reasoning = (
                f"BUY signal: RSI={latest_rsi:.1f} (oversold, below {self.rsi_buy_threshold}), "
                f"price ${latest_close:.2f} > MA50 ${latest_ma_long:.2f} (uptrend confirmed). "
                f"MACD histogram={latest_macd_hist:.4f}."
            )
            signal = Signal(
                action=SignalAction.BUY,
                asset=symbol,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                metadata=metadata,
            )

        # ── SELL logic ─────────────────────────────────────────────────────
        elif latest_rsi > self.rsi_sell_threshold:
            rsi_extreme = max(0.0, latest_rsi - self.rsi_sell_threshold)
            confidence = min(1.0, 0.4 + (rsi_extreme / (100 - self.rsi_sell_threshold)) * 0.6)

            # Bonus if MACD histogram is turning down
            if latest_macd_hist < 0:
                confidence = min(1.0, confidence + 0.1)

            reasoning = (
                f"SELL signal: RSI={latest_rsi:.1f} (overbought, above {self.rsi_sell_threshold}). "
                f"MACD histogram={latest_macd_hist:.4f}."
            )
            signal = Signal(
                action=SignalAction.SELL,
                asset=symbol,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                metadata=metadata,
            )

        # ── HOLD logic ─────────────────────────────────────────────────────
        else:
            reasoning = (
                f"HOLD: RSI={latest_rsi:.1f} is in neutral zone "
                f"[{self.rsi_buy_threshold}, {self.rsi_sell_threshold}]. "
                f"No clear entry/exit signal."
            )
            signal = Signal(
                action=SignalAction.HOLD,
                asset=symbol,
                confidence=0.0,
                reasoning=reasoning,
                metadata=metadata,
            )

        self.validate_signal(signal)
        return signal

    # ─── Technical Indicator Implementations ─────────────────────────────────

    @staticmethod
    def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute Wilder's RSI using exponential moving average of gains/losses.

        Returns a Series of RSI values (NaN for the first `period` entries).
        """
        delta = close.diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)

        avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
        avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()

        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)   # fill NaN with neutral RSI

    @staticmethod
    def _compute_macd(
        close: pd.Series,
        fast: int = 12,
        slow: int = 26,
        signal_period: int = 9,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute MACD line, signal line, and histogram.

        Returns:
            (macd_line, signal_line, histogram) as pandas Series.
        """
        ema_fast = close.ewm(span=fast, adjust=False).mean()
        ema_slow = close.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram
