"""
Mean Reversion Strategy
Uses Bollinger Bands to generate contrarian trading signals.
"""

import logging
import pandas as pd
import numpy as np

from strategies.base import BaseStrategy, Signal, SignalAction
from config.settings import MEAN_REVERSION_BB_PERIOD, MEAN_REVERSION_BB_STD

logger = logging.getLogger(__name__)


class MeanReversionStrategy(BaseStrategy):
    """
    Mean reversion strategy using Bollinger Bands.

    Entry rules:
        BUY  → price < lower Bollinger Band (oversold / below mean)
        SELL → price > upper Bollinger Band (overbought / above mean)
        HOLD → price is within the bands

    Confidence is proportional to how far the price has deviated from the band.
    """

    def __init__(
        self,
        bb_period: int = MEAN_REVERSION_BB_PERIOD,
        bb_std: float = MEAN_REVERSION_BB_STD,
    ) -> None:
        super().__init__("MeanReversionStrategy")
        self.bb_period = bb_period
        self.bb_std = bb_std

    # ─── Public Methods ───────────────────────────────────────────────────────

    def generate_signal(self, price_df: pd.DataFrame, symbol: str) -> Signal:
        """
        Generate a BUY/SELL/HOLD signal based on Bollinger Band deviation.

        Args:
            price_df: DataFrame with at least a 'close' column.
            symbol:   Ticker symbol.

        Returns:
            A Signal instance.
        """
        df = self._ensure_columns(price_df, ["close"])
        if df is None or len(df) < self.bb_period + 5:
            return self._hold_signal(
                symbol,
                f"Insufficient data: need at least {self.bb_period + 5} rows, "
                f"got {len(price_df) if price_df is not None else 0}.",
            )

        close = df["close"].astype(float)

        middle_band, upper_band, lower_band = self._compute_bollinger_bands(
            close, self.bb_period, self.bb_std
        )

        latest_close = float(close.iloc[-1])
        latest_upper = float(upper_band.iloc[-1])
        latest_lower = float(lower_band.iloc[-1])
        latest_middle = float(middle_band.iloc[-1])
        band_width = latest_upper - latest_lower

        # Percentage position within bands (0 = lower, 0.5 = middle, 1 = upper)
        if band_width > 0:
            band_pct = (latest_close - latest_lower) / band_width
        else:
            band_pct = 0.5

        metadata = {
            "close": round(latest_close, 2),
            "upper_band": round(latest_upper, 2),
            "middle_band": round(latest_middle, 2),
            "lower_band": round(latest_lower, 2),
            "band_width": round(band_width, 2),
            "band_pct": round(band_pct, 4),
        }

        # ── BUY logic ──────────────────────────────────────────────────────
        if latest_close < latest_lower:
            # How far below the lower band (as fraction of band_width)
            deviation = (latest_lower - latest_close) / max(band_width, 1e-9)
            confidence = min(1.0, 0.5 + deviation * 2.0)

            reasoning = (
                f"BUY signal: Price ${latest_close:.2f} is BELOW lower Bollinger Band "
                f"${latest_lower:.2f} (deviation={deviation:.2%}). "
                f"Mean reversion expected toward ${latest_middle:.2f}."
            )
            signal = Signal(
                action=SignalAction.BUY,
                asset=symbol,
                confidence=round(confidence, 3),
                reasoning=reasoning,
                metadata=metadata,
            )

        # ── SELL logic ─────────────────────────────────────────────────────
        elif latest_close > latest_upper:
            deviation = (latest_close - latest_upper) / max(band_width, 1e-9)
            confidence = min(1.0, 0.5 + deviation * 2.0)

            reasoning = (
                f"SELL signal: Price ${latest_close:.2f} is ABOVE upper Bollinger Band "
                f"${latest_upper:.2f} (deviation={deviation:.2%}). "
                f"Mean reversion expected toward ${latest_middle:.2f}."
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
                f"HOLD: Price ${latest_close:.2f} is within Bollinger Bands "
                f"[${latest_lower:.2f}, ${latest_upper:.2f}]. "
                f"No mean reversion signal."
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

    # ─── Technical Indicator Implementation ──────────────────────────────────

    @staticmethod
    def _compute_bollinger_bands(
        close: pd.Series,
        period: int = 20,
        num_std: float = 2.0,
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """
        Compute Bollinger Bands.

        Returns:
            (middle_band, upper_band, lower_band) as pandas Series.
        """
        middle = close.rolling(window=period).mean()
        std = close.rolling(window=period).std(ddof=0)
        upper = middle + num_std * std
        lower = middle - num_std * std
        return middle, upper, lower
