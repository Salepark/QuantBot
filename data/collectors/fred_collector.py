"""
FRED (Federal Reserve Economic Data) Collector
Fetches macroeconomic indicators from the St. Louis Fed's FRED API.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import pandas as pd

from config.settings import FRED_API_KEY, FRED_SERIES

logger = logging.getLogger(__name__)

# Mock values for key indicators when API is unavailable
MOCK_INDICATORS = {
    "FEDFUNDS": {"value": 5.33, "date": "2024-01-01", "description": "Federal Funds Rate (%)"},
    "CPIAUCSL": {"value": 314.2, "date": "2024-01-01", "description": "CPI All Urban Consumers"},
    "T10Y2Y":  {"value": -0.38, "date": "2024-01-01", "description": "10Y-2Y Treasury Spread (%)"},
    "VIXCLS":  {"value": 16.5, "date": "2024-01-01", "description": "CBOE VIX"},
    "UNRATE":  {"value": 3.7, "date": "2024-01-01", "description": "Unemployment Rate (%)"},
}


class FREDCollector:
    """
    Fetches macroeconomic data from the FRED API.
    Requires a FRED_API_KEY in the environment.
    Returns mock data if the key is missing or the API is unavailable.
    """

    def __init__(self) -> None:
        self._api_available = self._check_api()

    def _check_api(self) -> bool:
        """Check whether fredapi is installed and a key is configured."""
        if not FRED_API_KEY:
            logger.warning(
                "FRED_API_KEY not set – macroeconomic data will use mock values."
            )
            return False
        try:
            from fredapi import Fred  # noqa: F401
            return True
        except ImportError:
            logger.warning("fredapi not installed – using mock macro data.")
            return False

    # ─── Public Methods ───────────────────────────────────────────────────────

    def get_series(
        self,
        series_id: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Fetch a FRED data series as a DataFrame.

        Args:
            series_id:  FRED series identifier (e.g. "FEDFUNDS").
            start_date: ISO date string "YYYY-MM-DD" (default: 1 year ago).
            end_date:   ISO date string "YYYY-MM-DD" (default: today).

        Returns:
            DataFrame with columns: date (index), value.
        """
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        if end_date is None:
            end_date = datetime.now().strftime("%Y-%m-%d")

        if self._api_available:
            try:
                from fredapi import Fred

                fred = Fred(api_key=FRED_API_KEY)
                series = fred.get_series(series_id, observation_start=start_date, observation_end=end_date)
                df = series.reset_index()
                df.columns = ["date", "value"]
                df["date"] = pd.to_datetime(df["date"])
                df.set_index("date", inplace=True)
                df.dropna(inplace=True)
                return df

            except Exception as exc:
                logger.warning(
                    "FRED API error for series %s (%s) – using mock data.", series_id, exc
                )

        return self._mock_series(series_id, start_date, end_date)

    def get_key_indicators(self) -> dict[str, dict]:
        """
        Fetch the latest values for key macroeconomic indicators.

        Returns:
            Dict keyed by series_id with keys: value, date, description.
        """
        result: dict[str, dict] = {}

        if self._api_available:
            try:
                from fredapi import Fred

                fred = Fred(api_key=FRED_API_KEY)
                for series_id, description in FRED_SERIES.items():
                    try:
                        series = fred.get_series(series_id)
                        latest = series.dropna()
                        if len(latest) > 0:
                            latest_value = float(latest.iloc[-1])
                            latest_date = str(latest.index[-1].date())
                            result[series_id] = {
                                "value": latest_value,
                                "date": latest_date,
                                "description": description,
                            }
                        else:
                            result[series_id] = MOCK_INDICATORS[series_id]
                    except Exception as exc:
                        logger.warning("Could not fetch %s: %s", series_id, exc)
                        result[series_id] = MOCK_INDICATORS.get(
                            series_id,
                            {"value": 0.0, "date": "N/A", "description": description},
                        )
                return result

            except Exception as exc:
                logger.warning("FRED get_key_indicators failed (%s) – using mock.", exc)

        # Return mock data for all series
        for series_id, description in FRED_SERIES.items():
            mock = MOCK_INDICATORS.get(series_id, {}).copy()
            mock.setdefault("description", description)
            result[series_id] = mock
        return result

    # ─── Mock Data Generators ─────────────────────────────────────────────────

    def _mock_series(
        self,
        series_id: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Generate mock time-series data for a FRED series."""
        import numpy as np

        base = MOCK_INDICATORS.get(series_id, {}).get("value", 5.0)
        dates = pd.date_range(start=start_date, end=end_date, freq="MS")

        rng = np.random.default_rng(sum(ord(c) for c in series_id))
        noise = rng.normal(0, base * 0.02, len(dates))
        values = base + np.cumsum(noise * 0.1)

        df = pd.DataFrame({"value": values}, index=dates)
        df.index.name = "date"
        return df
