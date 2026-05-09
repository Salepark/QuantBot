"""
QuantBot Data Collectors
Individual data source collectors for market data aggregation.
"""

from .yahoo_finance import YahooFinanceCollector
from .binance_collector import BinanceCollector
from .fred_collector import FREDCollector
from .news_collector import NewsCollector

__all__ = [
    "YahooFinanceCollector",
    "BinanceCollector",
    "FREDCollector",
    "NewsCollector",
]
