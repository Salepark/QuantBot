"""
QuantBot Data Module
Provides market data collection and pipeline orchestration.
"""

from .pipeline import DataPipeline, MarketSnapshot

__all__ = ["DataPipeline", "MarketSnapshot"]
