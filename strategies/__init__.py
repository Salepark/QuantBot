"""
QuantBot Strategies Module
Contains base class and concrete trading strategy implementations.
"""

from .base import BaseStrategy, Signal, SignalAction
from .momentum import MomentumStrategy
from .mean_reversion import MeanReversionStrategy

__all__ = [
    "BaseStrategy",
    "Signal",
    "SignalAction",
    "MomentumStrategy",
    "MeanReversionStrategy",
]
