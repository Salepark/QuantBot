"""
QuantBot Configuration Settings
Loads environment variables and defines constants for the trading system.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ─── API Keys ────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
BINANCE_API_KEY: str = os.getenv("BINANCE_API_KEY", "")
BINANCE_SECRET_KEY: str = os.getenv("BINANCE_SECRET_KEY", "")
FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
SLACK_BOT_TOKEN: str = os.getenv("SLACK_BOT_TOKEN", "")
SLACK_CHANNEL: str = os.getenv("SLACK_CHANNEL", "#trading-alerts")

# ─── Trading Configuration ────────────────────────────────────────────────────
PAPER_TRADING: bool = os.getenv("PAPER_TRADING", "true").lower() == "true"
MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.1"))
MAX_DRAWDOWN_LIMIT: float = float(os.getenv("MAX_DRAWDOWN_LIMIT", "0.15"))

# ─── Allowed Assets Whitelist ─────────────────────────────────────────────────
# Top 10 Stocks (large-cap, highly liquid)
ALLOWED_STOCKS: list[str] = [
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "GOOGL",  # Alphabet Inc.
    "AMZN",   # Amazon.com Inc.
    "NVDA",   # NVIDIA Corporation
    "META",   # Meta Platforms Inc.
    "TSLA",   # Tesla Inc.
    "BRK-B",  # Berkshire Hathaway
    "JPM",    # JPMorgan Chase
    "V",      # Visa Inc.
]

# Top 5 Cryptocurrencies (by market cap)
ALLOWED_CRYPTOS: list[str] = [
    "BTCUSDT",  # Bitcoin
    "ETHUSDT",  # Ethereum
    "BNBUSDT",  # BNB
    "SOLUSDT",  # Solana
    "XRPUSDT",  # XRP
]

# Combined whitelist for validation
ALLOWED_ASSETS: list[str] = ALLOWED_STOCKS + ALLOWED_CRYPTOS

# Crypto symbol mapping (Binance format → display name)
CRYPTO_DISPLAY_NAMES: dict[str, str] = {
    "BTCUSDT": "Bitcoin",
    "ETHUSDT": "Ethereum",
    "BNBUSDT": "BNB",
    "SOLUSDT": "Solana",
    "XRPUSDT": "XRP",
}

# ─── Risk Level Configuration ─────────────────────────────────────────────────
RISK_LEVELS: dict[str, dict] = {
    "conservative": {
        "max_position_size": 0.05,       # 5% max per position
        "max_drawdown_limit": 0.08,      # 8% max drawdown
        "min_rsi_buy": 30,               # RSI must be very oversold to buy
        "max_rsi_sell": 65,              # Sell earlier
        "bb_entry_multiplier": 1.0,      # Enter exactly at Bollinger Band
        "stop_loss_pct": 0.03,           # 3% stop loss
        "take_profit_pct": 0.06,         # 6% take profit
        "max_open_positions": 3,
    },
    "neutral": {
        "max_position_size": 0.10,       # 10% max per position
        "max_drawdown_limit": 0.15,      # 15% max drawdown
        "min_rsi_buy": 35,               # RSI < 35 to buy
        "max_rsi_sell": 70,              # RSI > 70 to sell
        "bb_entry_multiplier": 1.0,
        "stop_loss_pct": 0.05,           # 5% stop loss
        "take_profit_pct": 0.10,         # 10% take profit
        "max_open_positions": 5,
    },
    "aggressive": {
        "max_position_size": 0.20,       # 20% max per position
        "max_drawdown_limit": 0.25,      # 25% max drawdown
        "min_rsi_buy": 40,               # Wider RSI range
        "max_rsi_sell": 75,
        "bb_entry_multiplier": 0.9,      # Enter slightly inside Bollinger Band
        "stop_loss_pct": 0.08,           # 8% stop loss
        "take_profit_pct": 0.20,         # 20% take profit
        "max_open_positions": 8,
    },
}

# Default risk level
DEFAULT_RISK_LEVEL: str = "neutral"

# ─── Data Collection Settings ─────────────────────────────────────────────────
DATA_CACHE_TTL_SECONDS: int = 300          # 5 minutes cache TTL
DEFAULT_PRICE_PERIOD: str = "1y"           # Default lookback period
DEFAULT_PRICE_INTERVAL: str = "1d"         # Default price interval
BACKTEST_COMMISSION_RATE: float = 0.001    # 0.1% commission per trade

# ─── Binance API Settings ─────────────────────────────────────────────────────
BINANCE_BASE_URL: str = "https://api.binance.com/api/v3"
BINANCE_RATE_LIMIT_REQUESTS: int = 1200    # Requests per minute
BINANCE_MAX_RETRIES: int = 3
BINANCE_RETRY_DELAY: float = 1.0           # Seconds between retries

# ─── FRED Series IDs ──────────────────────────────────────────────────────────
FRED_SERIES: dict[str, str] = {
    "FEDFUNDS": "Federal Funds Rate",
    "CPIAUCSL": "Consumer Price Index (CPI)",
    "T10Y2Y": "10Y-2Y Treasury Yield Spread",
    "VIXCLS": "CBOE Volatility Index (VIX)",
    "UNRATE": "Unemployment Rate",
}

# ─── News Sentiment Settings ──────────────────────────────────────────────────
SENTIMENT_LOOKBACK_DAYS: int = 7
MAX_ARTICLES_PER_QUERY: int = 20

# ─── Strategy Parameters ──────────────────────────────────────────────────────
MOMENTUM_RSI_PERIOD: int = 14
MOMENTUM_MACD_FAST: int = 12
MOMENTUM_MACD_SLOW: int = 26
MOMENTUM_MACD_SIGNAL: int = 9
MOMENTUM_MA_SHORT: int = 20
MOMENTUM_MA_LONG: int = 50

MEAN_REVERSION_BB_PERIOD: int = 20
MEAN_REVERSION_BB_STD: float = 2.0

# ─── Initial Capital ──────────────────────────────────────────────────────────
DEFAULT_INITIAL_CAPITAL: float = 100_000.0  # $100,000 paper trading capital
