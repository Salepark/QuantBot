"""
News Collector
Fetches financial news articles and performs keyword-based sentiment analysis.
"""

import logging
from datetime import datetime, timedelta
from typing import Optional
import requests

from config.settings import NEWS_API_KEY, SENTIMENT_LOOKBACK_DAYS, MAX_ARTICLES_PER_QUERY

logger = logging.getLogger(__name__)

# Sentiment keyword lists
BULLISH_WORDS = [
    "surge", "rally", "gain", "rise", "up", "bull", "bullish", "growth",
    "positive", "strong", "beat", "outperform", "upgrade", "buy", "record",
    "high", "breakout", "momentum", "boom", "soar", "climb", "profit",
    "earnings beat", "revenue growth", "market cap", "all-time high",
    "recovery", "rebound", "optimistic", "confident", "expansion",
]

BEARISH_WORDS = [
    "crash", "fall", "drop", "decline", "bear", "bearish", "recession",
    "negative", "weak", "miss", "underperform", "downgrade", "sell", "loss",
    "low", "breakdown", "plunge", "slump", "collapse", "debt", "deficit",
    "layoff", "bankruptcy", "default", "warning", "risk", "concern",
    "inflation", "downturn", "contraction", "pessimistic", "correction",
]


class NewsCollector:
    """
    Fetches financial news from NewsAPI and computes keyword-based sentiment.
    Falls back to mock data if the API key is missing or the API is unavailable.
    """

    NEWS_API_URL = "https://newsapi.org/v2/everything"

    def __init__(self) -> None:
        self._api_available = bool(NEWS_API_KEY)
        if not self._api_available:
            logger.warning(
                "NEWS_API_KEY not set – news data will use mock values."
            )

    # ─── Public Methods ───────────────────────────────────────────────────────

    def get_financial_news(
        self,
        query: str = "stock market finance trading",
        days_back: int = SENTIMENT_LOOKBACK_DAYS,
    ) -> list[dict]:
        """
        Fetch recent financial news articles for a query term.

        Args:
            query:     Search query string.
            days_back: Number of days to look back from today.

        Returns:
            List of article dicts with keys: title, description, url, publishedAt, source.
        """
        if self._api_available:
            try:
                from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
                params = {
                    "q": query,
                    "from": from_date,
                    "language": "en",
                    "sortBy": "relevancy",
                    "pageSize": min(MAX_ARTICLES_PER_QUERY, 100),
                    "apiKey": NEWS_API_KEY,
                }

                response = requests.get(self.NEWS_API_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()

                if data.get("status") != "ok":
                    logger.warning(
                        "NewsAPI returned non-ok status: %s", data.get("message", "unknown")
                    )
                    return self._mock_articles(query)

                articles = data.get("articles", [])
                return [
                    {
                        "title": a.get("title", ""),
                        "description": a.get("description", ""),
                        "url": a.get("url", ""),
                        "publishedAt": a.get("publishedAt", ""),
                        "source": a.get("source", {}).get("name", "Unknown"),
                    }
                    for a in articles
                    if a.get("title")
                ]

            except requests.exceptions.RequestException as exc:
                logger.warning("NewsAPI request failed (%s) – using mock data.", exc)
            except Exception as exc:
                logger.warning("Unexpected error fetching news (%s) – using mock data.", exc)

        return self._mock_articles(query)

    def analyze_sentiment(self, articles: list[dict]) -> dict:
        """
        Compute a keyword-based sentiment score from a list of articles.

        Args:
            articles: List of article dicts (from get_financial_news).

        Returns:
            Dict with keys:
                overall_sentiment (float, -1 to 1),
                article_count (int),
                sample_headlines (list[str]),
                bullish_count (int),
                bearish_count (int),
        """
        if not articles:
            return {
                "overall_sentiment": 0.0,
                "article_count": 0,
                "sample_headlines": [],
                "bullish_count": 0,
                "bearish_count": 0,
            }

        bullish_count = 0
        bearish_count = 0

        for article in articles:
            text = (
                (article.get("title") or "") + " " +
                (article.get("description") or "")
            ).lower()

            article_bull = sum(1 for w in BULLISH_WORDS if w in text)
            article_bear = sum(1 for w in BEARISH_WORDS if w in text)

            if article_bull > article_bear:
                bullish_count += 1
            elif article_bear > article_bull:
                bearish_count += 1

        total = bullish_count + bearish_count
        if total == 0:
            overall_sentiment = 0.0
        else:
            overall_sentiment = round((bullish_count - bearish_count) / total, 4)

        sample_headlines = [
            a["title"] for a in articles[:5] if a.get("title")
        ]

        return {
            "overall_sentiment": overall_sentiment,
            "article_count": len(articles),
            "sample_headlines": sample_headlines,
            "bullish_count": bullish_count,
            "bearish_count": bearish_count,
        }

    # ─── Mock Data Generators ─────────────────────────────────────────────────

    def _mock_articles(self, query: str) -> list[dict]:
        """Return a set of mock news articles for testing/fallback."""
        return [
            {
                "title": "Markets Rally as Tech Stocks Lead Gains",
                "description": "Major indices surge on strong earnings reports from technology sector.",
                "url": "https://example.com/article1",
                "publishedAt": datetime.now().isoformat(),
                "source": "Mock Financial News",
            },
            {
                "title": "Federal Reserve Holds Interest Rates Steady",
                "description": "The Fed signals a cautious approach amid mixed economic signals.",
                "url": "https://example.com/article2",
                "publishedAt": datetime.now().isoformat(),
                "source": "Mock Financial News",
            },
            {
                "title": "Inflation Data Shows Signs of Cooling",
                "description": "CPI growth slows, boosting investor optimism for rate cuts.",
                "url": "https://example.com/article3",
                "publishedAt": datetime.now().isoformat(),
                "source": "Mock Financial News",
            },
            {
                "title": "Bitcoin Breaks Above Key Resistance Level",
                "description": "Crypto markets show bullish momentum with Bitcoin leading the rally.",
                "url": "https://example.com/article4",
                "publishedAt": datetime.now().isoformat(),
                "source": "Mock Crypto News",
            },
            {
                "title": "Supply Chain Concerns Weigh on Market Outlook",
                "description": "Analysts warn of potential headwinds from global supply disruptions.",
                "url": "https://example.com/article5",
                "publishedAt": datetime.now().isoformat(),
                "source": "Mock Financial News",
            },
        ]
