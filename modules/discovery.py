"""
News discovery module using Tavily and configured sources.
Finds trending AI/tech topics suitable for tutorial videos.
"""
import json
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import yaml

try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False

from config.settings import TAVILY_API_KEY, BASE_DIR


@dataclass
class DiscoveredTopic:
    """A discovered topic for video creation."""
    title: str
    url: str
    source: str
    summary: str
    score: float
    discovered_at: str
    category: str
    keywords: list[str]


class NewsDiscovery:
    """Discovers trending AI/tech news for video content."""

    def __init__(self):
        self.sources = self._load_sources()
        if TAVILY_AVAILABLE and TAVILY_API_KEY:
            self.tavily = TavilyClient(api_key=TAVILY_API_KEY)
        else:
            self.tavily = None

    def _load_sources(self) -> dict:
        """Load source configuration."""
        sources_file = BASE_DIR / "config" / "sources.yaml"
        if sources_file.exists():
            with open(sources_file) as f:
                return yaml.safe_load(f)
        return {"sources": [], "tavily_queries": {}, "filters": {}}

    def discover(self, category: str = "ai_tools", count: int = 5) -> list[DiscoveredTopic]:
        """
        Discover trending topics in a category.

        Args:
            category: Category key from sources.yaml
            count: Number of topics to return

        Returns:
            List of DiscoveredTopic sorted by score
        """
        topics = []

        # Tavily search
        if self.tavily:
            tavily_topics = self._search_tavily(category, count)
            topics.extend(tavily_topics)

        # Sort by score and deduplicate
        topics = self._deduplicate(topics)
        topics.sort(key=lambda t: t.score, reverse=True)

        return topics[:count]

    def _search_tavily(self, category: str, count: int) -> list[DiscoveredTopic]:
        """Search using Tavily API."""
        queries = self.sources.get("tavily_queries", {}).get(category, [])
        topics = []

        for query in queries[:3]:  # Limit API calls
            try:
                response = self.tavily.search(
                    query=query,
                    search_depth="advanced",
                    max_results=count,
                    exclude_domains=self._get_excluded_domains()
                )

                for result in response.get("results", []):
                    topic = DiscoveredTopic(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        source="tavily",
                        summary=result.get("content", "")[:500],
                        score=result.get("score", 0.5),
                        discovered_at=datetime.now().isoformat(),
                        category=category,
                        keywords=self._extract_keywords(result.get("content", ""))
                    )
                    topics.append(topic)

            except Exception as e:
                print(f"Tavily search error: {e}")

        return topics

    def _get_excluded_domains(self) -> list[str]:
        """Get domains to exclude based on filters."""
        exclude_keywords = self.sources.get("filters", {}).get("exclude_keywords", [])
        # Map keywords to common domains
        keyword_domains = {
            "crypto": ["coindesk.com", "cointelegraph.com"],
            "nft": ["opensea.io", "rarible.com"],
            "blockchain": ["ethereum.org"],
        }
        excluded = []
        for keyword in exclude_keywords:
            excluded.extend(keyword_domains.get(keyword, []))
        return excluded

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract relevant keywords from text."""
        boost_keywords = self.sources.get("filters", {}).get("boost_keywords", [])
        found = []
        text_lower = text.lower()
        for keyword in boost_keywords:
            if keyword in text_lower:
                found.append(keyword)
        return found

    def _deduplicate(self, topics: list[DiscoveredTopic]) -> list[DiscoveredTopic]:
        """Remove duplicate topics by URL."""
        seen_urls = set()
        unique = []
        for topic in topics:
            if topic.url not in seen_urls:
                seen_urls.add(topic.url)
                unique.append(topic)
        return unique

    def save_discoveries(self, topics: list[DiscoveredTopic], filename: str = None):
        """Save discovered topics to JSON."""
        if not filename:
            filename = f"discoveries_{datetime.now().strftime('%Y%m%d')}.json"

        output_path = BASE_DIR / "output" / filename
        with open(output_path, "w") as f:
            json.dump([asdict(t) for t in topics], f, indent=2)

        return output_path


def discover_topics(category: str = "ai_tools", count: int = 5) -> list[dict]:
    """
    Convenience function to discover topics.

    Args:
        category: Category to search
        count: Number of results

    Returns:
        List of topic dicts
    """
    discovery = NewsDiscovery()
    topics = discovery.discover(category, count)
    return [asdict(t) for t in topics]
