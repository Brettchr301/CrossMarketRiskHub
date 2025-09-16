"""
T3 — News Pre-emption Pipeline
================================
Polls news sources (GNews API, RSS feeds) for commodity/geopolitical headlines,
classifies them into event types with impact scores, and triggers the event
pipeline when significant events are detected.

Runs as a background thread, polling every 15 minutes.
Posts to internal /v1/events/news endpoint to update prediction market weights.

Follows the provider pattern from app/providers/base.py.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Any, Optional, Sequence

import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

POLL_INTERVAL_SECONDS = 900  # 15 minutes

MONITOR_KEYWORDS = [
    "OPEC",
    "Panama Canal",
    "China stimulus",
    "SPR",
    "Strategic Petroleum Reserve",
    "refinery",
    "Strait of Hormuz",
    "Red Sea",
    "Houthi",
    "oil production",
    "crude oil",
    "Brent",
    "WTI",
    "natural gas",
    "LNG",
    "freight rates",
    "shipping rates",
    "tanker rates",
    "Baltic Dry",
    "contango",
    "backwardation",
]

# Maps headline keywords → EVENT_MAPPINGS keys in real_prediction.py
EVENT_TYPE_RULES: list[dict[str, Any]] = [
    {
        "keywords": ["OPEC", "production cut", "output cut", "barrel"],
        "event_type": "opec_production_cut",
        "base_impact": 4,
    },
    {
        "keywords": ["Panama Canal", "transit", "drought", "canal"],
        "event_type": "panama_canal_disruption",
        "base_impact": 3,
    },
    {
        "keywords": ["China", "stimulus", "yuan", "renminbi", "PBOC"],
        "event_type": "china_stimulus",
        "base_impact": 4,
    },
    {
        "keywords": ["SPR", "Strategic Petroleum Reserve", "reserve release"],
        "event_type": "us_spr_release",
        "base_impact": 3,
    },
    {
        "keywords": ["refinery", "utilization", "refinery outage", "refinery shutdown"],
        "event_type": "us_refinery_utilization_low",
        "base_impact": 3,
    },
    {
        "keywords": ["Hormuz", "strait", "Iran", "blockade"],
        "event_type": "hormuz_closure",
        "base_impact": 5,
    },
    {
        "keywords": ["Red Sea", "Houthi", "Yemen", "shipping attack"],
        "event_type": "red_sea_disruption",
        "base_impact": 4,
    },
    {
        "keywords": ["oil", "Brent", "crude", "WTI", "barrel", "$100"],
        "event_type": "oil_above_100",
        "base_impact": 3,
    },
]

# RSS feed sources (no API key required)
RSS_FEEDS = [
    "https://news.google.com/rss/search?q=OPEC+oil+crude+commodity&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=shipping+freight+tanker+rates&hl=en-US&gl=US&ceid=US:en",
    "https://news.google.com/rss/search?q=Panama+Canal+Strait+Hormuz+Red+Sea&hl=en-US&gl=US&ceid=US:en",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class NewsHeadline:
    """A news headline with metadata."""
    title: str
    source: str
    url: str
    published: datetime
    snippet: str = ""


@dataclass(slots=True)
class ClassifiedEvent:
    """A headline classified into an event type with impact score."""
    headline: NewsHeadline
    event_type: str
    impact_score: int       # 0-5 scale
    confidence: float       # 0.0 to 1.0
    matched_keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# News fetching
# ---------------------------------------------------------------------------

def _fetch_gnews(query: str, max_results: int = 10) -> list[NewsHeadline]:
    """
    Fetch headlines from GNews API (free tier: 100 req/day, no key for basic search).
    Falls back gracefully if API is unavailable.
    """
    api_key = os.environ.get("GNEWS_API_KEY", "")
    headlines: list[NewsHeadline] = []

    try:
        params = {
            "q": query,
            "lang": "en",
            "max": min(max_results, 10),
        }
        if api_key:
            params["apikey"] = api_key
            url = "https://gnews.io/api/v4/search"
        else:
            # Without key, use the free endpoint (limited)
            url = "https://gnews.io/api/v4/top-headlines"
            params["topic"] = "business"

        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            for article in data.get("articles", []):
                try:
                    published = datetime.fromisoformat(
                        article.get("publishedAt", "").replace("Z", "+00:00")
                    ).replace(tzinfo=None)
                except Exception:
                    published = datetime.now(UTC).replace(tzinfo=None)

                headlines.append(NewsHeadline(
                    title=article.get("title", ""),
                    source=article.get("source", {}).get("name", "GNews"),
                    url=article.get("url", ""),
                    published=published,
                    snippet=article.get("description", ""),
                ))
        else:
            logger.debug("GNews API returned %d, falling back to RSS", resp.status_code)

    except Exception as exc:
        logger.debug("GNews fetch failed: %s", exc)

    return headlines


def _fetch_rss_feeds() -> list[NewsHeadline]:
    """Fetch headlines from RSS feeds (always free, no key required)."""
    headlines: list[NewsHeadline] = []

    for feed_url in RSS_FEEDS:
        try:
            resp = requests.get(feed_url, timeout=15, headers={
                "User-Agent": "CrossMarketRiskHub/1.0",
            })
            if resp.status_code != 200:
                continue

            root = ET.fromstring(resp.text)
            channel = root.find("channel")
            if channel is None:
                continue

            for item in channel.findall("item")[:10]:
                title = (item.findtext("title") or "").strip()
                link = (item.findtext("link") or "").strip()
                pub_date_str = (item.findtext("pubDate") or "").strip()
                description = (item.findtext("description") or "").strip()
                source_el = item.find("source")
                source_name = source_el.text if source_el is not None else "RSS"

                # Parse pub date
                try:
                    # RSS date format: "Thu, 07 Mar 2026 15:30:00 GMT"
                    published = datetime.strptime(
                        pub_date_str, "%a, %d %b %Y %H:%M:%S %Z"
                    )
                except Exception:
                    published = datetime.now(UTC).replace(tzinfo=None)

                if title:
                    headlines.append(NewsHeadline(
                        title=title,
                        source=source_name,
                        url=link,
                        published=published,
                        snippet=re.sub(r"<[^>]+>", "", description)[:300],
                    ))

        except Exception as exc:
            logger.debug("RSS fetch failed for %s: %s", feed_url, exc)

    return headlines


def fetch_all_headlines() -> list[NewsHeadline]:
    """Fetch headlines from all configured sources."""
    headlines: list[NewsHeadline] = []

    # Try GNews first with commodity keywords
    for kw in ["OPEC oil crude", "shipping freight rates", "Panama Canal Hormuz"]:
        headlines.extend(_fetch_gnews(kw, max_results=5))

    # Always try RSS feeds as backup/supplement
    headlines.extend(_fetch_rss_feeds())

    # Deduplicate by title similarity
    seen_titles: set[str] = set()
    unique: list[NewsHeadline] = []
    for h in headlines:
        normalized = h.title.lower().strip()[:60]
        if normalized not in seen_titles:
            seen_titles.add(normalized)
            unique.append(h)

    return unique


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------

def _classify_headline_rules(headline: NewsHeadline) -> Optional[ClassifiedEvent]:
    """Classify a headline using rule-based keyword matching."""
    text = f"{headline.title} {headline.snippet}".lower()

    best_match: Optional[ClassifiedEvent] = None
    best_score = 0

    for rule in EVENT_TYPE_RULES:
        matched = [kw for kw in rule["keywords"] if kw.lower() in text]
        if not matched:
            continue

        # Score = base_impact * (number of matching keywords / total keywords)
        match_ratio = len(matched) / len(rule["keywords"])
        score = rule["base_impact"] * match_ratio

        if score > best_score:
            best_score = score
            impact = min(5, max(1, int(score + 0.5)))
            best_match = ClassifiedEvent(
                headline=headline,
                event_type=rule["event_type"],
                impact_score=impact,
                confidence=min(1.0, match_ratio),
                matched_keywords=matched,
            )

    return best_match


def _classify_headline_llm(headline: NewsHeadline) -> Optional[ClassifiedEvent]:
    """Use DeepSeek LLM to classify a headline (higher quality, higher cost)."""
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        return None

    event_types = [r["event_type"] for r in EVENT_TYPE_RULES]
    prompt = f"""Classify this news headline for commodity/energy market impact.

Headline: "{headline.title}"
Snippet: "{headline.snippet}"

Choose the best matching event_type from: {json.dumps(event_types)}
Or "none" if not relevant.

Return JSON only:
{{"event_type": "...", "impact_score": 1-5, "confidence": 0.0-1.0, "reasoning": "..."}}"""

    try:
        resp = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.1,
                "max_tokens": 256,
            },
            timeout=15,
        )
        resp.raise_for_status()
        content = resp.json()["choices"][0]["message"]["content"]
        json_match = re.search(r"\{[^}]+\}", content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            if data.get("event_type") and data["event_type"] != "none":
                return ClassifiedEvent(
                    headline=headline,
                    event_type=data["event_type"],
                    impact_score=min(5, max(0, int(data.get("impact_score", 2)))),
                    confidence=float(data.get("confidence", 0.5)),
                    matched_keywords=[],
                )
    except Exception as exc:
        logger.debug("LLM classification failed: %s", exc)

    return None


def classify_headlines(headlines: Sequence[NewsHeadline], use_llm: bool = False) -> list[ClassifiedEvent]:
    """Classify a batch of headlines. Uses rules by default, LLM if enabled."""
    events: list[ClassifiedEvent] = []

    for headline in headlines:
        # Rule-based classification first (fast, free)
        event = _classify_headline_rules(headline)

        # Try LLM for unclassified headlines if enabled
        if event is None and use_llm:
            event = _classify_headline_llm(headline)

        if event is not None:
            events.append(event)

    return events


# ---------------------------------------------------------------------------
# Event trigger (POST to internal API)
# ---------------------------------------------------------------------------

def _trigger_event_update(event: ClassifiedEvent, base_url: str = "http://127.0.0.1:8100") -> bool:
    """POST classified event to internal /v1/events/news endpoint."""
    try:
        payload = {
            "event_type": event.event_type,
            "impact_score": event.impact_score,
            "confidence": event.confidence,
            "headline": event.headline.title,
            "source": event.headline.source,
            "url": event.headline.url,
            "published": event.headline.published.isoformat(),
            "matched_keywords": event.matched_keywords,
        }
        resp = requests.post(
            f"{base_url}/v1/events/news",
            json=payload,
            timeout=10,
        )
        if resp.status_code in (200, 201, 202):
            logger.info("Event triggered: %s (impact=%d) from '%s'",
                        event.event_type, event.impact_score, event.headline.title[:60])
            return True
        else:
            logger.warning("Event trigger returned %d: %s", resp.status_code, resp.text[:200])
            return False
    except Exception as exc:
        logger.warning("Event trigger failed: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Background polling loop
# ---------------------------------------------------------------------------

class NewsMonitor:
    """
    Background news monitor that polls for commodity/geopolitical headlines,
    classifies them, and triggers event updates when significant.

    Usage:
        monitor = NewsMonitor(base_url="http://127.0.0.1:8100")
        monitor.start()   # starts background thread
        # ... later ...
        monitor.stop()
    """

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:8100",
        poll_interval: int = POLL_INTERVAL_SECONDS,
        impact_threshold: int = 3,
        use_llm: bool = False,
    ):
        self.base_url = base_url
        self.poll_interval = poll_interval
        self.impact_threshold = impact_threshold
        self.use_llm = use_llm
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._seen_urls: set[str] = set()
        self._last_poll: Optional[datetime] = None
        self._events_triggered: int = 0

    def start(self) -> None:
        """Start the background polling thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("NewsMonitor already running")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="news-monitor")
        self._thread.start()
        logger.info("NewsMonitor started (interval=%ds, threshold=%d, llm=%s)",
                     self.poll_interval, self.impact_threshold, self.use_llm)

    def stop(self) -> None:
        """Stop the background polling thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            logger.info("NewsMonitor stopped (total events triggered: %d)", self._events_triggered)

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> dict[str, Any]:
        return {
            "running": self.is_running,
            "last_poll": self._last_poll.isoformat() if self._last_poll else None,
            "events_triggered": self._events_triggered,
            "seen_headlines": len(self._seen_urls),
            "poll_interval_seconds": self.poll_interval,
            "impact_threshold": self.impact_threshold,
        }

    def poll_once(self) -> list[ClassifiedEvent]:
        """Run a single poll cycle. Returns new triggered events."""
        triggered: list[ClassifiedEvent] = []

        try:
            headlines = fetch_all_headlines()
            logger.info("Fetched %d headlines", len(headlines))

            # Filter out already-seen headlines
            new_headlines = [h for h in headlines if h.url not in self._seen_urls]
            for h in new_headlines:
                self._seen_urls.add(h.url)

            if not new_headlines:
                logger.debug("No new headlines")
                return triggered

            # Classify
            events = classify_headlines(new_headlines, use_llm=self.use_llm)
            logger.info("Classified %d events from %d new headlines", len(events), len(new_headlines))

            # Trigger events above threshold
            for event in events:
                if event.impact_score >= self.impact_threshold:
                    success = _trigger_event_update(event, base_url=self.base_url)
                    if success:
                        self._events_triggered += 1
                        triggered.append(event)

        except Exception as exc:
            logger.error("News poll error: %s", exc)

        self._last_poll = datetime.now(UTC).replace(tzinfo=None)
        return triggered

    def _poll_loop(self) -> None:
        """Internal polling loop running in background thread."""
        logger.info("NewsMonitor poll loop started")

        while not self._stop_event.is_set():
            self.poll_once()

            # Wait for next poll, but check stop event periodically
            for _ in range(self.poll_interval):
                if self._stop_event.is_set():
                    break
                time.sleep(1)

        logger.info("NewsMonitor poll loop exiting")


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------

_global_monitor: Optional[NewsMonitor] = None


def get_or_start_monitor(**kwargs: Any) -> NewsMonitor:
    """Get or create the global NewsMonitor singleton."""
    global _global_monitor
    if _global_monitor is None or not _global_monitor.is_running:
        _global_monitor = NewsMonitor(**kwargs)
        _global_monitor.start()
    return _global_monitor
