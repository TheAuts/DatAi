"""Social sentiment engine — Reddit + 4chan scan + VADER scoring for DatAi.

Graceful degradation: missing Reddit credentials / PRAW / 4chan HTTP / VADER →
stub path so tests and UI do not crash. Normalized scores are auditor-friendly
``[0, 1]``. Both sources are scanned in parallel and surfaced separately.
"""

from __future__ import annotations

import html
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# Reddit credentials (never hardcode secrets)
REDDIT_CLIENT_ID_ENV = "REDDIT_CLIENT_ID"
REDDIT_CLIENT_SECRET_ENV = "REDDIT_CLIENT_SECRET"
REDDIT_USER_AGENT_ENV = "REDDIT_USER_AGENT"

# 4chan public JSON API (no key); boards configurable
FOURCHAN_BOARDS_ENV = "FOURCHAN_BOARDS"
FOURCHAN_USER_AGENT_ENV = "FOURCHAN_USER_AGENT"
FOURCHAN_API_BASE = "https://a.4cdn.org"
DEFAULT_FOURCHAN_BOARDS: tuple[str, ...] = ("biz", "pol")
DEFAULT_FOURCHAN_USER_AGENT = "DatAiSentimentLab/1.0"

DEFAULT_SUBREDDITS: tuple[str, ...] = ("options", "stocks")
MIN_AUTHOR_KARMA = 100
MIN_FOURCHAN_TEXT_LEN = 15
BOT_NAME_MARKERS: tuple[str, ...] = ("bot", "auto", "moderator", "[deleted]", "automoderator")
UOA_PATTERNS: tuple[str, ...] = (
    r"\buoa\b",
    r"unusual\s+options?\s+activity",
    r"unusual\s+flow",
    r"\bdark\s+pool\b",
)
MENTION_SPIKE_Z = 2.0
SENTIMENT_UNSTABLE_MSG = "Sentiment Lab: normalized score unstable (require finite values in [0, 1])."
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_FOURCHAN_SOURCES = frozenset({"4chan", "4chan_stub", "fourchan"})
_REDDIT_SOURCES = frozenset({"praw", "reddit", "reddit_stub", "stub"})


def reddit_credentials_present(
    env: Mapping[str, str] | None = None,
) -> bool:
    """True when client id, secret, and user-agent are non-empty."""
    source = env if env is not None else os.environ
    client_id = str(source.get(REDDIT_CLIENT_ID_ENV, "") or "").strip()
    secret = str(source.get(REDDIT_CLIENT_SECRET_ENV, "") or "").strip()
    agent = str(source.get(REDDIT_USER_AGENT_ENV, "") or "").strip()
    return bool(client_id and secret and agent)


def configured_fourchan_boards(
    env: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """Parse ``FOURCHAN_BOARDS`` (comma-separated); default ``biz,pol``."""
    source = env if env is not None else os.environ
    raw = str(source.get(FOURCHAN_BOARDS_ENV, "") or "").strip()
    if not raw:
        return DEFAULT_FOURCHAN_BOARDS
    boards = tuple(b.strip().lstrip("/").rstrip("/").lower() for b in raw.split(",") if b.strip())
    return boards or DEFAULT_FOURCHAN_BOARDS


def normalize_compound_to_01(compound: Any) -> float:
    """Map VADER compound ``[-1, 1]`` → ``[0, 1]``; non-finite → ``0.5`` (neutral)."""
    try:
        x = float(compound)
    except (TypeError, ValueError):
        return 0.5
    if not np.isfinite(x):
        return 0.5
    return float(np.clip((x + 1.0) * 0.5, 0.0, 1.0))


def clip_sentiment_01(value: Any) -> float:
    """Bound a sentiment score to ``[0, 1]``; non-finite → ``0.5``."""
    try:
        x = float(value)
    except (TypeError, ValueError):
        return 0.5
    if not np.isfinite(x):
        return 0.5
    return float(np.clip(x, 0.0, 1.0))


def score_text_vader(text: str) -> dict[str, float]:
    """Score ``text`` with VADER when installed; else lightweight lexicon stub.

    Returns ``compound`` in ``[-1, 1]`` plus ``score_01`` in ``[0, 1]``.
    """
    body = str(text or "")
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

        raw = SentimentIntensityAnalyzer().polarity_scores(body)
        compound = float(raw.get("compound", 0.0))
    except Exception:
        compound = _lexicon_compound(body)
    return {
        "compound": float(np.clip(compound, -1.0, 1.0)),
        "score_01": normalize_compound_to_01(compound),
    }


def _lexicon_compound(text: str) -> float:
    """Tiny fallback scorer so CI works without VADER downloads."""
    lower = text.lower()
    bull = (
        "bull",
        "calls",
        "moon",
        "breakout",
        "long",
        "buy",
        "rally",
        "squeeze",
        "green",
    )
    bear = (
        "bear",
        "puts",
        "crash",
        "dump",
        "short",
        "sell",
        "tank",
        "red",
        "fade",
    )
    score = 0.0
    for w in bull:
        if w in lower:
            score += 0.25
    for w in bear:
        if w in lower:
            score -= 0.25
    return float(np.clip(score, -1.0, 1.0))


def strip_fourchan_html(raw: str) -> str:
    """Decode 4chan comment HTML to plain text."""
    text = html.unescape(str(raw or ""))
    text = _HTML_TAG_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def text_mentions_ticker(text: str, ticker: str) -> bool:
    """True when ``text`` mentions ``$TICKER`` or a whole-word ticker token."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return False
    body = str(text or "")
    if f"${symbol}" in body.upper():
        return True
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(symbol)}(?![A-Za-z0-9])", body, flags=re.IGNORECASE))


def is_bot_like_author(
    author: str | None,
    *,
    karma: Any = None,
    is_bot: bool = False,
    min_karma: int = MIN_AUTHOR_KARMA,
    allow_anonymous: bool = False,
) -> bool:
    """True for bot-like names, explicit bot flag, or low karma.

    When ``allow_anonymous`` (4chan), empty / Anonymous authors are kept.
    """
    if bool(is_bot):
        return True
    name = str(author or "").strip().lower()
    if allow_anonymous and (not name or name in {"anonymous", "anon"}):
        return False
    if not name or name in {"[deleted]", "none", "null"}:
        return True
    if any(marker in name for marker in BOT_NAME_MARKERS):
        return True
    try:
        k = float(karma) if karma is not None else None
    except (TypeError, ValueError):
        k = None
    if k is not None and np.isfinite(k) and k < float(min_karma):
        return True
    return False


def is_fourchan_noise(post: Mapping[str, Any], *, ticker: str | None = None) -> bool:
    """4chan spam / empty / bot-like trip noise filter."""
    if bool(post.get("is_bot", False)):
        return True
    author = str(post.get("author") or "").strip().lower()
    if author and any(marker in author for marker in BOT_NAME_MARKERS):
        return True
    text = f"{post.get('title', '')} {post.get('body', '')}".strip()
    plain = strip_fourchan_html(text) if "<" in text else text
    if len(plain) < MIN_FOURCHAN_TEXT_LEN:
        return True
    if ticker and not text_mentions_ticker(plain, ticker):
        return True
    return False


def filter_social_posts(
    posts: Sequence[Mapping[str, Any]] | pd.DataFrame,
    *,
    min_karma: int = MIN_AUTHOR_KARMA,
    ticker: str | None = None,
) -> list[dict[str, Any]]:
    """Drop bot-like / low-karma / 4chan-noise posts; return list of dicts."""
    if isinstance(posts, pd.DataFrame):
        rows: Iterable[Mapping[str, Any]] = posts.to_dict(orient="records")
    else:
        rows = posts
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        source = str(row.get("source") or "").strip().lower()
        if source in _FOURCHAN_SOURCES:
            if is_fourchan_noise(row, ticker=ticker):
                continue
            kept.append(dict(row))
            continue
        author = row.get("author")
        karma = row.get("karma", row.get("author_karma", row.get("link_karma")))
        bot_flag = bool(row.get("is_bot", False))
        if is_bot_like_author(
            str(author) if author is not None else None,
            karma=karma,
            is_bot=bot_flag,
            min_karma=min_karma,
        ):
            continue
        kept.append(dict(row))
    return kept


def detect_uoa_mention(text: str) -> bool:
    """True when text mentions Unusual Options Activity / flow shorthand."""
    body = str(text or "").lower()
    return any(re.search(pat, body, flags=re.IGNORECASE) for pat in UOA_PATTERNS)


def mention_spike_flags(
    mention_counts: Any,
    *,
    z_threshold: float = MENTION_SPIKE_Z,
) -> np.ndarray:
    """Flag sudden mention spikes via z-score vs series mean/std."""
    arr = np.asarray(mention_counts, dtype=np.float64)
    if arr.size == 0:
        return np.array([], dtype=bool)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return np.zeros(arr.shape, dtype=bool)
    mu = float(np.mean(finite))
    sigma = float(np.std(finite))
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.zeros(arr.shape, dtype=bool)
    z = (arr - mu) / sigma
    return np.asarray(z >= float(z_threshold), dtype=bool)


def stub_reddit_posts(ticker: str, *, n: int = 12, seed: int = 7) -> list[dict[str, Any]]:
    """Deterministic mock Reddit posts for offline / CI paths."""
    symbol = str(ticker or "SPY").strip().upper() or "SPY"
    rng = np.random.default_rng(int(seed) + sum(ord(c) for c in symbol))
    templates = (
        f"${symbol} looking bullish into earnings, buying calls",
        f"{symbol} flow heavy on puts — cautious near-term",
        f"Unusual options activity in {symbol} today",
        f"{symbol} range-bound; waiting for breakout",
        f"Long {symbol} shares, ignoring noise",
        f"{symbol} dump incoming? shorting the rally",
        f"Quiet tape on {symbol}, no edge",
        f"UOA: large call sweep on {symbol}",
    )
    posts: list[dict[str, Any]] = []
    for i in range(max(int(n), 0)):
        text = templates[i % len(templates)]
        karma = int(rng.integers(120, 5000))
        posts.append(
            {
                "id": f"reddit-stub-{symbol}-{i}",
                "author": f"trader_{i}",
                "karma": karma,
                "is_bot": False,
                "board": DEFAULT_SUBREDDITS[i % len(DEFAULT_SUBREDDITS)],
                "subreddit": DEFAULT_SUBREDDITS[i % len(DEFAULT_SUBREDDITS)],
                "title": text,
                "body": text,
                "score": int(rng.integers(1, 200)),
                "created_utc": 1_700_000_000.0 + float(i) * 3600.0,
                "source": "reddit_stub",
            }
        )
    posts.append(
        {
            "id": f"reddit-stub-{symbol}-bot",
            "author": "MarketNewsBot",
            "karma": 9999,
            "is_bot": True,
            "board": "stocks",
            "subreddit": "stocks",
            "title": f"{symbol} auto update",
            "body": f"{symbol} auto update",
            "score": 1,
            "created_utc": 1_700_000_000.0,
            "source": "reddit_stub",
        }
    )
    posts.append(
        {
            "id": f"reddit-stub-{symbol}-lowkarma",
            "author": "newbie42",
            "karma": 5,
            "is_bot": False,
            "board": "options",
            "subreddit": "options",
            "title": f"{symbol} to the moon",
            "body": f"{symbol} to the moon",
            "score": 1,
            "created_utc": 1_700_000_000.0,
            "source": "reddit_stub",
        }
    )
    return posts


# Backward-compatible alias
stub_social_posts = stub_reddit_posts


def stub_fourchan_posts(
    ticker: str,
    *,
    n: int = 10,
    seed: int = 11,
    boards: Sequence[str] | None = None,
) -> list[dict[str, Any]]:
    """Deterministic mock 4chan catalog posts for offline / CI paths."""
    symbol = str(ticker or "SPY").strip().upper() or "SPY"
    board_list = tuple(boards) if boards else DEFAULT_FOURCHAN_BOARDS
    rng = np.random.default_rng(int(seed) + 17 * sum(ord(c) for c in symbol))
    templates = (
        f"${symbol} calls printing, bulls eating good",
        f"{symbol} is dumping hard, buy puts",
        f"UOA on {symbol} large call sweep",
        f"anyone else long {symbol} into close?",
        f"{symbol} dead money, fade the rally",
        f"biz: {symbol} unusual options activity today",
    )
    posts: list[dict[str, Any]] = []
    for i in range(max(int(n), 0)):
        text = templates[i % len(templates)]
        board = board_list[i % len(board_list)]
        posts.append(
            {
                "id": f"4chan-stub-{board}-{symbol}-{i}",
                "author": "Anonymous",
                "karma": None,
                "is_bot": False,
                "board": board,
                "title": text if i % 3 == 0 else "",
                "body": text,
                "score": int(rng.integers(0, 50)),
                "created_utc": 1_700_100_000.0 + float(i) * 1800.0,
                "source": "4chan_stub",
            }
        )
    # Noise that filters must drop.
    posts.append(
        {
            "id": f"4chan-stub-{symbol}-short",
            "author": "Anonymous",
            "karma": None,
            "is_bot": False,
            "board": board_list[0],
            "title": "",
            "body": "lol",
            "score": 0,
            "created_utc": 1_700_100_000.0,
            "source": "4chan_stub",
        }
    )
    posts.append(
        {
            "id": f"4chan-stub-{symbol}-bot",
            "author": "CatalogBot",
            "karma": None,
            "is_bot": True,
            "board": board_list[0],
            "title": f"{symbol} auto",
            "body": f"{symbol} automated board mirror spam content here",
            "score": 0,
            "created_utc": 1_700_100_000.0,
            "source": "4chan_stub",
        }
    )
    return posts


def _praw_fetch_posts(
    ticker: str,
    *,
    subreddits: Sequence[str] = DEFAULT_SUBREDDITS,
    limit: int = 50,
) -> list[dict[str, Any]] | None:
    """Live Reddit fetch via PRAW. Returns ``None`` on any failure."""
    if not reddit_credentials_present():
        return None
    try:
        import praw  # type: ignore
    except Exception:
        return None
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    try:
        reddit = praw.Reddit(
            client_id=os.environ[REDDIT_CLIENT_ID_ENV].strip(),
            client_secret=os.environ[REDDIT_CLIENT_SECRET_ENV].strip(),
            user_agent=os.environ[REDDIT_USER_AGENT_ENV].strip(),
        )
        reddit.read_only = True
        query = f"${symbol} OR {symbol}"
        collected: list[dict[str, Any]] = []
        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(str(sub_name))
                for submission in sub.search(query, sort="new", time_filter="week", limit=int(limit)):
                    author = getattr(submission, "author", None)
                    author_name = str(getattr(author, "name", "") or "") if author else ""
                    karma = None
                    try:
                        if author is not None:
                            karma = int(getattr(author, "comment_karma", 0) or 0) + int(
                                getattr(author, "link_karma", 0) or 0
                            )
                    except Exception:
                        karma = None
                    title = str(getattr(submission, "title", "") or "")
                    body = str(getattr(submission, "selftext", "") or "")
                    collected.append(
                        {
                            "id": str(getattr(submission, "id", "")),
                            "author": author_name,
                            "karma": karma,
                            "is_bot": False,
                            "board": str(sub_name),
                            "subreddit": str(sub_name),
                            "title": title,
                            "body": body,
                            "score": int(getattr(submission, "score", 0) or 0),
                            "created_utc": float(getattr(submission, "created_utc", 0.0) or 0.0),
                            "source": "praw",
                        }
                    )
            except Exception:
                continue
        return collected
    except Exception:
        return None


def _fourchan_http_get(
    url: str,
    *,
    timeout: float = 8.0,
    opener: Callable[..., Any] | None = None,
) -> Any:
    """GET JSON from 4chan CDN; ``opener`` injectable for tests."""
    agent = str(os.environ.get(FOURCHAN_USER_AGENT_ENV, "") or DEFAULT_FOURCHAN_USER_AGENT).strip()
    req = urllib.request.Request(url, headers={"User-Agent": agent})
    fetch = opener or urllib.request.urlopen
    with fetch(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def parse_fourchan_catalog(
    catalog: Any,
    ticker: str,
    *,
    board: str,
    limit: int = 50,
) -> list[dict[str, Any]]:
    """Extract ticker-mention threads/posts from a 4chan catalog JSON payload."""
    symbol = str(ticker or "").strip().upper()
    collected: list[dict[str, Any]] = []
    if not isinstance(catalog, list):
        return collected
    for page in catalog:
        if not isinstance(page, Mapping):
            continue
        threads = page.get("threads") or []
        if not isinstance(threads, list):
            continue
        for thread in threads:
            if not isinstance(thread, Mapping):
                continue
            title = strip_fourchan_html(str(thread.get("sub") or ""))
            body = strip_fourchan_html(str(thread.get("com") or ""))
            blob = f"{title} {body}".strip()
            if not text_mentions_ticker(blob, symbol):
                continue
            author = str(thread.get("name") or "Anonymous")
            no = thread.get("no")
            collected.append(
                {
                    "id": f"{board}-{no}",
                    "author": author,
                    "karma": None,
                    "is_bot": False,
                    "board": str(board),
                    "title": title,
                    "body": body,
                    "score": int(thread.get("replies") or 0),
                    "created_utc": float(thread.get("time") or 0.0),
                    "source": "4chan",
                }
            )
            if len(collected) >= int(limit):
                return collected
    return collected


def _fourchan_fetch_posts(
    ticker: str,
    *,
    boards: Sequence[str] | None = None,
    limit: int = 50,
    opener: Callable[..., Any] | None = None,
) -> list[dict[str, Any]] | None:
    """Live 4chan catalog fetch. Returns ``None`` on total failure (caller stubs)."""
    symbol = str(ticker or "").strip().upper()
    if not symbol:
        return None
    board_list = tuple(boards) if boards else configured_fourchan_boards()
    collected: list[dict[str, Any]] = []
    any_ok = False
    per_board = max(int(limit) // max(len(board_list), 1), 5)
    for board in board_list:
        url = f"{FOURCHAN_API_BASE}/{board}/catalog.json"
        try:
            catalog = _fourchan_http_get(url, opener=opener)
            any_ok = True
            collected.extend(
                parse_fourchan_catalog(catalog, symbol, board=str(board), limit=per_board)
            )
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError, ValueError, json.JSONDecodeError):
            continue
        except Exception:
            continue
    if not any_ok:
        return None
    return collected


def fetch_reddit_posts(
    ticker: str,
    *,
    force_stub: bool = False,
    limit: int = 50,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Fetch Reddit posts for ``ticker``; stub when credentials/PRAW unavailable."""
    if not force_stub:
        live = _praw_fetch_posts(ticker, limit=limit)
        if live is not None:
            return live
    return stub_reddit_posts(ticker, n=min(max(int(limit), 4), 24), seed=seed)


def fetch_fourchan_posts(
    ticker: str,
    *,
    force_stub: bool = False,
    limit: int = 50,
    seed: int = 11,
    boards: Sequence[str] | None = None,
    opener: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Fetch 4chan posts for ``ticker``; stub when HTTP fails or ``force_stub``."""
    if not force_stub:
        live = _fourchan_fetch_posts(ticker, boards=boards, limit=limit, opener=opener)
        if live is not None:
            return live
    return stub_fourchan_posts(
        ticker,
        n=min(max(int(limit), 4), 24),
        seed=seed,
        boards=boards,
    )


def fetch_social_posts(
    ticker: str,
    *,
    force_stub: bool = False,
    limit: int = 50,
    seed: int = 7,
) -> list[dict[str, Any]]:
    """Fetch Reddit + 4chan posts (combined list). Prefer ``extract_ticker_sentiment``."""
    reddit = fetch_reddit_posts(ticker, force_stub=force_stub, limit=limit, seed=seed)
    fourchan = fetch_fourchan_posts(ticker, force_stub=force_stub, limit=limit, seed=seed + 4)
    return list(reddit) + list(fourchan)


def score_posts(
    posts: Sequence[Mapping[str, Any]],
    *,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Score filtered posts; columns include ``score_01``, ``uoa``, ``compound``."""
    filtered = filter_social_posts(posts, ticker=ticker)
    rows: list[dict[str, Any]] = []
    for post in filtered:
        raw_text = f"{post.get('title', '')} {post.get('body', '')}".strip()
        text = strip_fourchan_html(raw_text) if "<" in raw_text else raw_text
        scored = score_text_vader(text)
        rows.append(
            {
                **post,
                "text": text,
                "compound": scored["compound"],
                "score_01": scored["score_01"],
                "uoa": detect_uoa_mention(text),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "id",
                "author",
                "karma",
                "board",
                "subreddit",
                "text",
                "compound",
                "score_01",
                "uoa",
                "created_utc",
                "source",
            ]
        )
    return pd.DataFrame(rows)


def _source_label(scored: pd.DataFrame) -> str:
    if scored is None or scored.empty or "source" not in scored.columns:
        return "empty"
    vals = {str(v).lower() for v in scored["source"].dropna().tolist()}
    if vals & {"praw", "reddit"}:
        return "praw" if "praw" in vals else "reddit"
    if vals & {"4chan"}:
        return "4chan"
    if vals & {"reddit_stub", "stub"}:
        return "reddit_stub"
    if vals & {"4chan_stub"}:
        return "4chan_stub"
    return sorted(vals)[0] if vals else "empty"


def aggregate_sentiment(scored: pd.DataFrame) -> dict[str, Any]:
    """Aggregate post scores into mean sentiment, buzz, UOA / spike flags."""
    if scored is None or scored.empty or "score_01" not in scored.columns:
        return {
            "ok": True,
            "score_01": 0.5,
            "compound": 0.0,
            "buzz": 0.0,
            "mention_count": 0,
            "uoa_mentions": 0,
            "uoa_flag": False,
            "spike_flag": False,
            "source": "empty",
        }
    scores = pd.to_numeric(scored["score_01"], errors="coerce").to_numpy(dtype=np.float64)
    finite = scores[np.isfinite(scores)]
    mean_01 = float(np.mean(finite)) if finite.size else 0.5
    compounds = (
        pd.to_numeric(scored["compound"], errors="coerce").to_numpy(dtype=np.float64)
        if "compound" in scored.columns
        else (mean_01 * 2.0 - 1.0) * np.ones_like(scores)
    )
    c_finite = compounds[np.isfinite(compounds)]
    mean_c = float(np.mean(c_finite)) if c_finite.size else 0.0
    n = int(finite.size)
    buzz = float(np.clip(np.log1p(n) / np.log1p(100.0), 0.0, 1.0))
    uoa_count = int(scored["uoa"].fillna(False).astype(bool).sum()) if "uoa" in scored.columns else 0
    spike = bool(mention_spike_flags([n, 10, 10, 10, 12])[0]) if n > 0 else False
    return {
        "ok": True,
        "score_01": clip_sentiment_01(mean_01),
        "compound": float(np.clip(mean_c, -1.0, 1.0)),
        "buzz": buzz,
        "mention_count": n,
        "uoa_mentions": uoa_count,
        "uoa_flag": uoa_count > 0,
        "spike_flag": spike or uoa_count > 0,
        "source": _source_label(scored),
    }


def _split_by_source(scored: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if scored is None or scored.empty or "source" not in scored.columns:
        empty = scored.iloc[0:0].copy() if isinstance(scored, pd.DataFrame) else pd.DataFrame()
        return empty, empty.copy()
    src = scored["source"].astype(str).str.lower()
    reddit_mask = src.isin(_REDDIT_SOURCES)
    four_mask = src.isin(_FOURCHAN_SOURCES)
    return scored.loc[reddit_mask].copy(), scored.loc[four_mask].copy()


def combine_source_aggregates(
    reddit_agg: Mapping[str, Any],
    fourchan_agg: Mapping[str, Any],
) -> dict[str, Any]:
    """Mention-weighted combine of Reddit + 4chan aggregates (scores stay in [0, 1])."""
    n_r = int(reddit_agg.get("mention_count") or 0)
    n_f = int(fourchan_agg.get("mention_count") or 0)
    total = n_r + n_f
    if total <= 0:
        score = 0.5
        buzz = 0.0
    else:
        score = (
            float(reddit_agg.get("score_01", 0.5)) * n_r
            + float(fourchan_agg.get("score_01", 0.5)) * n_f
        ) / float(total)
        buzz = float(np.clip(np.log1p(total) / np.log1p(100.0), 0.0, 1.0))
    uoa = int(reddit_agg.get("uoa_mentions") or 0) + int(fourchan_agg.get("uoa_mentions") or 0)
    sources_used = []
    if n_r > 0:
        sources_used.append(str(reddit_agg.get("source") or "reddit"))
    if n_f > 0:
        sources_used.append(str(fourchan_agg.get("source") or "4chan"))
    return {
        "ok": True,
        "score_01": clip_sentiment_01(score),
        "buzz": clip_sentiment_01(buzz),
        "mention_count": total,
        "uoa_mentions": uoa,
        "uoa_flag": uoa > 0,
        "spike_flag": bool(reddit_agg.get("spike_flag")) or bool(fourchan_agg.get("spike_flag")) or uoa > 0,
        "source": "+".join(sources_used) if sources_used else "empty",
        "compound": float(
            np.clip(
                (
                    float(reddit_agg.get("compound", 0.0)) * n_r
                    + float(fourchan_agg.get("compound", 0.0)) * n_f
                )
                / float(total)
                if total
                else 0.0,
                -1.0,
                1.0,
            )
        ),
    }


def extract_ticker_sentiment(
    ticker: str,
    *,
    force_stub: bool = False,
    limit: int = 50,
    seed: int = 7,
    boards: Sequence[str] | None = None,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Scan Reddit + 4chan → filter → score → per-source + combined aggregate."""
    symbol = str(ticker or "").strip().upper() or "SPY"
    reddit_posts = fetch_reddit_posts(symbol, force_stub=force_stub, limit=limit, seed=seed)
    fourchan_posts = fetch_fourchan_posts(
        symbol,
        force_stub=force_stub,
        limit=limit,
        seed=seed + 4,
        boards=boards,
        opener=opener,
    )
    scored = score_posts(list(reddit_posts) + list(fourchan_posts), ticker=symbol)
    reddit_scored, fourchan_scored = _split_by_source(scored)
    reddit_agg = aggregate_sentiment(reddit_scored)
    fourchan_agg = aggregate_sentiment(fourchan_scored)
    combined = combine_source_aggregates(reddit_agg, fourchan_agg)
    combined["ticker"] = symbol
    combined["posts"] = scored
    combined["reddit"] = {**reddit_agg, "posts": reddit_scored}
    combined["fourchan"] = {**fourchan_agg, "posts": fourchan_scored}
    combined["sources"] = {
        "reddit": {
            "score_01": reddit_agg["score_01"],
            "buzz": reddit_agg["buzz"],
            "mention_count": reddit_agg["mention_count"],
            "source": reddit_agg["source"],
            "uoa_flag": reddit_agg["uoa_flag"],
        },
        "fourchan": {
            "score_01": fourchan_agg["score_01"],
            "buzz": fourchan_agg["buzz"],
            "mention_count": fourchan_agg["mention_count"],
            "source": fourchan_agg["source"],
            "uoa_flag": fourchan_agg["uoa_flag"],
            "boards": list(boards) if boards else list(configured_fourchan_boards()),
        },
    }
    combined["live"] = bool(
        reddit_agg.get("source") in {"praw", "reddit"} or fourchan_agg.get("source") == "4chan"
    )
    return combined


def extract_portfolio_sentiment(
    tickers: Sequence[str],
    *,
    force_stub: bool = False,
) -> dict[str, dict[str, Any]]:
    """Map each portfolio ticker → aggregated multi-source sentiment payload."""
    out: dict[str, dict[str, Any]] = {}
    for raw in tickers:
        symbol = str(raw or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = extract_ticker_sentiment(symbol, force_stub=force_stub)
    return out


def audit_sentiment_normalized(scores: Any) -> dict[str, Any]:
    """Auditor: require finite sentiment values in ``[0, 1]`` before UI gauge.

    Non-finite or out-of-range → ``halt_render`` with ``SENTIMENT_UNSTABLE_MSG``.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return {
            "ok": False,
            "message": SENTIMENT_UNSTABLE_MSG,
            "scores_01": np.array([], dtype=np.float64),
            "halt_render": True,
        }
    flat = arr.ravel()
    if not np.all(np.isfinite(flat)):
        return {
            "ok": False,
            "message": SENTIMENT_UNSTABLE_MSG,
            "scores_01": np.array(flat, dtype=np.float64, copy=True),
            "halt_render": True,
        }
    if float(np.min(flat)) < 0.0 or float(np.max(flat)) > 1.0:
        return {
            "ok": False,
            "message": SENTIMENT_UNSTABLE_MSG,
            "scores_01": np.array(flat, dtype=np.float64, copy=True),
            "halt_render": True,
        }
    return {
        "ok": True,
        "message": "ok",
        "scores_01": np.array(flat, dtype=np.float64, copy=True),
        "halt_render": False,
    }


def sentiment_factor_for_montecarlo(score_01: Any) -> dict[str, float]:
    """Map normalized sentiment → MC drift bias + vol multiplier.

    Neutral (0.5) → identity. Bullish → mild positive drift / lower vol.
    Bearish → mild negative drift / higher vol. Factor itself stays in ``[0, 1]``.
    """
    s = clip_sentiment_01(score_01)
    centered = s - 0.5
    drift_bias = float(0.04 * centered)
    vol_mult = float(np.clip(1.0 - 0.20 * centered, 0.80, 1.20))
    return {
        "sentiment_factor": s,
        "drift_bias": drift_bias,
        "vol_mult": vol_mult,
    }
