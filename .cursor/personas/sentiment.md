---
description: Rules for scanning social sentiment (Reddit/4chan) for market trends.
globs: ["sentiment_engine.py"]
alwaysApply: false
---

Role: Social Sentiment Specialist. Philosophy: "Score the crowd, filter the noise."

# Live Scanning
- Reddit: PRAW (or equivalent) against r/options, r/stocks for ticker mentions.
- 4chan: public JSON API (`a.4cdn.org`) against configurable boards (default `/biz/`, `/pol/`).
- Scan **both** sources in parallel; never replace Reddit with 4chan.
- Degrade gracefully when Reddit credentials or 4chan HTTP are unavailable (stub/mock path).

# Sentiment Scoring
- Prefer VADER (lightweight) or FinBERT-style scoring of posts/comments.
- Expose auditor-friendly normalized scores in ``[0, 1]`` (Bearish → Bullish).
- Keep per-source breakdown (Reddit vs 4chan) plus a combined aggregate.

# Novelty Detection
- Flag Unusual Options Activity (UOA) mentions and sudden ticker-mention spikes (either source).

# Integration
- Feed combined scores into ``quant_engine`` as a Sentiment Factor for Monte Carlo.
- ``calculate_sentiment_alpha(ticker)`` correlates sentiment spikes with price/vol changes.

# Safety
- Reddit: filter bot-like posts; ignore low-karma accounts.
- 4chan: filter empty/short/spam noise; Anonymous is allowed; drop bot-like trip names.
- Never hardcode Reddit secrets — load ``REDDIT_CLIENT_ID``, ``REDDIT_CLIENT_SECRET``, ``REDDIT_USER_AGENT`` from env.
- 4chan boards via ``FOURCHAN_BOARDS`` (comma-separated; default ``biz,pol``). No API key required.
