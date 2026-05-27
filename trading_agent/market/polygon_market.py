from __future__ import annotations

import os
from datetime import datetime, timedelta

import pandas as pd
from market_cgan.data.polygon import PolygonDataSource


def fetch_prices(
    symbol: str,
    start_date: str | None = None,
    end_date: str | None = None,
    timespan: str = "day",
) -> pd.Series:
    api_key = os.environ.get("POLYGON_API_KEY", "")
    if not api_key:
        raise ValueError("POLYGON_API_KEY environment variable is required")
    end = end_date or datetime.today().strftime("%Y-%m-%d")
    start = start_date or (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")
    source = PolygonDataSource(api_key)
    bars = source.fetch_aggregates(symbol, start, end, timespan)
    return pd.Series([b.close for b in bars])