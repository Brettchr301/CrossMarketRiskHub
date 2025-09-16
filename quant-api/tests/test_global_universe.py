from __future__ import annotations

from app.modeling.global_universe import GLOBAL_COMMODITY_UNIVERSE, global_universe_tickers


def test_global_universe_has_200_plus_distinct_tickers():
    tickers = global_universe_tickers()
    assert len(tickers) >= 200
    assert len(tickers) == len(set(tickers))
    countries = {row.country for row in GLOBAL_COMMODITY_UNIVERSE}
    assert len(countries) >= 10

