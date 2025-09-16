from __future__ import annotations

from datetime import date, datetime, UTC, timedelta

from app.modeling.options_implied import OptionsImpliedDistributionModel
from app.providers.base import OptionQuoteRow


def test_options_distribution_inference_works_from_chain():
    now = datetime.now(UTC).replace(tzinfo=None)
    expiry = date.today() + timedelta(days=60)
    chain = [
        OptionQuoteRow("TNK", expiry, 30, "call", 2.2, 2.6, 0.34, 900, now),
        OptionQuoteRow("TNK", expiry, 35, "call", 1.1, 1.4, 0.37, 1100, now),
        OptionQuoteRow("TNK", expiry, 40, "call", 0.6, 0.8, 0.42, 850, now),
    ]
    model = OptionsImpliedDistributionModel(horizon_days=60)
    inferred = model.infer("TNK", chain, spot_price=35.0)
    assert inferred.std_return > 0
    assert inferred.upside_p95 > inferred.downside_p05

