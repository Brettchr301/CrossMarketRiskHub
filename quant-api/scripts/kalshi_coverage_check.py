"""Step 3: Kalshi coverage check in election DB."""
from sqlalchemy import func, select

from app.election.db.session import get_session_factory
from app.election.db.historical_models import HistoricalQuote

db = get_session_factory()()

total = db.execute(select(func.count(HistoricalQuote.id))).scalar()
kalshi_count = db.execute(
    select(func.count(HistoricalQuote.id)).where(HistoricalQuote.platform == "kalshi")
).scalar()
print(f"Total historical quotes in DB: {total}")
print(f"Total Kalshi quotes: {kalshi_count}")

# By platform
by_platform = db.execute(
    select(HistoricalQuote.platform, func.count(HistoricalQuote.id))
    .group_by(HistoricalQuote.platform)
).all()
print("\nBy platform:")
for p, c in by_platform:
    print(f"  {p}: {c:,}")

if kalshi_count:
    by_market = db.execute(
        select(HistoricalQuote.question, func.count(HistoricalQuote.id))
        .where(HistoricalQuote.platform == "kalshi")
        .group_by(HistoricalQuote.question)
        .order_by(func.count(HistoricalQuote.id).desc())
        .limit(10)
    ).all()
    print("\nTop 10 Kalshi markets by quote count:")
    for q, c in by_market:
        print(f"  {c:,}: {str(q)[:80]}")

db.close()
