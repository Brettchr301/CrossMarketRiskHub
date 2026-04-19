from fastapi import APIRouter, Depends, Query
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime, timedelta
import os

from ..database import get_election_db
from ..models import Race, PriceHistory, WeatherData, ArbitrageData

router = APIRouter()

@router.get("/chart/price-history/{race_id}")
def get_price_history_chart(
    race_id: int,
    hours: int = 168,
    db: Session = Depends(get_election_db),
) -> dict:
    """Minute-level price history for charting."""
    race = db.query(Race).filter(Race.id == race_id).first()
    if not race:
        return {
            "race_id": race_id,
            "race_label": "Unknown Race",
            "platforms": {},
            "outcome": {}
        }
    
    cutoff = datetime.utcnow() - timedelta(hours=hours)
    
    price_data = db.query(PriceHistory).filter(
        PriceHistory.race_id == race_id,
        PriceHistory.timestamp >= cutoff
    ).order_by(PriceHistory.timestamp).all()
    
    platforms = {}
    for record in price_data:
        platform = record.platform
        if platform not in platforms:
            platforms[platform] = []
        platforms[platform].append({
            "ts": record.timestamp.isoformat(),
            "price": float(record.price)
        })
    
    return {
        "race_id": race_id,
        "race_label": f"{race.state} {race.office} {race.cycle}",
        "platforms": platforms,
        "outcome": {
            "winner_party": race.winner_party,
            "election_date": race.election_date.isoformat() if race.election_date else None
        }
    }

@router.get("/chart/arb-heatmap")
def get_arb_heatmap(
    cycle: int = 2024,
    db: Session = Depends(get_election_db),
) -> dict:
    """Arbitrage heatmap data."""
    races = db.query(Race).filter(Race.cycle == cycle).all()
    race_labels = [f"{r.state} {r.office}" for r in races]
    race_ids = [r.id for r in races]
    
    arb_data = db.query(ArbitrageData).filter(
        ArbitrageData.race_id.in_(race_ids)
    ).all()
    
    dates = sorted(set(d.date.isoformat() for d in arb_data))
    
    matrix = []
    for race_id in race_ids:
        row = []
        for date in dates:
            match = next((d for d in arb_data if d.race_id == race_id and d.date.isoformat() == date), None)
            row.append(float(match.net_edge_pct) if match else 0.0)
        matrix.append(row)
    
    return {
        "cycle": cycle,
        "races": race_labels,
        "dates": dates,
        "matrix": matrix
    }

@router.get("/chart/weather-scatter/{cycle}")
def get_weather_scatter(
    cycle: int,
    db: Session = Depends(get_election_db),
) -> List[dict]:
    """Weather-price correlation scatter data."""
    weather_data = db.query(
        WeatherData, Race
    ).join(
        Race, WeatherData.race_id == Race.id
    ).filter(
        Race.cycle == cycle
    ).all()
    
    result = []
    for weather, race in weather_data:
        result.append({
            "state": weather.state,
            "turnout_score": float(weather.turnout_score),
            "market_price": float(weather.market_price),
            "race_id": weather.race_id,
            "race_label": f"{race.state} {race.office}"
        })
    
    return result

@router.get("/dashboard/ui", response_class=HTMLResponse)
async def get_dashboard_ui():
    """Serve the dashboard HTML file."""
    dashboard_path = os.path.join(os.path.dirname(__file__), "..", "dashboard", "index.html")
    return FileResponse(dashboard_path)
