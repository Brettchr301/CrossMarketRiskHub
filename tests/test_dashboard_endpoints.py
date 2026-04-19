import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from unittest.mock import MagicMock
import os

from app.election.api.routes import router
from app.election.database import get_election_db
from app.election.models import Race, PriceHistory, ArbitrageData, WeatherData
from fastapi import FastAPI

app = FastAPI()
app.include_router(router, prefix="/v1/election")

client = TestClient(app)

class MockQuery:
    def __init__(self, model, db_mock):
        self.model = model
        self.db_mock = db_mock
        self._filters = []
        self._joins = []

    def filter(self, *args):
        self._filters.extend(args)
        return self

    def join(self, *args):
        self._joins.extend(args)
        return self

    def order_by(self, *args):
        return self

    def first(self):
        res = self.all()
        return res[0] if res else None

    def all(self):
        if self.model == Race:
            cycle_filter = next((f for f in self._filters if 'cycle' in str(f)), None)
            if cycle_filter:
                if '2024' in str(cycle_filter):
                    return [self.db_mock.race1]
                elif '2022' in str(cycle_filter):
                    return [self.db_mock.race2]
                else:
                    return []
            id_filter = next((f for f in self._filters if 'id' in str(f)), None)
            if id_filter and '1' in str(id_filter):
                return [self.db_mock.race1]
            return [self.db_mock.race1, self.db_mock.race2]
            
        elif self.model == PriceHistory:
            cutoff_filter = next((f for f in self._filters if '>=' in str(f) or 'timestamp' in str(f)), None)
            res = [self.db_mock.ph1, self.db_mock.ph2, self.db_mock.ph3]
            if cutoff_filter:
                try:
                    cutoff_dt = cutoff_filter.right.value
                    res = [p for p in res if p.timestamp >= cutoff_dt]
                except:
                    pass
            return res
            
        elif self.model == ArbitrageData:
            in_filter = next((f for f in self._filters if 'in_' in str(f)), None)
            if in_filter:
                try:
                    race_ids = in_filter.right.value
                    if 2 not in race_ids:
                        return []
                except:
                    pass
            return [self.db_mock.arb1, self.db_mock.arb2]
            
        elif self.model == WeatherData:
            cycle_filter = next((f for f in self._filters if 'cycle' in str(f)), None)
            if cycle_filter and '2024' not in str(cycle_filter):
                return []
            return [(self.db_mock.wd1, self.db_mock.race1)]

def override_get_db():
    db = MagicMock()
    now = datetime.utcnow()
    
    db.race1 = Race(id=1, state="PA", office="Senate", cycle=2024, winner_party="D", election_date=datetime(2024, 11, 5))
    db.race2 = Race(id=2, state="AZ", office="Governor", cycle=2022, winner_party="R", election_date=datetime(2022, 11, 8))
    
    db.ph1 = PriceHistory(race_id=1, platform="polymarket", timestamp=now - timedelta(hours=10), price=0.60)
    db.ph2 = PriceHistory(race_id=1, platform="polymarket", timestamp=now - timedelta(hours=2), price=0.65)
    db.ph3 = PriceHistory(race_id=1, platform="kalshi", timestamp=now - timedelta(hours=50), price=0.55)
    
    db.arb1 = ArbitrageData(race_id=2, date=datetime(2022, 10, 1).date(), net_edge_pct=1.5)
    db.arb2 = ArbitrageData(race_id=2, date=datetime(2022, 10, 2).date(), net_edge_pct=2.1)
    
    db.wd1 = WeatherData(race_id=1, state="PA", turnout_score=0.75, market_price=0.60)
    
    def mock_query(model, *args):
        return MockQuery(model, db)
        
    db.query = mock_query
    return db

app.dependency_overrides[get_election_db] = override_get_db

@pytest.fixture(autouse=True)
def setup_html():
    os.makedirs("app/election/dashboard", exist_ok=True)
    with open("app/election/dashboard/index.html", "w") as f:
        f.write("<html><body>Election Alpha Dashboard</body></html>")
    yield

def test_price_history_endpoint():
    response = client.get("/v1/election/chart/price-history/1")
    assert response.status_code == 200
    data = response.json()
    assert "platforms" in data
    assert "polymarket" in data["platforms"]
    assert len(data["platforms"]["polymarket"]) == 2
    assert "ts" in data["platforms"]["polymarket"][0]
    assert "price" in data["platforms"]["polymarket"][0]

def test_arb_heatmap_endpoint():
    response = client.get("/v1/election/chart/arb-heatmap?cycle=2022")
    assert response.status_code == 200
    data = response.json()
    assert "races" in data
    assert "dates" in data
    assert "matrix" in data
    assert len(data["races"]) == 1
    assert len(data["dates"]) == 2
    assert len(data["matrix"]) == 1
    assert len(data["matrix"][0]) == 2

def test_weather_scatter_endpoint():
    response = client.get("/v1/election/chart/weather-scatter/2024")
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) == 1
    assert "state" in data[0]
    assert "turnout_score" in data[0]
    assert "market_price" in data[0]
    assert "race_id" in data[0]
    assert "race_label" in data[0]

def test_dashboard_html_served():
    response = client.get("/v1/election/dashboard/ui")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Election Alpha Dashboard" in response.text

def test_empty_data_graceful():
    res1 = client.get("/v1/election/chart/price-history/9999")
    assert res1.status_code == 200
    assert res1.json()["platforms"] == {}
    
    res2 = client.get("/v1/election/chart/arb-heatmap?cycle=9999")
    assert res2.status_code == 200
    assert res2.json()["matrix"] == []
    
    res3 = client.get("/v1/election/chart/weather-scatter/9999")
    assert res3.status_code == 200
    assert res3.json() == []

def test_price_history_time_filter():
    def override_get_db_24h():
        db = override_get_db()
        def mock_query(model, *args):
            mq = MockQuery(model, db)
            if model == PriceHistory:
                mq.all = lambda: [db.ph1, db.ph2]
            return mq
        db.query = mock_query
        return db
        
    app.dependency_overrides[get_election_db] = override_get_db_24h
    
    response = client.get("/v1/election/chart/price-history/1?hours=24")
    assert response.status_code == 200
    data = response.json()
    assert "kalshi" not in data["platforms"]
    assert "polymarket" in data["platforms"]
    assert len(data["platforms"]["polymarket"]) == 2
    
    app.dependency_overrides[get_election_db] = override_get_db
