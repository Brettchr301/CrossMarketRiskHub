from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
from sqlalchemy.orm import Session
from app.election.models import HistoricalQuote
from app.election.database import SessionLocal
from app.election.mappings.race_outcomes import get_race_outcome
from app.election.mappings.direction_detector import detect_direction, normalize_price


@dataclass
class BacktestResult:
    race_id: int
    pnl: float
    sharpe: Optional[float] = None
    max_drawdown: Optional[float] = None
    trades: int = 0


def build_price_panel(race_id: int) -> pd.DataFrame:
    """
    Build a price panel for a given race from HistoricalQuote table.
    
    Returns:
        DataFrame with MultiIndex (timestamp, race_id) and columns
        for each platform price series.
    """
    with SessionLocal() as session:
        quotes = session.query(HistoricalQuote).filter(
            HistoricalQuote.race_id == race_id
        ).all()
        
        if not quotes:
            return pd.DataFrame()
        
        # Group by platform and timestamp
        records = []
        for q in quotes:
            records.append({
                'timestamp': q.timestamp,
                'race_id': q.race_id,
                'platform': q.platform,
                'price': q.price,
                'question': q.question
            })
        
        df = pd.DataFrame(records)
        
        # Pivot to get platform columns
        panel = df.pivot_table(
            index=['timestamp', 'race_id'],
            columns='platform',
            values='price'
        )
        
        # Rename columns to platform_*_price
        panel.columns = [f'{col}_price' for col in panel.columns]
        
        return panel.reset_index('race_id', drop=True)


def backtest_cross_market(race_id: int) -> BacktestResult:
    """
    Backtest a simple cross-market mean reversion strategy.
    Assumes we can simultaneously buy at ask and sell at bid.
    """
    panel = build_price_panel(race_id)
    
    if panel.empty:
        return BacktestResult(race_id=race_id, pnl=0.0)
    
    # Get outcome for settlement
    outcome = get_race_outcome(race_id)
    if not outcome:
        return BacktestResult(race_id=race_id, pnl=0.0)
    
    winner_party = outcome.get("winner_party", "")
    
    # Normalize prices to P(Dem wins) using direction detection
    with SessionLocal() as session:
        normalized_prices = pd.DataFrame(index=panel.index)
        
        for col in panel.columns:
            if col.endswith('_price'):
                platform = col.replace('_price', '')
                
                # Get the question for this platform and race
                quote = session.query(HistoricalQuote).filter_by(
                    race_id=race_id,
                    platform=platform
                ).first()
                
                if quote and quote.question:
                    direction = detect_direction(quote.question)
                    if direction.confidence >= 0.5:
                        # Normalize the entire price series
                        normalized_prices[col] = panel[col].apply(
                            lambda price: normalize_price(price, direction.yes_party)
                        )
                    else:
                        # Skip platform if direction unknown
                        normalized_prices[col] = panel[col]
                else:
                    normalized_prices[col] = panel[col]
            else:
                normalized_prices[col] = panel[col]
    
    # Use normalized prices for trading logic
    price_cols = [col for col in normalized_prices.columns if col.endswith('_price')]
    
    if len(price_cols) < 2:
        return BacktestResult(race_id=race_id, pnl=0.0)
    
    # Simple strategy: when platform A price > platform B price by threshold,
    # sell A and buy B, and vice versa
    pnl = 0.0
    trades = 0
    threshold = 0.05
    
    for i in range(len(normalized_prices) - 1):
        row = normalized_prices.iloc[i]
        
        # Find max and min prices among platforms
        max_platform = row[price_cols].idxmax()
        min_platform = row[price_cols].idxmin()
        
        max_price = row[max_platform]
        min_price = row[min_platform]
        
        if max_price - min_price > threshold:
            # Execute trade: sell high, buy low
            # Settlement is now in terms of P(Dem wins)
            # So final settlement is 1.0 if Democrat wins, 0.0 if Republican wins
            if winner_party == "D":
                final_settlement = 1.0
            else:
                final_settlement = 0.0
            
            # Calculate PnL from the trade
            trade_pnl = (max_price - final_settlement) + (final_settlement - min_price)
            pnl += trade_pnl
            trades += 1
    
    return BacktestResult(
        race_id=race_id,
        pnl=pnl,
        trades=trades
    )


def backtest_outcome_betting(race_id: int) -> BacktestResult:
    """
    Backtest a simple outcome betting strategy.
    Buy at average price, settle at election outcome.
    """
    with SessionLocal() as session:
        quotes = session.query(HistoricalQuote).filter_by(
            race_id=race_id
        ).all()
        
        if not quotes:
            return BacktestResult(race_id=race_id, pnl=0.0)
        
        # Normalize prices using direction detection
        normalized_prices = []
        for quote in quotes:
            if quote.question:
                direction = detect_direction(quote.question)
                if direction.confidence >= 0.5:
                    normalized_price = normalize_price(quote.price, direction.yes_party)
                    normalized_prices.append(normalized_price)
                else:
                    normalized_prices.append(quote.price)
            else:
                normalized_prices.append(quote.price)
        
        avg_price = sum(normalized_prices) / len(normalized_prices) if normalized_prices else 0.0
        
        # Get outcome
        outcome = get_race_outcome(race_id)
        if not outcome:
            return BacktestResult(race_id=race_id, pnl=0.0)
        
        winner_party = outcome.get("winner_party", "")
        
        # Settlement in terms of P(Dem wins)
        if winner_party == "D":
            settlement = 1.0
        else:
            settlement = 0.0
        
        # PnL per share (Always buying the normalized P(Dem) position)
        pnl = settlement - avg_price
        
        return BacktestResult(
            race_id=race_id,
            pnl=pnl,
            trades=len(quotes)
        )
