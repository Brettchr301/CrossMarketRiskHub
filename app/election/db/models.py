from sqlalchemy import Column, Integer, String, Float, Boolean, JSON
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class AltDataSignal(Base):
    __tablename__ = "alt_data_signals"
    
    id = Column(Integer, primary_key=True, index=True)
    signal_type = Column(String, index=True)
    state = Column(String, index=True)
    cycle = Column(Integer, index=True)
    race_type = Column(String, index=True)
    candidate = Column(String)
    party = Column(String)
    value = Column(Float)
    metadata_ = Column("metadata", JSON)
