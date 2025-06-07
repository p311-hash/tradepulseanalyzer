"""
Database models for the TradePulse Signals bot
"""

import os
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, ForeignKey, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship

Base = declarative_base()

class TradingSignal(Base):
    """
    Trading signal model to track generated signals and their outcomes
    """
    __tablename__ = "trading_signals"

    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), unique=True, nullable=False)  # Unique ID for the signal
    pair = Column(String(20), nullable=False)  # Currency pair
    timeframe = Column(String(10), nullable=False)  # Timeframe (15s, 30s, 1m, etc.)
    direction = Column(String(10), nullable=False)  # BUY, SELL, NEUTRAL
    confidence = Column(Float, nullable=False)  # Signal confidence percentage
    entry_price = Column(Float)  # Price at time of signal
    exit_price = Column(Float)  # Price at expiry time
    outcome = Column(String(10))  # WIN, LOSS, DRAW, PENDING
    profit_loss = Column(Float)  # Profit/loss percentage
    expiry_time = Column(DateTime)  # When the signal expires
    generated_at = Column(DateTime, default=datetime.utcnow)
    indicators = Column(Text)  # JSON string of indicators that triggered the signal
    pattern = Column(String(50))  # Candlestick pattern detected
    regime = Column(String(20))  # Market regime (trending, ranging, etc.)
    ml_prediction = Column(String(10))  # ML model prediction
    ml_confidence = Column(Float)  # ML model confidence
    volume_signal = Column(String(10))  # Volume analysis signal
    volume_confidence = Column(Float)  # Volume analysis confidence
    notes = Column(Text)  # Additional notes or observations

    def __repr__(self):
        return f"<TradingSignal(id={self.id}, pair='{self.pair}', direction='{self.direction}', outcome='{self.outcome}')>"


class UserFeedback(Base):
    """
    User feedback on signals
    """
    __tablename__ = "user_feedback"

    id = Column(Integer, primary_key=True)
    signal_id = Column(String(100), ForeignKey("trading_signals.signal_id"), nullable=False)
    user_id = Column(Integer, nullable=False)  # Telegram user ID
    rating = Column(Integer)  # 1-5 rating
    comment = Column(Text)  # User comment
    traded = Column(Boolean, default=False)  # Whether user traded on this signal
    user_outcome = Column(String(10))  # WIN, LOSS as reported by user
    created_at = Column(DateTime, default=datetime.utcnow)

    signal = relationship("TradingSignal", backref="feedback")

    def __repr__(self):
        return f"<UserFeedback(id={self.id}, signal_id='{self.signal_id}', rating={self.rating})>"


class PerformanceMetrics(Base):
    """
    Aggregated performance metrics by timeframe, pair, etc.
    """
    __tablename__ = "performance_metrics"

    id = Column(Integer, primary_key=True)
    metric_type = Column(String(20), nullable=False)  # daily, weekly, monthly, pair, timeframe
    metric_value = Column(String(50), nullable=False)  # The specific pair, timeframe, date etc.
    win_count = Column(Integer, default=0)
    loss_count = Column(Integer, default=0)
    draw_count = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)
    avg_profit = Column(Float, default=0.0)
    avg_loss = Column(Float, default=0.0)
    total_profit = Column(Float, default=0.0)
    profit_factor = Column(Float, default=0.0)  # total_profit / total_loss
    max_consecutive_wins = Column(Integer, default=0)
    max_consecutive_losses = Column(Integer, default=0)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f"<PerformanceMetrics(type='{self.metric_type}', value='{self.metric_value}', win_rate={self.win_rate})>"