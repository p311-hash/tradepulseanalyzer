"""
Trade validation system for MasterTrade Bot
"""

from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime, time

@dataclass
class ValidationResult:
    is_valid: bool
    reason: str
    confidence_adjustment: float = 0.0

class TradeValidator:
    def __init__(self, max_daily_trades: int = 10, 
                 min_confidence: float = 65.0,
                 max_loss_percent: float = 2.0):
        self.max_daily_trades = max_daily_trades
        self.min_confidence = min_confidence
        self.max_loss_percent = max_loss_percent
        self.daily_trades = 0
        self.last_reset = datetime.now().date()
        
    def validate_signal(self, signal_data: Dict) -> ValidationResult:
        """Validate trading signal"""
        # Reset daily counters if needed
        self._check_daily_reset()
        
        # Check trading hours
        if not self._is_valid_trading_time():
            return ValidationResult(
                is_valid=False,
                reason="Outside valid trading hours",
                confidence_adjustment=-100
            )
            
        # Check confidence threshold
        if signal_data.get('confidence', 0) < self.min_confidence:
            return ValidationResult(
                is_valid=False,
                reason=f"Confidence below minimum threshold of {self.min_confidence}%",
                confidence_adjustment=0
            )
            
        # Check daily trade limit
        if self.daily_trades >= self.max_daily_trades:
            return ValidationResult(
                is_valid=False,
                reason=f"Daily trade limit of {self.max_daily_trades} reached",
                confidence_adjustment=0
            )
            
        # Validate price levels
        if not self._validate_price_levels(signal_data):
            return ValidationResult(
                is_valid=False,
                reason="Invalid price levels",
                confidence_adjustment=0
            )
            
        # Signal passed all checks
        self.daily_trades += 1
        return ValidationResult(
            is_valid=True,
            reason="Signal validated successfully",
            confidence_adjustment=0
        )
        
    def validate_risk(self, position_size: float, 
                     account_balance: float,
                     stop_loss: float,
                     entry_price: float) -> ValidationResult:
        """Validate risk parameters"""
        # Calculate potential loss
        potential_loss = abs(stop_loss - entry_price) * position_size
        loss_percent = (potential_loss / account_balance) * 100
        
        if loss_percent > self.max_loss_percent:
            return ValidationResult(
                is_valid=False,
                reason=f"Risk exceeds maximum loss percentage of {self.max_loss_percent}%",
                confidence_adjustment=0
            )
            
        return ValidationResult(
            is_valid=True,
            reason="Risk parameters validated",
            confidence_adjustment=0
        )
        
    def _check_daily_reset(self):
        """Reset daily counters if it's a new day"""
        current_date = datetime.now().date()
        if current_date > self.last_reset:
            self.daily_trades = 0
            self.last_reset = current_date
            
    def _is_valid_trading_time(self) -> bool:
        """Check if current time is within valid trading hours"""
        current_time = datetime.now().time()
        
        # Define trading sessions (can be customized)
        london_session = (time(8, 0), time(16, 30))  # London hours
        ny_session = (time(13, 30), time(22, 0))     # New York hours
        
        # Check if current time is in either session
        in_london = london_session[0] <= current_time <= london_session[1]
        in_ny = ny_session[0] <= current_time <= ny_session[1]
        
        return in_london or in_ny
        
    def _validate_price_levels(self, signal_data: Dict) -> bool:
        """Validate price levels are reasonable"""
        entry = signal_data.get('entry_price', 0)
        sl = signal_data.get('stop_loss', 0)
        tp = signal_data.get('take_profit', 0)
        
        if not all([entry, sl, tp]):
            return False
            
        # Validate stop loss and take profit distances
        sl_distance = abs(entry - sl) / entry
        tp_distance = abs(entry - tp) / entry
        
        # Check if distances are reasonable (customizable)
        if sl_distance < 0.0001 or sl_distance > 0.05:  # 1-500 pips
            return False
        if tp_distance < 0.0001 or tp_distance > 0.1:   # 1-1000 pips
            return False
            
        return True
