"""
PocketOption trading module for binary options trading
"""

from typing import Dict, Optional, Tuple
import logging
from datetime import datetime, timedelta
import asyncio
from BinaryOptionsTools import PocketOption
from sentiment_analysis_fixed import SentimentAnalyzer
from technical_analysis import AdaptiveTechnicalAnalyzer

logger = logging.getLogger(__name__)


class PocketOptionTrader:
    def __init__(self, ssid: str, demo: bool = True):
        """Initialize PocketOption trader with session ID and mode."""
        self.api = PocketOption(ssid, demo)
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = AdaptiveTechnicalAnalyzer()
        
        # Trading parameters
        self.min_payout = 0.75  # Minimum 75% payout
        self.default_expiry = 300  # 5 minutes default
        self.max_risk_per_trade = 0.02  # 2% max risk
        self.min_confidence = 70  # Minimum confidence score to place trade
        
        # Connect to platform
        self.api.connect()
        self.initialize_trading_params()

    def initialize_trading_params(self):
        """Initialize trading parameters based on account balance."""
        balance = self.api.GetBalance()
        self.trade_amount = round(balance * self.max_risk_per_trade, 2)
        logger.info(f"Initialized with balance: {balance}, trade amount: {self.trade_amount}")
    
    def validate_entry(self, symbol: str, direction: str, expiry: int) -> bool:
        """Validate if trade entry meets requirements."""
        # Check minimum payout
        payout = self.api.GetPayout(symbol)
        if payout and payout < self.min_payout:
            logger.warning(f"Payout {payout} below minimum threshold for {symbol}")
            return False
            
        # Verify symbol is available for trading
        try:
            candles = self.api.GetCandles(symbol, 60)  # 1-minute candles
            if not candles:
                logger.error(f"Unable to fetch candles for {symbol}")
                return False
        except Exception as e:
            logger.error(f"Error validating symbol {symbol}: {str(e)}")
            return False
            
        return True
    
    def place_trade(self, symbol: str, direction: str, expiry: int = None) -> Tuple[bool, Optional[str]]:
        """Place a binary option trade."""
        if not expiry:
            expiry = self.default_expiry
            
        if not self.validate_entry(symbol, direction, expiry):
            return False, None
            
        try:
            # Place the trade
            if direction.upper() == "CALL":
                result = self.api.Call(
                    amount=self.trade_amount,
                    active=symbol,
                    expiration=expiry,
                    add_check_win=True
                )
            else:
                result = self.api.Put(
                    amount=self.trade_amount,
                    active=symbol,
                    expiration=expiry,
                    add_check_win=True
                )
                
            if isinstance(result, tuple):
                trade_id = result[1]
                logger.info(f"Trade placed successfully: {direction} {symbol} ID:{trade_id}")
                return True, trade_id
            else:
                logger.error("Trade placement failed")
                return False, None
                
        except Exception as e:
            logger.error(f"Error placing trade: {str(e)}")
            return False, None
    
    def check_trade_result(self, trade_id: str) -> Dict:
        """Check the result of a trade."""
        try:
            result = self.api.CheckWin(trade_id)
            if result:
                profit, status = result
                return {
                    'trade_id': trade_id,
                    'profit': profit,
                    'status': status,
                    'timestamp': datetime.now().isoformat()
                }
            return {
                'trade_id': trade_id,
                'profit': 0,
                'status': 'unknown',
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error checking trade result: {str(e)}")
            return {
                'trade_id': trade_id,
                'profit': 0,
                'status': 'error',
                'timestamp': datetime.now().isoformat()
            }
    
    async def analyze_trading_opportunity(self, symbol: str) -> Optional[Dict]:
        """Analyze trading opportunity using sentiment and technical analysis."""
        try:
            # Get technical data
            candles = self.api.GetCandles(symbol, 60, count=100)  # Get 100 1-minute candles
            if not candles:
                return None
                
            # Convert candles to dataframe format expected by analyzers
            df = self.technical_analyzer.prepare_data(candles)
            
            # Get technical sentiment
            tech_analysis = self.technical_analyzer.analyze_market_state(df)
            if not tech_analysis:
                return None
                
            # Get market sentiment
            sentiment = self.sentiment_analyzer.analyze_market_sentiment(symbol, df)
            
            # Combined analysis
            combined_score = (
                tech_analysis['trend_score'] * 0.7 +
                sentiment['combined_score'] * 0.3
            )
            
            confidence = (
                tech_analysis['confidence'] * 0.7 +
                sentiment['confidence'] * 0.3
            )
            
            if confidence < self.min_confidence:
                return None
                
            # Determine trade direction
            if combined_score > 25:
                direction = "CALL"
            elif combined_score < -25:
                direction = "PUT"
            else:
                return None
                
            return {
                'symbol': symbol,
                'direction': direction,
                'confidence': confidence,
                'score': combined_score,
                'expiry': self.default_expiry,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing opportunity: {str(e)}")
            return None
            
    def close(self):
        """Close connection to platform."""
        try:
            self.api.disconnect()
        except:
            pass
