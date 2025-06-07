"""
Enhanced signal validation and filtering module for TradePulseAnalyzer
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    # Use basic alternatives when talib is not available
import logging
from market_regime import MarketRegimeDetector
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    is_valid: bool
    confidence_adjustment: float
    reason: str
    additional_metrics: Dict

class SignalValidator:
    """Advanced signal validation and filtering system"""
    
    def __init__(self, data: pd.DataFrame, timeframes: List[str]):
        self.data = data
        self.timeframes = timeframes
        self.regime_detector = MarketRegimeDetector(data)
        
    def validate_signal(self, signal: Dict, min_confidence: float = 65.0) -> ValidationResult:
        """
        Comprehensive signal validation using multiple criteria
        
        Args:
            signal: Dictionary containing signal information
            min_confidence: Minimum confidence threshold
            
        Returns:
            ValidationResult object with validation details
        """
        try:
            # Initialize validation metrics
            confidence_adjustments = []
            validation_reasons = []
            metrics = {}
            
            # 1. Volume Profile Analysis
            volume_validation = self._validate_volume_profile()
            confidence_adjustments.append(volume_validation['confidence_adj'])
            validation_reasons.append(volume_validation['reason'])
            metrics['volume_score'] = volume_validation['score']
            
            # 2. Price Action Pattern Validation
            pattern_validation = self._validate_price_patterns(signal['direction'])
            confidence_adjustments.append(pattern_validation['confidence_adj'])
            validation_reasons.append(pattern_validation['reason'])
            metrics['pattern_score'] = pattern_validation['score']
            
            # 3. Market Structure Analysis
            structure_validation = self._analyze_market_structure(signal['direction'])
            confidence_adjustments.append(structure_validation['confidence_adj'])
            validation_reasons.append(structure_validation['reason'])
            metrics['structure_score'] = structure_validation['score']
            
            # 4. Cross-timeframe Confirmation
            timeframe_validation = self._validate_multiple_timeframes(signal['direction'])
            confidence_adjustments.append(timeframe_validation['confidence_adj'])
            validation_reasons.append(timeframe_validation['reason'])
            metrics['timeframe_alignment'] = timeframe_validation['score']
            
            # Calculate final confidence adjustment
            total_adjustment = sum(confidence_adjustments)
            base_confidence = signal.get('confidence', 50.0)
            final_confidence = base_confidence + total_adjustment
            
            # Determine if signal is valid
            is_valid = final_confidence >= min_confidence
            
            # Compile validation reasons
            valid_reasons = [r for r in validation_reasons if r]
            final_reason = " | ".join(valid_reasons) if valid_reasons else "No specific validation issues"
            
            return ValidationResult(
                is_valid=is_valid,
                confidence_adjustment=total_adjustment,
                reason=final_reason,
                additional_metrics=metrics
            )
            
        except Exception as e:
            logger.error(f"Error in signal validation: {str(e)}")
            return ValidationResult(
                is_valid=False,
                confidence_adjustment=-20.0,
                reason=f"Validation error: {str(e)}",
                additional_metrics={}
            )
    
    def _validate_volume_profile(self) -> Dict:
        """Analyze volume profile for signal confirmation"""
        try:
            if 'volume' not in self.data.columns:
                return {'confidence_adj': 0, 'reason': '', 'score': 0.5}
            
            # Get recent volume data
            recent_volume = self.data['volume'].tail(20)
            avg_volume = recent_volume.mean()
            current_volume = recent_volume.iloc[-1]
            
            # Calculate volume trend
            volume_sma = pd.Series(self.data['volume'].values, timeperiod=20)
            volume_trend = volume_sma[-1] > volume_sma[-2]
            
            # Volume analysis
            volume_ratio = current_volume / avg_volume
            
            if volume_ratio > 1.5 and volume_trend:
                return {
                    'confidence_adj': 10,
                    'reason': 'Strong volume confirmation',
                    'score': 0.9
                }
            elif volume_ratio > 1.2:
                return {
                    'confidence_adj': 5,
                    'reason': 'Above average volume',
                    'score': 0.7
                }
            elif volume_ratio < 0.8:
                return {
                    'confidence_adj': -5,
                    'reason': 'Below average volume',
                    'score': 0.3
                }
            
            return {'confidence_adj': 0, 'reason': 'Average volume', 'score': 0.5}
            
        except Exception as e:
            logger.error(f"Error in volume validation: {str(e)}")
            return {'confidence_adj': 0, 'reason': '', 'score': 0.5}
    
    def _validate_price_patterns(self, direction: str) -> Dict:
        """Validate price action patterns"""
        try:
            patterns = []
            scores = []
            
            # Check for candlestick patterns
            high = self.data['high'].values
            low = self.data['low'].values
            open = self.data['open'].values
            close = self.data['close'].values
            
            # Bullish patterns
            if direction == 'UP':
                hammer = talib.CDLHAMMER(open, high, low, close)
                morning_star = talib.CDLMORNINGSTAR(open, high, low, close)
                piercing = talib.CDLPIERCING(open, high, low, close)
                
                if hammer[-1] > 0:
                    patterns.append('Hammer')
                    scores.append(0.8)
                if morning_star[-1] > 0:
                    patterns.append('Morning Star')
                    scores.append(0.9)
                if piercing[-1] > 0:
                    patterns.append('Piercing Pattern')
                    scores.append(0.7)
                    
            # Bearish patterns
            elif direction == 'DOWN':
                shooting_star = talib.CDLSHOOTINGSTAR(open, high, low, close)
                evening_star = talib.CDLEVENINGSTAR(open, high, low, close)
                dark_cloud = talib.CDLDARKCLOUDCOVER(open, high, low, close)
                
                if shooting_star[-1] > 0:
                    patterns.append('Shooting Star')
                    scores.append(0.8)
                if evening_star[-1] > 0:
                    patterns.append('Evening Star')
                    scores.append(0.9)
                if dark_cloud[-1] > 0:
                    patterns.append('Dark Cloud Cover')
                    scores.append(0.7)
            
            if patterns:
                avg_score = sum(scores) / len(scores)
                return {
                    'confidence_adj': 10 * avg_score,
                    'reason': f"Confirmed by patterns: {', '.join(patterns)}",
                    'score': avg_score
                }
            
            return {
                'confidence_adj': 0,
                'reason': 'No significant patterns detected',
                'score': 0.5
            }
            
        except Exception as e:
            logger.error(f"Error in pattern validation: {str(e)}")
            return {'confidence_adj': 0, 'reason': '', 'score': 0.5}
    
    def _analyze_market_structure(self, direction: str) -> Dict:
        """Analyze market structure including S/R levels and trends"""
        try:
            # Calculate key levels
            high = self.data['high'].values
            low = self.data['low'].values
            close = self.data['close'].values
            
            # Pivot points
            pivot = (high[-1] + low[-1] + close[-1]) / 3
            r1 = 2 * pivot - low[-1]
            s1 = 2 * pivot - high[-1]
            
            current_price = close[-1]
            
            # Trend analysis
            sma20 = pd.Series(close, timeperiod=20)
            sma50 = pd.Series(close, timeperiod=50)
            
            # Structure analysis
            structure_score = 0.5
            reason = []
            
            if direction == 'UP':
                if current_price > sma20[-1] > sma50[-1]:
                    structure_score += 0.2
                    reason.append("Bullish moving average alignment")
                if current_price < r1:
                    structure_score += 0.2
                    reason.append("Room to resistance")
                    
            else:  # DOWN
                if current_price < sma20[-1] < sma50[-1]:
                    structure_score += 0.2
                    reason.append("Bearish moving average alignment")
                if current_price > s1:
                    structure_score += 0.2
                    reason.append("Room to support")
            
            confidence_adj = (structure_score - 0.5) * 20
            
            return {
                'confidence_adj': confidence_adj,
                'reason': ' | '.join(reason) if reason else "Neutral market structure",
                'score': structure_score
            }
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {str(e)}")
            return {'confidence_adj': 0, 'reason': '', 'score': 0.5}
    
    def _validate_multiple_timeframes(self, direction: str) -> Dict:
        """Validate signal across multiple timeframes"""
        try:
            alignments = []
            
            # Analyze trend direction on each timeframe
            for tf in self.timeframes:
                # Get trend direction using multiple indicators
                close = self.data['close'].values
                
                # EMA trend
                ema_fast = pd.Series(close, timeperiod=12)
                ema_slow = pd.Series(close, timeperiod=26)
                ema_trend = ema_fast[-1] > ema_slow[-1]
                
                # MACD trend
                macd, signal, _ = talib.MACD(close)
                macd_trend = macd[-1] > signal[-1]
                
                # RSI trend
                rsi = pd.Series(close)
                rsi_trend = rsi[-1] > 50
                
                # Calculate alignment score
                if direction == 'UP':
                    aligned = sum([ema_trend, macd_trend, rsi_trend])
                else:
                    aligned = sum([not ema_trend, not macd_trend, not rsi_trend])
                
                alignments.append(aligned / 3)
            
            # Calculate average alignment
            avg_alignment = sum(alignments) / len(alignments)
            
            if avg_alignment >= 0.8:
                return {
                    'confidence_adj': 15,
                    'reason': 'Strong multi-timeframe confirmation',
                    'score': avg_alignment
                }
            elif avg_alignment >= 0.6:
                return {
                    'confidence_adj': 8,
                    'reason': 'Moderate multi-timeframe confirmation',
                    'score': avg_alignment
                }
            elif avg_alignment <= 0.3:
                return {
                    'confidence_adj': -10,
                    'reason': 'Poor multi-timeframe alignment',
                    'score': avg_alignment
                }
            
            return {
                'confidence_adj': 0,
                'reason': 'Mixed timeframe signals',
                'score': avg_alignment
            }
            
        except Exception as e:
            logger.error(f"Error in timeframe validation: {str(e)}")
            return {'confidence_adj': 0, 'reason': '', 'score': 0.5}
