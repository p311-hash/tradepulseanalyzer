"""
Sequence pattern recognition module for binary options trading.
This module detects more complex candlestick patterns beyond single candles.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)

class SequencePatternRecognizer:
    """Class for recognizing advanced multi-candle patterns."""
    
    def __init__(self, data: pd.DataFrame, tolerance: float = 0.0001):
        """
        Initialize with price data and tolerance level.
        
        Args:
            data: DataFrame with OHLCV price data
            tolerance: Tolerance for price comparisons
        """
        self.data = data
        self.tolerance = tolerance
        self._results = {}
        
    def is_doji(self, candle: pd.Series) -> bool:
        """
        Check if a candle is a Doji (open and close prices are very close).
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            True if candle is a Doji, False otherwise
        """
        body_size = abs(candle['close'] - candle['open'])
        candle_range = candle['high'] - candle['low']
        
        if candle_range == 0:
            return False
            
        # Doji has a very small body compared to the total range
        return body_size / candle_range < 0.1
    
    def is_bullish(self, candle: pd.Series) -> bool:
        """
        Check if a candle is bullish (close > open).
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            True if candle is bullish, False otherwise
        """
        return candle['close'] > candle['open']
    
    def is_bearish(self, candle: pd.Series) -> bool:
        """
        Check if a candle is bearish (close < open).
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            True if candle is bearish, False otherwise
        """
        return candle['close'] < candle['open']
    
    def get_body_size(self, candle: pd.Series) -> float:
        """
        Get the size of the candle body.
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            Size of the candle body
        """
        return abs(candle['close'] - candle['open'])
    
    def get_upper_shadow(self, candle: pd.Series) -> float:
        """
        Get the size of the upper shadow.
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            Size of the upper shadow
        """
        if self.is_bullish(candle):
            return candle['high'] - candle['close']
        else:
            return candle['high'] - candle['open']
    
    def get_lower_shadow(self, candle: pd.Series) -> float:
        """
        Get the size of the lower shadow.
        
        Args:
            candle: Pandas Series with candle data
            
        Returns:
            Size of the lower shadow
        """
        if self.is_bullish(candle):
            return candle['open'] - candle['low']
        else:
            return candle['close'] - candle['low']
    
    def is_engulfing(self, current: pd.Series, previous: pd.Series) -> Tuple[bool, bool]:
        """
        Check if current candle is a bullish or bearish engulfing pattern.
        
        Args:
            current: Current candle data
            previous: Previous candle data
            
        Returns:
            Tuple of (is_bullish_engulfing, is_bearish_engulfing)
        """
        # Bullish engulfing
        bullish_engulfing = (
            self.is_bearish(previous) and 
            self.is_bullish(current) and
            current['open'] <= previous['close'] and
            current['close'] > previous['open']
        )
        
        # Bearish engulfing
        bearish_engulfing = (
            self.is_bullish(previous) and 
            self.is_bearish(current) and
            current['open'] >= previous['close'] and
            current['close'] < previous['open']
        )
        
        return bullish_engulfing, bearish_engulfing
    
    def is_morning_star(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a morning star pattern.
        
        Args:
            candles: List of three consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 3:
            return False
            
        first, second, third = candles
        
        # First candle is bearish
        if not self.is_bearish(first):
            return False
            
        # Second candle is small (ideally a doji)
        if self.get_body_size(second) > self.get_body_size(first) * 0.3:
            return False
            
        # Gap down from first to second
        if second['high'] > first['close']:
            return False
            
        # Third candle is bullish
        if not self.is_bullish(third):
            return False
            
        # Third candle closes above the midpoint of the first candle
        first_midpoint = (first['open'] + first['close']) / 2
        if third['close'] < first_midpoint:
            return False
            
        return True
    
    def is_evening_star(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms an evening star pattern.
        
        Args:
            candles: List of three consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 3:
            return False
            
        first, second, third = candles
        
        # First candle is bullish
        if not self.is_bullish(first):
            return False
            
        # Second candle is small (ideally a doji)
        if self.get_body_size(second) > self.get_body_size(first) * 0.3:
            return False
            
        # Gap up from first to second
        if second['low'] < first['close']:
            return False
            
        # Third candle is bearish
        if not self.is_bearish(third):
            return False
            
        # Third candle closes below the midpoint of the first candle
        first_midpoint = (first['open'] + first['close']) / 2
        if third['close'] > first_midpoint:
            return False
            
        return True
    
    def is_three_white_soldiers(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a three white soldiers pattern.
        
        Args:
            candles: List of three consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 3:
            return False
            
        # All three candles must be bullish
        if not all(self.is_bullish(candle) for candle in candles):
            return False
            
        # Each candle should close higher than the previous
        if not (candles[1]['close'] > candles[0]['close'] and candles[2]['close'] > candles[1]['close']):
            return False
            
        # Each candle should open within the body of the previous candle
        if not (candles[1]['open'] > candles[0]['open'] and candles[2]['open'] > candles[1]['open']):
            return False
            
        # Small upper shadows
        for candle in candles:
            if self.get_upper_shadow(candle) > self.get_body_size(candle) * 0.3:
                return False
                
        return True
    
    def is_three_black_crows(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a three black crows pattern.
        
        Args:
            candles: List of three consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 3:
            return False
            
        # All three candles must be bearish
        if not all(self.is_bearish(candle) for candle in candles):
            return False
            
        # Each candle should close lower than the previous
        if not (candles[1]['close'] < candles[0]['close'] and candles[2]['close'] < candles[1]['close']):
            return False
            
        # Each candle should open within the body of the previous candle
        if not (candles[1]['open'] < candles[0]['open'] and candles[2]['open'] < candles[1]['open']):
            return False
            
        # Small lower shadows
        for candle in candles:
            if self.get_lower_shadow(candle) > self.get_body_size(candle) * 0.3:
                return False
                
        return True
    
    def is_bullish_harami(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a bullish harami pattern.
        
        Args:
            candles: List of two consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 2:
            return False
            
        first, second = candles
        
        # First candle is bearish, second is bullish
        if not (self.is_bearish(first) and self.is_bullish(second)):
            return False
            
        # Second candle is contained within the body of the first
        if not (second['high'] <= first['open'] and second['low'] >= first['close']):
            return False
            
        return True
    
    def is_bearish_harami(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a bearish harami pattern.
        
        Args:
            candles: List of two consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 2:
            return False
            
        first, second = candles
        
        # First candle is bullish, second is bearish
        if not (self.is_bullish(first) and self.is_bearish(second)):
            return False
            
        # Second candle is contained within the body of the first
        if not (second['high'] <= first['close'] and second['low'] >= first['open']):
            return False
            
        return True
    
    def is_bullish_kicker(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a bullish kicker pattern.
        
        Args:
            candles: List of two consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 2:
            return False
            
        first, second = candles
        
        # First candle is bearish, second is bullish
        if not (self.is_bearish(first) and self.is_bullish(second)):
            return False
            
        # Second candle opens above first candle's open (gap up)
        if not (second['open'] > first['open']):
            return False
            
        # Second candle should have a strong body
        if not (self.get_body_size(second) > self.get_body_size(first)):
            return False
            
        return True
    
    def is_bearish_kicker(self, candles: List[pd.Series]) -> bool:
        """
        Check if the sequence forms a bearish kicker pattern.
        
        Args:
            candles: List of two consecutive candles
            
        Returns:
            True if pattern is found, False otherwise
        """
        if len(candles) != 2:
            return False
            
        first, second = candles
        
        # First candle is bullish, second is bearish
        if not (self.is_bullish(first) and self.is_bearish(second)):
            return False
            
        # Second candle opens below first candle's open (gap down)
        if not (second['open'] < first['open']):
            return False
            
        # Second candle should have a strong body
        if not (self.get_body_size(second) > self.get_body_size(first)):
            return False
            
        return True
    
    def recognize_patterns(self) -> Dict[str, bool]:
        """
        Recognize all sequence patterns in the latest candles.
        
        Returns:
            Dictionary mapping pattern names to boolean values (True if found)
        """
        try:
            # Ensure we have enough data
            if len(self.data) < 3:
                logger.warning("Not enough data for sequence pattern recognition")
                return {}
                
            # Get the most recent candles
            latest = self.data.iloc[-1]
            previous = self.data.iloc[-2]
            prev_prev = self.data.iloc[-3] if len(self.data) >= 3 else None
            
            # Initialize results
            patterns = {}
            
            # Two-candle patterns
            candles_2 = [previous, latest]
            patterns['bullish_engulfing'], patterns['bearish_engulfing'] = self.is_engulfing(latest, previous)
            patterns['bullish_harami'] = self.is_bullish_harami(candles_2)
            patterns['bearish_harami'] = self.is_bearish_harami(candles_2)
            patterns['bullish_kicker'] = self.is_bullish_kicker(candles_2)
            patterns['bearish_kicker'] = self.is_bearish_kicker(candles_2)
            
            # Three-candle patterns
            if prev_prev is not None:
                candles_3 = [prev_prev, previous, latest]
                patterns['morning_star'] = self.is_morning_star(candles_3)
                patterns['evening_star'] = self.is_evening_star(candles_3)
                patterns['three_white_soldiers'] = self.is_three_white_soldiers(candles_3)
                patterns['three_black_crows'] = self.is_three_black_crows(candles_3)
                
            # Filter out only found patterns
            found_patterns = {name: True for name, found in patterns.items() if found}
            
            if found_patterns:
                pattern_names = ', '.join(found_patterns.keys())
                logger.info(f"Found sequence patterns: {pattern_names}")
            else:
                logger.debug("No sequence patterns detected")
                
            return found_patterns
            
        except Exception as e:
            logger.error(f"Error recognizing sequence patterns: {str(e)}")
            return {}
    
    def generate_pattern_signal(self) -> Dict:
        """
        Generate trading signal based on detected patterns.
        
        Returns:
            Dictionary with signal details
        """
        try:
            patterns = self.recognize_patterns()
            
            if not patterns:
                return {
                    'pattern_signal': 'NEUTRAL',
                    'pattern_confidence': 0,
                    'patterns': {}
                }
                
            # Categorize patterns as bullish or bearish
            bullish_patterns = {
                'bullish_engulfing': 70,
                'morning_star': 80,
                'three_white_soldiers': 85,
                'bullish_harami': 60,
                'bullish_kicker': 75
            }
            
            bearish_patterns = {
                'bearish_engulfing': 70,
                'evening_star': 80,
                'three_black_crows': 85,
                'bearish_harami': 60,
                'bearish_kicker': 75
            }
            
            # Count bullish and bearish patterns
            found_bullish = [p for p in patterns if p in bullish_patterns]
            found_bearish = [p for p in patterns if p in bearish_patterns]
            
            bullish_confidence = sum(bullish_patterns[p] for p in found_bullish) / len(found_bullish) if found_bullish else 0
            bearish_confidence = sum(bearish_patterns[p] for p in found_bearish) / len(found_bearish) if found_bearish else 0
            
            # Generate signal
            if found_bullish and (not found_bearish or bullish_confidence > bearish_confidence):
                signal = 'BUY'
                confidence = bullish_confidence
            elif found_bearish and (not found_bullish or bearish_confidence > bullish_confidence):
                signal = 'SELL'
                confidence = bearish_confidence
            else:
                signal = 'NEUTRAL'
                confidence = 0
                
            result = {
                'pattern_signal': signal,
                'pattern_confidence': confidence,
                'patterns': patterns
            }
            
            logger.info(f"Generated pattern signal: {signal} with {confidence:.1f}% confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error generating pattern signal: {str(e)}")
            return {
                'pattern_signal': 'NEUTRAL',
                'pattern_confidence': 0,
                'patterns': {}
            }
    
    def get_pattern_description(self, pattern_name: str) -> str:
        """
        Get description for a pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Description text
        """
        descriptions = {
            'bullish_engulfing': "Bullish Engulfing: A larger bullish candle engulfs the previous bearish candle, indicating a potential reversal of a downtrend.",
            'bearish_engulfing': "Bearish Engulfing: A larger bearish candle engulfs the previous bullish candle, indicating a potential reversal of an uptrend.",
            'morning_star': "Morning Star: A three-candle pattern with a long bearish candle, followed by a small-bodied candle, and then a bullish candle. Signals a potential bullish reversal.",
            'evening_star': "Evening Star: A three-candle pattern with a long bullish candle, followed by a small-bodied candle, and then a bearish candle. Signals a potential bearish reversal.",
            'three_white_soldiers': "Three White Soldiers: Three consecutive bullish candles with each opening within the previous candle's body and closing higher. Strong bullish signal.",
            'three_black_crows': "Three Black Crows: Three consecutive bearish candles with each opening within the previous candle's body and closing lower. Strong bearish signal.",
            'bullish_harami': "Bullish Harami: A small bullish candle contained within the body of the previous larger bearish candle. Indicates potential bullish reversal.",
            'bearish_harami': "Bearish Harami: A small bearish candle contained within the body of the previous larger bullish candle. Indicates potential bearish reversal.",
            'bullish_kicker': "Bullish Kicker: Extremely strong bullish reversal signal with a bearish candle followed by a bullish candle that gaps up.",
            'bearish_kicker': "Bearish Kicker: Extremely strong bearish reversal signal with a bullish candle followed by a bearish candle that gaps down."
        }
        
        return descriptions.get(pattern_name, "Unknown pattern")
    
    def get_pattern_svg_path(self, pattern_name: str) -> str:
        """
        Get path to SVG image for a pattern.
        
        Args:
            pattern_name: Name of the pattern
            
        Returns:
            Path to SVG image or None if not available
        """
        pattern_images = {
            'bullish_engulfing': 'bullish_engulfing.svg',
            'bearish_engulfing': 'bearish_engulfing.svg',
            'morning_star': 'morning_star.svg',
            'evening_star': 'evening_star.svg',
            'three_white_soldiers': 'white_soldiers.svg',
            'three_black_crows': 'black_crows.svg',
            'bullish_harami': 'bullish_harami.svg',
            'bearish_harami': 'bearish_harami.svg',
            'bullish_kicker': 'bullish_kicker.svg',
            'bearish_kicker': 'bearish_kicker.svg'
        }
        
        image_name = pattern_images.get(pattern_name)
        if not image_name:
            return None
            
        # Check if image exists
        image_path = os.path.join('assets', 'images', image_name)
        if os.path.exists(image_path):
            return image_path
            
        # Fallback to neutral pattern
        return os.path.join('assets', 'images', 'neutral_pattern.svg')