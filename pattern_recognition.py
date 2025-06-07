"""Enhanced Pattern recognition module for candlestick analysis with 9 key patterns."""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

class EnhancedPatternRecognizer:
    """Enhanced class for recognizing 9 key candlestick patterns for binary options trading."""

    def __init__(self, data: pd.DataFrame, tolerance: float = 0.0001):
        """Initialize with price data and tolerance level."""
        self.data = data
        self.tolerance = tolerance
        self.min_body_size = 0.0001  # Minimum body size for pattern validity
        logger.info("EnhancedPatternRecognizer initialized with %d data points", len(data))

    def is_hammer(self, row: pd.Series) -> bool:
        """Identify Hammer pattern."""
        try:
            body = abs(row['open'] - row['close'])
            upper_shadow = row['high'] - max(row['open'], row['close'])
            lower_shadow = min(row['open'], row['close']) - row['low']

            # Hammer criteria
            if lower_shadow > (2 * body) and upper_shadow < body and body > 0:
                logger.debug(f"Hammer pattern detected - Body: {body:.5f}, Lower: {lower_shadow:.5f}, Upper: {upper_shadow:.5f}")
                return True
            logger.debug(f"No hammer pattern - Body: {body:.5f}, Lower: {lower_shadow:.5f}, Upper: {upper_shadow:.5f}")
            return False

        except Exception as e:
            logger.error("Error in hammer recognition: %s", str(e))
            return False

    def is_inverted_hammer(self, row: pd.Series) -> bool:
        """Identify Inverted Hammer pattern."""
        try:
            body = abs(row['open'] - row['close'])
            upper_shadow = row['high'] - max(row['open'], row['close'])
            lower_shadow = min(row['open'], row['close']) - row['low']

            # Inverted Hammer criteria
            if upper_shadow > (2 * body) and lower_shadow < body and body > 0:
                logger.debug(f"Inverted hammer pattern detected - Body: {body:.5f}, Upper: {upper_shadow:.5f}, Lower: {lower_shadow:.5f}")
                return True
            logger.debug(f"No inverted hammer pattern - Body: {body:.5f}, Upper: {upper_shadow:.5f}, Lower: {lower_shadow:.5f}")
            return False

        except Exception as e:
            logger.error("Error in inverted hammer recognition: %s", str(e))
            return False

    def is_tweezer_top(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Tweezer Top pattern."""
        try:
            high_diff = abs(current['high'] - previous['high'])
            if (high_diff <= self.tolerance and
                current['close'] < current['open'] and
                previous['close'] > previous['open']):
                logger.debug(f"Tweezer top pattern detected - High diff: {high_diff:.5f}")
                return True
            logger.debug(f"No tweezer top pattern - High diff: {high_diff:.5f}")
            return False

        except Exception as e:
            logger.error("Error in tweezer top recognition: %s", str(e))
            return False

    def is_tweezer_bottom(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Tweezer Bottom pattern."""
        try:
            low_diff = abs(current['low'] - previous['low'])
            if (low_diff <= self.tolerance and
                current['close'] > current['open'] and
                previous['close'] < previous['open']):
                logger.debug(f"Tweezer bottom pattern detected - Low diff: {low_diff:.5f}")
                return True
            logger.debug(f"No tweezer bottom pattern - Low diff: {low_diff:.5f}")
            return False

        except Exception as e:
            logger.error("Error in tweezer bottom recognition: %s", str(e))
            return False

    def is_doji(self, row: pd.Series) -> bool:
        """Identify Doji pattern (indecision/reversal)."""
        try:
            body = abs(row['open'] - row['close'])
            high_low_range = row['high'] - row['low']

            # Doji criteria: very small body relative to the range
            if high_low_range > 0 and body <= (high_low_range * 0.1):
                logger.debug(f"Doji pattern detected - Body: {body:.5f}, Range: {high_low_range:.5f}")
                return True
            return False
        except Exception as e:
            logger.error("Error in doji recognition: %s", str(e))
            return False

    def is_spinning_top(self, row: pd.Series) -> bool:
        """Identify Spinning Top pattern (indecision)."""
        try:
            body = abs(row['open'] - row['close'])
            upper_shadow = row['high'] - max(row['open'], row['close'])
            lower_shadow = min(row['open'], row['close']) - row['low']

            # Spinning top: small body with long shadows on both sides
            if (body > self.min_body_size and
                upper_shadow > body and lower_shadow > body and
                upper_shadow > body * 1.5 and lower_shadow > body * 1.5):
                logger.debug(f"Spinning top pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in spinning top recognition: %s", str(e))
            return False

    def is_bullish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Bullish Engulfing pattern."""
        try:
            # Previous candle should be bearish
            prev_bearish = previous['close'] < previous['open']
            # Current candle should be bullish
            curr_bullish = current['close'] > current['open']

            # Current candle should engulf previous candle
            engulfs = (current['open'] < previous['close'] and
                      current['close'] > previous['open'])

            if prev_bearish and curr_bullish and engulfs:
                logger.debug("Bullish engulfing pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in bullish engulfing recognition: %s", str(e))
            return False

    def is_bearish_engulfing(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Bearish Engulfing pattern."""
        try:
            # Previous candle should be bullish
            prev_bullish = previous['close'] > previous['open']
            # Current candle should be bearish
            curr_bearish = current['close'] < current['open']

            # Current candle should engulf previous candle
            engulfs = (current['open'] > previous['close'] and
                      current['close'] < previous['open'])

            if prev_bullish and curr_bearish and engulfs:
                logger.debug("Bearish engulfing pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in bearish engulfing recognition: %s", str(e))
            return False

    def is_piercing_line(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Piercing Line pattern (bullish reversal)."""
        try:
            # Previous candle should be bearish
            prev_bearish = previous['close'] < previous['open']
            # Current candle should be bullish
            curr_bullish = current['close'] > current['open']

            # Current should open below previous close and close above midpoint
            prev_midpoint = (previous['open'] + previous['close']) / 2
            piercing = (current['open'] < previous['close'] and
                       current['close'] > prev_midpoint and
                       current['close'] < previous['open'])

            if prev_bearish and curr_bullish and piercing:
                logger.debug("Piercing line pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in piercing line recognition: %s", str(e))
            return False

    def is_dark_cloud_cover(self, current: pd.Series, previous: pd.Series) -> bool:
        """Identify Dark Cloud Cover pattern (bearish reversal)."""
        try:
            # Previous candle should be bullish
            prev_bullish = previous['close'] > previous['open']
            # Current candle should be bearish
            curr_bearish = current['close'] < current['open']

            # Current should open above previous close and close below midpoint
            prev_midpoint = (previous['open'] + previous['close']) / 2
            dark_cloud = (current['open'] > previous['close'] and
                         current['close'] < prev_midpoint and
                         current['close'] > previous['open'])

            if prev_bullish and curr_bearish and dark_cloud:
                logger.debug("Dark cloud cover pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in dark cloud cover recognition: %s", str(e))
            return False

    def is_morning_star(self, candle1: pd.Series, candle2: pd.Series, candle3: pd.Series) -> bool:
        """Identify Morning Star pattern (bullish reversal)."""
        try:
            # First candle: bearish
            first_bearish = candle1['close'] < candle1['open']
            # Second candle: small body (star)
            star_body = abs(candle2['open'] - candle2['close'])
            star_range = candle2['high'] - candle2['low']
            small_star = star_range > 0 and star_body <= (star_range * 0.3)
            # Third candle: bullish
            third_bullish = candle3['close'] > candle3['open']

            # Gap conditions
            gap_down = candle2['high'] < candle1['close']
            gap_up = candle3['open'] > candle2['high']

            # Third candle closes well into first candle's body
            penetration = candle3['close'] > (candle1['open'] + candle1['close']) / 2

            if first_bearish and small_star and third_bullish and penetration:
                logger.debug("Morning star pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in morning star recognition: %s", str(e))
            return False

    def is_evening_star(self, candle1: pd.Series, candle2: pd.Series, candle3: pd.Series) -> bool:
        """Identify Evening Star pattern (bearish reversal)."""
        try:
            # First candle: bullish
            first_bullish = candle1['close'] > candle1['open']
            # Second candle: small body (star)
            star_body = abs(candle2['open'] - candle2['close'])
            star_range = candle2['high'] - candle2['low']
            small_star = star_range > 0 and star_body <= (star_range * 0.3)
            # Third candle: bearish
            third_bearish = candle3['close'] < candle3['open']

            # Gap conditions
            gap_up = candle2['low'] > candle1['close']
            gap_down = candle3['open'] < candle2['low']

            # Third candle closes well into first candle's body
            penetration = candle3['close'] < (candle1['open'] + candle1['close']) / 2

            if first_bullish and small_star and third_bearish and penetration:
                logger.debug("Evening star pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in evening star recognition: %s", str(e))
            return False

    def is_shooting_star(self, row: pd.Series) -> bool:
        """Identify Shooting Star pattern (bearish reversal)."""
        try:
            body = abs(row['open'] - row['close'])
            upper_shadow = row['high'] - max(row['open'], row['close'])
            lower_shadow = min(row['open'], row['close']) - row['low']

            # Shooting star criteria: long upper shadow, small body, little/no lower shadow
            if (upper_shadow > (2 * body) and
                lower_shadow < (body * 0.5) and
                body > self.min_body_size):
                logger.debug("Shooting star pattern detected")
                return True
            return False
        except Exception as e:
            logger.error("Error in shooting star recognition: %s", str(e))
            return False

    def recognize_patterns(self) -> Dict[str, any]:
        """Recognize all 9 key candlestick patterns."""
        try:
            if len(self.data) < 3:
                logger.warning("Insufficient data for pattern recognition (need at least 3 candles)")
                return self._empty_pattern_result()

            # Get the last 3 candles for pattern analysis
            candle3 = self.data.iloc[-1]  # Most recent
            candle2 = self.data.iloc[-2]  # Middle
            candle1 = self.data.iloc[-3]  # Oldest of the three

            logger.debug("Analyzing patterns with 3 candles")

            # Single candle patterns
            patterns = {
                'doji': self.is_doji(candle3),
                'spinning_top': self.is_spinning_top(candle3),
                'shooting_star': self.is_shooting_star(candle3),
                'hammer': self.is_hammer(candle3),
                'inverted_hammer': self.is_inverted_hammer(candle3)
            }

            # Two candle patterns
            patterns.update({
                'bullish_engulfing': self.is_bullish_engulfing(candle3, candle2),
                'bearish_engulfing': self.is_bearish_engulfing(candle3, candle2),
                'piercing_line': self.is_piercing_line(candle3, candle2),
                'dark_cloud_cover': self.is_dark_cloud_cover(candle3, candle2),
                'tweezer_top': self.is_tweezer_top(candle3, candle2),
                'tweezer_bottom': self.is_tweezer_bottom(candle3, candle2)
            })

            # Three candle patterns
            patterns.update({
                'morning_star': self.is_morning_star(candle1, candle2, candle3),
                'evening_star': self.is_evening_star(candle1, candle2, candle3)
            })

            # Calculate pattern significance and direction
            found_patterns = [k for k, v in patterns.items() if v]

            # Determine overall pattern signal
            bullish_patterns = ['morning_star', 'bullish_engulfing', 'piercing_line', 'hammer', 'tweezer_bottom']
            bearish_patterns = ['evening_star', 'shooting_star', 'bearish_engulfing', 'dark_cloud_cover', 'tweezer_top']
            indecision_patterns = ['doji', 'spinning_top']

            bullish_count = sum(1 for p in found_patterns if p in bullish_patterns)
            bearish_count = sum(1 for p in found_patterns if p in bearish_patterns)
            indecision_count = sum(1 for p in found_patterns if p in indecision_patterns)

            # Determine overall signal
            if bullish_count > bearish_count:
                overall_signal = 'BULLISH'
                confidence = min(0.9, 0.5 + (bullish_count * 0.2))
            elif bearish_count > bullish_count:
                overall_signal = 'BEARISH'
                confidence = min(0.9, 0.5 + (bearish_count * 0.2))
            elif indecision_count > 0:
                overall_signal = 'INDECISION'
                confidence = min(0.8, 0.4 + (indecision_count * 0.2))
            else:
                overall_signal = 'NEUTRAL'
                confidence = 0.0

            result = {
                'patterns': patterns,
                'found_patterns': found_patterns,
                'overall_signal': overall_signal,
                'confidence': confidence,
                'pattern_count': len(found_patterns),
                'bullish_count': bullish_count,
                'bearish_count': bearish_count,
                'indecision_count': indecision_count
            }

            if found_patterns:
                logger.info("Found patterns: %s (Signal: %s, Confidence: %.2f)",
                           ", ".join(found_patterns), overall_signal, confidence)
            else:
                logger.info("No patterns detected in current data")

            return result

        except Exception as e:
            logger.error("Error in pattern recognition: %s", str(e))
            return self._empty_pattern_result()

    def _empty_pattern_result(self) -> Dict[str, any]:
        """Return empty pattern result structure."""
        return {
            'patterns': {
                'morning_star': False, 'shooting_star': False, 'evening_star': False,
                'dark_cloud_cover': False, 'doji': False, 'spinning_top': False,
                'bullish_engulfing': False, 'bearish_engulfing': False, 'piercing_line': False,
                'hammer': False, 'inverted_hammer': False, 'tweezer_top': False, 'tweezer_bottom': False
            },
            'found_patterns': [],
            'overall_signal': 'NEUTRAL',
            'confidence': 0.0,
            'pattern_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'indecision_count': 0
        }

    def get_pattern_description(self, pattern_name: str) -> str:
        """Get human-readable description of a pattern."""
        descriptions = {
            'morning_star': 'Morning Star - Strong bullish reversal signal',
            'shooting_star': 'Shooting Star - Bearish reversal at resistance',
            'evening_star': 'Evening Star - Strong bearish reversal signal',
            'dark_cloud_cover': 'Dark Cloud Cover - Bearish reversal pattern',
            'doji': 'Doji - Market indecision, potential reversal',
            'spinning_top': 'Spinning Top - Indecision, weakening trend',
            'bullish_engulfing': 'Bullish Engulfing - Strong bullish reversal',
            'bearish_engulfing': 'Bearish Engulfing - Strong bearish reversal',
            'piercing_line': 'Piercing Line - Bullish reversal pattern',
            'hammer': 'Hammer - Bullish reversal at support',
            'inverted_hammer': 'Inverted Hammer - Potential bullish reversal',
            'tweezer_top': 'Tweezer Top - Bearish reversal at resistance',
            'tweezer_bottom': 'Tweezer Bottom - Bullish reversal at support'
        }
        return descriptions.get(pattern_name, f'{pattern_name} pattern')

class EnhancedPatternValidation:
    """Enhanced pattern validation class for verifying pattern reliability and context."""

    def __init__(self):
        """Initialize pattern validation with default settings."""
        self.min_reliability_score = 0.6
        self.min_volume_threshold = 1.5  # 1.5x average volume
        self.trend_lookback_period = 20
        logger.info("EnhancedPatternValidation initialized")

    def validate_pattern_reliability(self, patterns: List[str], market_regime: Dict = None) -> float:
        """Validate pattern reliability based on market regime and conditions.
        
        Args:
            patterns: List of detected patterns
            market_regime: Current market regime information
            
        Returns:
            float: Reliability score between 0 and 1
        """
        try:
            if not patterns:
                return 0.0

            reliability_score = 0.0
            total_patterns = len(patterns)

            # Base reliability calculation
            for pattern in patterns:
                # Pattern-specific reliability weights
                pattern_weights = {
                    'hammer': 0.8,
                    'inverted_hammer': 0.7,
                    'engulfing': 0.85,
                    'doji': 0.6,
                    'morning_star': 0.9,
                    'evening_star': 0.9,
                    'three_white_soldiers': 0.95,
                    'three_black_crows': 0.95,
                    'spinning_top': 0.5
                }
                
                weight = pattern_weights.get(pattern.lower(), 0.5)
                reliability_score += weight

            # Average reliability
            reliability_score /= total_patterns

            # Adjust based on market regime if available
            if market_regime and 'regime' in market_regime:
                regime_multipliers = {
                    'TRENDING': 1.2,
                    'RANGING': 1.0,
                    'VOLATILE': 0.8,
                    'BREAKOUT': 1.1
                }
                regime_mult = regime_multipliers.get(market_regime['regime'], 1.0)
                reliability_score *= regime_mult

            logger.debug(f"Pattern reliability score: {reliability_score:.2f}")
            return min(max(reliability_score, 0.0), 1.0)  # Ensure score is between 0 and 1

        except Exception as e:
            logger.error(f"Error in pattern reliability validation: {str(e)}")
            return 0.0

    def analyze_pattern_context(self, patterns: List[str], market_data: pd.DataFrame) -> Dict:
        """Analyze market context for pattern validation.
        
        Args:
            patterns: List of detected patterns
            market_data: DataFrame with OHLCV data
            
        Returns:
            Dict with context analysis results
        """
        try:
            if market_data.empty or not patterns:
                return {
                    'volume_confirmed': False,
                    'trend_aligned': False,
                    'support_resistance_validated': False
                }

            # Volume confirmation
            recent_volume = market_data['volume'].iloc[-1]
            avg_volume = market_data['volume'].rolling(window=20).mean().iloc[-1]
            volume_confirmed = recent_volume > (avg_volume * self.min_volume_threshold)

            # Trend alignment
            prices = market_data['close'].iloc[-self.trend_lookback_period:]
            trend = 'UP' if prices.iloc[-1] > prices.mean() else 'DOWN'
            
            # Check if patterns align with trend
            trend_aligned = True
            for pattern in patterns:
                if (pattern in ['hammer', 'morning_star', 'three_white_soldiers'] and trend == 'DOWN') or \
                   (pattern in ['inverted_hammer', 'evening_star', 'three_black_crows'] and trend == 'UP'):
                    trend_aligned = False
                    break

            # Support/Resistance validation
            highs = market_data['high'].rolling(window=20).max()
            lows = market_data['low'].rolling(window=20).min()
            current_price = market_data['close'].iloc[-1]
            
            near_support = abs(current_price - lows.iloc[-1]) / current_price < 0.02
            near_resistance = abs(current_price - highs.iloc[-1]) / current_price < 0.02
            support_resistance_validated = near_support or near_resistance

            return {
                'volume_confirmed': volume_confirmed,
                'trend_aligned': trend_aligned,
                'support_resistance_validated': support_resistance_validated
            }

        except Exception as e:
            logger.error(f"Error in pattern context analysis: {str(e)}")
            return {
                'volume_confirmed': False,
                'trend_aligned': False,
                'support_resistance_validated': False
            }

    def check_pattern_conflicts(self, patterns: List[str]) -> List[str]:
        """Check for conflicting patterns that shouldn't occur together.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            List of conflicting pattern pairs
        """
        conflicts = []
        bullish_patterns = {'hammer', 'morning_star', 'three_white_soldiers'}
        bearish_patterns = {'inverted_hammer', 'evening_star', 'three_black_crows'}
        
        detected_bullish = set(patterns) & bullish_patterns
        detected_bearish = set(patterns) & bearish_patterns
        
        if detected_bullish and detected_bearish:
            conflicts.extend([f"{b} vs {s}" for b in detected_bullish for s in detected_bearish])
            
        return conflicts

    def check_pattern_completion(self, patterns: List[str]) -> List[str]:
        """Check if patterns are fully formed or still developing.
        
        Args:
            patterns: List of detected patterns
            
        Returns:
            List of incomplete patterns
        """
        incomplete = []
        multi_candle_patterns = {
            'morning_star': 3,
            'evening_star': 3,
            'three_white_soldiers': 3,
            'three_black_crows': 3,
            'engulfing': 2
        }
        
        for pattern in patterns:
            if pattern.lower() in multi_candle_patterns:
                # In a real implementation, we would check the actual pattern formation
                # Here we're just identifying which patterns need multiple candles
                incomplete.append(pattern)
                
        return incomplete