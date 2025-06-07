"""
Signal confirmation system for validating and filtering trading signals.
Implements multiple confirmation layers to reduce false signals.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from market_regime import MarketRegimeDetector
from sentiment_analysis import SentimentAnalyzer
from volume_analysis import VolumeAnalyzer

logger = logging.getLogger(__name__)

class SignalConfirmationSystem:
    def __init__(self):
        self.sentiment_analyzer = SentimentAnalyzer()
        self.regime_detector = None  # Will be initialized with data when needed
        self.min_confidence = 65.0  # Minimum confidence threshold

    def confirm_signal(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """
        Validate and enhance trading signal through multiple confirmation layers.

        Args:
            signal: Original trading signal
            data: Price and indicator data

        Returns:
            Enhanced and confirmed signal
        """
        try:
            # Initialize confirmation metrics
            confirmations = []
            confidence_adjustments = []
            warnings = []

            # Check for false signals first
            false_signal_check = self._check_false_signal_patterns(data)
            timing_check = self._validate_signal_timing(data)

            # Apply false signal risk reduction
            if false_signal_check['false_signal_risk'] > 50:
                warnings.extend(false_signal_check['reasons'])
                confidence_adjustments.append(-false_signal_check['false_signal_risk'] * 0.5)

            # Apply timing validation
            if timing_check['timing_score'] < 70:
                warnings.extend(timing_check['issues'])
                confidence_adjustments.append(-((100 - timing_check['timing_score']) * 0.3))

            # 1. Trend Confirmation
            trend_conf = self._confirm_trend(signal, data)
            if trend_conf['confirmed']:
                confirmations.append('Trend Aligned')
                confidence_adjustments.append(trend_conf['confidence_adjustment'])

            # 2. Volume Confirmation
            volume_conf = self._confirm_volume(signal, data)
            if volume_conf['confirmed']:
                confirmations.append('Volume Supported')
                confidence_adjustments.append(volume_conf['confidence_adjustment'])

            # 3. Pattern Confirmation
            pattern_conf = self._confirm_patterns(signal, data)
            if pattern_conf['confirmed']:
                confirmations.append('Pattern Confirmed')
                confidence_adjustments.append(pattern_conf['confidence_adjustment'])

            # 4. Market Regime Check
            regime_conf = self._check_market_regime(signal, data)
            if regime_conf['confirmed']:
                confirmations.append('Regime Aligned')
                confidence_adjustments.append(regime_conf['confidence_adjustment'])

            # 5. Sentiment Alignment
            sentiment_conf = self._check_sentiment_alignment(signal, data)
            if sentiment_conf['confirmed']:
                confirmations.append('Sentiment Aligned')
                confidence_adjustments.append(sentiment_conf['confidence_adjustment'])

            # 6. False Signal Patterns Check
            false_signal_conf = self._check_false_signal_patterns(data)
            if false_signal_conf['false_signal_risk'] > 50.0:
                confirmations.append('False Signal Risk Detected')
                confidence_adjustments.append(-false_signal_conf['false_signal_risk'] * 0.5)

            # 7. Signal Timing Validation
            timing_conf = self._validate_signal_timing(data)
            if timing_conf['timing_score'] < 50.0:
                confirmations.append('Poor Timing Detected')
                confidence_adjustments.append(-50.0 + timing_conf['timing_score'] * 0.5)

            # Calculate final confidence adjustment
            total_adjustment = sum(confidence_adjustments)
            new_confidence = min(100, signal['confidence'] + total_adjustment)

            # Validate against minimum confidence threshold
            is_confirmed = new_confidence >= self.min_confidence
              # Prepare enhanced signal
            enhanced_signal = signal.copy()
            enhanced_signal.update({
                'confirmed': is_confirmed,
                'confidence': new_confidence,
                'confirmations': confirmations,
                'confirmation_count': len(confirmations),
                'original_confidence': signal['confidence'],
                'confidence_adjustment': total_adjustment,
                'warnings': warnings,
                'false_signal_risk': false_signal_check['false_signal_risk'],
                'timing_score': timing_check['timing_score'],
                'quality_metrics': {
                    'false_signal_patterns': false_signal_check['reasons'],
                    'timing_issues': timing_check['issues'],
                    'warning_count': len(warnings)
                }
            })

            logger.info(f"Signal confirmation complete - Confirmed: {is_confirmed}, "
                       f"Confidence: {new_confidence:.1f}%, "
                       f"Confirmations: {len(confirmations)}")

            return enhanced_signal

        except Exception as e:
            logger.error(f"Error in signal confirmation: {str(e)}")
            return signal

    def _confirm_trend(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Confirm signal aligns with overall trend."""
        try:
            # Calculate trend metrics
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()

            current_price = data['close'].iloc[-1]
            trend_bullish = current_price > sma20.iloc[-1] > sma50.iloc[-1]
            trend_bearish = current_price < sma20.iloc[-1] < sma50.iloc[-1]

            # Check alignment
            if (signal['direction'] == 'BUY' and trend_bullish) or \
               (signal['direction'] == 'SELL' and trend_bearish):
                return {
                    'confirmed': True,
                    'confidence_adjustment': 10.0
                }
            elif (signal['direction'] == 'BUY' and trend_bearish) or \
                 (signal['direction'] == 'SELL' and trend_bullish):
                return {
                    'confirmed': False,
                    'confidence_adjustment': -10.0
                }

            return {
                'confirmed': False,
                'confidence_adjustment': 0.0
            }

        except Exception as e:
            logger.error(f"Error in trend confirmation: {str(e)}")
            return {'confirmed': False, 'confidence_adjustment': 0.0}

    def _confirm_volume(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Confirm signal has volume support."""
        try:
            volume_analyzer = VolumeAnalyzer(data)
            volume_analysis = volume_analyzer.analyze_volume_profile()

            # Check volume criteria
            current_volume = data['volume'].iloc[-1]
            avg_volume = data['volume'].rolling(window=20).mean().iloc[-1]

            if current_volume > avg_volume * 1.5:
                return {
                    'confirmed': True,
                    'confidence_adjustment': 15.0
                }
            elif current_volume < avg_volume * 0.5:
                return {
                    'confirmed': False,
                    'confidence_adjustment': -10.0
                }

            return {
                'confirmed': False,
                'confidence_adjustment': 0.0
            }

        except Exception as e:
            logger.error(f"Error in volume confirmation: {str(e)}")
            return {'confirmed': False, 'confidence_adjustment': 0.0}

    def _confirm_patterns(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Confirm signal with pattern analysis."""
        try:
            # Check if signal has pattern information
            if 'patterns' in signal and signal['patterns']:
                pattern_count = len([p for p in signal['patterns'].values() if p])
                if pattern_count > 0:
                    return {
                        'confirmed': True,
                        'confidence_adjustment': min(20.0, pattern_count * 5.0)
                    }

            return {
                'confirmed': False,
                'confidence_adjustment': 0.0
            }

        except Exception as e:
            logger.error(f"Error in pattern confirmation: {str(e)}")
            return {'confirmed': False, 'confidence_adjustment': 0.0}

    def _check_market_regime(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Check if signal aligns with current market regime."""
        try:
            # Initialize regime detector with data if not already done
            if self.regime_detector is None:
                self.regime_detector = MarketRegimeDetector(data)

            regime = self.regime_detector.detect_regime()

            # Check regime alignment
            if regime['regime'] == 'TRENDING' and signal['direction'] != 'NEUTRAL':
                return {
                    'confirmed': True,
                    'confidence_adjustment': 10.0
                }
            elif regime['regime'] == 'VOLATILE' and signal['confidence'] > 80:
                return {
                    'confirmed': True,
                    'confidence_adjustment': 5.0
                }
            elif regime['regime'] == 'RANGING' and signal['direction'] != 'NEUTRAL':
                return {
                    'confirmed': False,
                    'confidence_adjustment': -5.0
                }

            return {
                'confirmed': False,
                'confidence_adjustment': 0.0
            }

        except Exception as e:
            logger.error(f"Error in regime check: {str(e)}")
            return {'confirmed': False, 'confidence_adjustment': 0.0}

    def _check_sentiment_alignment(self, signal: Dict, data: pd.DataFrame) -> Dict:
        """Check if signal aligns with market sentiment."""
        try:
            sentiment = self.sentiment_analyzer.analyze_market_sentiment(signal.get('pair', 'unknown'), data)

            # Check sentiment alignment
            if (signal['direction'] == 'BUY' and sentiment['overall_sentiment'] == 'BULLISH') or \
               (signal['direction'] == 'SELL' and sentiment['overall_sentiment'] == 'BEARISH'):
                return {
                    'confirmed': True,
                    'confidence_adjustment': sentiment['confidence'] * 0.2  # Scale sentiment confidence
                }
            elif (signal['direction'] == 'BUY' and sentiment['overall_sentiment'] == 'BEARISH') or \
                 (signal['direction'] == 'SELL' and sentiment['overall_sentiment'] == 'BULLISH'):
                return {
                    'confirmed': False,
                    'confidence_adjustment': -sentiment['confidence'] * 0.1
                }

            return {
                'confirmed': False,
                'confidence_adjustment': 0.0
            }

        except Exception as e:
            logger.error(f"Error in sentiment alignment check: {str(e)}")
            return {'confirmed': False, 'confidence_adjustment': 0.0}

    def _check_false_signal_patterns(self, data: pd.DataFrame) -> Dict:
        """
        Check for common false signal patterns.

        Args:
            data: Price and indicator data

        Returns:
            Dictionary with false signal detection results
        """
        try:
            # Initialize result
            false_signal_risk = 0.0
            reasons = []

            # Calculate required indicators
            rsi = data['rsi'] if 'rsi' in data else None
            macd = data['macd'] if 'macd' in data else None
            volume = data['volume'] if 'volume' in data else None

            if rsi is not None and macd is not None and volume is not None:
                # Check for RSI divergence with price
                price_higher = data['close'].iloc[-1] > data['close'].iloc[-2]
                rsi_lower = rsi.iloc[-1] < rsi.iloc[-2]
                if price_higher and rsi_lower:
                    false_signal_risk += 20.0
                    reasons.append("RSI Bearish Divergence")

                # Check for low volume signals
                avg_volume = volume.rolling(window=20).mean().iloc[-1]
                if volume.iloc[-1] < avg_volume * 0.7:
                    false_signal_risk += 15.0
                    reasons.append("Low Volume")

                # Check for overbought/oversold exhaustion
                if rsi.iloc[-1] > 80 or rsi.iloc[-1] < 20:
                    false_signal_risk += 10.0
                    reasons.append("Extreme RSI")

                # Check for choppy price action
                price_range = data['high'].rolling(window=10).max() - data['low'].rolling(window=10).min()
                avg_range = price_range.mean()
                current_range = data['high'].iloc[-1] - data['low'].iloc[-1]
                if current_range < avg_range * 0.5:
                    false_signal_risk += 10.0
                    reasons.append("Choppy Price Action")

                # Check for trend exhaustion
                if abs(macd.iloc[-1]) > abs(macd.iloc[-10:]).mean() * 2:
                    false_signal_risk += 15.0
                    reasons.append("Trend Exhaustion")

            return {
                'false_signal_risk': min(100.0, false_signal_risk),
                'reasons': reasons
            }

        except Exception as e:
            logger.error(f"Error checking false signal patterns: {str(e)}")
            return {'false_signal_risk': 0.0, 'reasons': []}

    def _validate_signal_timing(self, data: pd.DataFrame) -> Dict:
        """
        Validate signal timing using multiple timeframe analysis.

        Args:
            data: Price and indicator data

        Returns:
            Dictionary with timing validation results
        """
        try:
            timing_score = 100.0
            issues = []

            # Check for recent significant moves
            returns = data['close'].pct_change()
            recent_move = returns.iloc[-5:].abs().sum() * 100

            if recent_move > 2.0:  # More than 2% move in last 5 periods
                timing_score -= 20.0
                issues.append("Recent Large Move")

            # Check for trend consistency
            sma20 = data['close'].rolling(window=20).mean()
            sma50 = data['close'].rolling(window=50).mean()

            trend_changing = (sma20.diff().iloc[-1] * sma20.diff().iloc[-2]) < 0
            if trend_changing:
                timing_score -= 15.0
                issues.append("Changing Trend")

            # Check for high volatility
            volatility = returns.rolling(window=20).std() * np.sqrt(252)
            if volatility.iloc[-1] > volatility.mean() * 1.5:
                timing_score -= 15.0
                issues.append("High Volatility")

            # Check for news event proximity (if available)
            # This would require integration with a news API

            return {
                'timing_score': max(0.0, timing_score),
                'issues': issues
            }

        except Exception as e:
            logger.error(f"Error validating signal timing: {str(e)}")
            return {'timing_score': 0.0, 'issues': []}
