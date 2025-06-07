"""
Enhanced technical analysis module with adaptive algorithms and ML integration.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Tuple, Any
import talib_compat as talib
from sklearn.preprocessing import StandardScaler
import logging
from datetime import datetime
import json
import torch
from concurrent.futures import ThreadPoolExecutor
from enhanced_feature_engineering import EnhancedFeatureEngineer
from enhanced_ml_model import EnhancedMLPredictor
import config

logger = logging.getLogger(__name__)
import numpy as np
import logging
from typing import Dict, Optional, List, Tuple

# Configure logging
logger = logging.getLogger(__name__)

class TechnicalAnalyzer:
    """Technical analysis class for calculating indicators and generating signals"""

    def __init__(self, data=None):
        """Initialize with optional data"""
        self.data = data

    def analyze(self, data=None):
        """Run full analysis on data"""
        if data is not None:
            self.data = data

        if self.data is None or self.data.empty:
            logger.warning("No data provided for analysis")
            return {}

        # Calculate indicators
        self.data = calculate_indicators(self.data)

        # Analyze trend
        trend = analyze_trend(self.data)

        # Calculate support/resistance
        levels = calculate_support_resistance(self.data)

        # Detect patterns
        patterns = detect_patterns(self.data)

        # Generate signal
        signal = generate_signals(self.data)

        return {
            'trend': trend,
            'support_resistance': levels,
            'patterns': patterns,
            'signal': signal
        }

class AdaptiveTechnicalAnalyzer:
    """Advanced technical analysis with adaptive algorithms and ML integration."""

    def __init__(self,
                 model_path: Optional[str] = None,
                 use_ml: bool = True,
                 use_ensemble: bool = True,
                 feature_selection: bool = True):
        """
        Initialize the technical analyzer.

        Args:
            model_path: Path to saved ML model
            use_ml: Whether to use ML predictions
            use_ensemble: Whether to use ensemble predictions
            feature_selection: Whether to use adaptive feature selection
        """
        self.feature_engineer = EnhancedFeatureEngineer(
            use_pca=True,
            pca_components=15
        )

        if use_ml:
            # Initialize ML predictor with actual feature count
            feature_engineer = FeatureEngineer()
            feature_count = len(feature_engineer.get_feature_names())
            self.ml_predictor = EnhancedMLPredictor(
                input_size=feature_count,  # Use actual feature count
                model_path=model_path
            )
        else:
            self.ml_predictor = None

        self.use_ensemble = use_ensemble
        self.feature_selection = feature_selection
        self.market_state = {}
        self.performance_metrics = {
            'technical_accuracy': 0.0,
            'ml_accuracy': 0.0,
            'ensemble_accuracy': 0.0
        }

    def analyze(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive technical analysis with ML integration.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            Dictionary with analysis results
        """
        try:
            # Update market state
            self._update_market_state(data)

            # Engineer features
            features_df = self.feature_engineer.engineer_features(data)

            # Generate signals in parallel
            with ThreadPoolExecutor() as executor:
                futures = {
                    'technical': executor.submit(self._generate_technical_signals, features_df),
                    'patterns': executor.submit(self._analyze_patterns, features_df),
                    'volume': executor.submit(self._analyze_volume, features_df)
                }

                if self.ml_predictor:
                    futures['ml'] = executor.submit(
                        self._generate_ml_signals,
                        features_df
                    )

            # Collect results
            results = {
                name: future.result()
                for name, future in futures.items()
            }

            # Combine signals with adaptive weighting
            final_signal = self._combine_signals(results)

            # Prepare comprehensive analysis result
            analysis = {
                'signal': final_signal['signal'],
                'confidence': final_signal['confidence'],
                'market_state': self.market_state,
                'technical_signals': results['technical'],
                'pattern_signals': results['patterns'],
                'volume_analysis': results['volume'],
                'performance_metrics': self.performance_metrics
            }

            if self.ml_predictor:
                analysis['ml_signals'] = results['ml']

            return analysis

        except Exception as e:
            logger.error(f"Error in technical analysis: {str(e)}")
            return self._generate_safe_fallback()

    def _update_market_state(self, data: pd.DataFrame) -> None:
        """Update internal market state tracking."""
        try:
            latest = data.iloc[-1]
            recent = data.iloc[-20:]

            # Volatility state
            volatility = recent['close'].pct_change().std() * np.sqrt(252)
            volatility_regime = 'HIGH' if volatility > 0.2 else 'LOW'

            # Trend state
            sma_20 = talib.SMA(data['close'], timeperiod=20)
            sma_50 = talib.SMA(data['close'], timeperiod=50)
            trend = 'UP' if sma_20.iloc[-1] > sma_50.iloc[-1] else 'DOWN'

            # Volume state
            volume_sma = data['volume'].rolling(20).mean()
            volume_state = 'HIGH' if latest['volume'] > volume_sma.iloc[-1] * 1.5 else 'NORMAL'

            # Market regime detection
            adx = talib.ADX(data['high'], data['low'], data['close'])
            trending = adx.iloc[-1] > 25

            self.market_state = {
                'volatility': {
                    'value': volatility,
                    'regime': volatility_regime
                },
                'trend': {
                    'direction': trend,
                    'strength': adx.iloc[-1] / 100
                },
                'volume': {
                    'state': volume_state,
                    'ratio': latest['volume'] / volume_sma.iloc[-1]
                },
                'regime': 'TRENDING' if trending else 'RANGING'
            }

        except Exception as e:
            logger.error(f"Error updating market state: {str(e)}")
            self.market_state = {}

    def _generate_technical_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate technical analysis signals."""
        try:
            # Moving average signals
            ma_signals = self._analyze_moving_averages(data)

            # Momentum signals
            momentum_signals = self._analyze_momentum(data)

            # Volatility signals
            volatility_signals = self._analyze_volatility(data)

            # Combine signals with adaptive weights
            weights = self._calculate_signal_weights()

            technical_score = (
                weights['ma'] * ma_signals['score'] +
                weights['momentum'] * momentum_signals['score'] +
                weights['volatility'] * volatility_signals['score']
            )

            return {
                'signal': self._score_to_signal(technical_score),
                'score': technical_score,
                'confidence': self._calculate_signal_confidence([
                    ma_signals,
                    momentum_signals,
                    volatility_signals
                ]),
                'components': {
                    'moving_averages': ma_signals,
                    'momentum': momentum_signals,
                    'volatility': volatility_signals
                }
            }

        except Exception as e:
            logger.error(f"Error generating technical signals: {str(e)}")
            return self._generate_safe_signal()

    def _analyze_moving_averages(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze moving averages with adaptive parameters."""
        signals = {}

        try:
            # Dynamic MA periods based on volatility
            if self.market_state.get('volatility', {}).get('regime') == 'HIGH':
                periods = [5, 8, 13]  # Shorter periods for high volatility
            else:
                periods = [9, 21, 50]  # Standard periods for normal volatility

            close = data['close'].iloc[-1]
            signals['crosses'] = []

            for short_period, long_period in zip(periods[:-1], periods[1:]):
                short_ma = talib.EMA(data['close'], timeperiod=short_period).iloc[-1]
                long_ma = talib.EMA(data['close'], timeperiod=long_period).iloc[-1]

                if short_ma > long_ma:
                    signals['crosses'].append(1)  # Bullish
                else:
                    signals['crosses'].append(-1)  # Bearish

            # Calculate cumulative score
            score = sum(signals['crosses']) / len(signals['crosses'])

            return {
                'score': score,
                'signal': self._score_to_signal(score),
                'details': signals
            }

        except Exception as e:
            logger.error(f"Error in moving average analysis: {str(e)}")
            return {'score': 0, 'signal': 'NEUTRAL', 'details': {}}

    def _analyze_momentum(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum indicators with adaptive thresholds."""
        signals = {}

        try:
            # Adapt RSI thresholds based on volatility
            if self.market_state.get('volatility', {}).get('regime') == 'HIGH':
                rsi_oversold = 25  # More extreme for high volatility
                rsi_overbought = 75
            else:
                rsi_oversold = 30
                rsi_overbought = 70

            # RSI
            rsi = talib.RSI(data['close']).iloc[-1]
            signals['rsi'] = {
                'value': rsi,
                'signal': 1 if rsi < rsi_oversold else (-1 if rsi > rsi_overbought else 0)
            }

            # MACD
            macd, signal, hist = talib.MACD(data['close'])
            signals['macd'] = {
                'value': hist.iloc[-1],
                'signal': 1 if hist.iloc[-1] > 0 else -1
            }

            # Stochastic
            slowk, slowd = talib.STOCH(data['high'], data['low'], data['close'])
            signals['stoch'] = {
                'k': slowk.iloc[-1],
                'd': slowd.iloc[-1],
                'signal': 1 if slowk.iloc[-1] > slowd.iloc[-1] else -1
            }

            # Calculate weighted score
            weights = {
                'rsi': 0.4,
                'macd': 0.4,
                'stoch': 0.2
            }

            score = (
                weights['rsi'] * signals['rsi']['signal'] +
                weights['macd'] * signals['macd']['signal'] +
                weights['stoch'] * signals['stoch']['signal']
            )

            return {
                'score': score,
                'signal': self._score_to_signal(score),
                'details': signals
            }

        except Exception as e:
            logger.error(f"Error in momentum analysis: {str(e)}")
            return {'score': 0, 'signal': 'NEUTRAL', 'details': {}}

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volatility indicators with dynamic adaptation."""
        signals = {}

        try:
            # Bollinger Bands
            bb_upper, bb_middle, bb_lower = talib.BBANDS(
                data['close'],
                nbdevup=2,
                nbdevdn=2
            )

            latest_close = data['close'].iloc[-1]
            signals['bb'] = {
                'position': (latest_close - bb_lower.iloc[-1]) / (bb_upper.iloc[-1] - bb_lower.iloc[-1]),
                'width': (bb_upper.iloc[-1] - bb_lower.iloc[-1]) / bb_middle.iloc[-1]
            }

            # ATR
            atr = talib.ATR(data['high'], data['low'], data['close'])
            signals['atr'] = {
                'value': atr.iloc[-1],
                'ratio': atr.iloc[-1] / latest_close
            }

            # Generate signals based on volatility state
            if self.market_state.get('volatility', {}).get('regime') == 'HIGH':
                # More conservative in high volatility
                if signals['bb']['position'] < 0.2:  # Very oversold
                    score = 0.5  # Moderate buy
                elif signals['bb']['position'] > 0.8:  # Very overbought
                    score = -0.5  # Moderate sell
                else:
                    score = 0
            else:
                # More aggressive in normal volatility
                if signals['bb']['position'] < 0.1:
                    score = 1  # Strong buy
                elif signals['bb']['position'] > 0.9:
                    score = -1  # Strong sell
                else:
                    score = 0

            return {
                'score': score,
                'signal': self._score_to_signal(score),
                'details': signals
            }

        except Exception as e:
            logger.error(f"Error in volatility analysis: {str(e)}")
            return {'score': 0, 'signal': 'NEUTRAL', 'details': {}}

    def _analyze_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chart patterns with confidence scoring."""
        try:
            patterns = {}

            # Candlestick patterns
            pattern_functions = {
                'engulfing': talib.CDLENGULFING,
                'hammer': talib.CDLHAMMER,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR
            }

            pattern_weights = {
                'engulfing': 1.0,
                'hammer': 0.8,
                'shooting_star': 0.8,
                'morning_star': 1.2,
                'evening_star': 1.2
            }

            # Detect patterns
            for name, func in pattern_functions.items():
                patterns[name] = func(
                    data['open'],
                    data['high'],
                    data['low'],
                    data['close']
                ).iloc[-1]

            # Calculate weighted pattern score
            score = sum(
                patterns[name] * pattern_weights[name]
                for name in patterns
            ) / sum(pattern_weights.values())

            # Adjust confidence based on market regime
            confidence = abs(score)
            if self.market_state.get('trend', {}).get('strength', 0) > 0.7:
                confidence *= 1.2  # Higher confidence in strong trends

            return {
                'score': score,
                'signal': self._score_to_signal(score),
                'confidence': min(confidence, 1.0),
                'patterns': patterns
            }

        except Exception as e:
            logger.error(f"Error in pattern analysis: {str(e)}")
            return {'score': 0, 'signal': 'NEUTRAL', 'confidence': 0, 'patterns': {}}

    def _analyze_volume(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze volume patterns and trends."""
        try:
            volume_analysis = {}

            # Volume trend
            volume_sma = data['volume'].rolling(20).mean()
            current_vol_ratio = data['volume'].iloc[-1] / volume_sma.iloc[-1]

            # Volume-price relationship
            price_change = data['close'].pct_change().iloc[-1]
            volume_change = data['volume'].pct_change().iloc[-1]

            # Volume confirmation
            if abs(price_change) > 0:
                volume_confirms = (price_change > 0 and volume_change > 0) or \
                                (price_change < 0 and volume_change < 0)
            else:
                volume_confirms = False

            # Calculate volume score
            if current_vol_ratio > 1.5 and volume_confirms:
                score = np.sign(price_change) * min(current_vol_ratio / 2, 1)
            else:
                score = 0

            return {
                'score': score,
                'signal': self._score_to_signal(score),
                'volume_trend': {
                    'ratio': current_vol_ratio,
                    'confirms_price': volume_confirms
                }
            }

        except Exception as e:
            logger.error(f"Error in volume analysis: {str(e)}")
            return {'score': 0, 'signal': 'NEUTRAL', 'volume_trend': {}}

    def _generate_ml_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate ML-based predictions."""
        try:
            if not self.ml_predictor:
                return {'signal': 'NEUTRAL', 'confidence': 0}

            # Get ML prediction
            prediction = self.ml_predictor.predict(data)

            # Adjust confidence based on market conditions
            confidence = prediction['confidence']
            if self.market_state.get('volatility', {}).get('regime') == 'HIGH':
                confidence *= 0.8  # Reduce confidence in high volatility

            return {
                'signal': prediction['signal'],
                'confidence': confidence,
                'market_context': prediction['market_context'],
                'ensemble_agreement': prediction.get('agreement', 1.0)
            }

        except Exception as e:
            logger.error(f"Error generating ML signals: {str(e)}")
            return {'signal': 'NEUTRAL', 'confidence': 0}

    def _combine_signals(self, signals: Dict[str, Dict]) -> Dict[str, Any]:
        """Combine multiple signals with adaptive weighting."""
        try:
            # Base weights
            weights = {
                'technical': 0.3,
                'patterns': 0.2,
                'volume': 0.1,
                'ml': 0.4
            }

            # Adjust weights based on market state
            if self.market_state.get('volatility', {}).get('regime') == 'HIGH':
                # Reduce ML weight in high volatility
                weights['ml'] *= 0.8
                weights['technical'] *= 1.2

            if self.market_state.get('regime') == 'TRENDING':
                # Increase technical weight in trending markets
                weights['technical'] *= 1.2
                weights['patterns'] *= 0.8

            # Normalize weights
            total = sum(weights.values())
            weights = {k: v/total for k, v in weights.items()}

            # Calculate weighted signals
            composite_score = 0
            confidence_scores = []

            signal_map = {'BUY': 1, 'NEUTRAL': 0, 'SELL': -1}
            reverse_map = {1: 'BUY', 0: 'NEUTRAL', -1: 'SELL'}

            for signal_type, weight in weights.items():
                if signal_type in signals:
                    signal = signals[signal_type]
                    if isinstance(signal, dict) and 'signal' in signal:
                        score = signal_map.get(signal['signal'], 0)
                        confidence = signal.get('confidence', 0.5)

                        composite_score += score * weight * confidence
                        confidence_scores.append(confidence)

            # Calculate final signal
            final_score = composite_score / sum(weights.values())

            # Determine signal with hysteresis
            if abs(final_score) < 0.2:
                final_signal = 'NEUTRAL'
            else:
                final_signal = reverse_map[np.sign(final_score)]

            # Calculate overall confidence
            confidence = np.mean(confidence_scores) if confidence_scores else 0.5

            return {
                'signal': final_signal,
                'confidence': confidence,
                'score': final_score
            }

        except Exception as e:
            logger.error(f"Error combining signals: {str(e)}")
            return {'signal': 'NEUTRAL', 'confidence': 0, 'score': 0}

    def _calculate_signal_weights(self) -> Dict[str, float]:
        """Calculate adaptive weights for technical signals."""
        weights = {
            'ma': 0.4,
            'momentum': 0.3,
            'volatility': 0.3
        }

        # Adjust based on market regime
        if self.market_state.get('regime') == 'TRENDING':
            weights['ma'] *= 1.2
            weights['momentum'] *= 0.8
        else:
            weights['ma'] *= 0.8
            weights['momentum'] *= 1.2

        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _calculate_signal_confidence(self, signals: List[Dict]) -> float:
        """Calculate confidence score for technical signals."""
        try:
            # Get individual signal scores
            scores = [abs(s.get('score', 0)) for s in signals]

            # Calculate agreement between signals
            unique_signals = set(s.get('signal') for s in signals)
            agreement = 1 - (len(unique_signals) - 1) / (len(signals) - 1) if len(signals) > 1 else 1

            # Combine score magnitude and agreement
            confidence = (np.mean(scores) + agreement) / 2

            # Adjust for market conditions
            if self.market_state.get('trend', {}).get('strength', 0) > 0.7:
                confidence *= 1.1  # Boost confidence in strong trends

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating signal confidence: {str(e)}")
            return 0.5

    def _score_to_signal(self, score: float) -> str:
        """Convert numerical score to trading signal."""
        if score > 0.2:
            return 'BUY'
        elif score < -0.2:
            return 'SELL'
        else:
            return 'NEUTRAL'

    def _generate_safe_signal(self) -> Dict[str, Any]:
        """Generate safe fallback signal."""
        return {
            'signal': 'NEUTRAL',
            'score': 0,
            'confidence': 0,
            'details': {}
        }

    def _generate_safe_fallback(self) -> Dict[str, Any]:
        """Generate complete safe fallback analysis."""
        return {
            'signal': 'NEUTRAL',
            'confidence': 0,
            'market_state': {},
            'technical_signals': self._generate_safe_signal(),
            'pattern_signals': {'signal': 'NEUTRAL', 'confidence': 0, 'patterns': {}},
            'volume_analysis': {'signal': 'NEUTRAL', 'volume_trend': {}},
            'ml_signals': {'signal': 'NEUTRAL', 'confidence': 0},
            'performance_metrics': self.performance_metrics
        }

    def get_signal_direction(self, data: pd.DataFrame) -> str:
        """
        Get the trading signal direction (BUY/SELL/NEUTRAL) for a given timeframe.
        Uses multiple indicators for a consensus-based approach.
        """
        try:
            latest = data.iloc[-1]
            signals = []

            # Check moving averages
            ema_fast = talib.EMA(data['close'], timeperiod=9)
            ema_slow = talib.EMA(data['close'], timeperiod=21)
            if ema_fast.iloc[-1] > ema_slow.iloc[-1]:
                signals.append('BUY')
            elif ema_fast.iloc[-1] < ema_slow.iloc[-1]:
                signals.append('SELL')

            # Check RSI
            rsi = talib.RSI(data['close'])
            if rsi.iloc[-1] < 30:
                signals.append('BUY')
            elif rsi.iloc[-1] > 70:
                signals.append('SELL')

            # Check MACD
            macd, signal, hist = talib.MACD(data['close'])
            if hist.iloc[-1] > 0:
                signals.append('BUY')
            elif hist.iloc[-1] < 0:
                signals.append('SELL')

            # Determine consensus
            if signals:
                buy_count = sum(1 for s in signals if s == 'BUY')
                sell_count = sum(1 for s in signals if s == 'SELL')

                if buy_count > sell_count and buy_count >= len(signals) / 2:
                    return 'BUY'
                elif sell_count > buy_count and sell_count >= len(signals) / 2:
                    return 'SELL'

            return 'NEUTRAL'

        except Exception as e:
            logger.error(f"Error getting signal direction: {str(e)}")
            return 'NEUTRAL'

# Standalone functions that can be used directly or through the TechnicalAnalyzer class

def calculate_fractals(data: pd.DataFrame, window: int = 5) -> Tuple[pd.Series, pd.Series]:
    """
    Calculate Bill Williams' Fractals indicator

    Fractals identify potential reversal points in the market:
    - Bearish Fractal: A high with two lower highs on each side (local maximum)
    - Bullish Fractal: A low with two higher lows on each side (local minimum)

    Args:
        data: DataFrame with price data
        window: Window size for fractal detection (default is 5 for standard fractals)

    Returns:
        Tuple of (bullish_fractals, bearish_fractals) Series
    """
    if data.empty or len(data) < window:
        logger.warning(f"Insufficient data for fractal calculation. Need at least {window} candles.")
        return pd.Series(dtype=float), pd.Series(dtype=float)

    try:
        high_prices = data['high'].values
        low_prices = data['low'].values
        n = len(high_prices)

        # Initialize arrays
        bearish_fractals = np.zeros(n)
        bullish_fractals = np.zeros(n)

        # Half window size (e.g., for window=5, half_window=2)
        half_window = window // 2

        # Calculate fractals
        for i in range(half_window, n - half_window):
            # Bearish fractal (high with 2 lower highs on both sides)
            is_bearish = True
            current_high = high_prices[i]

            for j in range(1, half_window + 1):
                if high_prices[i-j] >= current_high or high_prices[i+j] >= current_high:
                    is_bearish = False
                    break

            if is_bearish:
                bearish_fractals[i] = current_high

            # Bullish fractal (low with 2 higher lows on both sides)
            is_bullish = True
            current_low = low_prices[i]

            for j in range(1, half_window + 1):
                if low_prices[i-j] <= current_low or low_prices[i+j] <= current_low:
                    is_bullish = False
                    break

            if is_bullish:
                bullish_fractals[i] = current_low

        return pd.Series(bullish_fractals, index=data.index), pd.Series(bearish_fractals, index=data.index)
    except Exception as e:
        logger.error(f"Error calculating fractals: {str(e)}")
        return pd.Series(dtype=float), pd.Series(dtype=float)

def calculate_parabolic_sar(data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """
    Calculate Parabolic SAR (Stop and Reverse) indicator

    Args:
        data: DataFrame with price data
        acceleration: Initial acceleration factor
        maximum: Maximum acceleration factor

    Returns:
        Series with Parabolic SAR values
    """
    if data.empty or len(data) < 2:
        logger.warning("Insufficient data for Parabolic SAR calculation")
        return pd.Series()

    try:
        high = data['high'].values
        low = data['low'].values
        psar = [0] * len(high)

        # Initialize trend and extreme point
        bull_market = True  # Initial trend direction (True=Bullish, False=Bearish)
        ep = high[0]  # Initial extreme point
        psar[0] = low[0]  # Initial PSAR value for bullish trend

        # Initialize acceleration factor (AF)
        af = acceleration

        # Calculate PSAR values
        for i in range(1, len(high)):
            # Previous PSAR value
            if i == 1:
                psar[i] = psar[0]
            else:
                if bull_market:
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])
                else:
                    psar[i] = psar[i-1] + af * (ep - psar[i-1])

            # Check for trend reversal
            reverse = False

            if bull_market:
                if low[i] < psar[i]:
                    bull_market = False
                    reverse = True
                    psar[i] = ep
                    ep = low[i]
                    af = acceleration
            else:
                if high[i] > psar[i]:
                    bull_market = True
                    reverse = True
                    psar[i] = ep
                    ep = high[i]
                    af = acceleration

            # If no reversal, update EP and AF if needed
            if not reverse:
                if bull_market:
                    if high[i] > ep:
                        ep = high[i]
                        af = min(af + acceleration, maximum)

                    # Ensure PSAR is below the lows of the last two candles
                    psar[i] = min(psar[i], low[i-1], low[i])
                else:
                    if low[i] < ep:
                        ep = low[i]
                        af = min(af + acceleration, maximum)

                    # Ensure PSAR is above the highs of the last two candles
                    psar[i] = max(psar[i], high[i-1], high[i])

        return pd.Series(psar, index=data.index)

    except Exception as e:
        logger.error(f"Error calculating Parabolic SAR: {str(e)}")
        return pd.Series()

def calculate_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate technical indicators for price data

    Args:
        data: DataFrame with OHLCV price data

    Returns:
        DataFrame with added indicators
    """
    if data.empty:
        logger.warning("Empty data provided for indicator calculation")
        return data

    # Make a copy to avoid modifying the original
    df = data.copy()

    try:
        # Parabolic SAR
        df['psar'] = calculate_parabolic_sar(df)

        # Moving Averages
        df['ma_5'] = df['close'].rolling(window=5).mean()
        df['ma_20'] = df['close'].rolling(window=20).mean()
        df['ma_50'] = df['close'].rolling(window=50).mean()

        # Exponential Moving Averages
        df['ema_5'] = df['close'].ewm(span=5, adjust=False).mean()
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()

        # MACD
        df['macd_line'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
        df['macd_signal'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['macd_signal']

        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['rsi'] = 100 - (100 / (1 + rs))

        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(window=20).mean()
        df['bb_std'] = df['close'].rolling(window=20).std()
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * 2)
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * 2)

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # CCI (Commodity Channel Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_dev = np.abs(typical_price - typical_price.rolling(window=20).mean()).rolling(window=20).mean()
        df['cci'] = (typical_price - typical_price.rolling(window=20).mean()) / (0.015 * mean_dev)

        # Average Directional Index (ADX)
        # Calculate True Range
        df['tr1'] = abs(df['high'] - df['low'])
        df['tr2'] = abs(df['high'] - df['close'].shift())
        df['tr3'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        df['atr'] = df['tr'].rolling(window=14).mean()

        # Directional Movement
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']

        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)

        df['plus_di'] = 100 * (df['plus_dm'].rolling(window=14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].rolling(window=14).mean() / df['atr'])

        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(window=14).mean()

        # Ichimoku Cloud
        # Conversion Line (Tenkan-sen)
        df['ichimoku_conversion'] = (df['high'].rolling(window=9).max() + df['low'].rolling(window=9).min()) / 2
        # Base Line (Kijun-sen)
        df['ichimoku_base'] = (df['high'].rolling(window=26).max() + df['low'].rolling(window=26).min()) / 2
        # Leading Span A (Senkou Span A)
        df['ichimoku_span_a'] = ((df['ichimoku_conversion'] + df['ichimoku_base']) / 2).shift(26)
        # Leading Span B (Senkou Span B)
        df['ichimoku_span_b'] = ((df['high'].rolling(window=52).max() + df['low'].rolling(window=52).min()) / 2).shift(26)
        # Lagging Span (Chikou Span)
        df['ichimoku_lagging'] = df['close'].shift(-26)

        # Awesome Oscillator
        df['ao'] = df['high'].rolling(window=5).mean() - df['high'].rolling(window=34).mean()

        # Money Flow Index
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        raw_money_flow = typical_price * df['volume']

        positive_flow = np.where(typical_price > typical_price.shift(), raw_money_flow, 0)
        negative_flow = np.where(typical_price < typical_price.shift(), raw_money_flow, 0)

        positive_mf = pd.Series(positive_flow).rolling(window=14).sum()
        negative_mf = pd.Series(negative_flow).rolling(window=14).sum()

        money_flow_ratio = positive_mf / negative_mf
        df['mfi'] = 100 - (100 / (1 + money_flow_ratio))

        # Volume Weighted Average Price (VWAP)
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()

        # On-Balance Volume (OBV)
        df['obv'] = np.where(df['close'] > df['close'].shift(),
                           df['volume'],
                           np.where(df['close'] < df['close'].shift(),
                                   -df['volume'], 0)).cumsum()

        # Chaikin Money Flow
        money_flow_multiplier = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        money_flow_volume = money_flow_multiplier * df['volume']
        df['cmf'] = money_flow_volume.rolling(window=20).sum() / df['volume'].rolling(window=20).sum()

        # Williams %R
        df['williams_r'] = ((df['high'].rolling(window=14).max() - df['close']) /
                          (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())) * -100

        # Commodity Channel Index (CCI)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        mean_tp = typical_price.rolling(window=20).mean()
        mean_deviation = abs(typical_price - mean_tp).rolling(window=20).mean()
        df['cci'] = (typical_price - mean_tp) / (0.015 * mean_deviation)

        # Bill Williams' Fractals
        bullish_fractals, bearish_fractals = calculate_fractals(df)
        df['bullish_fractal'] = bullish_fractals
        df['bearish_fractal'] = bearish_fractals

        # Create a fractal signal column
        # 1 = bullish fractal, -1 = bearish fractal, 0 = no fractal
        df['fractal_signal'] = 0
        df.loc[df['bullish_fractal'] > 0, 'fractal_signal'] = 1
        df.loc[df['bearish_fractal'] > 0, 'fractal_signal'] = -1

        logger.info(f"Calculated {len(df.columns) - 5} technical indicators on {len(df)} candles")
        return df

    except Exception as e:
        logger.error(f"Error calculating indicators: {str(e)}")
        return data

def fractal_strategy(data: pd.DataFrame) -> Dict:
    """
    Bill Williams' Fractal indicator strategy with increased confidence

    This strategy identifies potential reversal points using fractal patterns:
    - Buy when a bullish fractal forms (low with two higher lows on each side)
    - Sell when a bearish fractal forms (high with two lower highs on each side)

    The strategy combines fractal signals with the price position relative to moving averages
    to increase confidence in the signals.

    Args:
        data: DataFrame with price data and indicators including fractals

    Returns:
        Dictionary with signal direction and confidence
    """
    if data.empty or len(data) < 20:
        logger.warning("Insufficient data for Fractal strategy")
        return {'direction': 'NEUTRAL', 'confidence': 0}

    try:
        # Get recent data points
        recent = data.tail(5).copy()

        # Check if we have the required indicators
        if 'bullish_fractal' not in recent.columns or 'bearish_fractal' not in recent.columns:
            logger.warning("Missing fractal indicators for Fractal strategy")
            return {'direction': 'NEUTRAL', 'confidence': 0}

        # Start with base confidence
        base_confidence = 70  # Higher base confidence
        confidence_boost = 0
        direction = 'NEUTRAL'

        # Check for fractal signals in the most recent candles
        has_bullish_fractal = False
        has_bearish_fractal = False

        # Look for the most recent fractal (they appear with a 2-candle lag)
        for i in range(1, min(5, len(recent))):
            idx = -i
            if recent.iloc[idx]['bullish_fractal'] > 0:
                has_bullish_fractal = True
                break
            if recent.iloc[idx]['bearish_fractal'] > 0:
                has_bearish_fractal = True
                break

        # Add trend confirmation
        latest = data.iloc[-1]

        # Moving average confirmation
        ma_confirmation = False
        if 'ma_20' in latest and 'ma_50' in latest:
            ma_trend_up = latest['ma_20'] > latest['ma_50']
            ma_trend_down = latest['ma_20'] < latest['ma_50']

            if has_bullish_fractal and ma_trend_up:
                confidence_boost += 15
                ma_confirmation = True
            elif has_bearish_fractal and ma_trend_down:
                confidence_boost += 15
                ma_confirmation = True

        # RSI confirmation
        rsi_confirmation = False
        if 'rsi' in latest:
            rsi_bullish = latest['rsi'] > 50 and latest['rsi'] < 70
            rsi_bearish = latest['rsi'] < 50 and latest['rsi'] > 30

            if has_bullish_fractal and rsi_bullish:
                confidence_boost += 10
                rsi_confirmation = True
            elif has_bearish_fractal and rsi_bearish:
                confidence_boost += 10
                rsi_confirmation = True

        # Additional confirmation from Parabolic SAR
        psar_confirmation = False
        if 'psar' in latest:
            psar_bullish = latest['psar'] < latest['close']
            psar_bearish = latest['psar'] > latest['close']

            if has_bullish_fractal and psar_bullish:
                confidence_boost += 10
                psar_confirmation = True
            elif has_bearish_fractal and psar_bearish:
                confidence_boost += 10
                psar_confirmation = True

        # Generate signal with enhanced confidence
        if has_bullish_fractal:
            direction = 'BUY'
            # If at least two confirmation indicators agree, add another boost
            if (ma_confirmation and rsi_confirmation) or (ma_confirmation and psar_confirmation) or (rsi_confirmation and psar_confirmation):
                confidence_boost += 10

        elif has_bearish_fractal:
            direction = 'SELL'
            # If at least two confirmation indicators agree, add another boost
            if (ma_confirmation and rsi_confirmation) or (ma_confirmation and psar_confirmation) or (rsi_confirmation and psar_confirmation):
                confidence_boost += 10

        total_confidence = min(100, base_confidence + confidence_boost)

        return {'direction': direction, 'confidence': total_confidence}

    except Exception as e:
        logger.error(f"Error in Fractal strategy: {str(e)}")
        return {'direction': 'NEUTRAL', 'confidence': 0}

def parabolic_sar_cci_strategy(data: pd.DataFrame, m1_data: pd.DataFrame = None, m5_data: pd.DataFrame = None, cci_period: int = 45) -> Dict:
    """
    Enhanced scalping strategy using Parabolic SAR + CCI with multiple confirmations

    Rules:
    - Long position: PSAR below price AND CCI > 100 AND EMA validation
    - Short position: PSAR above price AND CCI < -100 AND EMA validation
    - EMA50 on M1 and EMA21 on M5 for trend confirmation
    - Volume confirmation
    - ATR-based dynamic targets
    - ADX trend strength validation

    Args:
        data: DataFrame with price data and indicators
        m1_data: M1 timeframe data for EMA50
        m5_data: M5 timeframe data for EMA21
        cci_period: Period for CCI calculation (default 45)

    Returns:
        Dictionary with signal direction, confidence, targets, and additional metrics
    """
    if data.empty or len(data) < 50:
        logger.warning("Insufficient data for Parabolic SAR + CCI strategy")
        return {'direction': 'NEUTRAL', 'confidence': 0, 'targets': {}, 'metrics': {}}

    try:
        # Get latest values
        latest = data.iloc[-1]
        current_price = latest['close']

        # Calculate required indicators if not present
        if 'psar' not in latest:
            data['psar'] = calculate_parabolic_sar(data, acceleration=0.02, maximum=0.2)
            latest = data.iloc[-1]

        if 'cci' not in latest:
            typical_price = (data['high'] + data['low'] + data['close']) / 3
            mean_dev = np.abs(typical_price - typical_price.rolling(window=cci_period).mean()).rolling(window=cci_period).mean()
            data['cci'] = (typical_price - typical_price.rolling(window=cci_period).mean()) / (0.015 * mean_dev)
            latest = data.iloc[-1]

        # Calculate ATR for dynamic targets
        atr = talib.ATR(data['high'], data['low'], data['close'], timeperiod=14)
        current_atr = atr.iloc[-1]

        # Calculate ADX for trend strength
        adx = talib.ADX(data['high'], data['low'], data['close'], timeperiod=14)
        current_adx = adx.iloc[-1]

        # Volume analysis
        volume_sma = data['volume'].rolling(window=20).mean()
        volume_ratio = latest['volume'] / volume_sma.iloc[-1]
        strong_volume = volume_ratio > 1.2

        # Calculate EMAs on respective timeframes
        ema_trend_aligned = True
        if m1_data is not None and m5_data is not None:
            m1_ema50 = m1_data['close'].ewm(span=50, adjust=False).mean()
            m5_ema21 = m5_data['close'].ewm(span=21, adjust=False).mean()

            above_ema50 = current_price > m1_ema50.iloc[-1]
            above_ema21 = current_price > m5_ema21.iloc[-1]
            ema_trend_aligned = above_ema50 == above_ema21

        # Check conditions
        psar_below_price = latest['psar'] < current_price
        psar_above_price = latest['psar'] > current_price
        cci_bullish = latest['cci'] > 100
        cci_bearish = latest['cci'] < -100
        strong_trend = current_adx > 25

        # Base confidence and metrics
        base_confidence = 70
        metrics = {
            'adx': current_adx,
            'volume_ratio': volume_ratio,
            'atr': current_atr,
            'ema_aligned': ema_trend_aligned,
            'trend_strength': 'STRONG' if strong_trend else 'WEAK'
        }
        if psar_below_price and cci_bullish:
            direction = 'BUY'
            confidence = base_confidence

            # Confidence adjustments
            if ema_trend_aligned:
                confidence += 10
            if strong_volume:
                confidence += 10
            if strong_trend:
                confidence += 10

            # Dynamic ATR-based targets
            tp1 = current_price + (current_atr * 1.0)
            tp2 = current_price + (current_atr * 1.5)
            sl = current_price - (current_atr * 0.8)

            targets = {
                'tp1': round(tp1, 5),
                'tp2': round(tp2, 5),
                'sl': round(sl, 5),
                'risk_reward': round((tp1 - current_price) / (current_price - sl), 2)
            }

            # Final confidence adjustment
            confidence = min(100, confidence + abs(latest['cci']) / 20)
            return {'direction': direction, 'confidence': confidence, 'targets': targets, 'metrics': metrics}

        elif psar_above_price and cci_bearish:
            direction = 'SELL'
            confidence = base_confidence

            # Confidence adjustments
            if ema_trend_aligned:
                confidence += 10
            if strong_volume:
                confidence += 10
            if strong_trend:
                confidence += 10

            # Dynamic ATR-based targets
            tp1 = current_price - (current_atr * 1.0)
            tp2 = current_price - (current_atr * 1.5)
            sl = current_price + (current_atr * 0.8)

            targets = {
                'tp1': round(tp1, 5),
                'tp2': round(tp2, 5),
                'sl': round(sl, 5),
                'risk_reward': round((current_price - tp1) / (sl - current_price), 2)
            }

            # Final confidence adjustment
            confidence = min(100, confidence + abs(latest['cci']) / 20)
            return {'direction': direction, 'confidence': confidence, 'targets': targets, 'metrics': metrics}

        return {'direction': 'NEUTRAL', 'confidence': 0, 'targets': {}, 'metrics': metrics}

    except Exception as e:
        logger.error(f"Error in Parabolic SAR + CCI strategy: {str(e)}")
        return {'direction': 'NEUTRAL', 'confidence': 0, 'targets': {}}

def analyze_trend(data: pd.DataFrame) -> Dict:
    """
    Analyze price trend using multiple technical indicators.

    Args:
        data: DataFrame with price data and technical indicators

    Returns:
        Dictionary with trend analysis results
    """
    try:
        trend_info = {
            'direction': 'NEUTRAL',
            'strength': 0,
            'duration': 0,
            'momentum': 0,
            'support': None,
            'resistance': None
        }

        if data.empty or len(data) < 20:
            return trend_info

        latest = data.iloc[-1]

        # Moving Average Analysis
        ma_trend = 0
        if ('ma_20' in latest) and ('ma_50' in latest):
            ma_trend = 1 if latest['ma_20'] > latest['ma_50'] else -1
            # Check moving average slope
            ma20_slope = (latest['ma_20'] - data['ma_20'].iloc[-5]) / data['ma_20'].iloc[-5]
            ma50_slope = (latest['ma_50'] - data['ma_50'].iloc[-5]) / data['ma_50'].iloc[-5]
            ma_trend *= (1 + abs(ma20_slope) * 5)  # Boost signal with slope steepness

        # ADX Trend Strength
        trend_strength = 0
        if 'adx' in latest:
            adx_value = latest['adx']
            trend_strength = min(adx_value / 50.0, 1.0)  # Normalize ADX to 0-1 range

            if 'plus_di' in latest and 'minus_di' in latest:
                di_trend = 1 if latest['plus_di'] > latest['minus_di'] else -1
                trend_strength *= di_trend

        # Price Action Analysis
        price_trend = 0
        recent_prices = data['close'].tail(20)
        price_direction = 1 if recent_prices.iloc[-1] > recent_prices.mean() else -1
        price_volatility = recent_prices.std() / recent_prices.mean()
        price_trend = price_direction * (1 - min(price_volatility * 2, 0.5))

        # Momentum Indicators
        momentum = 0
        if 'rsi' in latest:
            rsi = latest['rsi']
            if rsi > 70:
                momentum = 1
            elif rsi < 30:
                momentum = -1

        if 'macd_hist' in latest:
            momentum += np.sign(latest['macd_hist'])

        momentum = np.clip(momentum / 2, -1, 1)  # Normalize to -1 to 1

        # Volume Trend Confirmation
        volume_confirm = 0
        if 'volume' in latest:
            recent_volume = data['volume'].tail(10)
            avg_volume = recent_volume.mean()
            if latest['volume'] > avg_volume * 1.5:  # High volume
                volume_confirm = price_direction

        # Trend Score Calculation
        trend_score = (
            ma_trend * 0.3 +             # Moving average trend
            trend_strength * 0.3 +        # ADX strength
            price_trend * 0.2 +          # Price action
            momentum * 0.1 +             # Momentum
            volume_confirm * 0.1         # Volume confirmation
        )

        # Determine trend direction and strength
        trend_info['strength'] = abs(trend_score)
        if trend_score > 0.2:
            trend_info['direction'] = 'UPTREND'
        elif trend_score < -0.2:
            trend_info['direction'] = 'DOWNTREND'
        else:
            trend_info['direction'] = 'SIDEWAYS'

        # Calculate trend duration
        trend_changes = np.diff(np.signbit(data['macd_hist'].fillna(0)).astype(int))
        last_change = np.where(trend_changes != 0)[0][-1] if len(trend_changes) > 0 else 0
        trend_info['duration'] = len(data) - last_change

        # Set momentum
        trend_info['momentum'] = momentum

        # Find nearest support/resistance
        highs = data['high'].rolling(window=20).max()
        lows = data['low'].rolling(window=20).min()
        current_price = latest['close']

        trend_info['resistance'] = float(highs.iloc[-1])
        trend_info['support'] = float(lows.iloc[-1])

        return trend_info

    except Exception as e:
        logger.error(f"Error analyzing trend: {str(e)}")
        return {
            'direction': 'NEUTRAL',
            'strength': 0,
            'duration': 0,
            'momentum': 0,
            'support': None,
            'resistance': None
        }

def calculate_support_resistance(data: pd.DataFrame, window: int = 20) -> Dict[str, float]:
    """
    Calculate support and resistance levels

    Args:
        data: DataFrame with price data
        window: Lookback window for finding levels

    Returns:
        Dictionary with support and resistance levels
    """
    if data.empty or len(data) < window:
        logger.warning("Insufficient data for support/resistance calculation")
        return {'support': None, 'resistance': None}

    try:
        # Get recent data
        recent_data = data.iloc[-window:]

        # Find local minima and maxima
        local_min = []
        local_max = []

        for i in range(1, len(recent_data) - 1):
            if recent_data['low'].iloc[i] < recent_data['low'].iloc[i-1] and recent_data['low'].iloc[i] < recent_data['low'].iloc[i+1]:
                local_min.append(recent_data['low'].iloc[i])

            if recent_data['high'].iloc[i] > recent_data['high'].iloc[i-1] and recent_data['high'].iloc[i] > recent_data['high'].iloc[i+1]:
                local_max.append(recent_data['high'].iloc[i])

        # Get the strongest support and resistance
        if local_min:
            support = np.mean(local_min)
        else:
            support = recent_data['low'].min()

        if local_max:
            resistance = np.mean(local_max)
        else:
            resistance = recent_data['high'].max()

        logger.info(f"Support and resistance levels calculated for {data.index[-1]}")
        return {'support': support, 'resistance': resistance}

    except Exception as e:
        logger.error(f"Error calculating support/resistance: {str(e)}")
        return {'support': None, 'resistance': None}

def detect_patterns(data: pd.DataFrame) -> Dict[str, bool]:
    """
    Detect candlestick patterns in price data

    Args:
        data: DataFrame with price data

    Returns:
        Dictionary mapping pattern names to boolean detection
    """
    if data.empty or len(data) < 5:
        logger.warning("Insufficient data for pattern detection")
        return {}

    try:
        # Get the most recent candles
        recent = data.iloc[-5:].copy()
        patterns = {}

        # Doji
        recent['body'] = abs(recent['close'] - recent['open'])
        recent['range'] = recent['high'] - recent['low']
        recent['body_pct'] = recent['body'] / recent['range']

        latest = recent.iloc[-1]

        # Doji - very small body compared to range
        patterns['doji'] = latest['body_pct'] < 0.1

        # Hammer and Inverted Hammer
        if latest['body_pct'] < 0.3:
            lower_wick = min(latest['open'], latest['close']) - latest['low']
            upper_wick = latest['high'] - max(latest['open'], latest['close'])
            body_size = abs(latest['close'] - latest['open'])

            # Hammer: small body, little or no upper wick, lower wick at least 2x body
            patterns['hammer'] = (lower_wick > 2 * body_size) and (upper_wick < 0.5 * body_size)

            # Inverted Hammer: small body, little or no lower wick, upper wick at least 2x body
            patterns['inverted_hammer'] = (upper_wick > 2 * body_size) and (lower_wick < 0.5 * body_size)

        # Engulfing patterns (need at least 2 candles)
        if len(recent) >= 2:
            prev = recent.iloc[-2]

            # Bullish Engulfing
            patterns['bullish_engulfing'] = (
                latest['close'] > latest['open'] and  # Current candle is bullish
                prev['close'] < prev['open'] and      # Previous candle is bearish
                latest['close'] > prev['open'] and    # Current close is higher than previous open
                latest['open'] < prev['close']        # Current open is lower than previous close
            )

            # Bearish Engulfing
            patterns['bearish_engulfing'] = (
                latest['close'] < latest['open'] and  # Current candle is bearish
                prev['close'] > prev['open'] and      # Previous candle is bullish
                latest['close'] < prev['open'] and    # Current close is lower than previous open
                latest['open'] > prev['close']        # Current open is higher than previous close
            )

        # Morning Star and Evening Star (need at least 3 candles)
        if len(recent) >= 3:
            prev = recent.iloc[-2]
            prev_prev = recent.iloc[-3]

            # Morning Star
            patterns['morning_star'] = (
                prev_prev['close'] < prev_prev['open'] and  # First candle is bearish
                abs(prev['close'] - prev['open']) < 0.3 * prev['range'] and  # Middle candle is small
                latest['close'] > latest['open'] and  # Third candle is bullish
                latest['close'] > (prev_prev['open'] + prev_prev['close']) / 2  # Third candle closes above midpoint of first
            )

            # Evening Star
            patterns['evening_star'] = (
                prev_prev['close'] > prev_prev['open'] and  # First candle is bullish
                abs(prev['close'] - prev['open']) < 0.3 * prev['range'] and  # Middle candle is small
                latest['close'] < latest['open'] and  # Third candle is bearish
                latest['close'] < (prev_prev['open'] + prev_prev['close']) / 2  # Third candle closes below midpoint of first
            )

        # Check if any patterns were detected
        detected_patterns = [p for p, v in patterns.items() if v]
        if detected_patterns:
            logger.info(f"Detected patterns: {', '.join(detected_patterns)}")
        else:
            logger.info("No patterns detected in current data")

        return patterns

    except Exception as e:
        logger.error(f"Error detecting patterns: {str(e)}")
        return {}

def generate_signals(data: pd.DataFrame) -> Dict:
    """
    Generate trading signals from technical analysis

    Args:
        data: DataFrame with price data and indicators

    Returns:
        Dictionary with signal details
    """
    if data.empty or len(data) < 50:
        logger.warning("Insufficient data for signal generation")
        return {'direction': 'NEUTRAL', 'confidence': 0, 'indicators': {}}

    try:
        # Analyze trend
        trend_analysis = analyze_trend(data)

        # Detect patterns
        patterns = detect_patterns(data)

        # Calculate support/resistance
        sr_levels = calculate_support_resistance(data)

        # Get latest values
        latest = data.iloc[-1]

        # Count bullish and bearish signals
        bullish_signals = trend_analysis.get('bullish_signals', 0)
        bearish_signals = trend_analysis.get('bearish_signals', 0)
        total_signals = trend_analysis.get('total_signals', 0)

        # Add pattern signals
        if patterns.get('hammer', False) or patterns.get('bullish_engulfing', False) or patterns.get('morning_star', False):
            bullish_signals += 1
            total_signals += 1

        if patterns.get('inverted_hammer', False) or patterns.get('bearish_engulfing', False) or patterns.get('evening_star', False):
            bearish_signals += 1
            total_signals += 1

        # Analyze RSI conditions
        if 'rsi' in latest:
            if latest['rsi'] < 30:
                bullish_signals += 1
                total_signals += 1
            elif latest['rsi'] > 70:
                bearish_signals += 1
                total_signals += 1

        # Analyze position relative to support/resistance
        if sr_levels['support'] is not None and sr_levels['resistance'] is not None:
            price = latest['close']
            support = sr_levels['support']
            resistance = sr_levels['resistance']

            # Calculate distance from price to support/resistance as percentage
            support_distance = (price - support) / price * 100
            resistance_distance = (resistance - price) / price * 100

            # Near support or resistance
            if support_distance < 0.2:
                bullish_signals += 1
                total_signals += 1
            elif resistance_distance < 0.2:
                bearish_signals += 1
                total_signals += 1

        # Determine signal direction and confidence
        direction = 'NEUTRAL'
        confidence = 0

        if total_signals > 0:
            bullish_percentage = (bullish_signals / total_signals) * 100
            bearish_percentage = (bearish_signals / total_signals) * 100

            # Add trend strength to confidence calculation
            trend_strength = trend_analysis.get('strength', 0)

            if bullish_percentage > 60:
                direction = 'BUY'
                confidence = min(100, bullish_percentage * 0.7 + trend_strength * 0.3)
            elif bearish_percentage > 60:
                direction = 'SELL'
                confidence = min(100, bearish_percentage * 0.7 + trend_strength * 0.3)

            # Adjust confidence based on pattern strength
            if direction != 'NEUTRAL':
                strong_patterns = ['morning_star', 'evening_star', 'bullish_engulfing', 'bearish_engulfing']
                pattern_boost = sum(1 for p in strong_patterns if patterns.get(p, False)) * 5
                confidence = min(100, confidence + pattern_boost)

        logger.info(f"Generated signals for {data.index[-1]}: {direction}")
        return {
            'direction': direction,
            'confidence': confidence,
            'indicators': {
                'trend': trend_analysis.get('trend', 'NEUTRAL'),
                'trend_strength': trend_analysis.get('strength', 0),
                'patterns': [p for p, v in patterns.items() if v],
                'support': sr_levels.get('support'),
                'resistance': sr_levels.get('resistance'),
                'rsi': latest.get('rsi'),
                'macd': latest.get('macd_hist')
            }
        }

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}")
        return {'direction': 'NEUTRAL', 'confidence': 0, 'indicators': {}}

# Test example
if __name__ == "__main__":
    import pandas as pd
    import numpy as np

    # Create sample price data
    dates = pd.date_range(start='2023-01-01', periods=100, freq='1H')
    np.random.seed(42)

    data = pd.DataFrame({
        'open': np.random.normal(100, 5, 100),
        'high': np.random.normal(102, 5, 100),
        'low': np.random.normal(98, 5, 100),
        'close': np.random.normal(101, 5, 100),
        'volume': np.random.normal(1000, 200, 100)
    }, index=dates)

    # Make sure high is highest and low is lowest
    for i in range(len(data)):
        values = [data.iloc[i]['open'], data.iloc[i]['close']]
        data.iloc[i, data.columns.get_loc('high')] = max(values) + abs(np.random.normal(0, 1))
        data.iloc[i, data.columns.get_loc('low')] = min(values) - abs(np.random.normal(0, 1))

    # Calculate indicators
    data_with_indicators = calculate_indicators(data)

    # Analyze trend
    trend = analyze_trend(data_with_indicators)
    print(f"Trend Analysis: {trend}")

    # Detect patterns
    patterns = detect_patterns(data)
    print(f"Detected Patterns: {patterns}")

    # Calculate support/resistance
    sr_levels = calculate_support_resistance(data)
    print(f"Support: {sr_levels['support']:.2f}, Resistance: {sr_levels['resistance']:.2f}")

    # Generate signals
    signal = generate_signals(data_with_indicators)
    print(f"Signal: {signal['direction']} with {signal['confidence']:.2f}% confidence")
    print(f"Signal Indicators: {signal['indicators']}")

    # Or use the class
    analyzer = TechnicalAnalyzer(data)
    results = analyzer.analyze()
    print("\nResults from TechnicalAnalyzer:")
    print(f"Signal: {results['signal']['direction']} with {results['signal']['confidence']:.2f}% confidence")