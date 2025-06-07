"""
Enhanced Market Regime Detection module for TradePulseAnalyzer
Detects current market conditions and adapts trading strategy accordingly
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
import logging
from datetime import datetime, time
from enum import Enum, auto
from dataclasses import dataclass

# Optional import with fallback
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("TA-Lib not available, using basic technical indicators")

logger = logging.getLogger(__name__)

class MarketSession(Enum):
    ASIAN = "ASIAN"
    EUROPEAN = "EUROPEAN"
    US = "US"
    OVERLAP = "OVERLAP"
    OFF_HOURS = "OFF_HOURS"

class MarketRegime(Enum):
    STRONG_TREND_UP = auto()
    STRONG_TREND_DOWN = auto()
    WEAK_TREND_UP = auto()
    WEAK_TREND_DOWN = auto()
    RANGING = auto()
    VOLATILE = auto()
    VOLATILE_TREND_UP = auto()
    VOLATILE_TREND_DOWN = auto()
    BREAKOUT = auto()
    UNKNOWN = auto()
    TRENDING = "TRENDING"
    CONSOLIDATING = "CONSOLIDATING"

@dataclass
class RegimeMetrics:
    volatility: float
    trend_strength: float
    momentum: float
    mean_reversion: float
    range_bound: float
    breakout_probability: float

class MarketRegimeDetector:
    """Detects market regime (trend, range, volatile) and suggests optimal strategies"""

    def __init__(self, data: pd.DataFrame):
        """Initialize with market data"""
        self.data = data
        self.lookback_period = 20
        self.volatility_threshold = 1.5
        self.trend_threshold = 25
        self.breakout_threshold = 0.8

        self.session_times = {
            MarketSession.ASIAN: (time(0, 0), time(8, 0)),      # 00:00-08:00 UTC
            MarketSession.EUROPEAN: (time(7, 0), time(16, 0)),  # 07:00-16:00 UTC
            MarketSession.US: (time(13, 0), time(22, 0))        # 13:00-22:00 UTC
        }

    def calculate_metrics(self) -> RegimeMetrics:
        """Calculate comprehensive market metrics"""
        try:
            if len(self.data) < self.lookback_period:
                return RegimeMetrics(0.0, 0.0, 0.5, 0.5, 0.0, 0.0)

            # Volatility metrics
            returns = self.data['close'].pct_change().dropna()
            volatility = returns.rolling(self.lookback_period).std().iloc[-1] * np.sqrt(252)

            if TALIB_AVAILABLE:
                atr = talib.ATR(self.data['high'].values, self.data['low'].values, self.data['close'].values)
                normalized_atr = atr[-1] / self.data['close'].iloc[-1]

                # Trend metrics
                adx = talib.ADX(self.data['high'].values, self.data['low'].values, self.data['close'].values)
                plus_di = talib.PLUS_DI(self.data['high'].values, self.data['low'].values, self.data['close'].values)
                minus_di = talib.MINUS_DI(self.data['high'].values, self.data['low'].values, self.data['close'].values)
                trend_strength = adx[-1] / 100.0

                # Momentum
                rsi = talib.RSI(self.data['close'].values)
                macd, signal, _ = talib.MACD(self.data['close'].values)
                momentum = (rsi[-1] / 100.0 + (1 if macd[-1] > signal[-1] else 0)) / 2

                # Mean reversion
                upper, middle, lower = talib.BBANDS(self.data['close'].values)
                bb_width = (upper[-1] - lower[-1]) / middle[-1]
                price = self.data['close'].iloc[-1]
                bb_position = (price - lower[-1]) / (upper[-1] - lower[-1])
                mean_reversion = 1.0 - min(abs(bb_position - 0.5) * 2, 1.0)
            else:
                # Basic fallback calculations
                normalized_atr = volatility
                trend_strength = min(volatility, 1.0)
                momentum = 0.5  # Neutral momentum
                mean_reversion = 0.5  # Neutral mean reversion

            # Range analysis
            high_prices = self.data['high'].rolling(self.lookback_period).max()
            low_prices = self.data['low'].rolling(self.lookback_period).min()
            price_range = (high_prices - low_prices) / low_prices
            range_bound = 1.0 - min(price_range.iloc[-1], 1.0)

            # Breakout detection
            volume = self.data.get('volume', pd.Series([0] * len(self.data)))
            avg_volume = volume.rolling(self.lookback_period).mean()
            volume_surge = volume.iloc[-1] / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
            price_near_bounds = min(
                abs(price - high_prices.iloc[-1]),
                abs(price - low_prices.iloc[-1])
            ) / (high_prices.iloc[-1] - low_prices.iloc[-1])
            breakout_probability = min((volume_surge * (1 - price_near_bounds)), 1.0)

            return RegimeMetrics(
                volatility=volatility,
                trend_strength=trend_strength,
                momentum=momentum,
                mean_reversion=mean_reversion,
                range_bound=range_bound,
                breakout_probability=breakout_probability
            )

        except Exception as e:
            logger.error(f"Error calculating market metrics: {str(e)}")
            return RegimeMetrics(0.0, 0.0, 0.5, 0.5, 0.0, 0.0)

    def detect_regime(self, data: Optional[pd.DataFrame] = None) -> Dict:
        """
        Detect current market regime using multiple factors.

        Args:
            data: Optional DataFrame with OHLCV data. If not provided, uses self.data

        Returns:
            Dictionary with regime analysis results
        """
        try:
            # Use provided data or fallback to self.data
            data = data if data is not None else self.data

            # Get current session
            current_session = self._detect_market_session()

            # Calculate volatility regime
            volatility = self._analyze_volatility(data)

            # Detect market structure
            structure = self._detect_market_structure(data)

            # Analyze trend strength
            trend = self._analyze_trend_strength(data)

            # Combine analyses
            regime = self._determine_regime(volatility, structure, trend)            # Calculate metrics for strategy adaptation
            metrics = self.calculate_metrics()
            return {
                'regime': regime.name,  # Use name instead of value for regime
                'session': current_session.value,
                'volatility': volatility,
                'trend_strength': trend['strength'],
                'structure': structure,
                'confidence': self._calculate_regime_confidence(volatility, structure, trend),
                'metrics': {
                    'volatility': volatility['level'] if isinstance(volatility, dict) else 0,
                    'trend_strength': trend['strength'],
                    'breakout_probability': structure.get('breakout_probability', 0)
                }
            }

        except Exception as e:
            logger.error(f"Error detecting market regime: {str(e)}")
            return {'regime': MarketRegime.RANGING.value, 'confidence': 0}

    def detect_regime_transitions(self, data: pd.DataFrame) -> List[Dict]:
        """
        Detect regime transitions in historical data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            List of regime transitions with timestamps
        """
        transitions = []
        window = 20  # Window to detect transitions
        
        try:
            for i in range(window, len(data)):
                window_data = data.iloc[i-window:i]
                current_regime = self.detect_regime(window_data)
                
                if i > window:
                    prev_window = data.iloc[i-window-1:i-1]
                    prev_regime = self.detect_regime(prev_window)
                    
                    if current_regime['regime'] != prev_regime['regime']:
                        transitions.append({
                            'timestamp': data.index[i],
                            'from_regime': prev_regime['regime'],
                            'to_regime': current_regime['regime'],
                            'confidence': current_regime['confidence']
                        })
            
            return transitions
            
        except Exception as e:
            logger.error(f"Error detecting regime transitions: {str(e)}")
            return []

    def _detect_market_session(self) -> MarketSession:
        """Detect current market session based on UTC time."""
        current_time = datetime.utcnow().time()

        # Check for session overlaps first
        if (self._is_time_between(current_time, time(7, 0), time(8, 0)) or
            self._is_time_between(current_time, time(13, 0), time(16, 0))):
            return MarketSession.OVERLAP

        # Check individual sessions
        for session, (start, end) in self.session_times.items():
            if self._is_time_between(current_time, start, end):
                return session

        return MarketSession.OFF_HOURS

    def _analyze_volatility(self, data: pd.DataFrame) -> Dict:
        """Analyze market volatility regimes."""
        try:
            # Calculate multiple volatility measures
            returns = data['close'].pct_change()

            # Historical volatility (standard deviation of returns)
            hist_vol = returns.std() * np.sqrt(252)

            # ATR-based volatility
            atr = ta.volatility.AverageTrueRange(
                high=data['high'],
                low=data['low'],
                close=data['close']
            ).average_true_range()

            atr_vol = (atr.iloc[-1] / data['close'].iloc[-1])

            # Volatility regime classification
            if hist_vol > 0.20 or atr_vol > 0.015:
                regime = "HIGH"
            elif hist_vol < 0.10 or atr_vol < 0.005:
                regime = "LOW"
            else:
                regime = "NORMAL"

            return {
                'regime': regime,
                'historical': hist_vol,
                'atr_based': atr_vol,
                'level': (hist_vol + atr_vol * 100) / 2
            }

        except Exception as e:
            logger.error(f"Error analyzing volatility: {str(e)}")
            return {'regime': "NORMAL", 'level': 1.0}

    def _detect_market_structure(self, data: pd.DataFrame) -> Dict:
        """Detect market structure and patterns."""
        try:
            # Calculate key levels
            highs = data['high'].rolling(window=20).max()
            lows = data['low'].rolling(window=20).min()

            # Check for range formation
            range_height = (highs - lows) / data['close']
            is_ranging = range_height.std() < 0.01

            # Check for consolidation
            recent_range = range_height.iloc[-5:].mean()
            is_consolidating = recent_range < range_height.mean() * 0.7

            # Check for breakout
            volatility = data['close'].pct_change().std()
            recent_volatility = data['close'].pct_change().iloc[-5:].std()
            is_breaking_out = recent_volatility > volatility * 2

            if is_breaking_out:
                structure = "BREAKOUT"
            elif is_consolidating:
                structure = "CONSOLIDATION"
            elif is_ranging:
                structure = "RANGING"
            else:
                structure = "TRENDING"

            return {
                'pattern': structure,
                'range_height': range_height.iloc[-1],
                'breakout_probability': recent_volatility / volatility
            }

        except Exception as e:
            logger.error(f"Error detecting market structure: {str(e)}")
            return {'pattern': "UNDEFINED"}

    def _analyze_trend_strength(self, data: pd.DataFrame) -> Dict:
        """Analyze trend strength using multiple indicators."""
        try:
            if len(data) < 50:  # Minimum data required for calculations
                return {
                    'direction': "NEUTRAL",
                    'strength': 0.0,
                    'adx': 0.0
                }

            # ADX for trend strength
            adx = ta.trend.ADXIndicator(
                high=data['high'],
                low=data['low'],
                close=data['close'],
                window=14
            )
            adx_value = adx.adx().iloc[-1] if not pd.isna(adx.adx().iloc[-1]) else 0.0

            # Moving average alignment
            ema_short = ta.trend.EMAIndicator(
                close=data['close'], window=10
            ).ema_indicator()

            ema_medium = ta.trend.EMAIndicator(
                close=data['close'], window=20
            ).ema_indicator()

            ema_long = ta.trend.EMAIndicator(
                close=data['close'], window=50
            ).ema_indicator()

            # Get last valid values
            last_short = ema_short.iloc[-1] if not pd.isna(ema_short.iloc[-1]) else data['close'].iloc[-1]
            last_medium = ema_medium.iloc[-1] if not pd.isna(ema_medium.iloc[-1]) else data['close'].iloc[-1]
            last_long = ema_long.iloc[-1] if not pd.isna(ema_long.iloc[-1]) else data['close'].iloc[-1]

            # Check MA alignment
            bullish_alignment = last_short > last_medium > last_long
            bearish_alignment = last_short < last_medium < last_long

            # Determine trend direction and strength
            if adx_value > 25:
                if bullish_alignment:
                    direction = "BULLISH"
                    strength = min(adx_value / 100 * 1.5, 1.0)
                elif bearish_alignment:
                    direction = "BEARISH"
                    strength = min(adx_value / 100 * 1.5, 1.0)
                else:
                    direction = "MIXED"
                    strength = adx_value / 100
            else:
                direction = "NEUTRAL"
                strength = adx_value / 100

            return {
                'direction': direction,
                'strength': strength,
                'adx': adx_value
            }

        except Exception as e:
            logger.error(f"Error analyzing trend strength: {str(e)}")
            return {
                'direction': "NEUTRAL",
                'strength': 0.0,
                'adx': 0.0
            }

    def _determine_regime(self,
                         volatility: Dict,
                         structure: Dict,
                         trend: Dict) -> MarketRegime:
        """Determine overall market regime."""
        try:            # Check for high volatility conditions first
            if volatility['regime'] == "HIGH":
                if trend['strength'] > 0.7:
                    return MarketRegime.VOLATILE_TREND_UP if trend.get('direction') == 'BULLISH' else MarketRegime.VOLATILE_TREND_DOWN
                return MarketRegime.BREAKOUT

            # Check for trending conditions
            if trend['strength'] > 0.5:
                if trend.get('direction') == 'BULLISH':
                    return MarketRegime.STRONG_TREND_UP if trend['strength'] > 0.7 else MarketRegime.WEAK_TREND_UP
                elif trend.get('direction') == 'BEARISH':
                    return MarketRegime.STRONG_TREND_DOWN if trend['strength'] > 0.7 else MarketRegime.WEAK_TREND_DOWN
                else:
                    return MarketRegime.RANGING  # Default to ranging if direction is unclear

            # Check for consolidation
            if structure['pattern'] == "CONSOLIDATION":
                return MarketRegime.RANGING  # Use RANGING for consolidation periods

            # Default to ranging
            return MarketRegime.RANGING

        except Exception as e:
            logger.error(f"Error determining regime: {str(e)}")
            return MarketRegime.RANGING

    def _calculate_regime_confidence(self,
                                   volatility: Dict,
                                   structure: Dict,
                                   trend: Dict) -> float:
        """Calculate confidence level in regime detection."""
        try:
            # Base confidence on trend strength
            confidence = trend['strength']

            # Adjust based on volatility clarity
            if volatility['regime'] in ["HIGH", "LOW"]:
                confidence *= 1.2

            # Adjust based on structure clarity
            if structure['pattern'] in ["BREAKOUT", "CONSOLIDATION"]:
                confidence *= 1.1

            return min(confidence, 1.0)

        except Exception as e:
            logger.error(f"Error calculating regime confidence: {str(e)}")
            return 0.5    
    def get_optimal_strategy(self) -> Dict:
        """Get optimal trading strategy for current market regime"""
        regime_info = self.detect_regime()  # Remove redundant self.data argument
        regime = MarketRegime[regime_info['regime']]

        strategy = {
            'recommended_signal': 'NEUTRAL',
            'signal_confidence_boost': 0,
            'stop_loss_multiplier': 1.0,
            'take_profit_multiplier': 1.5,
            'regime': regime.name,
            'confidence': regime_info['confidence'],
            'description': '',
            'adaptations': []
        }

        # Strategy optimization based on regime
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            strategy.update({
                'stop_loss_multiplier': 1.5,
                'take_profit_multiplier': 2.0,
                'signal_confidence_boost': 15,
                'description': 'Strong trend following with wider stops',
                'adaptations': [
                    'Increased position size',
                    'Trailing stops recommended',
                    'Multiple entry points on pullbacks'
                ]
            })

        elif regime in [MarketRegime.VOLATILE_TREND_UP, MarketRegime.VOLATILE_TREND_DOWN]:
            strategy.update({
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 2.5,
                'signal_confidence_boost': 10,
                'description': 'Volatile trend following with very wide stops',
                'adaptations': [
                    'Reduced position size',
                    'Multiple partial profit targets',
                    'Aggressive trailing stops'
                ]
            })

        elif regime == MarketRegime.RANGING:
            strategy.update({
                'stop_loss_multiplier': 0.8,
                'take_profit_multiplier': 1.2,
                'signal_confidence_boost': 5,
                'description': 'Range trading with tight stops',
                'adaptations': [
                    'Mean reversion entries',
                    'Support/Resistance based targets',
                    'Quick profit taking'
                ]
            })

        elif regime == MarketRegime.BREAKOUT:
            strategy.update({
                'stop_loss_multiplier': 1.2,
                'take_profit_multiplier': 2.0,
                'signal_confidence_boost': 20,
                'description': 'Breakout trading strategy',
                'adaptations': [
                    'Aggressive entry on volume confirmation',
                    'Wide stops to accommodate volatility',
                    'Multiple scale-out levels'
                ]
            })

        elif regime == MarketRegime.VOLATILE:
            strategy.update({
                'stop_loss_multiplier': 2.0,
                'take_profit_multiplier': 1.8,
                'signal_confidence_boost': 8,
                'description': 'Volatile market strategy',
                'adaptations': [
                    'Tighten stops in high volatility',
                    'Consider options for hedging',
                    'Monitor news for sudden changes'
                ]
            })

        return strategy

    def calculate_adx(self, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculate ADX (Average Directional Index) with +DI and -DI.
        
        Args:
            period: The period for calculations, default is 14
            
        Returns:
            Tuple of (ADX, +DI, -DI) as numpy arrays
        """
        if TALIB_AVAILABLE:
            adx = talib.ADX(self.data['high'].values, self.data['low'].values, self.data['close'].values, timeperiod=period)
            plus_di = talib.PLUS_DI(self.data['high'].values, self.data['low'].values, self.data['close'].values, timeperiod=period)
            minus_di = talib.MINUS_DI(self.data['high'].values, self.data['low'].values, self.data['close'].values, timeperiod=period)
            return adx, plus_di, minus_di
        else:
            # Basic ADX calculation when TA-Lib is not available
            high = self.data['high'].values
            low = self.data['low'].values
            close = self.data['close'].values
            
            # Calculate True Range
            high_low = high - low
            high_close = np.abs(high - np.roll(close, 1))
            low_close = np.abs(low - np.roll(close, 1))
            
            tr = np.maximum(high_low, high_close)
            tr = np.maximum(tr, low_close)
            
            # Calculate directional movement
            up_move = high - np.roll(high, 1)
            down_move = np.roll(low, 1) - low
            
            plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
            minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
            
            # Calculate smoothed values
            tr_smooth = np.zeros_like(tr)
            plus_smooth = np.zeros_like(plus_dm)
            minus_smooth = np.zeros_like(minus_dm)
            
            # Initialize first values
            tr_smooth[period] = np.mean(tr[:period])
            plus_smooth[period] = np.mean(plus_dm[:period])
            minus_smooth[period] = np.mean(minus_dm[:period])
            
            # Calculate smoothed values
            for i in range(period + 1, len(tr)):
                tr_smooth[i] = (tr_smooth[i-1] * (period - 1) + tr[i]) / period
                plus_smooth[i] = (plus_smooth[i-1] * (period - 1) + plus_dm[i]) / period
                minus_smooth[i] = (minus_smooth[i-1] * (period - 1) + minus_dm[i]) / period
            
            # Calculate +DI and -DI
            plus_di = 100 * plus_smooth / tr_smooth
            minus_di = 100 * minus_smooth / tr_smooth
            
            # Calculate DX
            dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
            
            # Calculate ADX
            adx = np.zeros_like(dx)
            adx[2*period-1] = np.mean(dx[period:2*period])
            
            for i in range(2*period, len(dx)):
                adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period
            
            return adx, plus_di, minus_di

    def calculate_volatility(self) -> np.ndarray:
        """Calculate volatility using standard deviation of returns"""
        try:
            returns = self.data['close'].pct_change().dropna()
            volatility = returns.rolling(window=self.lookback_period).std() * np.sqrt(252)
            return volatility.to_numpy()
        except Exception as e:
            logger.error(f"Error calculating volatility: {str(e)}")
            return np.zeros(len(self.data))

    def calculate_bollinger_width(self) -> np.ndarray:
        """Calculate Bollinger Band width as a percentage of middle band"""
        try:
            if TALIB_AVAILABLE:
                upper, middle, lower = talib.BBANDS(self.data['close'].values, 
                                                  timeperiod=self.lookback_period,
                                                  nbdevup=2,
                                                  nbdevdn=2,
                                                  matype=0)
                width = (upper - lower) / middle
                return width
            else:
                # Basic implementation when TA-Lib is not available
                sma = self.data['close'].rolling(window=self.lookback_period).mean()
                std = self.data['close'].rolling(window=self.lookback_period).std()
                upper = sma + (2 * std)
                lower = sma - (2 * std)
                width = (upper - lower) / sma
                return width.to_numpy()
        except Exception as e:
            logger.error(f"Error calculating Bollinger width: {str(e)}")
            return np.zeros(len(self.data))

    def detect_support_resistance_test(self) -> Dict[str, bool]:
        """
        Detect if price is currently testing support or resistance levels.
        Uses Bollinger Bands and recent high/low levels.

        Returns:
            Dictionary indicating if price is testing support or resistance
        """
        try:
            # Get recent data window
            window = min(20, len(self.data))
            recent_data = self.data.iloc[-window:]
            
            # Get current price and recent highs/lows
            current_price = recent_data['close'].iloc[-1]
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            # Calculate Bollinger Bands
            bb_width = self.calculate_bollinger_width()
            sma = recent_data['close'].rolling(window=20).mean()
            std = recent_data['close'].rolling(window=20).std()
            upper_band = sma + (2 * std)
            lower_band = sma - (2 * std)
            
            # Calculate proximity thresholds (1% of price range)
            price_range = recent_high - recent_low
            proximity_threshold = 0.01 * price_range
            
            # Check if testing support
            is_testing_support = (
                abs(current_price - recent_low) < proximity_threshold or
                abs(current_price - lower_band.iloc[-1]) < proximity_threshold
            )
            
            # Check if testing resistance
            is_testing_resistance = (
                abs(current_price - recent_high) < proximity_threshold or
                abs(current_price - upper_band.iloc[-1]) < proximity_threshold
            )
            
            return {
                'is_testing_support': bool(is_testing_support),
                'is_testing_resistance': bool(is_testing_resistance),
                'support_level': float(max(recent_low, lower_band.iloc[-1])),
                'resistance_level': float(min(recent_high, upper_band.iloc[-1])),
                'current_price': float(current_price)
            }

        except Exception as e:
            logger.error(f"Error in support/resistance test detection: {str(e)}")
            return {
                'is_testing_support': False,
                'is_testing_resistance': False,
                'support_level': None,
                'resistance_level': None,
                'current_price': None
            }

    def _is_time_between(self, time: datetime.time, start: datetime.time, end: datetime.time) -> bool:
        """Check if a time is between start and end times, handling overnight periods."""
        if start <= end:
            return start <= time <= end
        else:  # crosses midnight
            return time >= start or time <= end
