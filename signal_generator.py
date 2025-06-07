from technical_analysis import TechnicalAnalyzer
from pattern_recognition import EnhancedPatternRecognizer
import pandas as pd
from typing import Dict, List, Optional, Union
import logging
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator, EMAIndicator, IchimokuIndicator, ADXIndicator, CCIIndicator
from ta.volatility import BollingerBands, AverageTrueRange
import numpy as np
import config
from datetime import datetime
from ml_model import MLPredictor
import os
import json
import random
from utils import fetch_market_data

# Import new advanced analysis modules
try:
    from volume_analysis import VolumeAnalyzer
    from market_regime import MarketRegimeDetector
    from sequence_patterns import SequencePatternRecognizer
    from signal_correlation import SignalCorrelationAnalyzer
    from sentiment_analysis import SentimentAnalyzer
    from signal_confirmation import SignalConfirmationSystem
    ENHANCED_ANALYSIS_ENABLED = True
except ImportError:
    ENHANCED_ANALYSIS_ENABLED = False

# Import the ContinuousLearningSystem class if the module exists
try:
    from continuous_learning import ContinuousLearningSystem
    CONTINUOUS_LEARNING_ENABLED = True
except ImportError:
    CONTINUOUS_LEARNING_ENABLED = False

logger = logging.getLogger(__name__)

# Initialize the continuous learning system as a singleton if enabled
continuous_learner = None
if CONTINUOUS_LEARNING_ENABLED:
    try:
        continuous_learner = ContinuousLearningSystem()
        # Load optimized parameters if available
        optimized_params = continuous_learner.optimize_parameters()
        logger.info(f"Initialized continuous learning system with optimized parameters: {optimized_params}")
    except Exception as e:
        logger.error(f"Error initializing continuous learning system: {str(e)}", exc_info=True)
        CONTINUOUS_LEARNING_ENABLED = False

# Signal storage for the application
_latest_signals = {}

def generate_signals() -> Dict[str, Dict]:
    """
    Generate trading signals for all supported assets.

    Returns:
        Dictionary mapping asset symbols to signal dictionaries
    """
    logger.info("Generating signals for all supported assets")
    signals = {}

    try:
        # Generate signals for each supported asset
        for asset in config.SUPPORTED_ASSETS:
            # Fetch market data
            data = fetch_market_data(pair=asset, timeframe=config.DEFAULT_TIMEFRAME)

            if data.empty:
                logger.warning(f"Failed to fetch data for {asset}")
                continue

            # Create signal generator
            generator = SignalGenerator(data, pair=asset, timeframe=config.DEFAULT_TIMEFRAME)

            # Generate signal
            signal = generator.generate_signal()

            # Only include actionable signals with sufficient confidence
            if signal['direction'] != 'NEUTRAL' and signal['confidence'] >= 50:
                # Format signal for display
                signals[asset] = {
                    'asset': asset,
                    'direction': signal['direction'],
                    'price': signal['price'],
                    'timestamp': signal['timestamp'],
                    'strength': f"{signal['confidence']:.1f}%",
                }

                # Enhance signal description based on confidence
                if signal['confidence'] >= 80:
                    signals[asset]['strength'] = f"Very Strong ({signal['confidence']:.1f}%)"
                elif signal['confidence'] >= 65:
                    signals[asset]['strength'] = f"Strong ({signal['confidence']:.1f}%)"
                elif signal['confidence'] >= 50:
                    signals[asset]['strength'] = f"Moderate ({signal['confidence']:.1f}%)"

                # Store in latest signals
                _latest_signals[asset] = signals[asset]

                logger.info(f"Generated {asset} signal: {signal['direction']} ({signal['confidence']:.1f}%)")
            else:
                logger.info(f"No actionable signal for {asset} (Direction: {signal['direction']}, Confidence: {signal['confidence']:.1f}%)")

    except Exception as e:
        logger.error(f"Error generating signals: {str(e)}", exc_info=True)

    return signals

def get_latest_signals(assets: List[str] = None) -> Dict[str, Dict]:
    """
    Get the latest signals for the specified assets.

    Args:
        assets: List of asset symbols to get signals for. If None, returns all signals.

    Returns:
        Dictionary mapping asset symbols to signal dictionaries
    """
    # If no signals have been generated yet, generate them
    if not _latest_signals:
        generate_signals()

    # If assets is None, return all signals
    if assets is None:
        return _latest_signals

    # Filter signals for the specified assets
    filtered_signals = {}
    for asset in assets:
        if asset in _latest_signals:
            filtered_signals[asset] = _latest_signals[asset]

    return filtered_signals

class SignalGenerator:
    def __init__(self, data: pd.DataFrame, pair: str = None, timeframe: str = None):
        """
        Initialize with price data and optionally specify currency pair and timeframe.

        Args:
            data: DataFrame with OHLCV price data
            pair: Currency pair symbol (e.g., 'EURUSD_otc')
            timeframe: Timeframe string (e.g., '15s', '30s', '1m')
        """
        self.data = data
        self.pair = pair
        self.timeframe = timeframe

        # Core analyzers
        self.technical_analyzer = TechnicalAnalyzer(data)
        self.pattern_recognizer = EnhancedPatternRecognizer(data)
        self.sentiment_analyzer = SentimentAnalyzer()

        # Initialize ML predictor if ML predictions are enabled
        self.ml_predictor = MLPredictor() if config.USE_ML_PREDICTIONS else None

        # Initialize advanced analysis components
        self.signal_confirmation = SignalConfirmationSystem() if ENHANCED_ANALYSIS_ENABLED else None

        # Cache for sentiment analysis
        self.sentiment_cache = {}

        # Initialize advanced analyzers
        try:
            # Check if volume data is available
            if 'volume' in self.data.columns:
                self.volume_analyzer = VolumeAnalyzer(data)
            else:
                self.volume_analyzer = None
                logger.warning(f"Volume data not available for {pair or 'unknown'}, volume analysis disabled")

            # Initialize market regime detector
            self.market_regime_detector = MarketRegimeDetector(data)

            # Initialize sequence pattern recognizer
            self.sequence_pattern_recognizer = SequencePatternRecognizer(data)

            # Initialize signal correlation analyzer
            self.correlation_analyzer = SignalCorrelationAnalyzer()

            # Store the recent signals from other pairs for correlation analysis
            self.recent_signals = {}

            logger.info(f"SignalGenerator initialized with enhanced features for {pair or 'unknown'} {timeframe or 'unknown'}")
        except Exception as e:
            logger.error(f"Error initializing signal enhancement features: {str(e)}", exc_info=True)
            self.volume_analyzer = None
            self.market_regime_detector = None
            self.sequence_pattern_recognizer = None
            self.correlation_analyzer = None
            self.recent_signals = {}

        logger.info(f"SignalGenerator initialized for {pair or 'unknown'} {timeframe or 'unknown'}")

    def add_indicators(self) -> pd.DataFrame:
        """Add technical indicators to the dataframe."""
        try:
            df = self.data.copy()

            # Calculate RSI
            df['rsi'] = RSIIndicator(close=df['close'], window=config.RSI_PERIOD).rsi()

            # Calculate MACD
            macd = MACD(
                close=df['close'],
                window_slow=config.MACD_SLOW,
                window_fast=config.MACD_FAST,
                window_sign=config.MACD_SIGNAL
            )
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()

            # Calculate Moving Averages
            df['sma20'] = SMAIndicator(close=df['close'], window=20).sma_indicator()
            df['sma50'] = SMAIndicator(close=df['close'], window=50).sma_indicator()
            df['sma200'] = SMAIndicator(close=df['close'], window=200).sma_indicator()
            df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
            df['ema21'] = EMAIndicator(close=df['close'], window=21).ema_indicator()
            df['ema50'] = EMAIndicator(close=df['close'], window=50).ema_indicator()

            # Calculate Parabolic SAR with strategy parameters
            try:
                from technical_analysis import calculate_parabolic_sar
                df['psar'] = calculate_parabolic_sar(
                    df,
                    acceleration=0.02,  # Strategy-specific parameter
                    maximum=0.2  # Strategy-specific parameter
                )
                logger.info(f"Added Parabolic SAR indicator for {self.pair or 'unknown'}")
            except Exception as e:
                logger.error(f"Error calculating Parabolic SAR: {str(e)}")
                df['psar'] = None

            # Calculate Stochastic Oscillator
            stoch = StochasticOscillator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=config.STOCHASTIC_K_PERIOD,
                smooth_window=config.STOCHASTIC_D_PERIOD
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()

            # Calculate Bollinger Bands
            bollinger = BollingerBands(
                close=df['close'],
                window=config.BOLLINGER_PERIOD,
                window_dev=config.BOLLINGER_STD
            )
            df['bb_upper'] = bollinger.bollinger_hband()
            df['bb_middle'] = bollinger.bollinger_mavg()
            df['bb_lower'] = bollinger.bollinger_lband()
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

            # Calculate ADX
            adx_indicator = ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            )
            df['adx'] = adx_indicator.adx()
            df['adx_pos'] = adx_indicator.adx_pos()
            df['adx_neg'] = adx_indicator.adx_neg()

            # Calculate ATR (Average True Range)
            df['atr'] = AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=14
            ).average_true_range()

            # Calculate CCI with strategy-specific period
            try:
                df['cci'] = CCIIndicator(
                    high=df['high'],
                    low=df['low'],
                    close=df['close'],
                    window=45  # Strategy-specific parameter
                ).cci()
                logger.info(f"Added CCI indicator for {self.pair or 'unknown'}")
            except Exception as e:
                logger.error(f"Error calculating CCI: {str(e)}")
                df['cci'] = None

            logger.info(f"Added indicators to DataFrame for {self.pair or 'unknown'} {self.timeframe or 'unknown'}")
            return df
        except Exception as e:
            logger.error(f"Error adding indicators: {str(e)}")
            return self.data

    def generate_signal(self) -> Dict:
        """Generate trading signal with enhanced analysis and confirmation."""
        try:
            logger.info(f"Starting signal generation for {self.pair or 'unknown'} {self.timeframe or 'unknown'}")

            # Add indicators to data
            df_with_indicators = self.add_indicators()

            # Get basic technical analysis signal
            analysis_results = self.technical_analyzer.analyze()
            tech_signal = self._evaluate_signals(
                analysis_results.get('signal', {}).get('trend', 'NEUTRAL'),
                analysis_results.get('patterns', {}),
                analysis_results.get('support_resistance', {}),
                df_with_indicators.iloc[-1]
            )

            # Generate sentiment analysis
            sentiment_result = self.sentiment_analyzer.analyze_market_sentiment(self.pair, df_with_indicators)

            # Generate ML prediction if enabled
            ml_prediction = None
            if self.ml_predictor and config.USE_ML_PREDICTIONS:
                logger.info("Generating ML prediction")
                ml_prediction = self.ml_predictor.predict(df_with_indicators)
                logger.info(f"ML prediction: {ml_prediction['direction']} ({ml_prediction['confidence']:.1f}%)")

            # Combine all signals
            final_signal = self._combine_signals_with_sentiment(tech_signal, ml_prediction, sentiment_result)

            # Run signal confirmation if available
            if ENHANCED_ANALYSIS_ENABLED and self.signal_confirmation:
                try:
                    final_signal = self.signal_confirmation.confirm_signal(final_signal, df_with_indicators)
                except Exception as e:
                    logger.error(f"Error in signal confirmation: {str(e)}")

            # Add additional context
            final_signal.update({
                'pair': self.pair,
                'timeframe': self.timeframe,
                'timestamp': datetime.now().isoformat(),
                'price': float(df_with_indicators['close'].iloc[-1]),
                'sentiment': sentiment_result,
                'ml_prediction': ml_prediction
            })

            # Log signal generation
            logger.info(f"Generated signal for {self.pair}: {final_signal['direction']} "
                       f"(Confidence: {final_signal['confidence']:.1f}%)")

            return final_signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'pair': self.pair,
                'timeframe': self.timeframe,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def _combine_signals_with_sentiment(self, tech_signal: Dict, ml_signal: Optional[Dict], sentiment: Dict) -> Dict:
        """Combine technical, ML, and sentiment signals into a final signal."""
        direction = 'NEUTRAL'
        confidence = 0
        signal_count = 0
        confidence_sum = 0

        # Technical Analysis (40% weight)
        if tech_signal['direction'] != 'NEUTRAL':
            signal_count += 1
            confidence_sum += tech_signal['confidence'] * 0.4

        # ML Prediction (30% weight if available)
        if ml_signal and ml_signal['direction'] != 'NEUTRAL':
            signal_count += 1
            confidence_sum += ml_signal['confidence'] * 0.3

        # Sentiment Analysis (30% weight)
        sentiment_score = sentiment['combined_score']
        sentiment_direction = sentiment['overall_sentiment']
        if sentiment_direction != 'NEUTRAL':
            signal_count += 1
            confidence_sum += sentiment['confidence'] * 0.3

        # Determine final direction
        if signal_count >= 2:  # Require at least 2 agreeing signals
            bullish_signals = sum(1 for s in [
                tech_signal['direction'] == 'BUY',
                ml_signal['direction'] == 'BUY' if ml_signal else False,
                sentiment_direction == 'BULLISH'
            ] if s)

            bearish_signals = sum(1 for s in [
                tech_signal['direction'] == 'SELL',
                ml_signal['direction'] == 'SELL' if ml_signal else False,
                sentiment_direction == 'BEARISH'
            ] if s)

            if bullish_signals > bearish_signals:
                direction = 'BUY'
            elif bearish_signals > bullish_signals:
                direction = 'SELL'

        # Calculate final confidence
        if signal_count > 0:
            confidence = confidence_sum / signal_count

        return {
            'direction': direction,
            'confidence': confidence,
            'technical_signal': tech_signal['direction'],
            'ml_signal': ml_signal['direction'] if ml_signal else 'NEUTRAL',
            'sentiment_signal': sentiment_direction,
            'agreement_level': signal_count
        }

    @staticmethod
    def generate_multi_timeframe_signal(timeframe_data, pair):
        """
        Generate signals across multiple timeframes and combine them.

        Args:
            timeframe_data (dict): Dictionary with timeframe as key and market data as value
            pair (str): Trading pair symbol

        Returns:
            dict: Combined multi-timeframe signal
        """
        signals = {}
        total_confidence = 0
        signal_count = 0

        # Generate signals for each timeframe
        for timeframe, data in timeframe_data.items():
            if data is not None and len(data) > 0:
                try:
                    generator = SignalGenerator(data, pair, timeframe)
                    signal = generator.generate_signal()
                    if signal and signal.get('direction') != 'NEUTRAL':
                        signals[timeframe] = signal
                        total_confidence += signal.get('confidence', 0)
                        signal_count += 1
                except Exception as e:
                    logger.warning(f"Error generating signal for {timeframe}: {e}")
                    continue

        # Combine signals
        if signal_count == 0:
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'timeframe_signals': signals,
                'pair': pair
            }

        # Determine overall direction based on majority
        buy_signals = sum(1 for s in signals.values() if s['direction'] == 'BUY')
        sell_signals = sum(1 for s in signals.values() if s['direction'] == 'SELL')

        if buy_signals > sell_signals:
            direction = 'BUY'
        elif sell_signals > buy_signals:
            direction = 'SELL'
        else:
            direction = 'NEUTRAL'

        # Calculate average confidence
        avg_confidence = total_confidence / signal_count if signal_count > 0 else 0

        return {
            'direction': direction,
            'confidence': avg_confidence,
            'timeframe_signals': signals,
            'pair': pair,
            'signal_count': signal_count
        }

    def _evaluate_signals(self, trend_signal, patterns, support_resistance, latest_data):
        """
        Evaluate technical signals and return a combined signal.

        Args:
            trend_signal (str): Trend direction from technical analysis
            patterns (dict): Detected patterns
            support_resistance (dict): Support and resistance levels
            latest_data (pd.Series): Latest price data with indicators

        Returns:
            dict: Evaluated signal with direction and confidence
        """
        direction = 'NEUTRAL'
        confidence = 0
        signal_strength = 0

        try:
            # Evaluate trend signal
            if trend_signal == 'BULLISH':
                signal_strength += 30
                direction = 'BUY'
            elif trend_signal == 'BEARISH':
                signal_strength += 30
                direction = 'SELL'

            # Evaluate RSI
            if 'rsi' in latest_data and not pd.isna(latest_data['rsi']):
                rsi = latest_data['rsi']
                if rsi < 30:  # Oversold
                    signal_strength += 20
                    if direction == 'NEUTRAL':
                        direction = 'BUY'
                elif rsi > 70:  # Overbought
                    signal_strength += 20
                    if direction == 'NEUTRAL':
                        direction = 'SELL'

            # Evaluate MACD
            if 'macd' in latest_data and 'macd_signal' in latest_data:
                if not pd.isna(latest_data['macd']) and not pd.isna(latest_data['macd_signal']):
                    if latest_data['macd'] > latest_data['macd_signal']:
                        signal_strength += 15
                        if direction == 'NEUTRAL':
                            direction = 'BUY'
                    elif latest_data['macd'] < latest_data['macd_signal']:
                        signal_strength += 15
                        if direction == 'NEUTRAL':
                            direction = 'SELL'

            # Evaluate moving averages
            if 'sma20' in latest_data and 'sma50' in latest_data:
                if not pd.isna(latest_data['sma20']) and not pd.isna(latest_data['sma50']):
                    if latest_data['sma20'] > latest_data['sma50']:
                        signal_strength += 10
                        if direction == 'NEUTRAL':
                            direction = 'BUY'
                    elif latest_data['sma20'] < latest_data['sma50']:
                        signal_strength += 10
                        if direction == 'NEUTRAL':
                            direction = 'SELL'

            # Evaluate patterns
            if patterns:
                bullish_patterns = ['morning_star', 'bullish_engulfing', 'piercing_line', 'hammer']
                bearish_patterns = ['evening_star', 'shooting_star', 'bearish_engulfing', 'dark_cloud_cover']

                for pattern in patterns:
                    if pattern in bullish_patterns:
                        signal_strength += 25
                        if direction == 'NEUTRAL':
                            direction = 'BUY'
                    elif pattern in bearish_patterns:
                        signal_strength += 25
                        if direction == 'NEUTRAL':
                            direction = 'SELL'

            # Calculate confidence (max 100%)
            confidence = min(signal_strength, 100)

            return {
                'direction': direction,
                'confidence': confidence,
                'signal_strength': signal_strength
            }

        except Exception as e:
            logger.error(f"Error evaluating signals: {str(e)}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'signal_strength': 0
            }

def get_latest_signals() -> Dict[str, Dict]:
    """
    Get the latest signals for all supported assets.

    Returns:
        Dictionary mapping asset symbols to their latest signals
    """
    try:
        # Check if signals file exists
        signals_file = 'signal_history.json'
        if not os.path.exists(signals_file):
            logger.warning("Signal history file not found, generating new signals")
            return generate_signals()

        # Load existing signals
        with open(signals_file, 'r') as f:
            signal_history = json.load(f)

        # Get the latest signal for each asset
        latest_signals = {}

        if 'signals' in signal_history and signal_history['signals']:
            # Group signals by asset
            signals_by_asset = {}
            for signal in signal_history['signals']:
                asset = signal.get('pair', 'unknown')
                if asset not in signals_by_asset:
                    signals_by_asset[asset] = []
                signals_by_asset[asset].append(signal)

            # Get the latest signal for each asset
            for asset, signals in signals_by_asset.items():
                # Sort by timestamp and get the latest
                signals.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
                latest_signal = signals[0]

                # Format for display
                latest_signals[asset] = {
                    'direction': latest_signal.get('direction', 'NEUTRAL'),
                    'confidence': latest_signal.get('confidence', 0),
                    'timestamp': latest_signal.get('timestamp', datetime.now().isoformat()),
                    'indicators': 'Enhanced Analysis',
                    'pair': asset
                }

        # If no signals found, generate new ones
        if not latest_signals:
            logger.info("No recent signals found, generating new signals")
            return generate_signals()

        return latest_signals

    except Exception as e:
        logger.error(f"Error getting latest signals: {str(e)}")
        # Fallback to generating new signals
        return generate_signals()