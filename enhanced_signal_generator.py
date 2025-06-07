"""
Enhanced signal generator integrating market regime detection,
signal validation, strategy templates, and advanced ML predictions.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from datetime import datetime
from market_regime import MarketRegimeDetector
from signal_validation import SignalValidator, ValidationResult
from enhanced_ml_model import EnhancedMLPredictor
from technical_analysis import TechnicalAnalyzer
from strategy_templates import StrategyTemplate, TradingStyle, MarketCondition
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class SignalStrength(Enum):
    STRONG = "STRONG"
    MODERATE = "MODERATE"
    WEAK = "WEAK"
    INVALID = "INVALID"

@dataclass
class SignalMetadata:
    regime: str
    regime_confidence: float
    validation_score: float
    validation_reasons: List[str]
    ml_confidence: float
    ml_uncertainty: float
    ensemble_agreement: float
    key_features: Dict[str, float]

@dataclass
class EnhancedSignal:
    direction: str  # BUY, SELL, NEUTRAL
    strength: SignalStrength
    confidence: float
    entry_price: float
    stop_loss: float
    take_profit: float
    metadata: SignalMetadata
    timestamp: pd.Timestamp

class EnhancedSignalGenerator:
    """
    Advanced signal generator that combines market regime detection,
    signal validation, ML predictions, and customizable strategy templates.
    """

    def __init__(self,
                 predictor: EnhancedMLPredictor,
                 pair: str,
                 timeframes: List[str] = None,
                 min_confidence: float = 65.0,
                 volatility_adjustment: bool = True,
                 api_key: str = None,
                 is_demo: bool = True):
                 
        self.predictor = predictor
        self.pair = pair
        self.api_key = api_key
        self.is_demo = is_demo
        self.available_pairs = [
            'EUR/USD', 'GBP/USD', 'USD/JPY', 'AUD/USD', 'USD/CAD',
            'EUR/GBP', 'USD/CHF', 'NZD/USD', 'EUR/JPY', 'GBP/JPY'
        ]
        # Default timeframes for scalping strategy
        default_timeframes = [
            '1m',  # M1 timeframe
            '5m',  # M5 timeframe
            # Additional timeframes can still be used
            '15m', '30m', '1h'
        ]
        
        self.timeframes = timeframes or default_timeframes
        self.min_confidence = min_confidence
        self.volatility_adjustment = volatility_adjustment
        self.technical_analyzer = TechnicalAnalyzer()
        self.strategy_template = StrategyTemplate(pair)
        
        # Parabolic SAR + CCI Strategy Parameters
        self.strategy_params = {
            'sar_step': 0.02,
            'sar_max': 0.2,
            'cci_period': 45,
            'ema50_period': 50,
            'ema21_period': 21,
            'cci_overbought': 100,
            'cci_oversold': -100
        }
        
        # Dynamic target calculation based on pair volatility
        self.target_multiplier = 1.5  # Adjustable based on pair characteristics
        self.stop_loss_multiplier = 1.0  # Will be adjusted based on EMA level
        
        # Parabolic SAR + CCI Strategy Parameters
        self.strategy_params = {
            'sar_step': 0.02,
            'sar_max': 0.2,
            'cci_period': 45,
            'ema50_period': 50,
            'ema21_period': 21,
            'cci_overbought': 100,
            'cci_oversold': -100
        }
        
        # Dynamic target calculation based on pair volatility
        self.target_multiplier = 1.5  # Adjustable based on pair characteristics
        self.stop_loss_multiplier = 1.0  # Will be adjusted based on EMA level

    def generate_signal(self, data: Dict[str, pd.DataFrame], strategy_name: Optional[str] = None) -> Optional[EnhancedSignal]:
            
        """  Generate an enhanced trading signal using multiple analysis layers and strategy templates
        
        Args:
        data: Dictionary of DataFrames with market data for different timeframes
        strategy_name: Optional name of strategy template to use
            
        Returns:
            EnhancedSignal object if a valid signal is found, None otherwise
        """
        try:
            # 1. Enhanced Market Regime Detection
            regime_detector = MarketRegimeDetector()
            regime_info = regime_detector.detect_regime(data[self.timeframes[0]])

            # 2. Get and optimize strategy parameters
            strategy_params = self.strategy_template.get_strategy(
                strategy_name or "swing_trader",
                self.risk_level
            )

            # Get market sentiment and volatility
            sentiment_data = self._get_market_sentiment()
            volatility = self._calculate_market_volatility(data[self.timeframes[0]])

            # Optimize strategy for current conditions
            optimized_params = self.strategy_template.optimize_strategy(
                strategy_params,
                sentiment_data,
                volatility
            )            # 3. Calculate Parabolic SAR + CCI signals
            scalping_signals = self._calculate_scalping_signals(data[self.timeframes[0]])  # M1 timeframe
            
            # 4. Generate Features with strategy-specific indicators
            features = self._prepare_features(data, optimized_params.indicators)

            # 5. ML Prediction with Uncertainty
            predictions = self.predictor.predict(features)
            
            # Combine ML prediction with scalping signals
            if scalping_signals['direction'] != 'NEUTRAL':
                primary_prediction = predictions[0]
                primary_prediction.signal = scalping_signals['direction']  # Override with strategy signal
                primary_prediction.confidence *= scalping_signals['cci_strength']  # Adjust confidence
              # 5. Signal Validation with multi-timeframe awareness
            timeframe_categories = self._categorize_timeframes()

            # Initialize validators for different timeframe categories
            ultra_short_validator = SignalValidator(data[timeframe_categories['ultra_short'][0]],
                                                 timeframe_categories['ultra_short'])
            short_validator = SignalValidator(data[timeframe_categories['short'][0]],
                                           timeframe_categories['short'])

            # Validate across timeframe categories
            ultra_short_validation = ultra_short_validator.validate_signal({
                'direction': primary_prediction.signal,
                'confidence': primary_prediction.confidence,
                'strategy_params': optimized_params
            })

            short_validation = short_validator.validate_signal({
                'direction': primary_prediction.signal,
                'confidence': primary_prediction.confidence,
                'strategy_params': optimized_params
            })

            # Combine validations with timeframe weights
            validation_score = (
                sum(self._get_timeframe_weight(tf) for tf in timeframe_categories['ultra_short']) * ultra_short_validation.confidence_adjustment +
                sum(self._get_timeframe_weight(tf) for tf in timeframe_categories['short']) * short_validation.confidence_adjustment
            ) / sum(self._get_timeframe_weight(tf) for tf in self.timeframes)

            # Create combined validation result
            validation = ultra_short_validation
            validation.confidence_adjustment = validation_score
            validation.reason = f"Ultra-short: {ultra_short_validation.reason} | Short: {short_validation.reason}"

            # 6. Combine and Adjust Signal
            final_confidence = self._calculate_final_confidence(
                base_confidence=primary_prediction.confidence,
                uncertainty=primary_prediction.uncertainty,
                validation_result=validation,
                regime_confidence=regime_info['confidence'],
                ensemble_agreement=primary_prediction.ensemble_agreement,
                sentiment_weight=optimized_params.sentiment_weight
            )

            # Determine signal strength with strategy thresholds
            strength = self._determine_signal_strength(
                confidence=final_confidence,
                uncertainty=primary_prediction.uncertainty,
                validation=validation,
                regime_alignment=(regime_info['regime'] == optimized_params.market_condition.value),
                confidence_threshold=optimized_params.signal_confidence_threshold
            )

            if strength == SignalStrength.INVALID:
                return None
              # Calculate price levels using multi-timeframe aware calculation
            price_levels = self._calculate_adjusted_price_levels(
                data=data,
                direction=primary_prediction.signal,
                strategy_params=optimized_params
            )

            # Adjust levels based on volatility if enabled
            if self.volatility_adjustment:
                price_levels = self._adjust_levels_for_volatility(price_levels, data)

            # Create enhanced metadata with strategy info
            metadata = SignalMetadata(
                regime=regime_info['regime'],
                regime_confidence=regime_info['confidence'],
                validation_score=validation.confidence_adjustment,
                validation_reasons=validation.reason.split(" | "),
                stop_loss=price_levels['stop_loss'],
                take_profit=price_levels['take_profit'],
                metadata=metadata,
                timestamp=pd.Timestamp.now()
            )
            
            # Create enhanced signal with all calculated data
            signal = EnhancedSignal(
                direction=primary_prediction.signal,
                strength=strength,
                confidence=final_confidence,
                entry_price=price_levels['entry_price'],
                stop_loss=price_levels['stop_loss'],
                take_profit=price_levels['take_profit'],
                metadata=metadata,
                timestamp=pd.Timestamp.now()
            )

            # Format signal with enhanced UI
            signal.formatted_message = self.format_signal_message(signal)
            return signal

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            return None

    def _prepare_features(self, data: Dict[str, pd.DataFrame], indicators: List[str]) -> pd.DataFrame:
        """Prepare features based on strategy indicators with timeframe awareness"""
        features = {}
        timeframe_categories = self._categorize_timeframes()

        # Calculate baseline features from ultra-short timeframes
        for tf in timeframe_categories['ultra_short']:
            tf_data = data[tf]
            for indicator in indicators:
                if hasattr(self.technical_analyzer, f"calculate_{indicator.lower()}"):
                    indicator_values = getattr(self.technical_analyzer, f"calculate_{indicator.lower()}")(tf_data)
                    if isinstance(indicator_values, pd.DataFrame):
                        features.update({f"{k}_ultra_{tf}": v for k, v in indicator_values.iloc[-1].to_dict().items()})
                    else:
                        features[f"{indicator.lower()}_ultra_{tf}"] = indicator_values.iloc[-1]

        # Add short timeframe features with higher weights
        for tf in timeframe_categories['short']:
            tf_data = data[tf]
            weight = self._get_timeframe_weight(tf)
            for indicator in indicators:
                if hasattr(self.technical_analyzer, f"calculate_{indicator.lower()}"):
                    tf_values = getattr(self.technical_analyzer, f"calculate_{indicator.lower()}")(tf_data)
                    if isinstance(tf_values, pd.DataFrame):
                        # Apply timeframe weight to feature values
                        weighted_values = {f"{k}_short_{tf}": v * weight
                                        for k, v in tf_values.iloc[-1].to_dict().items()}
                        features.update(weighted_values)
                    else:
                        features[f"{indicator.lower()}_short_{tf}"] = tf_values.iloc[-1] * weight

        # Add cross-timeframe correlation features
        features.update(self._calculate_timeframe_correlations(data, timeframe_categories))

        return pd.DataFrame([features])

    def _get_market_sentiment(self) -> Dict:
        """Get market sentiment data"""
        # Implement sentiment analysis here
        return {'composite_score': 0.5}  # Default neutral sentiment

    def _calculate_market_volatility(self, data: pd.DataFrame) -> float:
        """Calculate market volatility"""
        atr = self.technical_analyzer.calculate_atr(data)
        avg_price = data['close'].mean()
        return (atr.iloc[-1] / avg_price)

    def _calculate_final_confidence(self,
                                  base_confidence: float,
                                  uncertainty: float,
                                  validation_result: ValidationResult,
                                  regime_confidence: float,
                                  ensemble_agreement: float,
                                  sentiment_weight: float) -> float:
        """Calculate final confidence score with all factors"""

        # Start with base ML confidence
        confidence = base_confidence

        # Adjust for uncertainty (penalize high uncertainty)
        uncertainty_factor = 1 - (uncertainty / 100)
        confidence *= uncertainty_factor

        # Add validation adjustment
        confidence += validation_result.confidence_adjustment

        # Consider regime confidence
        regime_factor = regime_confidence / 100
        confidence *= (0.7 + 0.3 * regime_factor)  # Weight regime less than direct signal

        # Factor in ensemble agreement
        agreement_factor = ensemble_agreement / 100
        confidence *= (0.8 + 0.2 * agreement_factor)

        # Adjust based on market sentiment (if applicable)
        confidence += (sentiment_weight - 0.5) * 20  # Scale sentiment impact

        # Ensure confidence is within bounds
        return max(0, min(100, confidence))

    def _determine_signal_strength(self,
                                 confidence: float,
                                 uncertainty: float,
                                 validation: ValidationResult,
                                 regime_alignment: bool,
                                 confidence_threshold: float) -> SignalStrength:
        """Determine signal strength based on multiple factors"""

        if not validation.is_valid or confidence < self.min_confidence:
            return SignalStrength.INVALID

        # Base score considering confidence and uncertainty
        base_score = confidence * (1 - uncertainty / 100)

        if base_score >= confidence_threshold:
            return SignalStrength.STRONG
        elif base_score >= 70 and uncertainty <= 30 and validation.is_valid:
            return SignalStrength.MODERATE
        elif base_score >= self.min_confidence:
            return SignalStrength.WEAK
        else:
            return SignalStrength.INVALID

    def _calculate_price_levels(self,
                              entry_price: float,
                              direction: str,
                              strategy_params: dict,
                              data: pd.DataFrame) -> Dict[str, float]:
        """Calculate price levels based on strategy parameters"""

        atr = self.technical_analyzer.calculate_atr(data)

        # Get multipliers from strategy parameters
        sl_multiplier = strategy_params.stop_loss_multiplier
        tp_multiplier = strategy_params.take_profit_multiplier

        # Calculate levels using ATR
        atr_value = atr.iloc[-1]

        if direction == "BUY":
            stop_loss = entry_price - (atr_value * sl_multiplier)
            take_profit = entry_price + (atr_value * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (atr_value * sl_multiplier)
            take_profit = entry_price - (atr_value * tp_multiplier)

        return {
            'stop_loss': stop_loss,
            'take_profit': take_profit
        }

    def _calculate_adjusted_price_levels(self,
                              data: Dict[str, pd.DataFrame],
                              direction: str,
                              strategy_params: dict) -> Dict[str, float]:
        """Calculate price levels considering both ultra-short and short timeframes."""
        timeframe_categories = self._categorize_timeframes()

        # Calculate ATR for both timeframe categories
        ultra_short_atr = self.technical_analyzer.calculate_atr(
            data[timeframe_categories['ultra_short'][0]]
        ).iloc[-1]

        short_atr = self.technical_analyzer.calculate_atr(
            data[timeframe_categories['short'][0]]
        ).iloc[-1]

        # Weight ATRs based on timeframe category
        weighted_atr = (
            ultra_short_atr * 0.4 +  # Lower weight for ultra-short
            short_atr * 0.6          # Higher weight for short timeframe
        )

        entry_price = data[self.timeframes[0]]['close'].iloc[-1]

        # Get multipliers from strategy parameters
        sl_multiplier = strategy_params.stop_loss_multiplier
        tp_multiplier = strategy_params.take_profit_multiplier

        if direction == "BUY":
            stop_loss = entry_price - (weighted_atr * sl_multiplier)
            take_profit = entry_price + (weighted_atr * tp_multiplier)
        else:  # SELL
            stop_loss = entry_price + (weighted_atr * sl_multiplier)
            take_profit = entry_price - (weighted_atr * tp_multiplier)

        return {
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'atr': weighted_atr
        }

    def _adjust_levels_for_volatility(self,
                                    price_levels: Dict[str, float],
                                    data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
        """Adjust price levels based on current market volatility."""
        if not self.volatility_adjustment:
            return price_levels

        timeframe_categories = self._categorize_timeframes()

        # Calculate volatility for both timeframe categories
        ultra_short_vol = self._calculate_volatility(
            data[timeframe_categories['ultra_short'][0]]
        )
        short_vol = self._calculate_volatility(
            data[timeframe_categories['short'][0]]
        )

        # Weight volatilities
        weighted_vol = (
            ultra_short_vol * 0.35 +  # Lower weight for ultra-short
            short_vol * 0.65          # Higher weight for short timeframe
        )

        # Adjust levels based on volatility
        vol_factor = 1.0
        if weighted_vol > 1.5:  # High volatility
            vol_factor = 1.2    # Wider levels
        elif weighted_vol < 0.5:  # Low volatility
            vol_factor = 0.8    # Tighter levels

        return {
            'stop_loss': price_levels['stop_loss'] * vol_factor,
            'take_profit': price_levels['take_profit'] * vol_factor,
            'atr': price_levels['atr']
        }

    def _calculate_volatility(self, data: pd.DataFrame, window: int = 20) -> float:
        """Calculate normalized volatility for a given timeframe."""
        returns = data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        return volatility

    def _get_top_features(self, feature_importances: Dict[str, float],
                         top_n: int = 5) -> Dict[str, float]:
        """Get top n most important features"""
        sorted_features = sorted(
            feature_importances.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )
        return dict(sorted_features[:top_n])

    def _categorize_timeframes(self) -> Dict[str, List[str]]:
        """Categorize timeframes into ultra-short and short term."""
        categories = {
            'ultra_short': [],
            'short': []
        }

        for tf in self.timeframes:
            if tf in ['5s', '15s', '30s']:
                categories['ultra_short'].append(tf)
            else:  # 1m, 3m, 5m, 10m, 15m
                categories['short'].append(tf)

        return categories

    def _get_timeframe_weight(self, timeframe: str) -> float:
        """Get weight for timeframe in signal calculation."""
        # Ultra-short timeframes get lower weights
        if timeframe in ['5s', '15s', '30s']:
            return 0.5
        # Higher weights for standard timeframes
        weights = {
            '1m': 0.6,
            '3m': 0.7,
            '5m': 0.8,
            '10m': 0.9,
            '15m': 1.0
        }
        return weights.get(timeframe, 0.7)  # Default weight if not specified

    def _calculate_timeframe_correlations(self, data: Dict[str, pd.DataFrame],
                                     timeframe_categories: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate correlation features between different timeframe categories."""
        correlations = {}

        # Calculate correlations between ultra-short and short timeframes
        for ultra_tf in timeframe_categories['ultra_short']:
            for short_tf in timeframe_categories['short']:
                # Resample ultra-short data to match short timeframe
                ultra_data = data[ultra_tf]['close'].resample(short_tf).last()
                short_data = data[short_tf]['close']

                # Calculate correlation over last 20 periods
                correlation = ultra_data.tail(20).corr(short_data.tail(20))
                correlations[f'corr_{ultra_tf}_{short_tf}'] = correlation

        # Add momentum alignment features
        for ultra_tf in timeframe_categories['ultra_short']:
            ultra_momentum = data[ultra_tf]['close'].pct_change(5).tail(1).values[0]

            for short_tf in timeframe_categories['short']:
                short_momentum = data[short_tf]['close'].pct_change(5).tail(1).values[0]
                # Check if momentum is aligned (both positive or both negative)
                correlations[f'momentum_align_{ultra_tf}_{short_tf}'] = 1.0 if (
                    (ultra_momentum > 0 and short_momentum > 0) or
                    (ultra_momentum < 0 and short_momentum < 0)
                ) else 0.0

        return correlations

    def _validate_timeframe_alignment(self,
                                 data: Dict[str, pd.DataFrame],
                                 direction: str) -> Dict[str, float]:
        """
        Validate signal alignment across different timeframe categories.
        Returns alignment scores and confidence adjustments.
        """
        timeframe_categories = self._categorize_timeframes()
        alignments = {
            'ultra_short_alignment': 0.0,
            'short_alignment': 0.0,
            'cross_category_alignment': 0.0
        }

        # Check alignment within ultra-short timeframes
        ultra_short_signals = []
        for tf in timeframe_categories['ultra_short']:
            signal = self.technical_analyzer.get_signal_direction(data[tf])
            ultra_short_signals.append(1 if signal == direction else -1)

        alignments['ultra_short_alignment'] = sum(ultra_short_signals) / len(ultra_short_signals)

        # Check alignment within short timeframes
        short_signals = []
        for tf in timeframe_categories['short']:
            signal = self.technical_analyzer.get_signal_direction(data[tf])
            short_signals.append(1 if signal == direction else -1)

        alignments['short_alignment'] = sum(short_signals) / len(short_signals)

        # Calculate cross-category alignment
        cross_alignment = (
            alignments['ultra_short_alignment'] * 0.4 +  # Lower weight for ultra-short
            alignments['short_alignment'] * 0.6         # Higher weight for short timeframes
        )
        alignments['cross_category_alignment'] = cross_alignment

        return alignments

    def _calculate_scalping_signals(self, data: pd.DataFrame) -> Dict[str, float]:
        """Calculate signals based on Parabolic SAR and CCI strategy"""
        
        # Calculate indicators
        sar = self.technical_analyzer.calculate_sar(
            data,
            acceleration=self.strategy_params['sar_step'],
            maximum=self.strategy_params['sar_max']
        )
        
        cci = self.technical_analyzer.calculate_cci(
            data,
            period=self.strategy_params['cci_period']
        )
        
        ema50 = self.technical_analyzer.calculate_ema(
            data['close'],
            period=self.strategy_params['ema50_period']
        )
        
        ema21 = self.technical_analyzer.calculate_ema(
            data['close'],
            period=self.strategy_params['ema21_period']
        )
        
        # Get latest values
        current_price = data['close'].iloc[-1]
        current_sar = sar.iloc[-1]
        current_cci = cci.iloc[-1]
        current_ema50 = ema50.iloc[-1]
        current_ema21 = ema21.iloc[-1]
        
        # Generate signals
        long_signal = (
            current_price > current_sar and  # Price above SAR
            current_cci > self.strategy_params['cci_overbought'] and  # CCI above 100
            current_price > current_ema50 and  # Price above EMA50
            current_ema21 > current_ema50  # EMA21 above EMA50
        )
        
        short_signal = (
            current_price < current_sar and  # Price below SAR
            current_cci < self.strategy_params['cci_oversold'] and  # CCI below -100
            current_price < current_ema50 and  # Price below EMA50
            current_ema21 < current_ema50  # EMA21 below EMA50
        )
        
        return {
            'direction': 'BUY' if long_signal else 'SELL' if short_signal else 'NEUTRAL',
            'sar_distance': abs(current_price - current_sar) / current_price,
            'cci_strength': abs(current_cci) / 100,
            'ema_alignment': 1 if current_ema21 > current_ema50 else -1,
            'price_to_ema': (current_price - current_ema50) / current_price
        }

    def _adjust_targets_for_pair(self, pair: str) -> Dict[str, float]:
        """Adjust target and stop loss levels based on pair characteristics"""
        # Default targets
        default_targets = {
            'target_pips': 10,
            'stop_pips': 7
        }
        
        # Adjust based on pair volatility and characteristics
        major_pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF']
        if pair in major_pairs:
            return {
                'target_pips': default_targets['target_pips'] * 1.0,
                'stop_pips': default_targets['stop_pips'] * 1.0
            }
        else:
            # For more volatile pairs, increase targets
            return {
                'target_pips': default_targets['target_pips'] * 1.2,
                'stop_pips': default_targets['stop_pips'] * 1.2
            }

    def _calculate_optimal_entry(self, data: pd.DataFrame, direction: str) -> float:
        """Calculate optimal entry price based on current market conditions"""
        current_price = data['close'].iloc[-1]
        atr = self.technical_analyzer.calculate_atr(data).iloc[-1]
        
        if direction == 'BUY':
            # For buys, try to enter slightly above current price on momentum
            return current_price * (1 + (0.1 * atr / current_price))
        else:
            # For sells, try to enter slightly below current price on momentum
            return current_price * (1 - (0.1 * atr / current_price))   
    def format_signal_message(self, signal: EnhancedSignal) -> str:
        """Format signal information in a clear, bold UI format matching the Telegram bot style"""
        
        message = [
            "🎯 *MasterTrade Bot | Premium*\n",
            f"*BOT SIGNAL*\n*{signal.direction}*",
            "✓ Correct within 5 seconds of receipt\n",
            "\n⚜️ *Trading Mode:*",
            f"{'DEMO MODE' if self.is_demo else 'REAL TRADING'}\n",
            "\n🔧 *Signal information:*",
            f"{self.pair} (OTC) — {self.timeframes[0]}\n",
            "\n📰 *Market Setting:*",
            f"Info context: {signal.metadata.regime}",
            f"Volatility: {self._get_volatility_label(signal.metadata)}\n",
            "\n💻 *Technical overview:*",
            "Only for stock quotes\n",
            "\n📊 *Probabilities:*",
            f"Signal reliability: {int(signal.confidence)}%\n",
            "\n📍 *Bot signal:*",
            f"{signal.direction} ➡️\n",
            "\n🎮 *Actions:*",
            "• Refresh Signal",
            "• Toggle Demo/Real Mode",
            "• Back to Menu",
            f"\n💰 Take Profit: {signal.take_profit:.5f}",
            f"🛑 Stop Loss: {signal.stop_loss:.5f}"
        ]
        
        return "\n".join(message)
        
    def _get_volatility_label(self, metadata: SignalMetadata) -> str:
        """Convert volatility to human-readable label"""
        if metadata.regime_confidence > 0.7:
            return "High"
        elif metadata.regime_confidence > 0.4:
            return "Moderate"
        else:
            return "Low"
    
    def get_recommended_signals(self) -> Dict[str, Dict]:
        """Analyze all available pairs and return the most promising trading opportunities"""
        recommendations = {}
        
        for pair in self.available_pairs:
            try:
                # Get data for pair
                data = self.data_fetcher.fetch_data(pair, self.timeframes[0])
                
                # Calculate indicators
                signals = self._calculate_scalping_signals(data)
                
                # Only include if there's a clear signal
                if signals['direction'] != 'NEUTRAL':
                    confidence = signals['cci_strength'] * 100
                    if confidence >= self.min_confidence:
                        recommendations[pair] = {
                            'direction': signals['direction'],
                            'timeframe': self.timeframes[0],
                            'confidence': confidence,
                            'sar_alignment': signals['sar_distance'],
                            'ema_trend': signals['ema_alignment']
                        }
            except Exception as e:
                logger.error(f"Error analyzing {pair}: {str(e)}")
                continue
                
        # Sort by confidence
        sorted_recommendations = dict(
            sorted(recommendations.items(), 
                  key=lambda x: x[1]['confidence'], 
                  reverse=True)
        )
        
        return sorted_recommendations

    def format_recommendation_message(self, recommendations: Dict[str, Dict]) -> str:
        """Format the recommendations into a user-friendly message"""
        if not recommendations:
            return "🔍 No strong trading opportunities detected at the moment."
            
        message = ["🎯 *MasterTrade Bot | Signal Recommendations*\n"]
        message.append("*Best Trading Opportunities:*\n")
        
        for pair, details in recommendations.items():
            confidence = int(details['confidence'])
            direction = details['direction']
            timeframe = details['timeframe']
            
            message.append(f"💫 *{pair}*")
            message.append(f"Signal: {direction} on {timeframe}")
            message.append(f"Confidence: {confidence}%")
            message.append(f"Trend Strength: {self._get_trend_strength_label(details)}\n")
            
        message.append("\n⚡️ *Actions:*")
        message.append("• Select an asset to trade")
        message.append("• Refresh signals")
        message.append("• Toggle Demo/Real mode")
        message.append("• Back to menu")
        
        return "\n".join(message)
        
    def _get_trend_strength_label(self, details: Dict) -> str:
        """Convert numerical trend strength to label"""
        if details['ema_trend'] > 0.8:
            return "Very Strong"
        elif details['ema_trend'] > 0.5:
            return "Strong"
        elif details['ema_trend'] > 0.2:
            return "Moderate"
        else:
            return "Weak"
