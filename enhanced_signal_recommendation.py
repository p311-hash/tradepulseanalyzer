"""
Enhanced Signal Recommendation System with advanced weighting, confidence calculation,
and historical performance tracking.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import json

logger = logging.getLogger(__name__)

class SignalComponentWeight:
    """Adaptive weights for signal components."""
    def __init__(self, initial_weights: Optional[Dict[str, float]] = None):
        self.weights = initial_weights or {
            'technical': 0.30,  # Technical analysis
            'pattern': 0.20,    # Pattern recognition
            'ml': 0.25,         # Machine learning
            'sentiment': 0.15,  # Market sentiment
            'volume': 0.10      # Volume analysis
        }
        self.performance_history: List[Dict] = []
        self.adaptation_rate = 0.05  # Weight adjustment rate

    def adapt_weights(self, signal_performance: Dict):
        """Adapt weights based on component performance."""
        total_adjustment = 0
        best_component = max(signal_performance.items(), key=lambda x: x[1])[0]
        
        for component, performance in signal_performance.items():
            if component in self.weights:
                # Increase weight for better performing components
                adjustment = self.adaptation_rate * (performance - 0.5)
                self.weights[component] = max(0.05, min(0.4, self.weights[component] + adjustment))
                total_adjustment += adjustment

        # Normalize weights to sum to 1
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}

        logger.info(f"Adapted weights: {self.weights}")

class EnhancedSignalRecommender:
    """Enhanced signal recommender with advanced features."""
    
    def __init__(self):
        self.component_weights = SignalComponentWeight()
        self.signal_history: List[Dict] = []
        self.confidence_threshold = 65.0
        self.min_validation_signals = 3
        self.history_window = timedelta(days=30)
        
    def _calculate_advanced_confidence(self, components: Dict[str, Dict]) -> Tuple[float, List[str]]:
        """Calculate confidence with detailed reasoning."""
        confidence_scores = []
        explanations = []
        
        # Technical Analysis
        if 'technical' in components:
            tech_conf = components['technical'].get('confidence', 0) * self.component_weights.weights['technical']
            confidence_scores.append(tech_conf)
            if tech_conf > 0.2:  # Significant contribution
                explanations.append(f"Technical indicators show {'strong' if tech_conf > 0.3 else 'moderate'} confirmation")

        # Pattern Recognition
        if 'pattern' in components:
            pattern_conf = components['pattern'].get('confidence', 0) * self.component_weights.weights['pattern']
            confidence_scores.append(pattern_conf)
            if pattern_conf > 0.15:
                explanations.append(f"Detected {components['pattern'].get('pattern_count', 0)} confirming patterns")

        # Machine Learning
        if 'ml' in components:
            ml_conf = components['ml'].get('confidence', 0) * self.component_weights.weights['ml']
            confidence_scores.append(ml_conf)
            if ml_conf > 0.2:
                explanations.append(f"ML model predicts with {components['ml'].get('confidence', 0):.1f}% confidence")

        # Market Sentiment
        if 'sentiment' in components:
            sent_conf = components['sentiment'].get('confidence', 0) * self.component_weights.weights['sentiment']
            confidence_scores.append(sent_conf)
            if sent_conf > 0.1:
                explanations.append(f"Market sentiment aligns with prediction")

        # Volume Analysis
        if 'volume' in components:
            vol_conf = components['volume'].get('confidence', 0) * self.component_weights.weights['volume']
            confidence_scores.append(vol_conf)
            if vol_conf > 0.08:
                explanations.append(f"Volume analysis supports the signal")

        # Calculate final confidence
        final_confidence = sum(confidence_scores) * 100  # Convert to percentage
        
        return final_confidence, explanations

    def _validate_signal_quality(self, confidence: float, components: Dict) -> Tuple[bool, List[str]]:
        """Validate signal quality with multiple criteria."""
        validations = []
        warnings = []
        
        # Check base confidence
        if confidence < self.confidence_threshold:
            warnings.append(f"Signal confidence ({confidence:.1f}%) below threshold ({self.confidence_threshold}%)")
        
        # Check component agreement
        directions = [comp.get('direction', 'NEUTRAL') for comp in components.values()]
        unique_directions = set(directions) - {'NEUTRAL'}
        if len(unique_directions) > 1:
            warnings.append("Mixed signals from different components")
        
        # Check minimum validations
        valid_components = sum(1 for comp in components.values() if comp.get('confidence', 0) > 0.5)
        if valid_components < self.min_validation_signals:
            warnings.append(f"Insufficient signal validations ({valid_components}/{self.min_validation_signals})")
        
        # Market regime validation
        if 'market_regime' in components:
            regime = components['market_regime'].get('regime', 'UNKNOWN')
            if regime == 'HIGH_VOLATILITY':
                warnings.append("High market volatility detected - use caution")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings

    def record_signal_outcome(self, signal_id: str, outcome: str, profit_loss: float):
        """Record signal outcome for historical performance tracking."""
        try:
            # Find signal in history
            signal = next((s for s in self.signal_history if s['id'] == signal_id), None)
            if signal:
                signal['outcome'] = outcome
                signal['profit_loss'] = profit_loss
                signal['components_performance'] = self._calculate_component_performance(signal, outcome)
                
                # Update component weights based on performance
                self.component_weights.adapt_weights(signal['components_performance'])
                
                # Trim old history
                cutoff_time = datetime.now() - self.history_window
                self.signal_history = [s for s in self.signal_history 
                                     if datetime.fromisoformat(s['timestamp']) > cutoff_time]
                
                logger.info(f"Recorded outcome for signal {signal_id}: {outcome} ({profit_loss:.2f}%)")
            else:
                logger.warning(f"Signal {signal_id} not found in history")
        
        except Exception as e:
            logger.error(f"Error recording signal outcome: {str(e)}")

    def _calculate_component_performance(self, signal: Dict, outcome: str) -> Dict[str, float]:
        """Calculate performance contribution of each component."""
        performance = {}
        success = outcome == 'WIN'
        
        for component, data in signal['components'].items():
            # Calculate alignment score (1 for correct prediction, 0 for incorrect)
            predicted_direction = data.get('direction', 'NEUTRAL')
            actual_direction = signal.get('final_direction')
            
            if predicted_direction == actual_direction:
                alignment = 1.0
            elif predicted_direction == 'NEUTRAL':
                alignment = 0.5
            else:
                alignment = 0.0
                
            # Weight the performance by component confidence
            confidence = data.get('confidence', 0)
            performance[component] = alignment * confidence
            
        return performance

    def get_recent_performance(self, timeframe: str = '1d') -> Dict:
        """Get performance metrics for recent signals."""
        try:
            if not self.signal_history:
                return {'no_data': True}
            
            # Filter recent signals
            if timeframe == '1d':
                cutoff = datetime.now() - timedelta(days=1)
            elif timeframe == '1w':
                cutoff = datetime.now() - timedelta(weeks=1)
            elif timeframe == '1m':
                cutoff = datetime.now() - timedelta(days=30)
            else:
                cutoff = datetime.now() - timedelta(days=7)  # Default to 1 week
                
            recent_signals = [s for s in self.signal_history 
                            if datetime.fromisoformat(s['timestamp']) > cutoff
                            and 'outcome' in s]
            
            if not recent_signals:
                return {'no_recent_data': True}
                
            # Calculate metrics
            total_signals = len(recent_signals)
            wins = sum(1 for s in recent_signals if s['outcome'] == 'WIN')
            losses = sum(1 for s in recent_signals if s['outcome'] == 'LOSS')
            
            avg_profit = np.mean([s['profit_loss'] for s in recent_signals if s['outcome'] == 'WIN']) \
                        if wins > 0 else 0
            avg_loss = np.mean([s['profit_loss'] for s in recent_signals if s['outcome'] == 'LOSS']) \
                      if losses > 0 else 0
            
            win_rate = (wins / total_signals * 100) if total_signals > 0 else 0
            
            return {
                'timeframe': timeframe,
                'total_signals': total_signals,
                'win_rate': win_rate,
                'avg_profit': avg_profit,
                'avg_loss': avg_loss,
                'profit_factor': abs(avg_profit / avg_loss) if avg_loss != 0 else float('inf'),
                'component_weights': self.component_weights.weights
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {'error': str(e)}

    def generate_recommendation(self, components: Dict[str, Dict]) -> Dict:
        """Generate enhanced trading signal recommendation."""
        try:
            # Calculate confidence and get explanations
            confidence, explanations = self._calculate_advanced_confidence(components)
            
            # Validate signal quality
            is_valid, warnings = self._validate_signal_quality(confidence, components)
            
            # Determine final signal direction
            directions = [comp.get('direction', 'NEUTRAL') for comp in components.values()]
            weights = [self.component_weights.weights.get(comp_name, 0) 
                      for comp_name in components.keys()]
            
            # Weight the directions
            direction_scores = {'BUY': 0, 'SELL': 0, 'NEUTRAL': 0}
            for direction, weight in zip(directions, weights):
                direction_scores[direction] += weight
            
            final_direction = max(direction_scores.items(), key=lambda x: x[1])[0]
            
            # Create unique signal ID
            signal_id = f"SIG_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Prepare recommendation
            recommendation = {
                'id': signal_id,
                'direction': final_direction if is_valid else 'NEUTRAL',
                'confidence': confidence,
                'explanations': explanations,
                'warnings': warnings,
                'is_valid': is_valid,
                'components': components,
                'weights_used': self.component_weights.weights,
                'timestamp': datetime.now().isoformat(),
                'final_direction': final_direction
            }
            
            # Store in history
            self.signal_history.append(recommendation)
            
            # Add recent performance
            recommendation['recent_performance'] = self.get_recent_performance('1d')
            
            logger.info(f"Generated recommendation: {final_direction} with {confidence:.1f}% confidence")
            return recommendation
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'error': str(e)
            }
    
    def get_recommendation(self, market_data: Dict, ml_signal: Dict = None, 
                         regime_info: Dict = None, patterns: List = None,
                         correlations: Dict = None, risk_assessment: Dict = None) -> Dict:
        """Get a trading signal recommendation based on all available data."""
        try:
            components = {}
            
            # Add technical analysis
            if market_data:
                components['technical'] = self._analyze_technical_indicators(market_data)
            
            # Add ML predictions
            if ml_signal:
                components['ml'] = {
                    'signal': ml_signal.get('signal', 'NEUTRAL'),
                    'confidence': ml_signal.get('confidence', 0.5)
                }
            
            # Add pattern analysis
            if patterns:
                components['pattern'] = {
                    'pattern_count': len(patterns),
                    'patterns': patterns,
                    'confidence': sum(p.get('confidence', 0.5) for p in patterns) / max(len(patterns), 1)
                }
            
            # Add correlation analysis
            if correlations:
                components['correlation'] = correlations
            
            # Calculate confidence and get explanations
            confidence, explanations = self._calculate_advanced_confidence(components)
            
            # Validate signal quality
            is_valid, validation_notes = self._validate_signal_quality(confidence, components)
            
            if not is_valid:
                return {
                    'signal': 'NEUTRAL',
                    'confidence': confidence,
                    'explanation': 'Signal validation failed: ' + ', '.join(validation_notes),
                    'risk_level': 'HIGH'
                }
            
            # Determine final signal direction
            signal_direction = self._determine_signal_direction(components)
            
            # Add risk assessment
            risk_level = risk_assessment.get('risk_level', 'MEDIUM') if risk_assessment else 'MEDIUM'
            
            return {
                'signal': signal_direction,
                'confidence': confidence,
                'explanation': ' | '.join(explanations),
                'risk_level': risk_level,
                'components': components
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendation: {str(e)}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'explanation': f"Error: {str(e)}",
                'risk_level': 'HIGH'
            }

    def get_enhanced_recommendation(self, market_data: Dict, signal_data: Dict) -> Dict:
        """Generate an enhanced signal recommendation with comprehensive analysis.
        
        Args:
            market_data: Market data dictionary with OHLCV data
            signal_data: Base signal data with technical and pattern analysis
            
        Returns:
            Dict with enhanced recommendation containing:
            - direction: 'BUY', 'SELL', or 'NEUTRAL'
            - confidence: Float between 0 and 1
            - explanation: String explaining the recommendation
            - components: Dict with component-wise analysis
        """
        try:
            components = {}
            
            # Technical Analysis
            if 'technical' in signal_data:
                components['technical'] = {
                    'confidence': signal_data['technical'].get('confidence', 0),
                    'direction': signal_data['technical'].get('direction', 'NEUTRAL')
                }
            
            # Pattern Recognition
            if 'patterns' in signal_data:
                components['pattern'] = {
                    'confidence': signal_data['patterns'].get('confidence', 0),
                    'direction': signal_data['patterns'].get('direction', 'NEUTRAL')
                }
            
            # Calculate confidence and explanations
            confidence, explanations = self._calculate_advanced_confidence(components)
            
            # Validate signal quality
            is_valid, validation_notes = self._validate_signal_quality(confidence, components)
            
            # Get final direction based on component agreement
            if confidence > 0.7 and is_valid:
                direction = signal_data.get('direction', 'NEUTRAL')
            else:
                direction = 'NEUTRAL'
                
            return {
                'direction': direction,
                'confidence': confidence,
                'explanation': '\n'.join(explanations),
                'validation': validation_notes,
                'components': components
            }
            
        except Exception as e:
            logger.error(f"Error generating enhanced recommendation: {e}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'explanation': f"Error generating recommendation: {str(e)}",
                'validation': ['Error in signal validation'],
                'components': {}
            }
