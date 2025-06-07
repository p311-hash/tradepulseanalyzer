"""Advanced market regime adapter with automatic parameter tuning."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from dataclasses import dataclass
from sklearn.mixture import GaussianMixture
from enum import Enum
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class MarketRegimeState(Enum):
    TRENDING_UP = "TRENDING_UP"
    TRENDING_DOWN = "TRENDING_DOWN"
    RANGING_LOW_VOL = "RANGING_LOW_VOL"
    RANGING_HIGH_VOL = "RANGING_HIGH_VOL"
    BREAKOUT = "BREAKOUT"
    REVERSAL = "REVERSAL"
    CHOPPY = "CHOPPY"
    UNKNOWN = "UNKNOWN"

@dataclass
class RegimeMetrics:
    volatility: float
    trend_strength: float
    momentum: float
    mean_reversion: float
    liquidity: float
    regime_stability: float

class MarketRegimeAdapter:
    """Adapts trading parameters based on market regime and conditions."""
    
    def __init__(self):
        self.regime_history = []
        self.transition_matrix = {}
        self.regime_models = {}
        self.min_regime_duration = 20
        self.lookback_window = 100
        self.gmm_model = GaussianMixture(n_components=4, random_state=42)
        
        # Enhanced parameters
        self.volatility_threshold = 0.0015  # 15 pips
        self.trend_threshold = 25  # ADX threshold
        self.available_timeframes = ['1m', '2m', '5m', '15m', '30m', '1h', '4h']
        self.news_impact_window = 30  # minutes
        self.regime_confidence_threshold = 0.75
        self.parameter_history = {}
        
    def detect_and_adapt(self, market_data: pd.DataFrame, 
                        deep_structure: Dict) -> Dict[str, any]:
        """Detect current market regime and adapt trading parameters."""
        # Calculate regime metrics
        metrics = self._calculate_regime_metrics(market_data)
        
        # Enhanced regime detection
        regime_gmm = self._detect_regime_gmm(metrics)
        regime_rules = self._detect_regime_rules(metrics)
        regime_structure = self._detect_regime_structure(deep_structure)
        
        # Ensemble regime detection with confidence
        current_regime = self._ensemble_regime_detection(
            regime_gmm, regime_rules, regime_structure, metrics
        )
        
        # Calculate regime stability and transition probabilities
        stability = self._calculate_regime_stability(current_regime)
        transitions = self._get_transition_probabilities(current_regime)
        
        # Update regime history and transition matrix
        self._update_regime_history(current_regime, metrics)
        
        # Get adapted parameters
        adapted_params = self._get_regime_adapted_parameters(
            current_regime, metrics, stability
        )
        
        # Optimize timeframe selection
        optimal_timeframe = self._select_optimal_timeframe(
            metrics.volatility, metrics.trend_strength
        )
        
        return {
            'regime': current_regime,
            'metrics': metrics.__dict__,
            'stability': stability,
            'confidence': self._calculate_regime_confidence(metrics),
            'transitions': transitions,
            'adapted_params': adapted_params,
            'optimal_timeframe': optimal_timeframe
        }
        
    def _calculate_regime_metrics(self, data: pd.DataFrame) -> RegimeMetrics:
        """Calculate comprehensive regime metrics"""
        # Calculate returns and volatility
        returns = data['close'].pct_change()
        volatility = returns.std() * np.sqrt(252)
        
        # Trend strength using ADX
        high_low = data['high'] - data['low']
        high_close = abs(data['high'] - data['close'].shift())
        low_close = abs(data['low'] - data['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(14).mean()
        
        dx = pd.Series(np.where(
            (data['high'] - data['high'].shift(1)) > (data['low'].shift(1) - data['low']),
            data['high'] - data['high'].shift(1),
            0
        )).rolling(14).sum() / atr
        
        trend_strength = dx.iloc[-1]
        
        # Momentum and mean reversion
        price_ma = data['close'].rolling(20).mean()
        momentum = np.corrcoef(
            data['close'].diff(), 
            (data['close'] - price_ma)
        )[0, 1]
        
        # Additional metrics
        volume_ma = data['volume'].rolling(20).mean()
        liquidity = data['volume'].iloc[-1] / volume_ma.iloc[-1]
        
        returns_auto_corr = returns.autocorr()
        regime_stability = 1 - abs(returns_auto_corr)
        
        return RegimeMetrics(
            volatility=volatility,
            trend_strength=trend_strength,
            momentum=momentum,
            mean_reversion=1 - momentum,
            liquidity=liquidity,
            regime_stability=regime_stability
        )
        
    def _detect_regime_gmm(self, metrics: RegimeMetrics) -> MarketRegimeState:
        """Detect regime using Gaussian Mixture Model"""
        features = np.array([
            [metrics.volatility, metrics.trend_strength, 
             metrics.momentum, metrics.mean_reversion]
        ])
        
        if len(self.regime_history) > self.min_regime_duration:
            self.gmm_model.fit(features)
            
        regime_idx = self.gmm_model.predict(features)[0]
        
        # Map GMM cluster to regime state
        regime_map = {
            0: MarketRegimeState.RANGING_LOW_VOL,
            1: MarketRegimeState.TRENDING_UP,
            2: MarketRegimeState.RANGING_HIGH_VOL,
            3: MarketRegimeState.TRENDING_DOWN
        }
        
        return regime_map.get(regime_idx, MarketRegimeState.UNKNOWN)
        
    def _detect_regime_rules(self, metrics: RegimeMetrics) -> MarketRegimeState:
        """Detect regime using rule-based approach"""
        if metrics.volatility > 1.5:  # High volatility threshold
            if metrics.trend_strength > 0.7:
                return (MarketRegimeState.TRENDING_UP 
                       if metrics.momentum > 0 
                       else MarketRegimeState.TRENDING_DOWN)
            else:
                return MarketRegimeState.RANGING_HIGH_VOL
        else:
            if metrics.trend_strength > 0.7:
                return (MarketRegimeState.TRENDING_UP 
                       if metrics.momentum > 0 
                       else MarketRegimeState.TRENDING_DOWN)
            elif metrics.mean_reversion > 0.7:
                return MarketRegimeState.RANGING_LOW_VOL
            elif metrics.momentum > 0.5:
                return MarketRegimeState.BREAKOUT
            elif metrics.momentum < -0.5:
                return MarketRegimeState.REVERSAL
                
        return MarketRegimeState.CHOPPY
        
    def _detect_regime_structure(self, deep_structure: Dict) -> MarketRegimeState:
        """Detect regime using deep market structure"""
        if not deep_structure:
            return MarketRegimeState.UNKNOWN
            
        # Analyze market delta and order flow
        latest = deep_structure[-1] if isinstance(deep_structure, list) else deep_structure
        
        market_delta = latest.get('market_delta', 0)
        cum_delta = latest.get('cumulative_delta', 0)
        
        if abs(market_delta) > 0.7:  # Strong order flow
            if market_delta > 0:
                return MarketRegimeState.TRENDING_UP
            else:
                return MarketRegimeState.TRENDING_DOWN
                
        # Analyze auction zones
        zones = latest.get('zones', {})
        if 'institutional_zones' in zones:
            inst_zones = zones['institutional_zones']
            if inst_zones:
                latest_zone = inst_zones[-1]
                if latest_zone['type'] == 'accumulation':
                    return MarketRegimeState.RANGING_LOW_VOL
                else:
                    return MarketRegimeState.RANGING_HIGH_VOL
                    
        return MarketRegimeState.UNKNOWN
        
    def _ensemble_regime_detection(self, regime_gmm: MarketRegimeState,
                                 regime_rules: MarketRegimeState,
                                 regime_structure: MarketRegimeState,
                                 metrics: RegimeMetrics) -> MarketRegimeState:
        """Combine multiple regime detection methods"""
        # Weight the different methods based on their reliability
        weights = {
            'gmm': 0.3,
            'rules': 0.3,
            'structure': 0.4
        }
        
        # Convert regimes to numerical scores
        regime_scores = {}
        for regime in MarketRegimeState:
            score = 0
            if regime == regime_gmm:
                score += weights['gmm']
            if regime == regime_rules:
                score += weights['rules']
            if regime == regime_structure:
                score += weights['structure']
            regime_scores[regime] = score
            
        # Select regime with highest score
        return max(regime_scores.items(), key=lambda x: x[1])[0]
        
    def _calculate_regime_stability(self, current_regime: MarketRegimeState) -> float:
        """Calculate stability of current regime"""
        if len(self.regime_history) < self.min_regime_duration:
            return 0.5
            
        # Calculate how long the current regime has persisted
        regime_duration = 1
        for past_regime in reversed(self.regime_history[:-1]):
            if past_regime == current_regime:
                regime_duration += 1
            else:
                break
                
        # Calculate stability score
        stability = min(regime_duration / self.min_regime_duration, 1.0)
        return stability
        
    def _get_regime_adapted_parameters(self, regime: MarketRegimeState,
                                     metrics: RegimeMetrics,
                                     stability: float) -> Dict:
        """Get trading parameters adapted to current regime"""
        # Base parameters
        params = {
            'stop_loss_multiplier': 1.5,
            'take_profit_multiplier': 2.0,
            'position_size': 1.0,
            'entry_threshold': 0.7,
            'trend_following_weight': 0.5,
            'mean_reversion_weight': 0.5
        }
        
        # Adapt based on regime
        if regime == MarketRegimeState.TRENDING_UP or regime == MarketRegimeState.TRENDING_DOWN:
            params['trend_following_weight'] = 0.8
            params['mean_reversion_weight'] = 0.2
            params['take_profit_multiplier'] = 2.5
            params['position_size'] = 1.2 if stability > 0.7 else 1.0
            
        elif regime == MarketRegimeState.RANGING_LOW_VOL:
            params['trend_following_weight'] = 0.3
            params['mean_reversion_weight'] = 0.7
            params['stop_loss_multiplier'] = 1.2
            params['take_profit_multiplier'] = 1.5
            
        elif regime == MarketRegimeState.RANGING_HIGH_VOL:
            params['position_size'] = 0.7
            params['stop_loss_multiplier'] = 2.0
            params['entry_threshold'] = 0.8
            
        elif regime == MarketRegimeState.BREAKOUT:
            params['position_size'] = 1.3 if stability > 0.6 else 1.0
            params['take_profit_multiplier'] = 3.0
            params['entry_threshold'] = 0.6
            
        # Adjust for volatility
        vol_adjustment = np.clip(1 - metrics.volatility, 0.5, 1.5)
        params['position_size'] *= vol_adjustment
        
        # Store parameters for analysis
        self.parameter_history[datetime.now()] = {
            'regime': regime,
            'parameters': params.copy(),
            'metrics': metrics.__dict__
        }
        
        return params
        
    def _select_optimal_timeframe(self, volatility: float, 
                                trend_strength: float) -> str:
        """Select optimal timeframe based on market conditions"""
        if volatility > 0.02:  # High volatility
            return '5m' if trend_strength > 0.7 else '15m'
        elif volatility > 0.01:  # Medium volatility
            return '15m' if trend_strength > 0.7 else '30m'
        else:  # Low volatility
            return '30m' if trend_strength > 0.7 else '1h'
            
    def _calculate_regime_confidence(self, metrics: RegimeMetrics) -> float:
        """Calculate confidence in regime detection"""
        # Consider multiple factors for confidence
        stability_conf = metrics.regime_stability
        volatility_conf = 1 / (1 + metrics.volatility)  # Higher vol = lower conf
        trend_conf = abs(metrics.trend_strength)
        
        # Combine confidence scores
        confidence = 0.4 * stability_conf + 0.3 * volatility_conf + 0.3 * trend_conf
        return min(confidence, 1.0)
        
    def _get_transition_probabilities(self, 
                                    current_regime: MarketRegimeState) -> Dict:
        """Calculate regime transition probabilities"""
        if len(self.regime_history) < self.min_regime_duration:
            return {state: 1/len(MarketRegimeState) for state in MarketRegimeState}
            
        # Count transitions from current regime
        total_transitions = 0
        regime_transitions = {state: 0 for state in MarketRegimeState}
        
        for i in range(len(self.regime_history) - 1):
            if self.regime_history[i] == current_regime:
                next_regime = self.regime_history[i + 1]
                regime_transitions[next_regime] += 1
                total_transitions += 1
                
        # Calculate probabilities
        if total_transitions > 0:
            return {state: count/total_transitions 
                   for state, count in regime_transitions.items()}
        else:
            return {state: 1/len(MarketRegimeState) for state in MarketRegimeState}
            
    def _update_regime_history(self, regime: MarketRegimeState,
                             metrics: RegimeMetrics):
        """Update regime history and transition matrix"""
        self.regime_history.append(regime)
        
        # Keep history within lookback window
        if len(self.regime_history) > self.lookback_window:
            self.regime_history = self.regime_history[-self.lookback_window:]
