from typing import Dict, List
import pandas as pd
import numpy as np
from market_microstructure import MarketMicrostructureAnalyzer
from deep_market_structure import DeepMarketStructureAnalyzer
from market_regime_adapter import MarketRegimeAdapter
from risk_manager import RiskManager
from data_fetcher import EnhancedDataFetcher
from enhanced_indicators import EnhancedIndicators
from market_regime import MarketRegimeDetector
from signal_correlation import SignalCorrelationAnalyzer
from enhanced_ml_model import EnhancedMLModel
from sophisticated_execution import ExecutionOptimizer, SmartOrderRouter

class EnhancedTradingBot:
    def __init__(self, initial_capital: float = 100000):
        self.data_fetcher = EnhancedDataFetcher()
        self.indicators = EnhancedIndicators()
        self.regime_detector = MarketRegimeDetector()
        self.regime_adapter = MarketRegimeAdapter()
        self.signal_analyzer = SignalCorrelationAnalyzer()
        self.ml_model = EnhancedMLModel()
        self.market_analyzer = MarketMicrostructureAnalyzer()
        self.deep_analyzer = DeepMarketStructureAnalyzer()
        self.risk_manager = RiskManager(initial_capital=initial_capital)
        self.order_router = SmartOrderRouter()
        self.execution_optimizer = ExecutionOptimizer(self.order_router)
        
    def analyze_market_conditions(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Comprehensive market analysis combining all components with regime adaptation"""
        # Fetch market data
        market_data = self.data_fetcher.fetch_data(symbol, timeframe)
        trades_data = self.data_fetcher.fetch_trades(symbol, timeframe)
        
        # Deep market structure analysis
        deep_structure = self.deep_analyzer.analyze_market_auction(trades_data, timeframe)
        
        # Market regime detection and adaptation
        regime_analysis = self.regime_adapter.detect_and_adapt(market_data, deep_structure)
        
        # Technical analysis with regime-adapted parameters
        technical_indicators = self.indicators.calculate_all_indicators(
            market_data, 
            regime_analysis['adapted_params']
        )
        
        # Market microstructure analysis
        order_book = self.data_fetcher.fetch_order_book(symbol)
        microstructure_analysis = self.market_analyzer.analyze_order_book(
            order_book['bids'], order_book['asks']
        )
        
        # Risk analysis adjusted for regime
        position_size = self.risk_manager.calculate_position_size(
            symbol,
            market_data['close'].iloc[-1],
            technical_indicators['ATR'][-1],
            regime_multiplier=regime_analysis['adapted_params']['position_size']
        )
        
        # ML predictions with regime context
        ml_signals = self.ml_model.generate_signals(
            market_data,
            technical_indicators,
            regime_analysis,
            deep_structure,
            microstructure_analysis
        )
        
        # Optimize execution parameters based on regime
        execution_params = self.execution_optimizer.optimize_execution(
            {
                'symbol': symbol,
                'size': position_size['recommended_size'],
                'urgency': 'high' if regime_analysis['stability'] < 0.5 else 'normal'
            },
            {
                'regime': regime_analysis['regime'],
                'metrics': regime_analysis['metrics']
            },
            deep_structure
        )
        
        return {
            'technical_analysis': technical_indicators,
            'regime_analysis': regime_analysis,
            'deep_structure': deep_structure,
            'microstructure': microstructure_analysis,
            'ml_signals': ml_signals,
            'position_sizing': position_size,
            'execution_params': execution_params
        }
    
    def generate_trading_decision(self, symbol: str, timeframe: str = '1h') -> Dict:
        """Generate final trading decision combining all analyses"""
        analysis = self.analyze_market_conditions(symbol, timeframe)
        
        # Multi-timeframe correlation analysis
        timeframes = ['5m', '15m', '1h', '4h', '1d']
        correlation_analysis = self.signal_analyzer.analyze_timeframe_correlation(
            symbol, timeframes
        )
        
        # Combine signals with weights based on market regime
        regime = analysis['regime_analysis']['regime']
        if regime == 'high_volatility':
            weights = {
                'technical': 0.2,
                'ml': 0.2,
                'microstructure': 0.3,
                'deep_structure': 0.3
            }
        elif regime == 'trending':
            weights = {
                'technical': 0.3,
                'ml': 0.3,
                'microstructure': 0.2,
                'deep_structure': 0.2
            }
        else:  # range_bound
            weights = {
                'technical': 0.25,
                'ml': 0.25,
                'microstructure': 0.25,
                'deep_structure': 0.25
            }
            
        # Calculate final signal
        technical_signal = self._aggregate_technical_signals(analysis['technical_analysis'])
        ml_signal = analysis['ml_signals']['primary_signal']
        microstructure_signal = self._calculate_microstructure_signal(
            analysis['microstructure']
        )
        deep_structure_signal = self._calculate_deep_structure_signal(
            analysis.get('deep_structure', [])
        )
        
        # Combine signals with weights
        final_signal = (
            weights['technical'] * technical_signal +
            weights['ml'] * ml_signal +
            weights['microstructure'] * microstructure_signal +
            weights['deep_structure'] * deep_structure_signal
        )
        
        # Adjust position size based on confidence and risk
        confidence = self._calculate_signal_confidence(
            technical_signal,
            ml_signal,
            microstructure_signal,
            deep_structure_signal,
            correlation_analysis
        )
        
        position_size = analysis['position_sizing']['recommended_size'] * confidence
        
        return {
            'signal': final_signal,
            'confidence': confidence,
            'position_size': position_size,
            'analysis': analysis,
            'correlation': correlation_analysis,
            'weights': weights
        }

    def _aggregate_technical_signals(self, indicators: Dict) -> float:
        """Aggregate technical indicators into a single signal"""
        signals = []
        
        # Trend following signals
        if indicators['ADX'][-1] > 25:  # Strong trend
            if indicators['MACD'][-1] > 0:
                signals.append(1)
            else:
                signals.append(-1)
                
        # Mean reversion signals
        if indicators['RSI'][-1] < 30:
            signals.append(1)
        elif indicators['RSI'][-1] > 70:
            signals.append(-1)
            
        # Volume confirmation
        if indicators['OBV_slope'][-1] > 0:
            signals.append(1)
        else:
            signals.append(-1)
            
        return sum(signals) / len(signals) if signals else 0
    
    def _calculate_microstructure_signal(self, microstructure: Dict) -> float:
        """Calculate trading signal from market microstructure"""
        order_book = microstructure['order_book']
        volume_profile = microstructure['volume_profile']
        
        signals = []
        
        # Order book imbalance signal
        if abs(order_book['imbalance']) > 0.2:  # Significant imbalance
            signals.append(np.sign(order_book['imbalance']))
            
        # Volume profile signal
        current_price = volume_profile['poc_price']
        if current_price < volume_profile['va_low']:
            signals.append(1)  # Price below value area
        elif current_price > volume_profile['va_high']:
            signals.append(-1)  # Price above value area
            
        return sum(signals) / len(signals) if signals else 0
    
    def _calculate_deep_structure_signal(self, deep_structure: List[Dict]) -> float:
        """Calculate trading signal from deep market structure analysis"""
        if not deep_structure:
            return 0
            
        latest = deep_structure[-1]
        signal = 0
        
        # Analyze market delta
        if abs(latest['delta']) > 0:
            signal += np.sign(latest['delta']) * 0.3
            
        # Analyze price acceptance
        if latest['price_acceptance']['price_position'] == 'above_value':
            signal += 0.3
        elif latest['price_acceptance']['price_position'] == 'below_value':
            signal -= 0.3
            
        # Consider institutional activity
        inst_zones = latest['zones']['institutional_zones']
        if inst_zones:
            latest_zone = inst_zones[-1]
            if latest_zone['type'] == 'accumulation':
                signal += 0.4
            else:  # distribution
                signal -= 0.4
                
        return np.clip(signal, -1, 1)

    def _calculate_signal_confidence(self, technical: float, ml: float, 
                                  microstructure: float, deep_structure: float,
                                  correlation: Dict) -> float:
        """Calculate overall confidence in the trading signal"""
        # Base confidence on signal agreement
        signals = [technical, ml, microstructure, deep_structure]
        signal_agreement = len([s for s in signals if abs(s) > 0.5]) / len(signals)
        
        # Consider signal correlation across timeframes
        timeframe_agreement = correlation.get('agreement_ratio', 0.5)
        
        # Weight the components
        confidence = (
            0.4 * signal_agreement +
            0.3 * timeframe_agreement +
            0.3 * abs(sum(signals) / len(signals))
        )
        
        return min(max(confidence, 0.1), 1.0)  # Bound between 0.1 and 1.0
