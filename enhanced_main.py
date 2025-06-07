"""
Enhanced main application integrating all TradePulseAnalyzer improvements
"""

import logging
from typing import Dict, Optional
import pandas as pd
from market_regime import MarketRegimeDetector
from signal_validation import SignalValidator
from enhanced_ml_model import EnhancedMLPredictor
from enhanced_signal_generator import EnhancedSignalGenerator, EnhancedSignal
from feature_engineering import FeatureEngineer
from technical_analysis import TechnicalAnalyzer
import config
import json
from pathlib import Path
import torch

logger = logging.getLogger(__name__)

class EnhancedTradePulseAnalyzer:
    """Enhanced main class for trading signal generation and analysis"""
    
    def __init__(self, 
                 model_dir: str = "models",
                 min_confidence: float = 65.0,
                 enable_online_learning: bool = True):
        
        self.model_dir = Path(model_dir)
        self.min_confidence = min_confidence
        self.enable_online_learning = enable_online_learning
        
        # Initialize components
        self.feature_engineer = FeatureEngineer()
        self.technical_analyzer = TechnicalAnalyzer()
        
        # Load or initialize ML model
        self.predictor = self._initialize_ml_model()
        
        # Initialize signal generator
        self.signal_generator = EnhancedSignalGenerator(
            predictor=self.predictor,
            timeframes=config.TIMEFRAMES,
            min_confidence=min_confidence
        )
        
        # Performance tracking
        self.performance_history = {
            'signals': [],
            'accuracy': 0.0,
            'regime_changes': [],
            'model_updates': []
        }
    
    def analyze_market(self, market_data: Dict[str, pd.DataFrame]) -> Optional[EnhancedSignal]:
        """
        Analyze market data and generate trading signals with comprehensive analysis
        
        Args:
            market_data: Dictionary of DataFrames with market data for different timeframes
            
        Returns:
            EnhancedSignal object if a valid signal is found, None otherwise
        """
        try:
            # Generate signal with enhanced analysis
            signal = self.signal_generator.generate_signal(market_data)
            
            if signal:
                # Log signal for performance tracking
                self._log_signal(signal)
                
                # Update model if online learning is enabled
                if self.enable_online_learning:
                    self._update_model_online(market_data, signal)
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in market analysis: {str(e)}")
            return None
    
    def _initialize_ml_model(self) -> EnhancedMLPredictor:
        """Initialize or load the ML model"""
        try:
            # Get feature names from feature engineer
            sample_features = self.feature_engineer.get_feature_names()
            
            # Initialize model
            predictor = EnhancedMLPredictor(
                input_size=len(sample_features),
                feature_names=sample_features,
                model_path=self.model_dir / "latest_model" if self.model_dir.exists() else None
            )
            
            return predictor
            
        except Exception as e:
            logger.error(f"Error initializing ML model: {str(e)}")
            raise
    
    def _update_model_online(self, market_data: Dict[str, pd.DataFrame], signal: EnhancedSignal):
        """Update model with new market data for online learning"""
        try:
            # Prepare features for update
            features = self.signal_generator._prepare_features(market_data)
            
            # Create target based on signal direction
            target_map = {'BUY': 2, 'NEUTRAL': 1, 'SELL': 0}
            target = pd.Series([target_map[signal.direction]])
            
            # Update model
            self.predictor.update_online(features, target)
            
            # Log update
            self.performance_history['model_updates'].append({
                'timestamp': pd.Timestamp.now(),
                'features_shape': features.shape,
                'signal_confidence': signal.confidence
            })
            
        except Exception as e:
            logger.error(f"Error in online model update: {str(e)}")
    
    def _log_signal(self, signal: EnhancedSignal):
        """Log signal for performance tracking"""
        signal_data = {
            'timestamp': signal.timestamp,
            'direction': signal.direction,
            'strength': signal.strength.value,
            'confidence': signal.confidence,
            'entry_price': signal.entry_price,
            'stop_loss': signal.stop_loss,
            'take_profit': signal.take_profit,
            'regime': signal.metadata.regime,
            'validation_score': signal.metadata.validation_score,
            'ml_uncertainty': signal.metadata.ml_uncertainty,
            'ensemble_agreement': signal.metadata.ensemble_agreement
        }
        
        self.performance_history['signals'].append(signal_data)
        
        # Save to JSON file
        try:
            with open('signal_history.json', 'w') as f:
                json.dump(self.performance_history, f, 
                         default=str,  # Handle datetime serialization
                         indent=2)
        except Exception as e:
            logger.error(f"Error saving signal history: {str(e)}")
    
    def get_performance_metrics(self) -> Dict:
        """Get current performance metrics"""
        try:
            if not self.performance_history['signals']:
                return {
                    'total_signals': 0,
                    'accuracy': 0.0,
                    'regime_changes': 0,
                    'model_updates': 0
                }
            
            signals = pd.DataFrame(self.performance_history['signals'])
            
            return {
                'total_signals': len(signals),
                'accuracy': self.performance_history.get('accuracy', 0.0),
                'signals_by_regime': signals['regime'].value_counts().to_dict(),
                'average_confidence': signals['confidence'].mean(),
                'average_uncertainty': signals['ml_uncertainty'].mean(),
                'regime_changes': len(self.performance_history['regime_changes']),
                'model_updates': len(self.performance_history['model_updates'])
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
