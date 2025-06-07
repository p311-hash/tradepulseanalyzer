"""
Comprehensive testing of all TradePulseAnalyzer features.
"""
import os
import pandas as pd
import numpy as np
import torch
from datetime import datetime
import logging
from typing import Dict, List, Any

# Import enhanced components
from feature_engineering import FeatureEngineer
from enhanced_ml_model import EnhancedMLPredictor
from market_regime import MarketRegimeDetector
from risk_manager import RiskManager
from continuous_learning import ContinuousLearningSystem
from enhanced_signal_generator import EnhancedSignalGenerator
from technical_analysis import AdaptiveTechnicalAnalyzer
from market_microstructure import MarketMicrostructureAnalyzer
from deep_market_structure import DeepMarketStructureAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data with realistic patterns."""
    np.random.seed(42)
    # Create trending price data
    trend = np.linspace(0, 1, periods)
    noise = np.random.normal(0, 0.1, periods)
    close = 100 * (1 + trend + noise)
    
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(periods) * 0.002),
        'high': close * (1 + abs(np.random.randn(periods) * 0.003)),
        'low': close * (1 - abs(np.random.randn(periods) * 0.003)),
        'close': close,
        'volume': np.random.randint(1000, 10000, periods) * (1 + trend)  # Increasing volume trend
    })
    return df

def test_feature_engineering(data: pd.DataFrame) -> Dict[str, Any]:
    """Test feature engineering capabilities."""
    try:
        logger.info("Testing Feature Engineering...")
        
        feature_engineer = FeatureEngineer()
        features = feature_engineer.engineer_features(data)
        
        # Verify all expected features are present
        feature_names = feature_engineer.get_feature_names()
        missing_features = set(feature_names) - set(features.columns)
        
        # Check for NaN values
        nan_columns = features.columns[features.isna().any()].tolist()
        
        # Calculate basic statistics
        stats = features.describe()
        
        return {
            "success": True,
            "feature_count": len(features.columns),
            "missing_features": missing_features,
            "nan_columns": nan_columns,
            "stats": stats,
            "features": features
        }
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

def test_market_regime_detection(data: pd.DataFrame) -> Dict[str, Any]:
        """Test market regime detection."""
        try:
            logger.info("Testing Market Regime Detection...")
            
            # Initialize with data
            regime_detector = MarketRegimeDetector(data)
            regime_info = regime_detector.detect_regime()  # Now uses internal data
            
            # Test regime transitions
            transitions = regime_detector.detect_regime_transitions(data)
            
            return {
                "success": True,
                "current_regime": regime_info,
                "regime_transitions": transitions
            }
        except Exception as e:
            logger.error(f"Market regime detection test failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

def test_risk_management(data: pd.DataFrame) -> Dict[str, Any]:
        """Test risk management system."""
        try:
            logger.info("Testing Risk Management...")
            
            risk_manager = RiskManager(initial_capital=10000)
            
            # Test position sizing with correct parameters
            position_size = risk_manager.calculate_position_size(
                symbol="TEST",
                current_price=data['close'].iloc[-1],
                atr=data['high'].iloc[-1] - data['low'].iloc[-1],  # Simple ATR approximation
                regime_multiplier=1.0
            )
            
            # Test risk metrics
            risk_metrics = risk_manager.calculate_portfolio_risk_metrics(data)
            
            return {
                "success": True,
                "position_size": position_size,
                "risk_metrics": risk_metrics
            }
        except Exception as e:
            logger.error(f"Risk management test failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

def test_signal_generation(features: pd.DataFrame) -> Dict[str, Any]:
        """Test signal generation with ML model."""
        try:
            logger.info("Testing Signal Generation...")
              # Get feature names to ensure correct order and all features
            feature_engineer = FeatureEngineer()
            feature_names = feature_engineer.get_feature_names()
            
            # Initialize predictor with correct feature count
            predictor = EnhancedMLPredictor(
                input_size=len(feature_names),
                feature_names=feature_names
            )
              # Handle categorical features before conversion to tensor
            feature_data = features[feature_names].copy()
            
            # Convert categorical variables to numeric
            categorical_columns = feature_data.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                # Convert categorical values to numeric codes
                if col == 'regime':
                    regime_map = {'TRENDING': 0, 'RANGING': 1, 'VOLATILE': 2}
                    feature_data[col] = feature_data[col].map(regime_map).fillna(1)  # Default to RANGING if unknown
                else:
                    # For any other categorical columns, use simple label encoding
                    unique_values = feature_data[col].unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    feature_data[col] = feature_data[col].map(value_map)
            
            # Use only last row of features and convert to tensor
            last_features = feature_data.iloc[-1:].values.astype(np.float32)
            feature_tensor = torch.tensor(last_features, dtype=torch.float32)
            
            # Generate signal
            prediction = predictor.predict(feature_tensor)
            
            return {
                "success": True,
                "prediction": prediction
            }
        except Exception as e:
            logger.error(f"Signal generation test failed: {str(e)}", exc_info=True)
            return {"success": False, "error": str(e)}

def test_continuous_learning(data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
    """Test continuous learning system."""
    try:
        logger.info("Testing Continuous Learning...")
        
        learning_system = ContinuousLearningSystem(
            model_path="models/latest_model",
            history_path="data/signal_history.json",
            feedback_path="data/feedback.json"
        )
        
        # Test online learning
        learning_metrics = learning_system.process_new_data(data, features)
        
        return {
            "success": True,
            "learning_metrics": learning_metrics
        }
    except Exception as e:
        logger.error(f"Continuous learning test failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

def run_comprehensive_tests():
    """Run all system tests."""
    try:
        # Create test data
        logger.info("Creating test data...")
        data = create_sample_data()
        
        # Run all tests
        results = {}
        
        # 1. Test Feature Engineering
        results["feature_engineering"] = test_feature_engineering(data)
        
        if results["feature_engineering"]["success"]:
            features = results["feature_engineering"]["features"]
            
            # 2. Test Market Regime Detection
            results["market_regime"] = test_market_regime_detection(data)
            
            # 3. Test Risk Management
            results["risk_management"] = test_risk_management(data)
            
            # 4. Test Signal Generation
            results["signal_generation"] = test_signal_generation(features)
            
            # 5. Test Continuous Learning
            results["continuous_learning"] = test_continuous_learning(data, features)
        
        # Print results summary
        logger.info("\n=== Test Results Summary ===")
        for component, result in results.items():
            status = "✓" if result["success"] else "✗"
            logger.info(f"{component}: {status}")
            if not result["success"]:
                logger.error(f"  Error: {result.get('error', 'Unknown error')}")
        
        return results
        
    except Exception as e:
        logger.error(f"Comprehensive testing failed: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    run_comprehensive_tests()
