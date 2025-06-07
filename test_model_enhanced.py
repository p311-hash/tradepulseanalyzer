"""
Test the ML model with sample data.
"""
import os
import pandas as pd
import numpy as np
import torch
from feature_engineering import FeatureEngineer
from enhanced_ml_model import EnhancedMLPredictor
import logging

print("Starting test script...")  # Direct print for verification

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

def create_sample_data(periods: int = 100) -> pd.DataFrame:
    """Create sample OHLCV data."""
    np.random.seed(42)
    close = 100 * (1 + np.random.randn(periods).cumsum() * 0.02)
    df = pd.DataFrame({
        'open': close * (1 + np.random.randn(periods) * 0.002),
        'high': close * (1 + abs(np.random.randn(periods) * 0.003)),
        'low': close * (1 - abs(np.random.randn(periods) * 0.003)),
        'close': close,
        'volume': np.random.randint(1000, 10000, periods)
    })
    return df

def test_model():
    """Test the ML model with sample data."""
    try:
        # Create sample data
        logger.info("Creating sample data...")
        data = create_sample_data()
        logger.info(f"Created sample data with shape: {data.shape}")
        
        # Initialize feature engineer
        logger.info("Initializing feature engineer...")
        feature_engineer = FeatureEngineer()
        
        # Generate features
        logger.info("Generating features...")
        try:
            features = feature_engineer.engineer_features(data)
            logger.info(f"Generated features with shape: {features.shape}")
        except Exception as e:
            logger.error(f"Error generating features: {str(e)}", exc_info=True)
            raise
        
        # Get feature names and verify
        try:
            feature_names = feature_engineer.get_feature_names()
            logger.info(f"Using features: {feature_names}")
        except Exception as e:
            logger.error(f"Error getting feature names: {str(e)}", exc_info=True)
            raise
        
        # Load the model
        logger.info("Loading model...")
        model_path = "models/latest_model"
        try:
            predictor = EnhancedMLPredictor(
                input_size=len(feature_names),
                model_path=model_path if os.path.exists(model_path) else None
            )
        except Exception as e:
            logger.error(f"Error loading predictor: {str(e)}", exc_info=True)
            raise
        
        # Make predictions
        logger.info("Making predictions...")
        available_features = [f for f in feature_names if f in features.columns]
        if len(available_features) != len(feature_names):
            missing = set(feature_names) - set(features.columns)
            logger.warning(f"Missing features: {missing}")
            raise ValueError(f"Missing required features: {missing}")
            
        feature_values = features[available_features].fillna(0)
        logger.info(f"Feature data shape: {feature_values.shape}")
        
        try:
            prediction = predictor.predict(feature_values)
            logger.info(f"Prediction results: {prediction}")
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}", exc_info=True)
            raise
            
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_model()
