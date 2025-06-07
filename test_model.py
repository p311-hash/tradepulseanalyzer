"""
Test the ML model with sample data.
"""
import pandas as pd
import numpy as np
import torch
from feature_engineering import FeatureEngineer
from enhanced_ml_model import EnhancedMLPredictor
import logging

logging.basicConfig(level=logging.INFO)
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
        features = feature_engineer.engineer_features(data)
        logger.info(f"Generated features with shape: {features.shape}")
        
        # Get feature names
        feature_names = feature_engineer.get_feature_names()
        logger.info(f"Using features: {feature_names}")
        
        # Load the model
        logger.info("Loading model...")
        predictor = EnhancedMLPredictor(
            input_size=len(feature_names),
            feature_names=feature_names,
            model_path="models/latest_model"
        )
        
        # Convert features to tensor and make predictions
        logger.info("Making predictions...")
        feature_values = features[feature_names].fillna(0).values  # Handle any NaN values
        
        # Use last timestep for prediction
        last_features = feature_values[-1:]  # Shape: [1, n_features]
        feature_tensor = torch.tensor(last_features, dtype=torch.float32)
        logger.info(f"Feature tensor shape: {feature_tensor.shape}")
        
        prediction = predictor.predict(feature_tensor)
        logger.info(f"Prediction results: {prediction}")
        return True
        
    except Exception as e:
        logger.error(f"Error testing model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_model()
