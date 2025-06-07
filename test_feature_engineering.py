"""
Test feature engineering in isolation.
"""
import pandas as pd
import numpy as np
from feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create sample OHLCV data."""
    np.random.seed(42)
    periods = 100
    close = 100 * (1 + np.random.randn(periods).cumsum() * 0.02)
    data = pd.DataFrame({
        'open': close * (1 + np.random.randn(periods) * 0.002),
        'high': close * (1 + abs(np.random.randn(periods) * 0.003)),
        'low': close * (1 - abs(np.random.randn(periods) * 0.003)),
        'close': close,
        'volume': np.random.randint(1000, 10000, periods)
    })
    return data

def test_feature_engineering():
    """Test the feature engineering process."""
    try:
        # Create test data
        data = create_test_data()
        logger.info(f"Created test data with shape: {data.shape}")

        # Initialize feature engineer
        fe = FeatureEngineer()
        
        # Generate features
        features = fe.engineer_features(data)
        logger.info(f"Generated features with shape: {features.shape}")
        logger.info(f"Feature names: {features.columns.tolist()}")
        
        # Verify results
        feature_names = fe.get_feature_names()
        missing_features = set(feature_names) - set(features.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False
            
        logger.info("Feature engineering test passed!")
        return True
        
    except Exception as e:
        logger.error(f"Feature engineering test failed: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    test_feature_engineering()
