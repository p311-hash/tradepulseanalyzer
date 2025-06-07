"""
Initialize and save the enhanced ML model for TradePulseAnalyzer.
"""
import os
import sys
import logging
import torch
from enhanced_ml_model import EnhancedMLPredictor
from feature_engineering import FeatureEngineer
from pathlib import Path

# Setup logging with console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('model_init.log')
    ]
)
logger = logging.getLogger(__name__)

def initialize_model():
    """Initialize and save the ML model."""
    try:
        logger.info("Starting model initialization...")
        
        # Initialize feature engineer to get feature names
        logger.info("Initializing feature engineer...")
        feature_engineer = FeatureEngineer()
        feature_names = feature_engineer.get_feature_names()
        logger.info(f"Generated {len(feature_names)} features")
        
        # Initialize model with default feature size
        logger.info("Creating ML model...")
        predictor = EnhancedMLPredictor(
            input_size=len(feature_names),
            feature_names=feature_names
        )
        
        # Create models directory if it doesn't exist
        model_dir = Path("models")
        model_dir.mkdir(exist_ok=True)
        logger.info(f"Created models directory at {model_dir.absolute()}")
            
        # Save model
        model_path = model_dir / "latest_model"
        predictor.save_model(str(model_path))
        logger.info(f"Model initialized and saved to {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    success = initialize_model()
    sys.exit(0 if success else 1)