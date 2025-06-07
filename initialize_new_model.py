"""
Initialize a new model with the correct architecture.
"""
import os
import torch
from enhanced_ml_model import EnhancedMLPredictor
from feature_engineering import FeatureEngineer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def initialize_new_model():
    try:
        # Get feature dimensions from feature engineer
        feature_engineer = FeatureEngineer()
        feature_names = feature_engineer.get_feature_names()
        input_size = len(feature_names)
        
        logger.info(f"Initializing new model with {input_size} features")
        
        # Initialize model
        predictor = EnhancedMLPredictor(
            input_size=input_size,
            feature_names=feature_names
        )
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        # Save model with new architecture
        model_path = "models/latest_model"
        predictor.save_model(model_path)
        logger.info(f"Saved new model to {model_path}")
        
        return True
    except Exception as e:
        logger.error(f"Error initializing new model: {str(e)}", exc_info=True)
        return False

if __name__ == "__main__":
    initialize_new_model()
