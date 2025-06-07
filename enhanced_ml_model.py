"""
Enhanced ML model with LSTM networks and ensemble methods.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime
import json
import joblib
from enhanced_logging import logger as enhanced_logger

# Configure logging
logger = logging.getLogger(__name__)

class LSTMModel(nn.Module):
    """LSTM network for time series prediction."""
    
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=4,
            dropout=0.1
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 3)  # 3 classes: Buy, Sell, Neutral
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # LSTM layers
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, _ = self.attention(
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1),
            lstm_out.transpose(0, 1)
        )
        
        # Use last output with attention for prediction
        out = self.fc(attn_out[-1])
        return out

class EnhancedMLPredictor:
    """Enhanced ML predictor with ensemble methods and LSTM."""
    
    def __init__(self,
                 input_size: int,
                 feature_names: Optional[List[str]] = None,
                 model_path: Optional[str] = None):
        """
        Initialize the predictor.
        
        Args:
            input_size: Number of input features
            feature_names: Optional list of feature names
            model_path: Optional path to saved model
        """
        self.input_size = input_size
        self.feature_names = feature_names
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.lstm_model = LSTMModel(input_size=input_size).to(self.device)
        
        # Initialize ensemble models
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=42
        )
        
        if model_path:
            self.load_models(model_path)
            
        # Log initialization
        enhanced_logger.logger.info(
            f"Initialized EnhancedMLPredictor with input size: {input_size}",
            extra={
                "model_path": model_path,
                "device": str(self.device),
                "feature_count": input_size
            }
        )
    
    def train(self,
             train_data: pd.DataFrame,
             labels: np.ndarray,
             validation_split: float = 0.2) -> Dict:
        """
        Train all models in the ensemble.
        
        Args:
            train_data: DataFrame with features
            labels: Array with labels (0: Sell, 1: Neutral, 2: Buy)
            validation_split: Fraction of data to use for validation
        """
        try:
            # Prepare data
            X = torch.FloatTensor(train_data.values).to(self.device)
            y = torch.LongTensor(labels).to(self.device)
            
            # Split data for time series validation
            tscv = TimeSeriesSplit(n_splits=5)
            split = list(tscv.split(X))[-1]  # Use last split
            train_idx, val_idx = split
            
            # Train LSTM
            lstm_metrics = self._train_lstm(
                X[train_idx],
                y[train_idx],
                X[val_idx],
                y[val_idx]
            )
            
            # Train ensemble models
            X_np = train_data.values
            y_np = labels
            
            self.rf_model.fit(X_np[train_idx], y_np[train_idx])
            self.gb_model.fit(X_np[train_idx], y_np[train_idx])
            
            # Validate ensemble models
            rf_accuracy = self.rf_model.score(X_np[val_idx], y_np[val_idx])
            gb_accuracy = self.gb_model.score(X_np[val_idx], y_np[val_idx])
            
            metrics = {
                'lstm': lstm_metrics,
                'random_forest': {'accuracy': rf_accuracy},
                'gradient_boosting': {'accuracy': gb_accuracy}
            }
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error training models: {str(e)}")
            return {}
    
    def predict(self, features) -> Dict:
        """Generate predictions using the ensemble."""
        component = "EnhancedMLPredictor"
        try:
            # Convert different input types to numpy array
            if isinstance(features, pd.DataFrame):
                features_array = features.values
            elif isinstance(features, torch.Tensor):
                features_array = features.cpu().numpy()
            elif isinstance(features, np.ndarray):
                features_array = features
            else:
                error_msg = f"Unsupported feature type: {type(features)}"
                enhanced_logger.log_error(
                    ValueError(error_msg),
                    component,
                    additional_data={"feature_type": str(type(features))}
                )
                raise ValueError(error_msg)

            # Log feature shape
            enhanced_logger.logger.info(f"Processing features with shape: {features_array.shape}")

            # Ensure 2D array
            if features_array.ndim == 1:
                features_array = features_array.reshape(1, -1)

            # Verify feature dimensions
            if features_array.shape[1] != self.input_size:
                error_msg = f"Feature dimension mismatch. Expected {self.input_size}, got {features_array.shape[1]}"
                enhanced_logger.log_error(
                    ValueError(error_msg),
                    component,
                    additional_data={
                        "expected_size": self.input_size,
                        "actual_size": features_array.shape[1]
                    }
                )
                raise ValueError(error_msg)

            # Convert to tensor
            X = torch.FloatTensor(features_array).to(self.device)
            
            # LSTM prediction
            self.lstm_model.eval()
            with torch.no_grad():
                lstm_out = self.lstm_model(X)
                lstm_probs = torch.softmax(lstm_out, dim=1)
                lstm_pred = torch.argmax(lstm_probs, dim=1).item()
            
            # Ensemble predictions
            rf_pred = self.rf_model.predict(features_array)[0]
            rf_probs = self.rf_model.predict_proba(features_array)[0]
            
            gb_pred = self.gb_model.predict(features_array)[0]
            gb_probs = self.gb_model.predict_proba(features_array)[0]
            
            # Combine predictions
            ensemble_probs = (
                lstm_probs.cpu().numpy() +
                rf_probs +
                gb_probs
            ) / 3
            
            final_pred = np.argmax(ensemble_probs)
            
            # Calculate prediction uncertainty
            uncertainty = 1 - np.max(ensemble_probs)
            
            # Calculate ensemble agreement
            predictions = [lstm_pred, rf_pred, gb_pred]
            agreement = predictions.count(final_pred) / len(predictions)
            
            # Get market context
            if isinstance(features, pd.DataFrame):
                market_context = self._get_market_context(features)
            else:
                market_context = self._get_market_context(pd.DataFrame(features_array))
            
            return {
                'signal': self._index_to_signal(final_pred),
                'confidence': float(np.max(ensemble_probs) * 100),
                'uncertainty': float(uncertainty * 100),
                'ensemble_agreement': float(agreement * 100),
                'market_context': market_context
            }
            
        except Exception as e:
            error_msg = str(e)
            enhanced_logger.log_error(
                e,
                component,
                additional_data={
                    "feature_type": str(type(features)),
                    "feature_shape": getattr(features, 'shape', None)
                }
            )
            logger.error(f"Error generating prediction: {error_msg}")
            return {
                'signal': 'NEUTRAL',
                'confidence': 0,
                'uncertainty': 100,
                'ensemble_agreement': 0,
                'market_context': {
                    'trend_strength': 0.0,
                    'volatility': 0.0,
                    'range_bound': 0.0
                }            }
    
    def save_model(self, path: str):
        """Save all models to disk."""
        try:
            # Save LSTM model
            torch.save(self.lstm_model.state_dict(), f"{path}_lstm.pt")
              # Save ensemble models using joblib for scikit-learn models
            models = {
                'random_forest': self.rf_model,
                'gradient_boosting': self.gb_model
            }
            
            for name, model in models.items():
                joblib.dump(model, f"{path}_{name}.joblib")
                    
        except Exception as e:
            logger.error(f"Error saving models: {str(e)}")
    
    def load_models(self, path: str):
        """Load all models from disk."""
        try:
            # Load LSTM model
            self.lstm_model.load_state_dict(
                torch.load(f"{path}_lstm.pt", map_location=self.device)
            )
              # Load ensemble models using joblib for scikit-learn models
            models = {
                'random_forest': self.rf_model,
                'gradient_boosting': self.gb_model
            }
            
            for name, model in models.items():
                loaded_model = joblib.load(f"{path}_{name}.joblib")
                if name in models:
                    setattr(self, f"{name}_model", loaded_model)
                    
        except Exception as e:
            logger.error(f"Error loading models: {str(e)}")
    
    def _train_lstm(self,
                    X_train: torch.Tensor,
                    y_train: torch.Tensor,
                    X_val: torch.Tensor,
                    y_val: torch.Tensor,
                    epochs: int = 100,
                    batch_size: int = 32) -> Dict:
        """Train LSTM model."""
        try:
            criterion = nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(self.lstm_model.parameters())
            
            best_val_loss = float('inf')
            patience = 10
            patience_counter = 0
            
            train_losses = []
            val_losses = []
            
            for epoch in range(epochs):
                # Training
                self.lstm_model.train()
                total_loss = 0
                
                for i in range(0, len(X_train), batch_size):
                    batch_X = X_train[i:i+batch_size]
                    batch_y = y_train[i:i+batch_size]
                    
                    optimizer.zero_grad()
                    outputs = self.lstm_model(batch_X)
                    loss = criterion(outputs, batch_y)
                    
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                
                avg_train_loss = total_loss / (len(X_train) / batch_size)
                train_losses.append(avg_train_loss)
                
                # Validation
                self.lstm_model.eval()
                with torch.no_grad():
                    val_outputs = self.lstm_model(X_val)
                    val_loss = criterion(val_outputs, y_val)
                    val_losses.append(val_loss.item())
                    
                    # Early stopping
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        
                    if patience_counter >= patience:
                        logger.info(f"Early stopping at epoch {epoch}")
                        break
            
            return {
                'train_loss': train_losses[-1],
                'val_loss': val_losses[-1],
                'epochs': len(train_losses)
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM: {str(e)}")
            return {}
    
    def _index_to_signal(self, index: int) -> str:
        """Convert predicted index to signal string."""
        signals = ['BUY', 'NEUTRAL', 'SELL']
        return signals[index]

    def _get_market_context(self, features: pd.DataFrame) -> Dict[str, float]:
        """Extract market context from features."""
        try:
            # Calculate basic market context metrics
            context = {
                'trend_strength': 0.0,
                'volatility': 0.0,
                'range_bound': 0.0
            }
            
            # If we have RSI in features, use it for trend strength
            rsi_cols = [col for col in features.columns if 'rsi' in col.lower()]
            if rsi_cols:
                rsi_values = features[rsi_cols].mean(axis=1).values
                context['trend_strength'] = abs(50 - rsi_values[0]) / 50
                
            # If we have ATR or similar in features, use it for volatility
            vol_cols = [col for col in features.columns if any(x in col.lower() for x in ['atr', 'volatility', 'std'])]
            if vol_cols:
                context['volatility'] = features[vol_cols].mean(axis=1).values[0]
                
            # Check for range-bound market using Bollinger Band position if available
            bb_cols = [col for col in features.columns if 'bb_pos' in col.lower()]
            if bb_cols:
                bb_pos = features[bb_cols].mean(axis=1).values[0]
                context['range_bound'] = 1 - abs(bb_pos - 0.5) * 2
                
            return context
            
        except Exception as e:
            logger.error(f"Error getting market context: {str(e)}")
            return {
                'trend_strength': 0.0,
                'volatility': 0.0,
                'range_bound': 0.0
            }
