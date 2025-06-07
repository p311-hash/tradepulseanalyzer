"""
Enhanced ML model with transformer architecture and advanced features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from torch.distributions import Normal
import logging

logger = logging.getLogger(__name__)

@dataclass
class PredictionWithUncertainty:
    prediction: float
    uncertainty: float
    attention_weights: np.ndarray
    feature_importance: Dict[str, float]

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, 
                value: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(0)
        
        # Linear transformations
        Q = self.q_linear(query)
        K = self.k_linear(key)
        V = self.v_linear(value)
        
        # Split into heads
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Calculate attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attention = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        out = torch.matmul(attention, V)
        
        # Combine heads
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        return self.out(out), attention

class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attention_out, attention_weights = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attention_out))
        
        ff_out = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_out))
        
        return x, attention_weights

class MarketTransformer(nn.Module):
    def __init__(self, input_dim: int = 64, d_model: int = 256, 
                 num_heads: int = 8, num_layers: int = 6, 
                 dropout: float = 0.1):
        super().__init__()
        
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = nn.Embedding(1000, d_model)  # Max sequence length of 1000
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_model * 4, dropout)
            for _ in range(num_layers)
        ])
        
        self.market_structure_encoder = nn.Sequential(
            nn.Linear(32, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.microstructure_encoder = nn.Sequential(
            nn.Linear(16, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 2)  # Mean and log variance
        )
        
        self.output_layer = nn.Linear(d_model, 1)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, market_structure: torch.Tensor,
                microstructure: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        # Input embedding and positional encoding
        x = self.input_projection(x)
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = x + self.pos_encoder(positions)
        
        # Process market structure and microstructure
        ms_features = self.market_structure_encoder(market_structure)
        micro_features = self.microstructure_encoder(microstructure)
        
        # Combine features
        x = torch.cat([x, ms_features.unsqueeze(1), micro_features.unsqueeze(1)], dim=1)
        
        # Apply transformer blocks
        attention_weights = []
        for transformer in self.transformer_blocks:
            x, attn = transformer(x)
            attention_weights.append(attn)
            
        # Extract features for prediction
        features = x.mean(dim=1)
        
        # Predict mean and uncertainty
        uncertainty_params = self.uncertainty_estimator(features)
        mean, log_var = uncertainty_params.chunk(2, dim=-1)
        
        # Output prediction
        prediction = self.output_layer(features)
        
        return prediction, torch.exp(log_var), attention_weights

class EnhancedMLPredictor:
    def __init__(self, input_dim: int = 64, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = torch.device(device)
        self.model = MarketTransformer(input_dim=input_dim).to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        
        self.feature_names = []
        self.regime_models = {}
        self.ensemble_weights = {}
        
    def predict(self, features: torch.Tensor, market_structure: torch.Tensor,
                microstructure: torch.Tensor) -> PredictionWithUncertainty:
        """Generate prediction with uncertainty estimation"""
        self.model.eval()
        with torch.no_grad():
            prediction, uncertainty, attention_weights = self.model(
                features, market_structure, microstructure
            )
            
            # Calculate feature importance from attention
            importance = self._calculate_feature_importance(attention_weights[0])
            
            return PredictionWithUncertainty(
                prediction=prediction.cpu().numpy(),
                uncertainty=uncertainty.cpu().numpy(),
                attention_weights=attention_weights[-1].cpu().numpy(),
                feature_importance=dict(zip(self.feature_names, importance))
            )
            
    def adapt_to_regime(self, regime: str, market_data: pd.DataFrame):
        """Adapt model to current market regime using transfer learning"""
        if regime not in self.regime_models:
            # Initialize new regime-specific model
            self.regime_models[regime] = MarketTransformer(
                input_dim=self.model.input_projection.in_features
            ).to(self.device)
            
            # Transfer weights from main network
            self.regime_models[regime].load_state_dict(self.model.state_dict())
            
        # Fine-tune regime-specific model
        regime_model = self.regime_models[regime]
        regime_model.train()
        
        optimizer = torch.optim.Adam(regime_model.parameters(), lr=0.0001)
        
        # Process data
        features = self._create_regime_specific_features(market_data)
        x = torch.tensor(features, dtype=torch.float32).to(self.device)
        
        # Fine-tune for a few steps
        for _ in range(10):
            optimizer.zero_grad()
            prediction, uncertainty, _ = regime_model(x)
            loss = self._calculate_regime_loss(prediction, uncertainty, market_data)
            loss.backward()
            optimizer.step()
            
    def _create_regime_specific_features(self, market_data: pd.DataFrame) -> np.ndarray:
        """Create features specific to market regime"""
        features = []
        
        # Price-based features
        if 'close' in market_data.columns:
            returns = market_data['close'].pct_change()
            volatility = returns.rolling(20).std()
            features.extend([
                returns.fillna(0).values,
                volatility.fillna(0).values
            ])
            
        # Volume-based features
        if 'volume' in market_data.columns:
            volume_ma = market_data['volume'].rolling(20).mean()
            volume_std = market_data['volume'].rolling(20).std()
            features.extend([
                volume_ma.fillna(0).values,
                volume_std.fillna(0).values
            ])
            
        features = np.column_stack(features)
        return features
        
    def _calculate_regime_loss(self, prediction: torch.Tensor, 
                             uncertainty: torch.Tensor,
                             market_data: pd.DataFrame) -> torch.Tensor:
        """Calculate regime-specific loss with uncertainty"""
        # Calculate returns
        returns = torch.tensor(
            market_data['close'].pct_change().fillna(0).values,
            dtype=torch.float32
        ).to(self.device)
        
        # Negative log likelihood loss with uncertainty
        dist = Normal(prediction.squeeze(), torch.sqrt(uncertainty.squeeze()))
        nll_loss = -dist.log_prob(returns).mean()
        
        # Add regularization for uncertainty
        uncertainty_reg = 0.1 * uncertainty.mean()
        
        return nll_loss + uncertainty_reg
        
    def _calculate_feature_importance(self, attention_weights: torch.Tensor) -> np.ndarray:
        """Calculate feature importance from attention weights"""
        # Average attention weights across heads
        importance = attention_weights.mean(dim=1).mean(dim=1)
        
        # Normalize
        importance = importance / importance.sum()
        
        return importance.cpu().numpy()
        
    def save_model(self, path: str):
        """Save model state and learned parameters"""
        torch.save({
            'model_state': self.model.state_dict(),
            'regime_models': {
                regime: model.state_dict()
                for regime, model in self.regime_models.items()
            },
            'ensemble_weights': self.ensemble_weights,
            'feature_names': self.feature_names
        }, path)
        
    def load_model(self, path: str):
        """Load model state and learned parameters"""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        
        for regime, state_dict in checkpoint['regime_models'].items():
            if regime not in self.regime_models:
                self.regime_models[regime] = MarketTransformer(
                    input_dim=self.model.input_projection.in_features
                ).to(self.device)
            self.regime_models[regime].load_state_dict(state_dict)
            
        self.ensemble_weights = checkpoint['ensemble_weights']
        self.feature_names = checkpoint['feature_names']
