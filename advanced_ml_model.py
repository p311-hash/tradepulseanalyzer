import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import pickle
from collections import deque
import random

@dataclass
class TradingState:
    price_history: np.ndarray
    market_features: Dict
    deep_structure: Dict
    positions: List[Dict]
    balance: float

class MarketTransformer(nn.Module):
    def __init__(self, input_dim: int, d_model: int = 128, nhead: int = 4, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, d_model)
        self.position_encoding = nn.Parameter(torch.randn(1000, d_model))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, 
                                     dim_feedforward=d_model*4,
                                     dropout=dropout),
            num_layers=num_layers
        )
        self.output_projection = nn.Linear(d_model, 3)  # Buy, Hold, Sell
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        seq_len = x.size(1)
        x = x + self.position_encoding[:seq_len, :]
        x = self.transformer(x.transpose(0, 1)).transpose(0, 1)
        return self.output_projection(x[:, -1, :])  # Only use last sequence output

class ExperienceBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
        
    def add(self, state: TradingState, action: int, reward: float, 
            next_state: TradingState, done: bool):
        self.buffer.append((state, action, reward, next_state, done))
        
    def sample(self, batch_size: int) -> List:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
    def __len__(self) -> int:
        return len(self.buffer)

class AdvancedMLModel:
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99,
                 batch_size: int = 32, update_target_every: int = 100):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.input_dim = 256  # Adjusted based on feature engineering
        
        # Main network and target network (for DQN stability)
        self.main_network = MarketTransformer(self.input_dim).to(self.device)
        self.target_network = MarketTransformer(self.input_dim).to(self.device)
        self.target_network.load_state_dict(self.main_network.state_dict())
        
        self.optimizer = optim.Adam(self.main_network.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # RL parameters
        self.gamma = gamma
        self.batch_size = batch_size
        self.update_target_every = update_target_every
        self.steps = 0
        
        # Experience replay
        self.experience_buffer = ExperienceBuffer()
        
        # Transfer learning state
        self.market_embeddings = {}
        self.regime_models = {}
        
    def preprocess_state(self, state: TradingState) -> torch.Tensor:
        """Convert state to tensor format for the network"""
        # Process price history
        price_features = torch.tensor(state.price_history, dtype=torch.float32)
        
        # Process market features
        market_features = []
        for key, value in state.market_features.items():
            if isinstance(value, (int, float)):
                market_features.append(value)
        market_features = torch.tensor(market_features, dtype=torch.float32)
        
        # Process deep structure features
        deep_features = []
        if 'market_delta' in state.deep_structure:
            deep_features.extend([
                state.deep_structure['market_delta'],
                state.deep_structure['cumulative_delta']
            ])
        deep_features = torch.tensor(deep_features, dtype=torch.float32)
        
        # Combine all features
        combined = torch.cat([
            price_features.flatten(),
            market_features,
            deep_features
        ]).unsqueeze(0)  # Add batch dimension
        
        return combined.to(self.device)
        
    def get_action(self, state: TradingState, epsilon: float = 0.1) -> int:
        """Get action using epsilon-greedy policy"""
        if random.random() < epsilon:
            return random.randint(0, 2)  # Random action
            
        with torch.no_grad():
            state_tensor = self.preprocess_state(state)
            q_values = self.main_network(state_tensor)
            return q_values.argmax().item()
            
    def update(self, batch: List[Tuple]) -> float:
        """Update the model using a batch of experiences"""
        if len(batch) < self.batch_size:
            return 0.0
            
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        state_tensors = torch.cat([self.preprocess_state(s) for s in states])
        action_tensors = torch.tensor(actions, dtype=torch.long).to(self.device)
        reward_tensors = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_state_tensors = torch.cat([self.preprocess_state(s) for s in next_states])
        done_tensors = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        # Get current Q values
        current_q_values = self.main_network(state_tensors)
        current_q_values = current_q_values.gather(1, action_tensors.unsqueeze(1))
        
        # Get next Q values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_state_tensors)
            max_next_q = next_q_values.max(1)[0]
            target_q_values = reward_tensors + (1 - done_tensors) * self.gamma * max_next_q
            
        # Update main network
        loss = self.criterion(current_q_values.squeeze(), target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network periodically
        self.steps += 1
        if self.steps % self.update_target_every == 0:
            self.target_network.load_state_dict(self.main_network.state_dict())
            
        return loss.item()
        
    def adapt_to_regime(self, regime: str, market_data: pd.DataFrame):
        """Adapt model to current market regime using transfer learning"""
        if regime not in self.regime_models:
            # Initialize new regime-specific model
            self.regime_models[regime] = MarketTransformer(self.input_dim).to(self.device)
            # Transfer weights from main network as starting point
            self.regime_models[regime].load_state_dict(self.main_network.state_dict())
            
        # Fine-tune regime-specific model
        regime_model = self.regime_models[regime]
        regime_model.train()
        
        # Create regime-specific dataset and train
        # This is a simplified version - in practice, you'd want to create a proper
        # dataset and training loop
        optimizer = optim.Adam(regime_model.parameters(), lr=0.0001)
        
        for _ in range(10):  # Limited fine-tuning steps
            # Process market data and create features
            features = self._create_regime_specific_features(market_data)
            predictions = regime_model(features)
            
            # Calculate regime-specific loss
            target = self._calculate_regime_specific_targets(market_data)
            loss = self.criterion(predictions, target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def _create_regime_specific_features(self, market_data: pd.DataFrame) -> torch.Tensor:
        """Create features specific to market regime"""
        # Example feature creation - expand based on regime characteristics
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
        return torch.tensor(features, dtype=torch.float32).to(self.device)
        
    def _calculate_regime_specific_targets(self, market_data: pd.DataFrame) -> torch.Tensor:
        """Calculate targets specific to market regime"""
        # Example target calculation - customize based on regime
        if 'close' in market_data.columns:
            future_returns = market_data['close'].pct_change(5).shift(-5)
            targets = np.where(future_returns > 0.01, 1,  # Buy
                             np.where(future_returns < -0.01, 2, 0))  # Sell or Hold
            return torch.tensor(targets, dtype=torch.long).to(self.device)
        return torch.zeros(len(market_data), dtype=torch.long).to(self.device)
        
    def save_model(self, path: str):
        """Save model state and learned embeddings"""
        torch.save({
            'main_network': self.main_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'market_embeddings': self.market_embeddings,
            'regime_models': {k: v.state_dict() for k, v in self.regime_models.items()},
            'steps': self.steps
        }, path)
        
    def load_model(self, path: str):
        """Load model state and learned embeddings"""
        checkpoint = torch.load(path)
        self.main_network.load_state_dict(checkpoint['main_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.market_embeddings = checkpoint['market_embeddings']
        
        # Load regime models
        for regime, state_dict in checkpoint['regime_models'].items():
            if regime not in self.regime_models:
                self.regime_models[regime] = MarketTransformer(self.input_dim).to(self.device)
            self.regime_models[regime].load_state_dict(state_dict)
            
        self.steps = checkpoint['steps']
