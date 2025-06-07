"""
Enhanced ML model v2 with Deep Reinforcement Learning and Transfer Learning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class MarketState:
    technical_features: np.ndarray
    market_structure: Dict
    microstructure: Dict
    regime: str
    action_history: List[int]

@dataclass
class TradingEnv:
    commission: float = 0.001
    slippage: float = 0.0002
    position_limit: float = 1.0

class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        super().__init__()
        
        # Shared feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        
        # Actor network (policy)
        self.actor_mean = nn.Linear(hidden_size, action_dim)
        self.actor_log_std = nn.Linear(hidden_size, action_dim)
        
        # Critic network (value function)
        self.critic = nn.Linear(hidden_size, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
            nn.init.zeros_(module.bias)
            
    def forward(self, state: torch.Tensor) -> Tuple[Normal, torch.Tensor]:
        features = self.feature_net(state)
        
        # Actor outputs
        action_mean = self.actor_mean(features)
        action_log_std = self.actor_log_std(features)
        action_log_std = torch.clamp(action_log_std, -20, 2)  # Prevent numerical instability
        
        # Create normal distribution
        action_std = torch.exp(action_log_std)
        dist = Normal(action_mean, action_std)
        
        # Critic output
        value = self.critic(features)
        
        return dist, value

class MarketTransformer(nn.Module):
    def __init__(self, d_model: int = 256, nhead: int = 8, num_layers: int = 6):
        super().__init__()
        
        self.d_model = d_model
        self.pos_encoder = nn.Embedding(1000, d_model)  # Maximum sequence length of 1000
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.market_structure_encoder = nn.Sequential(
            nn.Linear(64, d_model),  # Adjust input size based on market structure features
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
        self.microstructure_encoder = nn.Sequential(
            nn.Linear(32, d_model),  # Adjust input size based on microstructure features
            nn.LayerNorm(d_model),
            nn.ReLU()
        )
        
    def forward(self, x: torch.Tensor, market_structure: torch.Tensor,
                microstructure: torch.Tensor) -> torch.Tensor:
        # Add positional encoding
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        x = x + self.pos_encoder(positions)
        
        # Encode market structure and microstructure data
        ms_features = self.market_structure_encoder(market_structure)
        micro_features = self.microstructure_encoder(microstructure)
        
        # Combine features
        x = torch.cat([x, ms_features.unsqueeze(1), micro_features.unsqueeze(1)], dim=1)
        
        # Apply transformer
        x = self.transformer(x)
        
        return x

class EnhancedTradingAgent:
    def __init__(self, state_dim: int, action_dim: int, 
                 learning_rate: float = 3e-4, gamma: float = 0.99):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize networks
        self.actor_critic = ActorCritic(state_dim, action_dim).to(self.device)
        self.transformer = MarketTransformer().to(self.device)
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            list(self.actor_critic.parameters()) + list(self.transformer.parameters()),
            lr=learning_rate
        )
        
        # Training parameters
        self.gamma = gamma
        self.gae_lambda = 0.95
        self.clip_param = 0.2
        self.max_grad_norm = 0.5
        
        # Memory buffers
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.log_probs = []
        
    def select_action(self, state: MarketState) -> Tuple[float, Dict]:
        """Select trading action based on current market state"""
        with torch.no_grad():
            # Convert state components to tensors
            technical = torch.FloatTensor(state.technical_features).unsqueeze(0)
            market_structure = self._process_market_structure(state.market_structure)
            microstructure = self._process_microstructure(state.microstructure)
            
            # Get transformer features
            features = self.transformer(
                technical.to(self.device),
                market_structure.to(self.device),
                microstructure.to(self.device)
            )
            
            # Get action distribution and value
            action_dist, value = self.actor_critic(features.mean(dim=1))
            
            # Sample action
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            
            return (
                action.cpu().numpy()[0],
                {
                    'value': value.cpu().numpy()[0],
                    'log_prob': log_prob.cpu().numpy()[0],
                    'action_mean': action_dist.mean.cpu().numpy()[0],
                    'action_std': action_dist.stddev.cpu().numpy()[0]
                }
            )
            
    def update(self, batch_size: int = 64) -> Dict[str, float]:
        """Update policy and value networks using PPO"""
        device = self.device
        states = torch.FloatTensor(self.states).to(device)
        actions = torch.FloatTensor(self.actions).to(device)
        rewards = torch.FloatTensor(self.rewards).to(device)
        old_values = torch.FloatTensor(self.values).to(device)
        old_log_probs = torch.FloatTensor(self.log_probs).to(device)
        
        # Calculate advantages
        advantages = self._compute_gae(rewards, old_values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO update
        for _ in range(10):  # Number of PPO epochs
            # Sample random mini-batch
            indices = torch.randperm(len(states))[:batch_size]
            state_batch = states[indices]
            action_batch = actions[indices]
            advantage_batch = advantages[indices]
            old_value_batch = old_values[indices]
            old_log_prob_batch = old_log_probs[indices]
            
            # Get current policy distribution and value
            action_dist, value = self.actor_critic(state_batch)
            
            # Calculate policy loss
            log_prob = action_dist.log_prob(action_batch)
            ratio = torch.exp(log_prob - old_log_prob_batch)
            
            policy_loss1 = -advantage_batch * ratio
            policy_loss2 = -advantage_batch * torch.clamp(
                ratio,
                1 - self.clip_param,
                1 + self.clip_param
            )
            policy_loss = torch.max(policy_loss1, policy_loss2).mean()
            
            # Calculate value loss
            value_loss = F.mse_loss(value.squeeze(), old_value_batch + advantage_batch)
            
            # Calculate entropy bonus
            entropy_loss = -action_dist.entropy().mean()
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss
            
            # Update networks
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.actor_critic.parameters()) + list(self.transformer.parameters()),
                self.max_grad_norm
            )
            self.optimizer.step()
            
        # Clear memory buffers
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.values.clear()
        self.log_probs.clear()
        
        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': -entropy_loss.item()
        }
        
    def _compute_gae(self, rewards: torch.Tensor, 
                    values: torch.Tensor) -> torch.Tensor:
        """Compute Generalized Advantage Estimation"""
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
                
            delta = rewards[t] + self.gamma * next_value - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * last_advantage
            last_advantage = advantages[t]
            
        return advantages
        
    def _process_market_structure(self, market_structure: Dict) -> torch.Tensor:
        """Convert market structure data to tensor format"""
        features = []
        
        # Extract relevant features from market structure
        features.extend([
            market_structure['delta'],
            market_structure['cumulative_delta'],
            *self._flatten_zones(market_structure['zones']),
            *self._encode_price_acceptance(market_structure['price_acceptance'])
        ])
        
        return torch.FloatTensor(features).unsqueeze(0)
        
    def _process_microstructure(self, microstructure: Dict) -> torch.Tensor:
        """Convert microstructure data to tensor format"""
        features = []
        
        # Extract relevant features from microstructure
        features.extend([
            microstructure['spread'],
            microstructure['imbalance'],
            microstructure['market_depth'],
            *self._process_orderbook(microstructure.get('order_book', {}))
        ])
        
        return torch.FloatTensor(features).unsqueeze(0)
        
    def _flatten_zones(self, zones: Dict) -> List[float]:
        """Flatten market structure zones into feature vector"""
        features = []
        features.extend([
            zones['value_area']['va_high'],
            zones['value_area']['va_low'],
            zones['poc_price']
        ])
        
        # Process institutional zones
        inst_zones = zones.get('institutional_zones', [])
        if inst_zones:
            latest_zone = inst_zones[-1]
            features.extend([
                1 if latest_zone['type'] == 'accumulation' else -1,
                latest_zone['volume'],
                latest_zone['delta']
            ])
        else:
            features.extend([0, 0, 0])
            
        return features
        
    def _encode_price_acceptance(self, acceptance: Dict) -> List[float]:
        """Encode price acceptance information"""
        position_encoding = {
            'above_value': [1, 0, 0],
            'inside_value': [0, 1, 0],
            'below_value': [0, 0, 1]
        }
        
        return [
            *position_encoding.get(acceptance['price_position'], [0, 0, 0]),
            acceptance['acceptance_ratio']
        ]
        
    def _process_orderbook(self, order_book: Dict) -> List[float]:
        """Process order book into feature vector"""
        if not order_book:
            return [0] * 10  # Return zeros if no order book data
            
        bid_volumes = [level['volume'] for level in order_book.get('bids', [])[:5]]
        ask_volumes = [level['volume'] for level in order_book.get('asks', [])[:5]]
        
        # Pad with zeros if needed
        bid_volumes = (bid_volumes + [0] * 5)[:5]
        ask_volumes = (ask_volumes + [0] * 5)[:5]
        
        return bid_volumes + ask_volumes
        
    def save_model(self, path: str):
        """Save model state"""
        torch.save({
            'actor_critic_state': self.actor_critic.state_dict(),
            'transformer_state': self.transformer.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)
        
    def load_model(self, path: str):
        """Load model state"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state'])
        self.transformer.load_state_dict(checkpoint['transformer_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
