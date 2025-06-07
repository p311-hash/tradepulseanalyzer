"""Market microstructure analysis module for real-time impact analysis."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import logging
from scipy import stats
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

@dataclass
class OrderBookLevel:
    price: float
    volume: float
    count: int = 1

class MarketMicrostructureAnalyzer:
    def __init__(self):
        self.order_book_cache = {}
        self.volume_profile = {}
        self.impact_threshold = 0.0001  # 1 pip
        self.volume_window = 20
        self.tick_buffer = []
        self.order_flow_buffer = []
        self.institutional_threshold = 100  # Large order threshold
        self.liquidity_window = 100  # Window for liquidity analysis
        
    def analyze_order_book(self, bids: List[OrderBookLevel], asks: List[OrderBookLevel]) -> Dict:
        """Analyzes order book for market impact and liquidity"""
        # Basic metrics
        spread = asks[0].price - bids[0].price if asks and bids else 0
        bid_volume = sum(level.volume for level in bids)
        ask_volume = sum(level.volume for level in asks)
        
        # Advanced liquidity analysis
        bid_depth = self._calculate_market_depth(bids)
        ask_depth = self._calculate_market_depth(asks)
        
        # Order book imbalance
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume) if (bid_volume + ask_volume) > 0 else 0
        
        # Detect institutional activity
        inst_footprint = self._detect_institutional_footprint(bids, asks)
        
        # Order flow toxicity
        toxicity = self._calculate_order_flow_toxicity()
        
        return {
            'spread': spread,
            'bid_volume': bid_volume,
            'ask_volume': ask_volume,
            'imbalance': imbalance,
            'market_depth': len(bids) + len(asks),
            'bid_depth': bid_depth,
            'ask_depth': ask_depth,
            'institutional_presence': inst_footprint,
            'order_flow_toxicity': toxicity,
            'liquidity_score': self._calculate_liquidity_score(spread, bid_depth, ask_depth)
        }

    def _calculate_market_depth(self, levels: List[OrderBookLevel]) -> float:
        """Calculate market depth with price impact"""
        depth = 0
        reference_price = levels[0].price if levels else 0
        
        for level in levels:
            price_impact = abs(level.price - reference_price) / reference_price
            depth += level.volume * (1 - price_impact)
            
        return depth

    def _detect_institutional_footprint(self, bids: List[OrderBookLevel], 
                                     asks: List[OrderBookLevel]) -> Dict:
        """Detect potential institutional activity"""
        large_bids = [level for level in bids if level.volume >= self.institutional_threshold]
        large_asks = [level for level in asks if level.volume >= self.institutional_threshold]
        
        return {
            'buy_side': len(large_bids) > 0,
            'sell_side': len(large_asks) > 0,
            'buy_volume': sum(level.volume for level in large_bids),
            'sell_volume': sum(level.volume for level in large_asks),
            'price_levels': {
                'buy': [level.price for level in large_bids],
                'sell': [level.price for level in large_asks]
            }
        }

    def _calculate_order_flow_toxicity(self) -> float:
        """Calculate order flow toxicity using VPIN"""
        if len(self.order_flow_buffer) < self.volume_window:
            return 0.0
            
        buy_volume = sum(flow['buy_volume'] for flow in self.order_flow_buffer[-self.volume_window:])
        sell_volume = sum(flow['sell_volume'] for flow in self.order_flow_buffer[-self.volume_window:])
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return 0.0
            
        return abs(buy_volume - sell_volume) / total_volume

    def _calculate_liquidity_score(self, spread: float, bid_depth: float, 
                                ask_depth: float) -> float:
        """Calculate overall liquidity score"""
        spread_score = 1 / (1 + spread)  # Normalize spread
        depth_score = (bid_depth + ask_depth) / 2  # Average depth
        
        # Combine scores with weights
        return 0.4 * spread_score + 0.6 * depth_score

    def analyze_market_impact(self, data: pd.DataFrame, volume_profile: Dict) -> Dict:
        """Analyze real-time market impact."""
        if len(self.tick_buffer) < 2:
            return {}
            
        # Calculate price impact
        price_changes = np.diff([tick['price'] for tick in self.tick_buffer])
        volume_changes = np.diff([tick['volume'] for tick in self.tick_buffer])
        
        # Calculate impact coefficients
        impact_coef = np.corrcoef(price_changes, volume_changes)[0,1]
        
        # Analyze relative to value areas
        current_price = data['close'].iloc[-1]
        value_area_impact = 0
        
        if volume_profile:
            va_high = volume_profile.get('va_high', current_price)
            va_low = volume_profile.get('va_low', current_price)
            
            if current_price > va_high:
                value_area_impact = (current_price - va_high) / va_high
            elif current_price < va_low:
                value_area_impact = (va_low - current_price) / va_low
        
        return {
            'price_impact': impact_coef,
            'value_area_impact': value_area_impact,
            'tick_size': len(self.tick_buffer),
            'avg_trade_size': np.mean([tick['volume'] for tick in self.tick_buffer])
        }
