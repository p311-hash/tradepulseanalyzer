import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from market_microstructure import OrderBookLevel, MarketMicrostructureAnalyzer

@dataclass
class MarketDelta:
    timestamp: pd.Timestamp
    delta: float
    cumulative_delta: float
    aggressor_side: str
    price: float
    volume: float

class DeepMarketStructureAnalyzer:
    def __init__(self):
        self.micro_analyzer = MarketMicrostructureAnalyzer()
        self.deltas_cache = {}
        self.auction_zones = {}
        
    def analyze_market_auction(self, trades: pd.DataFrame, timeframe: str = '1h') -> Dict:
        """
        Analyzes market auction theory components including:
        - Market Delta calculations
        - Auction zones identification
        - Price acceptance/rejection levels
        - Institutional footprint detection
        """
        # Group trades by timeframe
        trades['timestamp'] = pd.to_datetime(trades['timestamp'])
        grouped = trades.groupby(pd.Grouper(key='timestamp', freq=timeframe))
        
        results = []
        for _, period in grouped:
            if period.empty:
                continue
                
            # Calculate market delta
            deltas = self._calculate_market_delta(period)
            
            # Identify auction zones
            zones = self._identify_auction_zones(period, deltas)
            
            # Analyze price acceptance
            acceptance = self._analyze_price_acceptance(period, zones)
            
            results.append({
                'timestamp': period.iloc[0]['timestamp'],
                'delta': deltas[-1].delta if deltas else 0,
                'cumulative_delta': deltas[-1].cumulative_delta if deltas else 0,
                'zones': zones,
                'price_acceptance': acceptance
            })
            
        return results

    def _calculate_market_delta(self, trades: pd.DataFrame) -> List[MarketDelta]:
        """
        Calculates trade-by-trade market delta showing aggressive buying vs selling
        """
        deltas = []
        cum_delta = 0
        
        for _, trade in trades.iterrows():
            # Determine trade aggressor (buyer or seller)
            aggressor = 'buy' if trade['side'].lower() == 'buy' else 'sell'
            
            # Calculate delta (positive for buys, negative for sells)
            delta = trade['volume'] if aggressor == 'buy' else -trade['volume']
            cum_delta += delta
            
            deltas.append(MarketDelta(
                timestamp=trade['timestamp'],
                delta=delta,
                cumulative_delta=cum_delta,
                aggressor_side=aggressor,
                price=trade['price'],
                volume=trade['volume']
            ))
            
        return deltas

    def _identify_auction_zones(self, trades: pd.DataFrame, deltas: List[MarketDelta]) -> Dict:
        """
        Identifies key auction zones including:
        - Value Area
        - Point of Control
        - High Volume Nodes
        - Low Volume Nodes
        """
        # Use existing volume profile analysis
        profile = self.micro_analyzer.analyze_volume_profile(trades)
        
        # Enhance with delta volume analysis
        delta_profile = self._calculate_delta_profile(deltas, profile['price_levels'])
        
        # Identify institutional zones (large delta accumulation)
        inst_zones = self._identify_institutional_zones(deltas)
        
        return {
            'value_area': profile['value_area'],
            'poc_price': profile['poc_price'],
            'delta_profile': delta_profile,
            'institutional_zones': inst_zones
        }

    def _calculate_delta_profile(self, deltas: List[MarketDelta], price_levels: List[float]) -> Dict:
        """
        Creates a volume profile using delta values instead of raw volume
        """
        delta_vol = np.zeros(len(price_levels) - 1)
        
        for delta in deltas:
            # Find price level index
            for i in range(len(price_levels) - 1):
                if price_levels[i] <= delta.price < price_levels[i + 1]:
                    delta_vol[i] += delta.delta
                    break
                    
        return {
            'price_levels': price_levels,
            'delta_distribution': delta_vol.tolist()
        }

    def _identify_institutional_zones(self, deltas: List[MarketDelta]) -> List[Dict]:
        """
        Identifies zones with significant institutional activity based on:
        - Large delta accumulation
        - High volume nodes
        - Price acceptance within zone
        """
        zones = []
        current_zone = None
        
        # Parameters for institutional activity detection
        MIN_ZONE_VOLUME = 100  # Minimum volume to consider
        MIN_DELTA_THRESHOLD = 50  # Minimum delta change to consider significant
        
        for delta in deltas:
            if current_zone is None:
                if abs(delta.delta) >= MIN_ZONE_VOLUME:
                    current_zone = {
                        'start_time': delta.timestamp,
                        'start_price': delta.price,
                        'volume': delta.volume,
                        'delta': delta.delta
                    }
            else:
                current_zone['volume'] += delta.volume
                current_zone['delta'] += delta.delta
                
                # Check if zone should be closed
                if abs(current_zone['delta']) >= MIN_DELTA_THRESHOLD:
                    current_zone.update({
                        'end_time': delta.timestamp,
                        'end_price': delta.price,
                        'type': 'accumulation' if current_zone['delta'] > 0 else 'distribution'
                    })
                    zones.append(current_zone)
                    current_zone = None
                
        return zones

    def _analyze_price_acceptance(self, trades: pd.DataFrame, zones: Dict) -> Dict:
        """
        Analyzes whether current price action is being accepted or rejected
        at key market structure levels
        """
        current_price = trades['price'].iloc[-1]
        va_high = zones['value_area']['va_high']
        va_low = zones['value_area']['va_low']
        poc = zones['poc_price']
        
        # Calculate time spent above/below POC
        above_poc_time = len(trades[trades['price'] > poc])
        below_poc_time = len(trades[trades['price'] < poc])
        
        # Determine acceptance/rejection
        price_position = 'above_value' if current_price > va_high else \
                        'below_value' if current_price < va_low else 'inside_value'
                        
        return {
            'price_position': price_position,
            'above_poc_time': above_poc_time,
            'below_poc_time': below_poc_time,
            'acceptance_ratio': above_poc_time / (above_poc_time + below_poc_time) \
                              if (above_poc_time + below_poc_time) > 0 else 0.5
        }
