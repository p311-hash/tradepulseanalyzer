import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

@dataclass
class ExecutionMetrics:
    slippage: float
    market_impact: float
    execution_time: float
    fill_ratio: float
    price_improvement: float

class SmartOrderRouter:
    def __init__(self):
        self.execution_history = deque(maxlen=1000)
        self.venue_performance = {}
        self.impact_models = {}
        
    def route_order(self, order: Dict, market_state: Dict) -> Dict:
        """Route order to optimal venue based on execution metrics"""
        venues = self._analyze_venues(order, market_state)
        best_venue = max(venues.items(), key=lambda x: x[1]['score'])[0]
        
        return {
            'venue': best_venue,
            'strategy': self._get_execution_strategy(order, market_state),
            'params': self._optimize_execution_params(order, market_state)
        }
        
    def _analyze_venues(self, order: Dict, market_state: Dict) -> Dict:
        """Analyze available venues for optimal execution"""
        venues = {}
        for venue in self._get_active_venues():
            metrics = self._get_venue_metrics(venue)
            liquidity = self._estimate_venue_liquidity(venue, market_state)
            cost = self._estimate_execution_cost(venue, order)
            
            score = self._calculate_venue_score(metrics, liquidity, cost)
            venues[venue] = {
                'score': score,
                'metrics': metrics,
                'liquidity': liquidity,
                'cost': cost
            }
            
        return venues

class ExecutionOptimizer:
    def __init__(self, order_router: SmartOrderRouter):
        self.router = order_router
        self.execution_cache = {}
        self.impact_history = deque(maxlen=1000)
        self.twap_window = 20
        self.vwap_window = 20
        
    def optimize_execution(self, order: Dict, market_state: Dict,
                         deep_structure: Dict) -> Dict:
        """Optimize order execution using market microstructure and ML signals"""
        # Analyze market impact
        impact = self._estimate_market_impact(order, deep_structure)
        
        # Calculate optimal execution schedule
        schedule = self._calculate_execution_schedule(order, market_state, impact)
        
        # Route order parts
        routed_orders = []
        for child_order in schedule['orders']:
            route = self.router.route_order(child_order, market_state)
            child_order.update(route)
            routed_orders.append(child_order)
            
        return {
            'orders': routed_orders,
            'schedule': schedule['timing'],
            'expected_impact': impact
        }
        
    def _estimate_market_impact(self, order: Dict, deep_structure: Dict) -> float:
        """Estimate market impact using deep market structure analysis"""
        # Get relevant market structure features
        market_delta = deep_structure.get('market_delta', 0)
        cumulative_delta = deep_structure.get('cumulative_delta', 0)
        auction_zones = deep_structure.get('zones', {})
        
        # Calculate base impact
        order_size = order['size']
        base_impact = self._calculate_base_impact(order_size)
        
        # Adjust for market structure
        impact_multiplier = 1.0
        
        # Adjust for order book imbalance
        if market_delta != 0:
            impact_multiplier *= (1 + abs(market_delta))
            
        # Adjust for institutional activity
        if 'institutional_zones' in auction_zones:
            inst_zones = auction_zones['institutional_zones']
            if inst_zones and inst_zones[-1]['type'] == 'accumulation':
                impact_multiplier *= 0.8  # Reduce impact if institutions are accumulating
                
        return base_impact * impact_multiplier
        
    def _calculate_execution_schedule(self, order: Dict, market_state: Dict,
                                   estimated_impact: float) -> Dict:
        """Calculate optimal execution schedule based on market conditions"""
        total_size = order['size']
        urgency = order.get('urgency', 'normal')
        
        if urgency == 'high':
            # Aggressive execution
            return self._generate_aggressive_schedule(total_size, market_state)
        elif urgency == 'low':
            # Passive execution
            return self._generate_passive_schedule(total_size, market_state)
        else:
            # Balanced execution
            return self._generate_balanced_schedule(total_size, market_state, estimated_impact)
            
    def _generate_aggressive_schedule(self, total_size: float, 
                                   market_state: Dict) -> Dict:
        """Generate aggressive execution schedule"""
        # Use larger order sizes and shorter intervals
        interval = timedelta(minutes=1)
        num_orders = 5
        
        base_size = total_size / num_orders
        orders = []
        current_time = datetime.now()
        
        for i in range(num_orders):
            size = base_size * (1.2 if i == 0 else 1.0)  # Front-load first order
            orders.append({
                'size': size,
                'time': current_time + interval * i
            })
            
        return {
            'orders': orders,
            'timing': {
                'start': current_time,
                'end': current_time + interval * num_orders,
                'interval': interval
            }
        }
        
    def _generate_passive_schedule(self, total_size: float,
                                market_state: Dict) -> Dict:
        """Generate passive execution schedule"""
        # Use smaller order sizes and longer intervals
        interval = timedelta(minutes=5)
        num_orders = 10
        
        base_size = total_size / num_orders
        orders = []
        current_time = datetime.now()
        
        for i in range(num_orders):
            size = base_size * (0.8 if i == 0 else 1.0)  # Reduce first order size
            orders.append({
                'size': size,
                'time': current_time + interval * i
            })
            
        return {
            'orders': orders,
            'timing': {
                'start': current_time,
                'end': current_time + interval * num_orders,
                'interval': interval
            }
        }
        
    def _generate_balanced_schedule(self, total_size: float, market_state: Dict,
                                 estimated_impact: float) -> Dict:
        """Generate balanced execution schedule"""
        # Adjust schedule based on estimated impact
        interval = timedelta(minutes=2)
        num_orders = 8
        
        if estimated_impact > 0.02:  # High impact threshold
            num_orders = 12  # More orders to reduce impact
            interval = timedelta(minutes=3)
            
        base_size = total_size / num_orders
        orders = []
        current_time = datetime.now()
        
        for i in range(num_orders):
            size = base_size
            orders.append({
                'size': size,
                'time': current_time + interval * i
            })
            
        return {
            'orders': orders,
            'timing': {
                'start': current_time,
                'end': current_time + interval * num_orders,
                'interval': interval
            }
        }
        
    def _calculate_base_impact(self, size: float) -> float:
        """Calculate base market impact"""
        # Simple square-root model
        return 0.1 * np.sqrt(size / 100.0)  # Adjust constants based on asset
        
    def update_execution_metrics(self, order: Dict, execution: Dict):
        """Update execution metrics for future optimization"""
        metrics = ExecutionMetrics(
            slippage=execution['price'] - order['price'],
            market_impact=execution['market_impact'],
            execution_time=execution['duration'].total_seconds(),
            fill_ratio=execution['filled'] / order['size'],
            price_improvement=execution['price_improvement']
        )
        
        # Update history
        self.execution_cache[order['id']] = metrics
        self.impact_history.append(execution['market_impact'])
        
        # Update venue performance
        venue = order['venue']
        if venue not in self.router.venue_performance:
            self.router.venue_performance[venue] = []
        self.router.venue_performance[venue].append(metrics)
        
        # Trim old data
        if len(self.router.venue_performance[venue]) > 100:
            self.router.venue_performance[venue].pop(0)
            
    def get_execution_analytics(self) -> Dict:
        """Get execution analytics for monitoring and optimization"""
        if not self.execution_cache:
            return {}
            
        metrics = list(self.execution_cache.values())
        return {
            'avg_slippage': np.mean([m.slippage for m in metrics]),
            'avg_impact': np.mean([m.market_impact for m in metrics]),
            'avg_fill_ratio': np.mean([m.fill_ratio for m in metrics]),
            'price_improvement': np.mean([m.price_improvement for m in metrics]),
            'execution_speed': np.mean([m.execution_time for m in metrics]),
            'venue_analysis': self._analyze_venue_performance()
        }
        
    def _analyze_venue_performance(self) -> Dict:
        """Analyze venue performance metrics"""
        venue_analysis = {}
        for venue, metrics in self.router.venue_performance.items():
            if not metrics:
                continue
                
            venue_analysis[venue] = {
                'avg_slippage': np.mean([m.slippage for m in metrics]),
                'fill_ratio': np.mean([m.fill_ratio for m in metrics]),
                'cost_score': np.mean([m.price_improvement for m in metrics]),
                'reliability': len(metrics) / max(len(v) for v in self.router.venue_performance.values())
            }
            
        return venue_analysis
