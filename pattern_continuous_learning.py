"""
Advanced continuous learning system for pattern and signal validation.
"""
import logging
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import os

logger = logging.getLogger(__name__)

class PatternContinuousLearning:
    """Advanced continuous learning system for pattern validation and signal improvement."""
    
    def __init__(self, history_file: str = "data/pattern_history.json"):
        self.history_file = history_file
        self.pattern_history = self._load_history()
        self.validation_weights = {
            'pattern_reliability': 0.3,
            'market_regime': 0.2,
            'volume_confirmation': 0.2,
            'trend_alignment': 0.2,
            'support_resistance': 0.1
        }
        
    def _load_history(self) -> List[Dict]:
        """Load pattern history from file."""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return []
        except Exception as e:
            logger.error(f"Error loading pattern history: {str(e)}")
            return []
            
    def _save_history(self):
        """Save pattern history to file."""
        try:
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)
            with open(self.history_file, 'w') as f:
                json.dump(self.pattern_history, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving pattern history: {str(e)}")
            
    def record_pattern(self, 
                      patterns: Dict,
                      market_context: Dict,
                      signal_result: str,
                      profit_loss: float = None) -> None:
        """
        Record pattern occurrence and result for learning.
        
        Args:
            patterns: Dictionary of detected patterns
            market_context: Market context when patterns were detected
            signal_result: Result of the trade ('WIN', 'LOSS', or 'NEUTRAL')
            profit_loss: Optional profit/loss percentage
        """
        try:
            record = {
                'timestamp': datetime.now().isoformat(),
                'patterns': patterns,
                'market_context': market_context,
                'result': signal_result,
                'profit_loss': profit_loss,
                'validation_scores': self._calculate_validation_scores(patterns, market_context)
            }
            
            self.pattern_history.append(record)
            
            # Trim old history (keep last 1000 records)
            if len(self.pattern_history) > 1000:
                self.pattern_history = self.pattern_history[-1000:]
                
            self._save_history()
            
        except Exception as e:
            logger.error(f"Error recording pattern: {str(e)}")
            
    def _calculate_validation_scores(self, patterns: Dict, market_context: Dict) -> Dict:
        """Calculate validation scores for different components."""
        scores = {}
        try:
            # Pattern reliability score
            if patterns:
                reliability_scores = [p.get('reliability', 0.7) for p in patterns.values()]
                scores['pattern_reliability'] = sum(reliability_scores) / len(reliability_scores)
            else:
                scores['pattern_reliability'] = 0.0
                
            # Market regime score
            regime = market_context.get('market_regime', {}).get('regime', 'UNKNOWN')
            scores['market_regime'] = {
                'TRENDING': 0.9,
                'RANGING': 0.7,
                'VOLATILE': 0.5,
                'UNKNOWN': 0.3
            }.get(regime, 0.3)
            
            # Volume confirmation
            scores['volume_confirmation'] = 1.0 if market_context.get('volume_confirmed', False) else 0.5
            
            # Trend alignment
            scores['trend_alignment'] = 1.0 if market_context.get('trend_aligned', False) else 0.5
            
            # Support/Resistance validation
            scores['support_resistance'] = 1.0 if market_context.get('support_resistance_validated', False) else 0.5
            
        except Exception as e:
            logger.error(f"Error calculating validation scores: {str(e)}")
            scores = {k: 0.5 for k in self.validation_weights.keys()}
            
        return scores
        
    def get_pattern_performance(self, pattern_name: str, timeframe: str = '7d') -> Dict:
        """
        Get historical performance metrics for a specific pattern.
        
        Args:
            pattern_name: Name of the pattern to analyze
            timeframe: Timeframe to analyze ('1d', '7d', '30d')
            
        Returns:
            Dict containing performance metrics
        """
        try:
            # Calculate cutoff time
            cutoff_time = datetime.now() - {
                '1d': timedelta(days=1),
                '7d': timedelta(days=7),
                '30d': timedelta(days=30)
            }.get(timeframe, timedelta(days=7))
            
            # Filter relevant history
            pattern_records = [
                record for record in self.pattern_history
                if (datetime.fromisoformat(record['timestamp']) > cutoff_time and
                    pattern_name in record['patterns'])
            ]
            
            if not pattern_records:
                return {
                    'win_rate': 0.0,
                    'avg_profit': 0.0,
                    'occurrence_count': 0,
                    'reliability_score': 0.0
                }
                
            # Calculate metrics
            wins = sum(1 for r in pattern_records if r['result'] == 'WIN')
            total = len(pattern_records)
            profits = [r['profit_loss'] for r in pattern_records if r['profit_loss'] is not None]
            
            return {
                'win_rate': wins / total if total > 0 else 0.0,
                'avg_profit': sum(profits) / len(profits) if profits else 0.0,
                'occurrence_count': total,
                'reliability_score': self._calculate_reliability_score(pattern_records)
            }
            
        except Exception as e:
            logger.error(f"Error calculating pattern performance: {str(e)}")
            return {
                'win_rate': 0.0,
                'avg_profit': 0.0,
                'occurrence_count': 0,
                'reliability_score': 0.0
            }
            
    def _calculate_reliability_score(self, pattern_records: List[Dict]) -> float:
        """Calculate overall reliability score based on historical performance."""
        try:
            if not pattern_records:
                return 0.0
                
            total_score = 0.0
            for record in pattern_records:
                validation_scores = record['validation_scores']
                weighted_score = sum(
                    score * self.validation_weights[component]
                    for component, score in validation_scores.items()
                )
                total_score += weighted_score
                
            return total_score / len(pattern_records)
            
        except Exception as e:
            logger.error(f"Error calculating reliability score: {str(e)}")
            return 0.0
            
    def get_optimal_conditions(self, pattern_name: str) -> Dict:
        """
        Analyze historical data to determine optimal conditions for pattern success.
        
        Args:
            pattern_name: Name of the pattern to analyze
            
        Returns:
            Dict containing optimal conditions
        """
        try:
            pattern_records = [
                record for record in self.pattern_history
                if pattern_name in record['patterns']
            ]
            
            if not pattern_records:
                return {}
                
            # Analyze successful trades
            winning_records = [r for r in pattern_records if r['result'] == 'WIN']
            if not winning_records:
                return {}
                
            # Analyze market conditions in winning trades
            regimes = []
            volume_confirmed = 0
            trend_aligned = 0
            sr_validated = 0
            
            for record in winning_records:
                context = record['market_context']
                regimes.append(context.get('market_regime', {}).get('regime', 'UNKNOWN'))
                if context.get('volume_confirmed', False):
                    volume_confirmed += 1
                if context.get('trend_aligned', False):
                    trend_aligned += 1
                if context.get('support_resistance_validated', False):
                    sr_validated += 1
                    
            total_wins = len(winning_records)
            
            return {
                'best_regime': max(set(regimes), key=regimes.count),
                'volume_confirmation_important': volume_confirmed / total_wins > 0.7,
                'trend_alignment_important': trend_aligned / total_wins > 0.7,
                'sr_validation_important': sr_validated / total_wins > 0.7
            }
            
        except Exception as e:
            logger.error(f"Error analyzing optimal conditions: {str(e)}")
            return {}
