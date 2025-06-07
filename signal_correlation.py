"""
Signal correlation analysis module for binary options trading.
This module analyzes correlations between signals on different currency pairs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import json
import os
from datetime import datetime, timedelta
from scipy import stats
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Define currency pair correlations
# These are approximate correlations that should be updated with real data
DEFAULT_CORRELATIONS = {
    'EURUSD_otc': {
        'GBPUSD_otc': 0.85,  # High positive correlation
        'EURGBP_otc': 0.65,  # Moderate positive correlation
        'USDCHF_otc': -0.90, # High negative correlation
        'USDJPY_otc': -0.60, # Moderate negative correlation
        'AUDUSD_otc': 0.65,  # Moderate positive correlation
        'NZDUSD_otc': 0.60,  # Moderate positive correlation
        'USDCAD_otc': -0.70, # High negative correlation
        'EURCAD_otc': 0.55,  # Moderate positive correlation
        'EURCHF_otc': 0.75,  # High positive correlation
    },
    'GBPUSD_otc': {
        'EURUSD_otc': 0.85,  # High positive correlation
        'EURGBP_otc': -0.50, # Moderate negative correlation
        'USDCHF_otc': -0.75, # High negative correlation
        'USDJPY_otc': -0.55, # Moderate negative correlation
        'AUDUSD_otc': 0.60,  # Moderate positive correlation
        'NZDUSD_otc': 0.55,  # Moderate positive correlation
        'USDCAD_otc': -0.65, # Moderate negative correlation
        'GBPJPY_otc': 0.75,  # High positive correlation
    },
    'USDJPY_otc': {
        'EURUSD_otc': -0.60, # Moderate negative correlation
        'GBPUSD_otc': -0.55, # Moderate negative correlation
        'EURJPY_otc': 0.70,  # High positive correlation
        'GBPJPY_otc': 0.75,  # High positive correlation
        'AUDJPY_otc': 0.80,  # High positive correlation
        'CADJPY_otc': 0.75,  # High positive correlation
        'CHFJPY_otc': 0.65,  # Moderate positive correlation
    },
    # Add more correlations as needed
}

class CorrelationType(Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

@dataclass
class TimeframeCorrelation:
    timeframe1: str
    timeframe2: str
    correlation: float
    significance: float
    type: CorrelationType
    confidence: float

class SignalCorrelationAnalyzer:
    """Class for analyzing correlations between signals on different pairs."""
    
    def __init__(self, signal_history_path: str = 'signal_history.json', min_correlation: float = 0.5):
        """
        Initialize with path to signal history file.
        
        Args:
            signal_history_path: Path to JSON file containing signal history
        """
        self.signal_history_path = signal_history_path
        self.correlations = DEFAULT_CORRELATIONS.copy()
        self.signal_history = self._load_signal_history()
        self._results = {}
        self.min_correlation = min_correlation
        self.timeframe_correlations: Dict[str, TimeframeCorrelation] = {}
        
    def _load_signal_history(self) -> Dict:
        """
        Load signal history from file.
        
        Returns:
            Dictionary with signal history
        """
        try:
            if os.path.exists(self.signal_history_path):
                with open(self.signal_history_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            logger.error(f"Error loading signal history: {str(e)}")
            return {}
    
    def calculate_dynamic_correlations(self, lookback_days: int = 7) -> Dict[str, Dict[str, float]]:
        """
        Calculate dynamic correlations based on recent signal performance.
        
        Args:
            lookback_days: Number of days to look back for signal correlation
            
        Returns:
            Dictionary with dynamic correlation values
        """
        try:
            if not self.signal_history:
                logger.warning("No signal history available for correlation calculation")
                return self.correlations
                
            # Get recent signals
            cutoff_time = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
            
            # Extract signals with timestamps and outcomes
            recent_signals = {}
            for signal_id, signal in self.signal_history.items():
                if isinstance(signal, dict):  # Make sure signal is a dictionary
                    if 'timestamp' not in signal or 'pair' not in signal or 'signal' not in signal:
                        continue
                        
                    # Skip signals without timestamp or before cutoff
                    if signal['timestamp'] < cutoff_time:
                        continue
                        
                    pair = signal['pair']
                    if pair not in recent_signals:
                        recent_signals[pair] = []
                        
                    recent_signals[pair].append({
                        'timestamp': signal['timestamp'],
                        'direction': signal['signal'],
                        'outcome': signal.get('outcome', 'unknown')
                    })
                
            # Too few pairs with signals
            if len(recent_signals) < 2:
                logger.warning("Not enough pairs with signals for correlation calculation")
                return self.correlations
                
            # Create correlation matrix
            dynamic_correlations = {}
            pairs = list(recent_signals.keys())
            
            for i, pair1 in enumerate(pairs):
                if pair1 not in dynamic_correlations:
                    dynamic_correlations[pair1] = {}
                    
                signals1 = recent_signals[pair1]
                
                for j, pair2 in enumerate(pairs[i+1:], i+1):
                    signals2 = recent_signals[pair2]
                    
                    # Match signals by timestamp (approximately)
                    matched_signals = []
                    for s1 in signals1:
                        for s2 in signals2:
                            # Simple timestamp matching
                            if s1['timestamp'][:16] == s2['timestamp'][:16]:  # Match up to minutes
                                direction_match = (s1['direction'] == s2['direction'])
                                matched_signals.append((direction_match, s1['outcome'], s2['outcome']))
                                break
                                
                    # Calculate signal correlation
                    if matched_signals:
                        # Count matching directions
                        match_count = sum(1 for m in matched_signals if m[0])
                        mismatch_count = len(matched_signals) - match_count
                        
                        # Calculate correlation (-1 to 1)
                        if len(matched_signals) > 0:
                            correlation = (match_count - mismatch_count) / len(matched_signals)
                        else:
                            correlation = 0
                            
                        # Add to correlations (both directions)
                        dynamic_correlations[pair1][pair2] = correlation
                        
                        if pair2 not in dynamic_correlations:
                            dynamic_correlations[pair2] = {}
                        dynamic_correlations[pair2][pair1] = correlation
                    else:
                        # Fallback to default correlation
                        correlation = self.correlations.get(pair1, {}).get(pair2, 0)
                        dynamic_correlations[pair1][pair2] = correlation
                        
                        if pair2 not in dynamic_correlations:
                            dynamic_correlations[pair2] = {}
                        dynamic_correlations[pair2][pair1] = correlation
                        
            # Fill in any missing correlations with defaults
            for pair1, pair1_data in self.correlations.items():
                if pair1 not in dynamic_correlations:
                    dynamic_correlations[pair1] = {}
                    
                for pair2, correlation in pair1_data.items():
                    if pair2 not in dynamic_correlations.get(pair1, {}):
                        dynamic_correlations[pair1][pair2] = correlation
                        
                        if pair2 not in dynamic_correlations:
                            dynamic_correlations[pair2] = {}
                        if pair1 not in dynamic_correlations[pair2]:
                            dynamic_correlations[pair2][pair1] = correlation
                            
            logger.info("Calculated dynamic correlations based on recent signals")
            return dynamic_correlations
            
        except Exception as e:
            logger.error(f"Error calculating dynamic correlations: {str(e)}")
            return self.correlations
    
    def get_correlated_pairs(self, pair: str, min_correlation: float = 0.6) -> List[Tuple[str, float]]:
        """
        Get list of pairs that are highly correlated with the given pair.
        
        Args:
            pair: Currency pair to find correlations for
            min_correlation: Minimum absolute correlation value
            
        Returns:
            List of tuples (pair, correlation)
        """
        try:
            # Use dynamic correlations if available
            if not hasattr(self, '_dynamic_correlations'):
                self._dynamic_correlations = self.calculate_dynamic_correlations()
                
            correlations = self._dynamic_correlations.get(pair, {})
            
            if not correlations:
                correlations = self.correlations.get(pair, {})
                
            if not correlations:
                logger.warning(f"No correlation data available for {pair}")
                return []
                
            # Filter by minimum absolute correlation
            correlated_pairs = [(p, c) for p, c in correlations.items() if abs(c) >= min_correlation]
            
            # Sort by absolute correlation (highest first)
            correlated_pairs.sort(key=lambda x: abs(x[1]), reverse=True)
            
            logger.debug(f"Found {len(correlated_pairs)} correlated pairs for {pair}")
            return correlated_pairs
            
        except Exception as e:
            logger.error(f"Error getting correlated pairs: {str(e)}")
            return []
    
    def analyze_correlated_signals(self, pair: str, signal: str, recent_signals: Dict[str, Dict] = None) -> Dict:
        """
        Analyze signals on correlated pairs to confirm or contradict the given signal.
        
        Args:
            pair: Currency pair
            signal: Signal direction ('BUY' or 'SELL')
            recent_signals: Dictionary with recent signals for other pairs
            
        Returns:
            Dictionary with correlation analysis results
        """
        try:
            if signal not in ['BUY', 'SELL']:
                return {
                    'correlated_confirmation': False,
                    'confirmation_level': 0,
                    'contradicting_pairs': [],
                    'confirming_pairs': [],
                    'correlation_boost': 0
                }
                
            # Get correlated pairs
            correlated_pairs = self.get_correlated_pairs(pair)
            
            if not correlated_pairs:
                return {
                    'correlated_confirmation': False,
                    'confirmation_level': 0,
                    'contradicting_pairs': [],
                    'confirming_pairs': [],
                    'correlation_boost': 0
                }
                
            # No recent signals provided
            if not recent_signals:
                return {
                    'correlated_confirmation': False,
                    'confirmation_level': 0,
                    'contradicting_pairs': [],
                    'confirming_pairs': [],
                    'correlation_boost': 0
                }
                
            # Analyze each correlated pair
            confirming_pairs = []
            contradicting_pairs = []
            
            for corr_pair, correlation in correlated_pairs:
                if corr_pair not in recent_signals:
                    continue
                    
                corr_signal = recent_signals[corr_pair].get('signal', 'NEUTRAL')
                
                if corr_signal == 'NEUTRAL':
                    continue
                    
                # Determine if the correlated signal confirms or contradicts
                if correlation > 0:  # Positive correlation
                    if corr_signal == signal:
                        confirming_pairs.append((corr_pair, correlation))
                    else:
                        contradicting_pairs.append((corr_pair, correlation))
                else:  # Negative correlation
                    if corr_signal != signal:
                        confirming_pairs.append((corr_pair, abs(correlation)))
                    else:
                        contradicting_pairs.append((corr_pair, abs(correlation)))
                        
            # Calculate confirmation level
            total_weight = sum(abs(c) for _, c in correlated_pairs if c != 0)
            confirm_weight = sum(c for _, c in confirming_pairs)
            contradict_weight = sum(c for _, c in contradicting_pairs)
            
            if total_weight > 0:
                confirmation_level = (confirm_weight - contradict_weight) / total_weight * 100
            else:
                confirmation_level = 0
                
            # Determine correlation boost
            if confirmation_level > 50:
                correlation_boost = min(20, confirmation_level / 5)  # Max 20% boost
            elif confirmation_level < -50:
                correlation_boost = max(-20, confirmation_level / 5)  # Max -20% boost
            else:
                correlation_boost = 0
                
            # Format results
            result = {
                'correlated_confirmation': confirmation_level > 0,
                'confirmation_level': confirmation_level,
                'confirming_pairs': [(p, f"{c:.2f}") for p, c in confirming_pairs],
                'contradicting_pairs': [(p, f"{c:.2f}") for p, c in contradicting_pairs],
                'correlation_boost': correlation_boost
            }
            
            logger.info(f"Correlation analysis for {pair} {signal}: " + 
                       f"confirmation {confirmation_level:.1f}%, boost {correlation_boost:.1f}%")
            return result
            
        except Exception as e:
            logger.error(f"Error analyzing correlated signals: {str(e)}")
            return {
                'correlated_confirmation': False,
                'confirmation_level': 0,
                'contradicting_pairs': [],
                'confirming_pairs': [],
                'correlation_boost': 0
            }
    
    def update_correlation_from_history(self) -> None:
        """Update correlation data based on historical signal performance."""
        try:
            # Calculate dynamic correlations with longer lookback
            dynamic_correlations = self.calculate_dynamic_correlations(lookback_days=30)
            
            # Update the stored correlations
            self._dynamic_correlations = dynamic_correlations
            
            logger.info("Updated correlation data from signal history")
            
        except Exception as e:
            logger.error(f"Error updating correlation from history: {str(e)}")
    
    def get_divergence(self, pair: str, data: pd.DataFrame, recent_signals: Dict[str, Dict] = None) -> Dict:
        """
        Detect divergence between the pair and its correlated pairs.
        
        Args:
            pair: Currency pair
            data: DataFrame with price data
            recent_signals: Dictionary with recent signals for other pairs
            
        Returns:
            Dictionary with divergence analysis
        """
        try:
            # Get correlated pairs
            correlated_pairs = self.get_correlated_pairs(pair, min_correlation=0.7)
            
            if not correlated_pairs or not recent_signals:
                return {
                    'divergence_detected': False,
                    'divergence_pairs': [],
                    'divergence_potential': 0
                }
                
            # Calculate price changes
            if 'close' in data.columns:
                price_change_pct = (data['close'].iloc[-1] / data['close'].iloc[-10] - 1) * 100
            else:
                return {
                    'divergence_detected': False,
                    'divergence_pairs': [],
                    'divergence_potential': 0
                }
                
            # Check for divergence with correlated pairs
            divergence_pairs = []
            
            for corr_pair, correlation in correlated_pairs:
                if corr_pair not in recent_signals or 'price_change_pct' not in recent_signals[corr_pair]:
                    continue
                    
                corr_price_change = recent_signals[corr_pair]['price_change_pct']
                
                # Calculate expected correlation
                expected_change = price_change_pct * correlation
                
                # Check for significant divergence
                divergence = corr_price_change - expected_change
                
                if abs(divergence) > 0.5:  # Threshold for significance
                    divergence_pairs.append({
                        'pair': corr_pair,
                        'correlation': correlation,
                        'divergence': divergence,
                        'expected_change': expected_change,
                        'actual_change': corr_price_change
                    })
                    
            # Calculate divergence potential
            if divergence_pairs:
                # Higher negative divergence suggests potential for mean reversion
                avg_divergence = sum(d['divergence'] for d in divergence_pairs) / len(divergence_pairs)
                divergence_potential = -avg_divergence  # Negative divergence means potential upward reversion
            else:
                divergence_potential = 0
                
            result = {
                'divergence_detected': len(divergence_pairs) > 0,
                'divergence_pairs': divergence_pairs,
                'divergence_potential': divergence_potential
            }
            
            if divergence_pairs:
                logger.info(f"Detected divergence for {pair} with {len(divergence_pairs)} pairs")
            return result
            
        except Exception as e:
            logger.error(f"Error detecting divergence: {str(e)}")
            return {
                'divergence_detected': False,
                'divergence_pairs': [],
                'divergence_potential': 0
            }
    
    def analyze_timeframe_correlations(self,
                                     data: Dict[str, pd.DataFrame],
                                     timeframes: List[str]) -> Dict[str, TimeframeCorrelation]:
        """
        Analyze correlations between different timeframes.
        
        Args:
            data: Dictionary of DataFrames for different timeframes
            timeframes: List of timeframes to analyze
            
        Returns:
            Dictionary of timeframe correlations
        """
        try:
            correlations = {}
            
            for i, tf1 in enumerate(timeframes):
                for tf2 in timeframes[i+1:]:
                    # Resample the shorter timeframe to match the longer one
                    tf1_data = self._resample_data(data[tf1], tf2)
                    
                    # Calculate correlation
                    correlation = self._calculate_correlation(tf1_data, data[tf2])
                    
                    # Store results
                    key = f"{tf1}_{tf2}"
                    correlations[key] = correlation
                    
            self.timeframe_correlations = correlations
            return correlations
            
        except Exception as e:
            logger.error(f"Error analyzing timeframe correlations: {str(e)}")
            return {}
    
    def get_aligned_timeframes(self,
                             min_correlation: Optional[float] = None) -> List[List[str]]:
        """
        Get groups of aligned timeframes based on correlation threshold.
        
        Args:
            min_correlation: Minimum correlation to consider timeframes aligned
            
        Returns:
            List of timeframe groups that are aligned
        """
        try:
            threshold = min_correlation or self.min_correlation
            aligned_groups = []
            processed = set()
            
            for key, corr in self.timeframe_correlations.items():
                if corr.correlation >= threshold:
                    tf1, tf2 = key.split('_')
                    
                    # Find or create group
                    added = False
                    for group in aligned_groups:
                        if tf1 in group or tf2 in group:
                            group.extend([tf for tf in [tf1, tf2] if tf not in group])
                            added = True
                            break
                            
                    if not added:
                        aligned_groups.append([tf1, tf2])
                        
                    processed.add(tf1)
                    processed.add(tf2)
            
            return aligned_groups
            
        except Exception as e:
            logger.error(f"Error getting aligned timeframes: {str(e)}")
            return []
    
    def calculate_signal_agreement(self,
                                 signals: Dict[str, str],
                                 timeframes: List[str]) -> float:
        """
        Calculate agreement score between signals across timeframes.
        
        Args:
            signals: Dictionary of signals for each timeframe
            timeframes: List of timeframes to analyze
            
        Returns:
            Agreement score between 0 and 1
        """
        try:
            if not signals or not timeframes:
                return 0.0
                
            # Count signal types
            signal_counts = {
                'BUY': 0,
                'SELL': 0,
                'NEUTRAL': 0
            }
            
            for tf in timeframes:
                if tf in signals:
                    signal_counts[signals[tf]] += 1
            
            # Calculate agreement score
            max_count = max(signal_counts.values())
            total = sum(signal_counts.values())
            
            return max_count / total if total > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating signal agreement: {str(e)}")
            return 0.0
    
    @staticmethod
    def _resample_data(data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Resample data to target timeframe."""
        try:
            # Extract numeric value and unit from timeframe
            value = int(''.join(filter(str.isdigit, target_timeframe)))
            unit = ''.join(filter(str.isalpha, target_timeframe))
            
            # Convert to pandas frequency string
            freq_map = {'s': 'S', 'm': 'T', 'h': 'H', 'd': 'D'}
            freq = f"{value}{freq_map.get(unit, 'T')}"
            
            # Resample OHLCV data
            resampled = data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            })
            
            return resampled
            
        except Exception as e:
            logger.error(f"Error resampling data: {str(e)}")
            return data
    
    def _calculate_correlation(self,
                             data1: pd.DataFrame,
                             data2: pd.DataFrame) -> TimeframeCorrelation:
        """Calculate correlation between two timeframes."""
        try:
            # Align data
            df1 = data1['close']
            df2 = data2['close']
            df1, df2 = df1.align(df2, join='inner')
            
            if len(df1) < 2:
                return TimeframeCorrelation(
                    timeframe1='',
                    timeframe2='',
                    correlation=0,
                    significance=0,
                    type=CorrelationType.NEUTRAL,
                    confidence=0
                )
            
            # Calculate correlation and p-value
            correlation, p_value = stats.pearsonr(df1, df2)
            
            # Determine correlation type
            if abs(correlation) < 0.3:
                corr_type = CorrelationType.NEUTRAL
            elif correlation > 0:
                corr_type = CorrelationType.POSITIVE
            else:
                corr_type = CorrelationType.NEGATIVE
            
            # Calculate confidence based on p-value and correlation strength
            confidence = (1 - p_value) * abs(correlation)
            
            return TimeframeCorrelation(
                timeframe1=str(df1.index.freq),
                timeframe2=str(df2.index.freq),
                correlation=correlation,
                significance=1 - p_value,
                type=corr_type,
                confidence=confidence
            )
            
        except Exception as e:
            logger.error(f"Error calculating correlation: {str(e)}")
            return TimeframeCorrelation(
                timeframe1='',
                timeframe2='',
                correlation=0,
                significance=0,
                type=CorrelationType.NEUTRAL,
                confidence=0
            )