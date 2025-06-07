#!/usr/bin/env python3
"""
Advanced Signal Analytics Dashboard
Provides comprehensive performance tracking and analytics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import json
import logging

logger = logging.getLogger(__name__)

class SignalAnalyticsDashboard:
    """Advanced analytics dashboard for signal performance tracking."""
    
    def __init__(self):
        self.signal_history = []
        self.performance_metrics = {}
        self.pair_performance = {}
        self.timeframe_performance = {}
        
    def record_signal(self, signal_data: Dict):
        """Record a generated signal for analytics."""
        try:
            signal_record = {
                'timestamp': datetime.now().isoformat(),
                'pair': signal_data.get('pair'),
                'timeframe': signal_data.get('timeframe'),
                'direction': signal_data.get('direction'),
                'confidence': signal_data.get('confidence'),
                'entry_price': signal_data.get('entry_price'),
                'stop_loss': signal_data.get('stop_loss'),
                'take_profit': signal_data.get('take_profit'),
                'pattern': signal_data.get('pattern'),
                'ml_prediction': signal_data.get('ml_prediction'),
                'regime': signal_data.get('regime', 'UNKNOWN')
            }
            
            self.signal_history.append(signal_record)
            
            # Keep only last 1000 signals
            if len(self.signal_history) > 1000:
                self.signal_history = self.signal_history[-1000:]
                
            logger.info(f"Recorded signal: {signal_data.get('pair')} {signal_data.get('direction')}")
            
        except Exception as e:
            logger.error(f"Error recording signal: {e}")
    
    def get_performance_summary(self) -> Dict:
        """Get comprehensive performance summary."""
        try:
            if not self.signal_history:
                return {'error': 'No signal history available'}
            
            recent_signals = [
                s for s in self.signal_history
                if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days <= 7
            ]
            
            if not recent_signals:
                return {'error': 'No recent signals available'}
            
            # Calculate metrics
            total_signals = len(recent_signals)
            avg_confidence = np.mean([s['confidence'] for s in recent_signals])
            
            # Direction distribution
            buy_signals = len([s for s in recent_signals if s['direction'] == 'BUY'])
            sell_signals = len([s for s in recent_signals if s['direction'] == 'SELL'])
            
            # Pair distribution
            pair_counts = {}
            for signal in recent_signals:
                pair = signal['pair']
                pair_counts[pair] = pair_counts.get(pair, 0) + 1
            
            # Timeframe distribution
            timeframe_counts = {}
            for signal in recent_signals:
                tf = signal['timeframe']
                timeframe_counts[tf] = timeframe_counts.get(tf, 0) + 1
            
            # Confidence distribution
            high_confidence = len([s for s in recent_signals if s['confidence'] >= 90])
            medium_confidence = len([s for s in recent_signals if 80 <= s['confidence'] < 90])
            low_confidence = len([s for s in recent_signals if s['confidence'] < 80])
            
            return {
                'period': '7 days',
                'total_signals': total_signals,
                'average_confidence': round(avg_confidence, 1),
                'direction_distribution': {
                    'BUY': buy_signals,
                    'SELL': sell_signals,
                    'BUY_percentage': round(buy_signals / total_signals * 100, 1),
                    'SELL_percentage': round(sell_signals / total_signals * 100, 1)
                },
                'confidence_distribution': {
                    'high_confidence': high_confidence,
                    'medium_confidence': medium_confidence,
                    'low_confidence': low_confidence,
                    'high_percentage': round(high_confidence / total_signals * 100, 1)
                },
                'top_pairs': dict(sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'top_timeframes': dict(sorted(timeframe_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
                'last_updated': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating performance summary: {e}")
            return {'error': f'Analytics error: {str(e)}'}
    
    def get_pair_analytics(self, pair: str) -> Dict:
        """Get detailed analytics for specific pair."""
        try:
            pair_signals = [s for s in self.signal_history if s['pair'] == pair]
            
            if not pair_signals:
                return {'error': f'No signals found for {pair}'}
            
            recent_signals = [
                s for s in pair_signals
                if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days <= 30
            ]
            
            if not recent_signals:
                return {'error': f'No recent signals for {pair}'}
            
            avg_confidence = np.mean([s['confidence'] for s in recent_signals])
            
            # Best timeframes for this pair
            tf_performance = {}
            for signal in recent_signals:
                tf = signal['timeframe']
                if tf not in tf_performance:
                    tf_performance[tf] = {'count': 0, 'total_confidence': 0}
                tf_performance[tf]['count'] += 1
                tf_performance[tf]['total_confidence'] += signal['confidence']
            
            # Calculate average confidence per timeframe
            for tf in tf_performance:
                tf_performance[tf]['avg_confidence'] = tf_performance[tf]['total_confidence'] / tf_performance[tf]['count']
            
            # Sort by average confidence
            best_timeframes = sorted(tf_performance.items(), key=lambda x: x[1]['avg_confidence'], reverse=True)
            
            return {
                'pair': pair,
                'total_signals': len(recent_signals),
                'average_confidence': round(avg_confidence, 1),
                'best_timeframes': [(tf, round(data['avg_confidence'], 1)) for tf, data in best_timeframes[:3]],
                'signal_frequency': len(recent_signals) / 30,  # signals per day
                'last_signal': recent_signals[-1]['timestamp'] if recent_signals else None
            }
            
        except Exception as e:
            logger.error(f"Error generating pair analytics: {e}")
            return {'error': f'Pair analytics error: {str(e)}'}
    
    def get_recommendations(self) -> Dict:
        """Get AI-powered recommendations based on analytics."""
        try:
            if len(self.signal_history) < 50:
                return {
                    'recommendation': 'Collect more signal data for meaningful recommendations',
                    'min_signals_needed': 50,
                    'current_signals': len(self.signal_history)
                }
            
            recent_signals = [
                s for s in self.signal_history
                if (datetime.now() - datetime.fromisoformat(s['timestamp'])).days <= 14
            ]
            
            # Analyze performance patterns
            pair_performance = {}
            timeframe_performance = {}
            
            for signal in recent_signals:
                pair = signal['pair']
                tf = signal['timeframe']
                confidence = signal['confidence']
                
                if pair not in pair_performance:
                    pair_performance[pair] = []
                pair_performance[pair].append(confidence)
                
                if tf not in timeframe_performance:
                    timeframe_performance[tf] = []
                timeframe_performance[tf].append(confidence)
            
            # Find best performing combinations
            best_pair = max(pair_performance.items(), key=lambda x: np.mean(x[1]))
            best_timeframe = max(timeframe_performance.items(), key=lambda x: np.mean(x[1]))
            
            # Generate recommendations
            recommendations = []
            
            if best_pair[1] and np.mean(best_pair[1]) > 88:
                recommendations.append(f"🎯 **Best Pair**: {best_pair[0]} (avg confidence: {np.mean(best_pair[1]):.1f}%)")
            
            if best_timeframe[1] and np.mean(best_timeframe[1]) > 88:
                recommendations.append(f"⏰ **Best Timeframe**: {best_timeframe[0]} (avg confidence: {np.mean(best_timeframe[1]):.1f}%)")
            
            # Market condition recommendations
            high_conf_signals = [s for s in recent_signals if s['confidence'] >= 90]
            if len(high_conf_signals) / len(recent_signals) > 0.6:
                recommendations.append("📈 **Market Condition**: Excellent signal quality detected!")
            
            return {
                'recommendations': recommendations,
                'best_pair': best_pair[0],
                'best_timeframe': best_timeframe[0],
                'analysis_period': '14 days',
                'signals_analyzed': len(recent_signals)
            }
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return {'error': f'Recommendations error: {str(e)}'}