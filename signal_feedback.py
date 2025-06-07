"""
Feedback handler for signal continuous learning.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Optional, List
from continuous_learning import ContinuousLearningSystem
import asyncio
import threading
import time

logger = logging.getLogger(__name__)

class SignalFeedbackManager:
    """Manages signal feedback and integrates with continuous learning."""
    
    def __init__(self, continuous_learner: Optional[ContinuousLearningSystem] = None):
        self.continuous_learner = continuous_learner
        self.active_signals: Dict[str, Dict] = {}
        self.feedback_stats = {
            'total_signals': 0,
            'total_feedback': 0,
            'profitable_trades': 0,
            'loss_trades': 0
        }
        
        # Start cleanup thread
        self._start_cleanup_thread()
    
    def _start_cleanup_thread(self):
        """Start background thread for periodic cleanup of old signals."""
        def cleanup_loop():
            while True:
                try:
                    self.clean_old_signals()
                    # Sleep for 1 hour before next cleanup
                    time.sleep(3600)
                except Exception as e:
                    logger.error(f"Error in cleanup thread: {str(e)}")
                    time.sleep(3600)  # Wait an hour after error
        
        cleanup_thread = threading.Thread(target=cleanup_loop, daemon=True)
        cleanup_thread.start()
        logger.info("Started signal cleanup thread")

    def store_signal(self, signal_id: str, signal_data: Dict) -> None:
        """Store a signal for later feedback."""
        self.active_signals[signal_id] = {
            'data': signal_data,
            'timestamp': datetime.now().isoformat(),
            'status': 'pending'
        }
        self.feedback_stats['total_signals'] += 1

    def record_feedback(self, signal_id: str, feedback: Dict) -> None:
        """Record user feedback for a signal."""
        if signal_id not in self.active_signals:
            logger.warning(f"Signal {signal_id} not found in active signals")
            return
            
        signal = self.active_signals[signal_id]
        signal['feedback'] = feedback
        signal['status'] = feedback['result']  # 'win' or 'loss'
        
        # Update feedback statistics
        self.feedback_stats['total_feedback'] += 1
        if feedback['result'] == 'win':
            self.feedback_stats['profitable_trades'] += 1
        elif feedback['result'] == 'loss':
            self.feedback_stats['loss_trades'] += 1

        # Update continuous learning if available
        if self.continuous_learner:
            try:
                full_feedback = {
                    'signal_id': signal_id,
                    'timestamp': datetime.now().isoformat(),
                    'result': feedback['result'],
                    'profit_loss': 1.0 if feedback['result'] == 'win' else -1.0,
                    'signal_data': signal['data'],
                    'feedback_type': 'user',
                    'confidence_validation': True if feedback['result'] == 'win' else False
                }
                
                self.continuous_learner.record_user_feedback(signal_id, full_feedback)
                
                logger.info(f"Recorded feedback for signal {signal_id}: {feedback['result']}")
            except Exception as e:
                logger.error(f"Error recording feedback: {str(e)}")

            # Clean up processed signal
            del self.active_signals[signal_id]

    def get_signal(self, signal_id: str) -> Optional[Dict]:
        """Get stored signal data."""
        return self.active_signals.get(signal_id)

    def get_feedback_stats(self) -> Dict:
        """Get current feedback statistics."""
        stats = self.feedback_stats.copy()
        stats['success_rate'] = (
            (stats['profitable_trades'] / stats['total_feedback'] * 100)
            if stats['total_feedback'] > 0 else 0
        )
        return stats

    def clean_old_signals(self, max_age_hours: int = 24) -> None:
        """Clean up signals older than specified hours."""
        now = datetime.now()
        old_signals = []
        
        for signal_id, signal in self.active_signals.items():
            signal_time = datetime.fromisoformat(signal['timestamp'])
            if (now - signal_time).total_seconds() > max_age_hours * 3600:
                old_signals.append(signal_id)
        
        for signal_id in old_signals:
            del self.active_signals[signal_id]
