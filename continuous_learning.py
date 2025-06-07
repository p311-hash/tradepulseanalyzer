"""
Continuous learning module for binary options trading bot.

This module handles:
1. Tracking signal performance
2. Collecting user feedback
3. Storing historical predictions and outcomes
4. Periodically retraining the ML model
5. Optimizing signal generation parameters
"""

import os
import json
import time
import pickle
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Union, Optional, Any
from ml_model import BinaryOptionsModel, MLPredictor
from threading import Thread, Lock

# Configure logging
logger = logging.getLogger(__name__)

class ContinuousLearningSystem:
    """Continuous learning system for improving signal predictions over time."""
    
    def __init__(self, 
                 model_path: str = None,
                 history_path: str = None,
                 feedback_path: str = None,
                 performance_path: str = None,
                 learning_interval: int = 24 * 60 * 60):  # Default: retrain once a day
        """
        Initialize the continuous learning system.
        
        Args:
            model_path: Path to the ML model file
            history_path: Path to store signal history
            feedback_path: Path to store user feedback
            performance_path: Path to store model performance metrics
            learning_interval: Time between model retraining (in seconds)
        """
        # Initialize with absolute paths
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = model_path or os.path.join(base_dir, 'models', 'latest_model')
        self.history_path = history_path or os.path.join(base_dir, 'data', 'signal_history.json')
        self.feedback_path = feedback_path or os.path.join(base_dir, 'data', 'feedback.json')
        self.performance_path = performance_path or os.path.join(base_dir, 'data', 'model_performance.json')
        self.learning_interval = learning_interval
        
        # Initialize data storage
        self.signal_history = []
        self.user_feedback = []
        self.performance_metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1_score': [],
            'win_rate': [],
            'timestamps': []
        }
        
        # Initialize locks for thread safety
        self.history_lock = Lock()
        self.feedback_lock = Lock()
        
        # Create data directory if it doesn't exist
        for path in [self.history_path, self.feedback_path, self.performance_path]:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            if not os.path.exists(path):
                with open(path, 'w') as f:
                    json.dump([], f)
        
        # Load existing data
        self.load_data()
        
        # Flag to track if the model has been updated
        self.model_updated = False
    
    def load_data(self) -> None:
        """Load historical data from files."""
        try:
            # Load signal history
            if os.path.exists(self.history_path):
                try:
                    with open(self.history_path, 'r') as f:
                        self.signal_history = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted signal history file: {self.history_path}")
                    self.signal_history = []
            
            # Load user feedback
            if os.path.exists(self.feedback_path):
                try:
                    with open(self.feedback_path, 'r') as f:
                        self.user_feedback = json.load(f)
                except json.JSONDecodeError:
                    logger.error(f"Corrupted feedback file: {self.feedback_path}")
                    self.user_feedback = []
            
            # Load performance metrics
            if os.path.exists(self.performance_path):
                try:
                    with open(self.performance_path, 'r') as f:
                        loaded_metrics = json.load(f)
                        # Handle both list and dict formats for backwards compatibility
                        if isinstance(loaded_metrics, list):
                            # Convert old list format to new dict format
                            self.performance_metrics = {
                                'accuracy': [],
                                'precision': [],
                                'recall': [],
                                'f1_score': [],
                                'win_rate': [],
                                'timestamps': []
                            }
                        else:
                            # Initialize any missing metrics
                            self.performance_metrics = {
                                'accuracy': loaded_metrics.get('accuracy', []),
                                'precision': loaded_metrics.get('precision', []),
                                'recall': loaded_metrics.get('recall', []),
                                'f1_score': loaded_metrics.get('f1_score', []),
                                'win_rate': loaded_metrics.get('win_rate', []),
                                'timestamps': loaded_metrics.get('timestamps', [])
                            }
                except json.JSONDecodeError:
                    logger.error(f"Corrupted performance metrics file: {self.performance_path}")
                    self.performance_metrics = {
                        'accuracy': [],
                        'precision': [],
                        'recall': [],
                        'f1_score': [],
                        'win_rate': [],
                        'timestamps': []
                    }
            
            logger.info("Successfully loaded continuous learning data")
        except Exception as e:
            logger.error(f"Error loading continuous learning data: {str(e)}", exc_info=True)
            # Initialize empty data structures on error
            self.signal_history = []
            self.user_feedback = []
            self.performance_metrics = {
                'accuracy': [],
                'precision': [],
                'recall': [],
                'f1_score': [],
                'win_rate': [],
                'timestamps': []
            }
    
    def save_data(self):
        """Save all learning data to disk."""
        try:
            with self.history_lock:
                with open(self.history_path, 'w') as f:
                    json.dump(self.signal_history, f, indent=2)
                    
            with self.feedback_lock:
                with open(self.feedback_path, 'w') as f:
                    json.dump(self.user_feedback, f, indent=2)
                
            with open(self.performance_path, 'w') as f:
                json.dump(self.performance_metrics, f, indent=2)
                
            logger.info("Successfully saved continuous learning data")
        except Exception as e:
            logger.error(f"Error saving continuous learning data: {str(e)}", exc_info=True)
    
    def record_signal(self, signal: Dict) -> str:
        """
        Record a generated trading signal for future evaluation.
        
        Args:
            signal: Trading signal dictionary with prediction details
            
        Returns:
            signal_id: Unique ID for this signal
        """
        try:
            with self.history_lock:
                # Generate unique ID for this signal
                signal_id = f"{signal['pair']}_{signal['timeframe']}_{int(time.time())}"
                
                # Add signal to history with status "pending"
                signal_record = {
                    'id': signal_id,
                    'timestamp': datetime.now().isoformat(),
                    'pair': signal['pair'],
                    'timeframe': signal['timeframe'],
                    'price': signal.get('price', 0),
                    'prediction': signal.get('signal', 'NEUTRAL'),
                    'confidence': signal.get('confidence', 0),
                    'indicators': signal.get('indicators', {}),
                    'patterns': signal.get('patterns', {}),
                    'ml_prediction': signal.get('ml_prediction', {}),
                    'status': 'pending',  # pending, correct, incorrect
                    'outcome_price': None,
                    'expiration': None,  # When the option would expire
                    'profit_loss': None  # Profit/loss if traded
                }
                
                # Set expiration time based on timeframe
                timeframe = signal['timeframe']
                expiration_minutes = 1  # Default 1 minute
                if timeframe == '15s':
                    expiration_minutes = 0.25
                elif timeframe == '30s':
                    expiration_minutes = 0.5
                elif timeframe == '1m':
                    expiration_minutes = 1
                elif timeframe == '5m':
                    expiration_minutes = 5
                elif timeframe == '15m':
                    expiration_minutes = 15
                
                signal_record['expiration'] = (datetime.now() + timedelta(minutes=expiration_minutes)).isoformat()
                
                self.signal_history.append(signal_record)
                
                # Limit history size to prevent excessive growth
                if len(self.signal_history) > 10000:
                    self.signal_history = self.signal_history[-10000:]
                
                # Save after each new signal
                with open(self.history_path, 'w') as f:
                    json.dump(self.signal_history, f, indent=2)
                
                logger.info(f"Recorded signal {signal_id} for {signal['pair']} {signal['timeframe']}")
                return signal_id
        except Exception as e:
            logger.error(f"Error recording signal: {str(e)}", exc_info=True)
            return ""
    
    def update_signal_outcome(self, signal_id: str, outcome_price: float) -> None:
        """
        Update a signal with its actual outcome once the expiration time is reached.
        
        Args:
            signal_id: Unique ID of the signal to update
            outcome_price: Actual price at expiration time
        """
        try:
            with self.history_lock:
                # Find the signal in history
                for i, signal in enumerate(self.signal_history):
                    if signal['id'] == signal_id and signal['status'] == 'pending':
                        # Calculate if prediction was correct
                        initial_price = signal['price']
                        prediction = signal['prediction']
                        
                        if prediction == 'BUY':
                            is_correct = outcome_price > initial_price
                        elif prediction == 'SELL':
                            is_correct = outcome_price < initial_price
                        else:  # NEUTRAL
                            is_correct = abs(outcome_price - initial_price) / initial_price < 0.0001  # Very small change
                        
                        # Update signal record
                        self.signal_history[i]['outcome_price'] = outcome_price
                        self.signal_history[i]['status'] = 'correct' if is_correct else 'incorrect'
                        
                        # Calculate profit/loss (assuming fixed 80% return on correct predictions)
                        profit_loss = 0.8 if is_correct else -1.0
                        self.signal_history[i]['profit_loss'] = profit_loss
                        
                        logger.info(f"Updated signal {signal_id} outcome: {'correct' if is_correct else 'incorrect'}")
                        
                        # Save updates
                        with open(self.history_path, 'w') as f:
                            json.dump(self.signal_history, f, indent=2)
                        
                        return
                
                logger.warning(f"Signal {signal_id} not found or already updated")
        except Exception as e:
            logger.error(f"Error updating signal outcome: {str(e)}", exc_info=True)

    def _calculate_performance_metrics(self) -> Dict:
        """Internal method to calculate current performance metrics."""
        try:
            # Filter out pending signals
            completed_signals = [s for s in self.signal_history if s['status'] != 'pending']
            
            if not completed_signals:
                return {
                    'accuracy': 0,
                    'precision': 0,
                    'recall': 0,
                    'f1_score': 0,
                    'win_rate': 0,
                    'signal_count': 0
                }
                
            # Calculate metrics
            correct_count = sum(1 for s in completed_signals if s['status'] == 'correct')
            total_count = len(completed_signals)
            accuracy = correct_count / total_count if total_count > 0 else 0
            
            # Calculate precision, recall for BUY and SELL signals
            buy_signals = [s for s in completed_signals if s['prediction'] == 'BUY']
            sell_signals = [s for s in completed_signals if s['prediction'] == 'SELL']
            
            # Precision: correct predictions / total predictions for that class
            buy_correct = sum(1 for s in buy_signals if s['status'] == 'correct')
            sell_correct = sum(1 for s in sell_signals if s['status'] == 'correct')
            
            buy_precision = buy_correct / len(buy_signals) if buy_signals else 0
            sell_precision = sell_correct / len(sell_signals) if sell_signals else 0
            
            # Average precision
            precision = (buy_precision + sell_precision) / 2 if buy_signals and sell_signals else (buy_precision or sell_precision)
            
            # Simplified recall and F1 (not truly applicable but included for completeness)
            recall = precision  # Simplified for this context
            f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_score,
                'win_rate': accuracy,
                'signal_count': total_count
            }
            
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}", exc_info=True)
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'win_rate': 0,
                'signal_count': 0
            }
    
    def record_user_feedback(self, signal_id: str, feedback: Dict) -> None:
        """
        Record user feedback on a signal and update the learning system.
        
        Args:
            signal_id: Unique ID of the signal
            feedback: Dictionary with feedback data including:
                - result: 'win' or 'loss'
                - profit_loss: 1.0 for win, -1.0 for loss 
                - signal_data: Original signal data
                - feedback_type: Type of feedback ('user' or 'system')
                - confidence_validation: Whether signal confidence was validated
        """
        try:
            with self.feedback_lock:
                # Create feedback record with enhanced metadata
                feedback_record = {
                    'signal_id': signal_id,
                    'timestamp': datetime.now().isoformat(),
                    'result': feedback.get('result'),
                    'profit_loss': feedback.get('profit_loss', 0.0),
                    'signal_data': feedback.get('signal_data', {}),
                    'feedback_type': feedback.get('feedback_type', 'user'),
                    'confidence_validated': feedback.get('confidence_validation', False)
                }
                
                self.user_feedback.append(feedback_record)
                
                # Update signal history with outcome
                for signal in self.signal_history:
                    if signal.get('id') == signal_id:
                        signal['outcome'] = feedback['result']
                        signal['profit_loss'] = feedback['profit_loss']
                        break
                
                # Trigger model update on significant feedback volume
                if len(self.user_feedback) % 10 == 0:  # Every 10 feedback records
                    self._trigger_model_update()
                
                # Save updated data
                self.save_data()
                
                logger.info(f"Recorded enhanced feedback for signal {signal_id}")
                
        except Exception as e:
            logger.error(f"Error recording user feedback: {str(e)}", exc_info=True)

    def _trigger_model_update(self) -> None:
        """Trigger model update based on recent feedback."""
        try:
            # Calculate recent performance metrics
            recent_feedback = self.user_feedback[-10:]
            wins = sum(1 for f in recent_feedback if f['result'] == 'win')
            accuracy = wins / len(recent_feedback)
            
            # If accuracy below threshold, force retrain
            if accuracy < 0.7 and len(self.signal_history) >= 50:
                logger.info("Low recent accuracy detected, triggering model retraining")
                success = self.retrain_model()
                if success:
                    logger.info("Model successfully retrained after accuracy drop")
                
        except Exception as e:
            logger.error(f"Error in model update trigger: {str(e)}", exc_info=True)

    def calculate_performance_metrics(self) -> Dict:
        """
        Calculate performance metrics based on signal history.
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            with self.history_lock:
                # Filter out pending signals
                completed_signals = [s for s in self.signal_history if s['status'] != 'pending']
                
                if not completed_signals:
                    return {
                        'accuracy': 0,
                        'precision': 0,
                        'recall': 0,
                        'f1_score': 0,
                        'win_rate': 0,
                        'signal_count': 0
                    }
                
                # Calculate metrics
                correct_count = sum(1 for s in completed_signals if s['status'] == 'correct')
                total_count = len(completed_signals)
                accuracy = correct_count / total_count if total_count > 0 else 0
                
                # Calculate precision, recall for BUY and SELL signals
                buy_signals = [s for s in completed_signals if s['prediction'] == 'BUY']
                sell_signals = [s for s in completed_signals if s['prediction'] == 'SELL']
                
                # Precision: correct predictions / total predictions for that class
                buy_correct = sum(1 for s in buy_signals if s['status'] == 'correct')
                sell_correct = sum(1 for s in sell_signals if s['status'] == 'correct')
                
                buy_precision = buy_correct / len(buy_signals) if buy_signals else 0
                sell_precision = sell_correct / len(sell_signals) if sell_signals else 0
                
                # Average precision
                precision = (buy_precision + sell_precision) / 2 if buy_signals and sell_signals else (buy_precision or sell_precision)
                
                # Simplified recall and F1 (not truly applicable in this context but included for completeness)
                recall = precision  # Simplified for binary trading
                f1_score = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
                
                # Record these metrics
                timestamp = datetime.now().isoformat()
                
                self.performance_metrics['accuracy'].append(accuracy)
                self.performance_metrics['precision'].append(precision)
                self.performance_metrics['recall'].append(recall)
                self.performance_metrics['f1_score'].append(f1_score)
                self.performance_metrics['win_rate'].append(accuracy)  # Same as accuracy in this context
                self.performance_metrics['timestamps'].append(timestamp)
                
                # Keep only the last 100 metric points
                for key in self.performance_metrics:
                    if key != 'timestamps' and len(self.performance_metrics[key]) > 100:
                        self.performance_metrics[key] = self.performance_metrics[key][-100:]
                
                if len(self.performance_metrics['timestamps']) > 100:
                    self.performance_metrics['timestamps'] = self.performance_metrics['timestamps'][-100:]
                
                # Save updated metrics
                with open(self.performance_path, 'w') as f:
                    json.dump(self.performance_metrics, f, indent=2)
                
                return {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1_score,
                    'win_rate': accuracy,
                    'signal_count': total_count
                }
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}", exc_info=True)
            return {
                'accuracy': 0,
                'precision': 0,
                'recall': 0,
                'f1_score': 0,
                'win_rate': 0,
                'signal_count': 0
            }
    
    def prepare_training_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepare training data from signal history.
        
        Returns:
            Tuple of (features_tensor, labels_tensor)
        """
        try:
            with self.history_lock:
                # Filter completed signals
                completed_signals = [s for s in self.signal_history if s['status'] != 'pending']
                
                if not completed_signals:
                    logger.warning("No completed signals available for training")
                    return None, None
                
                # Extract features and labels
                features = []
                labels = []
                
                for signal in completed_signals:
                    # Extract indicators as features
                    feature_vector = []
                    indicators = signal.get('indicators', {})
                    
                    # Add basic technical indicators
                    feature_vector.append(indicators.get('rsi', 50))
                    feature_vector.append(indicators.get('macd', 0))
                    feature_vector.append(indicators.get('macd_signal', 0))
                    feature_vector.append(indicators.get('adx', 20))
                    feature_vector.append(indicators.get('vi_plus', 1))
                    feature_vector.append(indicators.get('vi_minus', 1))
                    
                    # Add Bollinger Band positioning
                    if 'close' in indicators and 'bb_upper' in indicators and 'bb_lower' in indicators:
                        close = indicators['close']
                        upper = indicators['bb_upper']
                        lower = indicators['bb_lower']
                        middle = (upper + lower) / 2
                        
                        # Normalized position within bands
                        bb_position = (close - lower) / (upper - lower) if upper != lower else 0.5
                        feature_vector.append(bb_position)
                        
                        # Distance from middle band
                        middle_distance = (close - middle) / middle if middle != 0 else 0
                        feature_vector.append(middle_distance)
                    else:
                        feature_vector.extend([0.5, 0])  # Default values
                    
                    # Add pattern information as binary features
                    patterns = signal.get('patterns', {})
                    feature_vector.append(1 if patterns.get('hammer', False) else 0)
                    feature_vector.append(1 if patterns.get('inverted_hammer', False) else 0)
                    feature_vector.append(1 if patterns.get('tweezer_top', False) else 0)
                    feature_vector.append(1 if patterns.get('tweezer_bottom', False) else 0)
                    
                    # Add price action info
                    price_action = signal.get('price_action', {})
                    feature_vector.append(price_action.get('trend_strength', 0))
                    feature_vector.append(price_action.get('volatility', 0))
                    
                    # Determine label (1 for correct prediction, 0 for incorrect)
                    label = 1 if signal['status'] == 'correct' else 0
                    
                    features.append(feature_vector)
                    labels.append(label)
                
                # Convert to tensors
                if features and labels:
                    # Convert lists to numpy arrays first
                    features_np = np.array(features, dtype=np.float32)
                    labels_np = np.array(labels, dtype=np.int64)
                    
                    # Handle missing or NaN values
                    features_np = np.nan_to_num(features_np, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Convert to PyTorch tensors
                    features_tensor = torch.tensor(features_np, dtype=torch.float32)
                    labels_tensor = torch.tensor(labels_np, dtype=torch.long)
                    
                    logger.info(f"Prepared training data with {len(features)} samples")
                    return features_tensor, labels_tensor
                else:
                    logger.warning("No valid training data available")
                    return None, None
                
        except Exception as e:
            logger.error(f"Error preparing training data: {str(e)}", exc_info=True)
            return None, None
    
    def retrain_model(self) -> bool:
        """
        Retrain the ML model using collected signal history.
        
        Returns:
            True if retraining was successful, False otherwise
        """
        try:
            # Prepare training data
            X, y = self.prepare_training_data()
            
            if X is None or y is None or len(X) < 50:
                logger.warning(f"Insufficient data for retraining (samples: {len(X) if X is not None else 0})")
                return False
            
            # Load the current model
            predictor = MLPredictor(self.model_path)
            
            if predictor.model is None:
                logger.error("Failed to load ML model for retraining")
                return False
            
            # Set model to training mode
            predictor.model.train()
            
            # Define loss function and optimizer
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(predictor.model.parameters(), lr=0.001)
            
            # Training loop
            num_epochs = 50
            batch_size = 16
            n_samples = len(X)
            
            for epoch in range(num_epochs):
                # Shuffle data
                indices = torch.randperm(n_samples)
                X_shuffled = X[indices]
                y_shuffled = y[indices]
                
                # Mini-batch training
                for i in range(0, n_samples, batch_size):
                    if i + batch_size <= n_samples:
                        X_batch = X_shuffled[i:i+batch_size]
                        y_batch = y_shuffled[i:i+batch_size]
                        
                        # Forward pass
                        outputs = predictor.model(X_batch)
                        loss = criterion(outputs, y_batch)
                        
                        # Backward and optimize
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                
                # Log progress every 10 epochs
                if (epoch+1) % 10 == 0:
                    logger.info(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
            
            # Save the retrained model
            torch.save(predictor.model.state_dict(), self.model_path)
            
            # Set flag to indicate model has been updated
            self.model_updated = True
            
            logger.info("Model successfully retrained and saved")
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model: {str(e)}", exc_info=True)
            return False
    
    def optimize_parameters(self) -> Dict:
        """
        Optimize signal generation parameters based on historical performance.
        
        Returns:
            Dictionary with optimized parameters
        """
        try:
            metrics = self.calculate_performance_metrics()
            
            # Start with default parameters
            optimized_params = {
                'technical_weight': 0.6,
                'ml_weight': 0.4,
                'pattern_weight': 0.5,
                'trend_confirmation_threshold': 0.6,
                'confidence_threshold': 50.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
            
            # Only optimize if we have sufficient data
            if metrics['signal_count'] >= 100:
                # Analyze performance and adjust weights
                current_accuracy = metrics['accuracy']
                
                # If accuracy is low, adjust weights to favor more reliable indicators
                if current_accuracy < 0.5:
                    # Investigate which component is more reliable
                    with self.history_lock:
                        # Check ML prediction accuracy
                        signals_with_ml = [s for s in self.signal_history if s['status'] != 'pending' and s.get('ml_prediction')]
                        
                        ml_correct = sum(1 for s in signals_with_ml 
                                         if s['status'] == 'correct' and 
                                         s.get('ml_prediction', {}).get('direction') == s['prediction'])
                        
                        ml_accuracy = ml_correct / len(signals_with_ml) if signals_with_ml else 0
                        
                        # Check technical accuracy
                        tech_signals = [s for s in self.signal_history if s['status'] != 'pending']
                        tech_indicators = ['adx', 'rsi', 'macd', 'vi_plus', 'vi_minus']
                        
                        # Count signals with strong technical indicators that were correct
                        tech_correct = 0
                        total_tech = 0
                        
                        for signal in tech_signals:
                            indicators = signal.get('indicators', {})
                            
                            # Check if basic indicators are present
                            if all(ind in indicators for ind in tech_indicators):
                                total_tech += 1
                                
                                # Simple heuristic to determine if technical indicators were "strong"
                                rsi = indicators.get('rsi', 50)
                                adx = indicators.get('adx', 20)
                                
                                if ((rsi < 30 or rsi > 70) and adx > 25) and signal['status'] == 'correct':
                                    tech_correct += 1
                        
                        tech_accuracy = tech_correct / total_tech if total_tech else 0
                        
                        # Adjust weights based on relative accuracy
                        if ml_accuracy > tech_accuracy:
                            optimized_params['ml_weight'] = min(0.7, optimized_params['ml_weight'] + 0.1)
                            optimized_params['technical_weight'] = 1.0 - optimized_params['ml_weight']
                        else:
                            optimized_params['technical_weight'] = min(0.7, optimized_params['technical_weight'] + 0.1)
                            optimized_params['ml_weight'] = 1.0 - optimized_params['technical_weight']
                        
                        # Adjust confidence threshold
                        if current_accuracy < 0.45:
                            # Increase threshold to be more selective
                            optimized_params['confidence_threshold'] = min(65.0, optimized_params['confidence_threshold'] + 5.0)
                        
                        # Adjust RSI levels based on performance
                        rsi_signals = [s for s in self.signal_history if s['status'] != 'pending' and 'rsi' in s.get('indicators', {})]
                        
                        if rsi_signals:
                            buy_signals_rsi = [s['indicators']['rsi'] for s in rsi_signals 
                                              if s['prediction'] == 'BUY' and s['status'] == 'correct']
                            
                            sell_signals_rsi = [s['indicators']['rsi'] for s in rsi_signals 
                                               if s['prediction'] == 'SELL' and s['status'] == 'correct']
                            
                            if buy_signals_rsi:
                                avg_buy_rsi = sum(buy_signals_rsi) / len(buy_signals_rsi)
                                optimized_params['rsi_oversold'] = max(20, min(40, int(avg_buy_rsi)))
                            
                            if sell_signals_rsi:
                                avg_sell_rsi = sum(sell_signals_rsi) / len(sell_signals_rsi)
                                optimized_params['rsi_overbought'] = max(60, min(80, int(avg_sell_rsi)))
            
            logger.info(f"Generated optimized parameters: {optimized_params}")
            return optimized_params
            
        except Exception as e:
            logger.error(f"Error optimizing parameters: {str(e)}", exc_info=True)
            return {
                'technical_weight': 0.6,
                'ml_weight': 0.4,
                'pattern_weight': 0.5,
                'trend_confirmation_threshold': 0.6,
                'confidence_threshold': 50.0,
                'rsi_oversold': 30,
                'rsi_overbought': 70
            }
    
    def run_learning_cycle(self):
        """Perform a full learning cycle: evaluate performance, optimize parameters, retrain model."""
        try:
            logger.info("Starting continuous learning cycle")
            
            # Calculate current performance
            metrics = self.calculate_performance_metrics()
            logger.info(f"Current performance metrics: {metrics}")
            
            # Optimize parameters
            optimized_params = self.optimize_parameters()
            logger.info(f"Optimized parameters: {optimized_params}")
            
            # Retrain model if enough data is available
            if metrics['signal_count'] >= 50:
                logger.info("Retraining model...")
                success = self.retrain_model()
                if success:
                    logger.info("Model successfully retrained")
                else:
                    logger.warning("Model retraining failed or skipped")
            else:
                logger.info(f"Insufficient data for retraining ({metrics['signal_count']}/50 signals required)")
            
            # Save all data
            self.save_data()
            
            logger.info("Continuous learning cycle completed")
            
        except Exception as e:
            logger.error(f"Error in learning cycle: {str(e)}", exc_info=True)
    
    def start_learning_thread(self):
        """Start continuous learning in a background thread."""
        def learning_loop():
            while True:
                try:
                    self.run_learning_cycle()
                    # Sleep until next learning cycle
                    time.sleep(self.learning_interval)
                except Exception as e:
                    logger.error(f"Error in learning thread: {str(e)}", exc_info=True)
                    time.sleep(3600)  # Wait an hour before retrying after error
        
        learning_thread = Thread(target=learning_loop, daemon=True)
        learning_thread.start()
        logger.info("Started continuous learning thread")
        
        return learning_thread
    
    def get_performance_summary(self) -> Dict:
        """
        Get a summary of current performance metrics.
        
        Returns:
            Dictionary with performance summary
        """
        try:
            metrics = self.calculate_performance_metrics()
            
            # Get recent trends
            accuracy_trend = self.performance_metrics['accuracy'][-10:] if len(self.performance_metrics['accuracy']) >= 10 else self.performance_metrics['accuracy']
            
            # Calculate trend direction
            if len(accuracy_trend) >= 2:
                recent_avg = sum(accuracy_trend[-3:]) / min(3, len(accuracy_trend))
                earlier_avg = sum(accuracy_trend[:3]) / min(3, len(accuracy_trend))
                
                if recent_avg > earlier_avg:
                    trend_direction = "improving"
                elif recent_avg < earlier_avg:
                    trend_direction = "declining"
                else:
                    trend_direction = "stable"
            else:
                trend_direction = "insufficient data"
            
            # Return complete summary
            return {
                'current_metrics': metrics,
                'trend_direction': trend_direction,
                'signals_recorded': len(self.signal_history),
                'signals_evaluated': metrics['signal_count'],
                'feedback_count': len(self.user_feedback),
                'learning_cycles': len(self.performance_metrics['timestamps']),
                'model_updated': self.model_updated
            }
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}", exc_info=True)
            return {
                'current_metrics': {
                    'accuracy': 0,
                    'win_rate': 0,
                    'signal_count': 0
                },
                'trend_direction': "error",
                'signals_recorded': 0,
                'signals_evaluated': 0,
                'feedback_count': 0,
                'learning_cycles': 0,
                'model_updated': False
            }
    
    def process_new_data(self, data: pd.DataFrame, features: pd.DataFrame) -> Dict[str, Any]:
        """Process new market data and features for continuous learning."""
        try:
            # Add new data to history with status
            with self.history_lock:
                self.signal_history.append({
                    'timestamp': datetime.now().isoformat(),
                    'data': data.tail(1).to_dict('records')[0],
                    'features': features.tail(1).to_dict('records')[0],
                    'status': 'pending',  # Initial status
                    'performance': None,
                    'feedback': None
                })
            
            # Calculate current performance metrics
            metrics = self._calculate_performance_metrics()
            
            # Save updated data
            self.save_data()
            
            return {
                'signal_count': len(self.signal_history),
                'performance_metrics': metrics,
                'feedback_count': len(self.user_feedback)
            }
            
        except Exception as e:
            logger.error(f"Error processing new data: {str(e)}")
            return {
                'signal_count': 0,
                'performance_metrics': {},
                'feedback_count': 0,
                'error': str(e)
            }