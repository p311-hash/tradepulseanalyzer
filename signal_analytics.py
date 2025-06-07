"""
Signal analytics module for tracking and analyzing trading signal performance
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from sqlalchemy import create_engine, func, desc, and_, or_
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.sql import text
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import io
import base64

from models import Base, TradingSignal, UserFeedback, PerformanceMetrics

# Configure logging
logger = logging.getLogger(__name__)

class SignalAnalytics:
    """
    Signal analytics service for tracking and analyzing trading signal performance
    """
    def __init__(self, db_url: Optional[str] = None):
        """
        Initialize the analytics service
        
        Args:
            db_url: Database connection URL. If None, uses DATABASE_URL environment variable
        """
        if db_url is None:
            db_url = os.environ.get('DATABASE_URL')
            
        if not db_url:
            raise ValueError("Database URL is required for signal analytics")
            
        self.engine = create_engine(db_url)
        self.Session = sessionmaker(bind=self.engine)
        self._create_tables_if_not_exist()
        
    def _create_tables_if_not_exist(self):
        """Create database tables if they don't exist"""
        Base.metadata.create_all(self.engine)
        logger.info("Signal analytics database tables created or verified")
    
    def record_signal(self, signal_data: Dict) -> str:
        """
        Record a new trading signal in the database
        
        Args:
            signal_data: Dictionary with signal details
            
        Returns:
            signal_id: The ID of the recorded signal
        """
        try:
            session = self.Session()
            
            # Extract relevant fields from signal_data
            signal_id = signal_data.get('signal_id', f"{signal_data.get('pair')}_{signal_data.get('timeframe')}_{int(datetime.now().timestamp())}")
            pair = signal_data.get('pair', '')
            timeframe = signal_data.get('timeframe', '')
            direction = signal_data.get('direction', 'NEUTRAL')
            confidence = float(signal_data.get('confidence', 0))
            entry_price = float(signal_data.get('entry_price', 0)) if signal_data.get('entry_price') else None
            expiry_time = signal_data.get('expiry_time')
            
            # Convert string dates to datetime if necessary
            if isinstance(expiry_time, str):
                expiry_time = datetime.fromisoformat(expiry_time)
                
            # Handle optional fields
            indicators = json.dumps(signal_data.get('indicators', {})) if isinstance(signal_data.get('indicators'), dict) else signal_data.get('indicators', '')
            pattern = signal_data.get('pattern', '')
            regime = signal_data.get('regime', '')
            ml_prediction = signal_data.get('ml_prediction', '')
            ml_confidence = float(signal_data.get('ml_confidence', 0)) if signal_data.get('ml_confidence') is not None else None
            volume_signal = signal_data.get('volume_signal', '')
            volume_confidence = float(signal_data.get('volume_confidence', 0)) if signal_data.get('volume_confidence') is not None else None
            notes = signal_data.get('notes', '')
            
            # Create new signal record
            new_signal = TradingSignal(
                signal_id=signal_id,
                pair=pair,
                timeframe=timeframe,
                direction=direction,
                confidence=confidence,
                entry_price=entry_price,
                expiry_time=expiry_time,
                outcome='PENDING',
                indicators=indicators,
                pattern=pattern,
                regime=regime,
                ml_prediction=ml_prediction,
                ml_confidence=ml_confidence,
                volume_signal=volume_signal,
                volume_confidence=volume_confidence,
                notes=notes
            )
            
            # Check if signal already exists
            existing_signal = session.query(TradingSignal).filter_by(signal_id=signal_id).first()
            if existing_signal:
                logger.warning(f"Signal with ID {signal_id} already exists. Skipping.")
                session.close()
                return signal_id
            
            session.add(new_signal)
            session.commit()
            logger.info(f"Recorded new signal {signal_id} for {pair} {timeframe}: {direction}")
            
            session.close()
            return signal_id
            
        except Exception as e:
            logger.error(f"Error recording signal: {str(e)}")
            if session:
                session.rollback()
                session.close()
            raise
            
    def update_signal_outcome(self, signal_id: str, exit_price: float, outcome: Optional[str] = None) -> bool:
        """
        Update a signal with its outcome after expiry
        
        Args:
            signal_id: Unique ID of the signal
            exit_price: Price at signal expiry
            outcome: Optional outcome override (WIN, LOSS, DRAW)
            
        Returns:
            success: Whether the update was successful
        """
        try:
            session = self.Session()
            
            # Find the signal
            signal = session.query(TradingSignal).filter_by(signal_id=signal_id).first()
            if not signal:
                logger.warning(f"Signal with ID {signal_id} not found for outcome update")
                session.close()
                return False
                
            # Update exit price
            signal.exit_price = exit_price
            
            # Calculate outcome if not provided
            if outcome is None:
                if signal.direction == 'BUY':
                    # For BUY signals, win if exit_price > entry_price
                    if exit_price > signal.entry_price:
                        outcome = 'WIN'
                    elif exit_price < signal.entry_price:
                        outcome = 'LOSS'
                    else:
                        outcome = 'DRAW'
                elif signal.direction == 'SELL':
                    # For SELL signals, win if exit_price < entry_price
                    if exit_price < signal.entry_price:
                        outcome = 'WIN'
                    elif exit_price > signal.entry_price:
                        outcome = 'LOSS'
                    else:
                        outcome = 'DRAW'
                else:
                    # For NEUTRAL signals, always DRAW
                    outcome = 'DRAW'
            
            # Update outcome
            signal.outcome = outcome
            
            # Calculate profit/loss percentage
            if signal.entry_price and signal.entry_price > 0:
                if signal.direction == 'BUY':
                    profit_loss = (exit_price - signal.entry_price) / signal.entry_price * 100
                elif signal.direction == 'SELL':
                    profit_loss = (signal.entry_price - exit_price) / signal.entry_price * 100
                else:
                    profit_loss = 0
                signal.profit_loss = profit_loss
            
            session.commit()
            logger.info(f"Updated signal {signal_id} outcome: {outcome}, P/L: {signal.profit_loss:.2f}%")
            
            # Update performance metrics
            self._update_performance_metrics(session)
            
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating signal outcome: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def record_user_feedback(self, signal_id: str, user_id: int, 
                           rating: Optional[int] = None, comment: Optional[str] = None,
                           traded: Optional[bool] = None, user_outcome: Optional[str] = None) -> bool:
        """
        Record user feedback for a signal
        
        Args:
            signal_id: Unique ID of the signal
            user_id: Telegram user ID
            rating: User rating from 1-5
            comment: User comment
            traded: Whether user traded on this signal
            user_outcome: User-reported outcome (WIN or LOSS)
            
        Returns:
            success: Whether the feedback was recorded successfully
        """
        try:
            session = self.Session()
            
            # Check if the signal exists
            signal = session.query(TradingSignal).filter_by(signal_id=signal_id).first()
            if not signal:
                logger.warning(f"Signal with ID {signal_id} not found for feedback")
                session.close()
                return False
            
            # Check if feedback already exists from this user
            existing_feedback = session.query(UserFeedback).filter_by(
                signal_id=signal_id, user_id=user_id).first()
                
            if existing_feedback:
                # Update existing feedback
                if rating is not None:
                    existing_feedback.rating = rating
                if comment is not None:
                    existing_feedback.comment = comment
                if traded is not None:
                    existing_feedback.traded = traded
                if user_outcome is not None:
                    existing_feedback.user_outcome = user_outcome
                    
                session.commit()
                logger.info(f"Updated feedback for signal {signal_id} from user {user_id}")
            else:
                # Create new feedback
                new_feedback = UserFeedback(
                    signal_id=signal_id,
                    user_id=user_id,
                    rating=rating,
                    comment=comment,
                    traded=traded if traded is not None else False,
                    user_outcome=user_outcome
                )
                
                session.add(new_feedback)
                session.commit()
                logger.info(f"Recorded new feedback for signal {signal_id} from user {user_id}")
            
            session.close()
            return True
            
        except Exception as e:
            logger.error(f"Error recording user feedback: {str(e)}")
            if session:
                session.rollback()
                session.close()
            return False
    
    def get_pending_signals(self) -> List[Dict]:
        """
        Get signals that are pending outcome updates
        
        Returns:
            List of signal dictionaries
        """
        try:
            session = self.Session()
            
            # Find signals that are PENDING and have expired
            now = datetime.utcnow()
            pending_signals = session.query(TradingSignal).filter(
                and_(
                    TradingSignal.outcome == 'PENDING',
                    TradingSignal.expiry_time < now
                )
            ).all()
            
            result = []
            for signal in pending_signals:
                result.append({
                    'signal_id': signal.signal_id,
                    'pair': signal.pair,
                    'timeframe': signal.timeframe,
                    'direction': signal.direction,
                    'entry_price': signal.entry_price,
                    'expiry_time': signal.expiry_time.isoformat() if signal.expiry_time else None,
                    'generated_at': signal.generated_at.isoformat() if signal.generated_at else None
                })
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting pending signals: {str(e)}")
            if session:
                session.close()
            return []
            
    def _update_performance_metrics(self, session: Optional[Session] = None) -> bool:
        """
        Update all performance metrics based on signal outcomes
        
        Args:
            session: Optional SQLAlchemy session
            
        Returns:
            success: Whether the update was successful
        """
        close_session = False
        if session is None:
            session = self.Session()
            close_session = True
            
        try:
            # Get all completed signals
            completed_signals = session.query(TradingSignal).filter(
                TradingSignal.outcome.in_(['WIN', 'LOSS', 'DRAW'])
            ).all()
            
            if not completed_signals:
                logger.info("No completed signals for performance metrics update")
                if close_session:
                    session.close()
                return True
                
            # Organize signals by different dimensions
            metrics_dimensions = {
                'overall': {'': completed_signals},
                'pair': {},
                'timeframe': {},
                'daily': {},
                'weekly': {},
                'monthly': {}
            }
            
            # Group by pair
            for signal in completed_signals:
                if signal.pair not in metrics_dimensions['pair']:
                    metrics_dimensions['pair'][signal.pair] = []
                metrics_dimensions['pair'][signal.pair].append(signal)
                
                # Group by timeframe
                if signal.timeframe not in metrics_dimensions['timeframe']:
                    metrics_dimensions['timeframe'][signal.timeframe] = []
                metrics_dimensions['timeframe'][signal.timeframe].append(signal)
                
                # Group by date periods
                if signal.generated_at:
                    date_str = signal.generated_at.strftime('%Y-%m-%d')
                    if date_str not in metrics_dimensions['daily']:
                        metrics_dimensions['daily'][date_str] = []
                    metrics_dimensions['daily'][date_str].append(signal)
                    
                    week_str = f"{signal.generated_at.year}-W{signal.generated_at.isocalendar()[1]}"
                    if week_str not in metrics_dimensions['weekly']:
                        metrics_dimensions['weekly'][week_str] = []
                    metrics_dimensions['weekly'][week_str].append(signal)
                    
                    month_str = signal.generated_at.strftime('%Y-%m')
                    if month_str not in metrics_dimensions['monthly']:
                        metrics_dimensions['monthly'][month_str] = []
                    metrics_dimensions['monthly'][month_str].append(signal)
                    
            # Calculate metrics for each dimension
            for dimension, values in metrics_dimensions.items():
                for value, signals in values.items():
                    if not signals:
                        continue
                        
                    # Count outcomes
                    win_count = sum(1 for s in signals if s.outcome == 'WIN')
                    loss_count = sum(1 for s in signals if s.outcome == 'LOSS')
                    draw_count = sum(1 for s in signals if s.outcome == 'DRAW')
                    total_count = win_count + loss_count + draw_count
                    
                    # Calculate win rate and other metrics
                    win_rate = win_count / total_count * 100 if total_count > 0 else 0
                    
                    # Profit metrics
                    profits = [s.profit_loss for s in signals if s.profit_loss and s.profit_loss > 0]
                    losses = [abs(s.profit_loss) for s in signals if s.profit_loss and s.profit_loss < 0]
                    
                    avg_profit = sum(profits) / len(profits) if profits else 0
                    avg_loss = sum(losses) / len(losses) if losses else 0
                    total_profit = sum(profits) - sum(losses)
                    profit_factor = sum(profits) / sum(losses) if sum(losses) > 0 else float('inf')
                    
                    # Consecutive metrics
                    outcomes = [s.outcome for s in sorted(signals, key=lambda x: x.generated_at)]
                    max_consecutive_wins = self._calculate_max_streak(outcomes, 'WIN')
                    max_consecutive_losses = self._calculate_max_streak(outcomes, 'LOSS')
                    
                    # Update or create metrics record
                    metric = session.query(PerformanceMetrics).filter_by(
                        metric_type=dimension, metric_value=value).first()
                        
                    if metric:
                        # Update existing record
                        metric.win_count = win_count
                        metric.loss_count = loss_count
                        metric.draw_count = draw_count
                        metric.win_rate = win_rate
                        metric.avg_profit = avg_profit
                        metric.avg_loss = avg_loss
                        metric.total_profit = total_profit
                        metric.profit_factor = profit_factor
                        metric.max_consecutive_wins = max_consecutive_wins
                        metric.max_consecutive_losses = max_consecutive_losses
                        metric.updated_at = datetime.utcnow()
                    else:
                        # Create new record
                        new_metric = PerformanceMetrics(
                            metric_type=dimension,
                            metric_value=value,
                            win_count=win_count,
                            loss_count=loss_count,
                            draw_count=draw_count,
                            win_rate=win_rate,
                            avg_profit=avg_profit,
                            avg_loss=avg_loss,
                            total_profit=total_profit,
                            profit_factor=profit_factor,
                            max_consecutive_wins=max_consecutive_wins,
                            max_consecutive_losses=max_consecutive_losses
                        )
                        session.add(new_metric)
                        
            session.commit()
            logger.info("Performance metrics updated successfully")
            
            if close_session:
                session.close()
            return True
            
        except Exception as e:
            logger.error(f"Error updating performance metrics: {str(e)}")
            if close_session and session:
                session.rollback()
                session.close()
            return False
            
    def _calculate_max_streak(self, outcomes: List[str], outcome_type: str) -> int:
        """Calculate the maximum consecutive streak of a specific outcome"""
        max_streak = 0
        current_streak = 0
        
        for outcome in outcomes:
            if outcome == outcome_type:
                current_streak += 1
                max_streak = max(max_streak, current_streak)
            else:
                current_streak = 0
                
        return max_streak
        
    def get_performance_summary(self) -> Dict:
        """
        Get overall performance summary
        
        Returns:
            Dictionary with performance metrics
        """
        try:
            session = self.Session()
            
            # Get overall metrics
            overall = session.query(PerformanceMetrics).filter_by(
                metric_type='overall', metric_value='').first()
                
            if not overall:
                logger.warning("No overall performance metrics available")
                session.close()
                return {
                    'win_rate': 0,
                    'total_signals': 0,
                    'win_count': 0,
                    'loss_count': 0,
                    'draw_count': 0,
                    'avg_profit': 0,
                    'avg_loss': 0,
                    'total_profit': 0,
                    'profit_factor': 0,
                    'max_consecutive_wins': 0,
                    'max_consecutive_losses': 0
                }
                
            result = {
                'win_rate': overall.win_rate,
                'total_signals': overall.win_count + overall.loss_count + overall.draw_count,
                'win_count': overall.win_count,
                'loss_count': overall.loss_count,
                'draw_count': overall.draw_count,
                'avg_profit': overall.avg_profit,
                'avg_loss': overall.avg_loss,
                'total_profit': overall.total_profit,
                'profit_factor': overall.profit_factor,
                'max_consecutive_wins': overall.max_consecutive_wins,
                'max_consecutive_losses': overall.max_consecutive_losses
            }
            
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {str(e)}")
            if session:
                session.close()
            return {
                'win_rate': 0,
                'total_signals': 0,
                'win_count': 0,
                'loss_count': 0,
                'draw_count': 0,
                'avg_profit': 0,
                'avg_loss': 0,
                'total_profit': 0,
                'profit_factor': 0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0,
                'error': str(e)
            }
            
    def get_performance_by_dimension(self, dimension: str) -> List[Dict]:
        """
        Get performance metrics grouped by a specific dimension
        
        Args:
            dimension: 'pair', 'timeframe', 'daily', 'weekly', or 'monthly'
            
        Returns:
            List of metric dictionaries
        """
        try:
            session = self.Session()
            
            # Validate dimension
            if dimension not in ['pair', 'timeframe', 'daily', 'weekly', 'monthly']:
                logger.error(f"Invalid dimension: {dimension}")
                session.close()
                return []
                
            # Get metrics for the dimension
            metrics = session.query(PerformanceMetrics).filter(
                PerformanceMetrics.metric_type == dimension
            ).order_by(desc(PerformanceMetrics.total_profit)).all()
            
            result = []
            for metric in metrics:
                result.append({
                    'dimension_value': metric.metric_value,
                    'win_rate': metric.win_rate,
                    'total_signals': metric.win_count + metric.loss_count + metric.draw_count,
                    'win_count': metric.win_count,
                    'loss_count': metric.loss_count,
                    'draw_count': metric.draw_count,
                    'avg_profit': metric.avg_profit,
                    'avg_loss': metric.avg_loss,
                    'total_profit': metric.total_profit,
                    'profit_factor': metric.profit_factor,
                    'max_consecutive_wins': metric.max_consecutive_wins,
                    'max_consecutive_losses': metric.max_consecutive_losses
                })
                
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting performance by dimension: {str(e)}")
            if session:
                session.close()
            return []
            
    def get_recent_signals(self, limit: int = 20, offset: int = 0, 
                          pair: Optional[str] = None, 
                          timeframe: Optional[str] = None,
                          outcome: Optional[str] = None) -> List[Dict]:
        """
        Get recent signals with optional filtering
        
        Args:
            limit: Maximum number of signals to return
            offset: Number of signals to skip
            pair: Optional filter by pair
            timeframe: Optional filter by timeframe
            outcome: Optional filter by outcome
            
        Returns:
            List of signal dictionaries
        """
        try:
            session = self.Session()
            
            # Build query with filters
            query = session.query(TradingSignal)
            
            if pair:
                query = query.filter(TradingSignal.pair == pair)
                
            if timeframe:
                query = query.filter(TradingSignal.timeframe == timeframe)
                
            if outcome:
                query = query.filter(TradingSignal.outcome == outcome)
                
            # Order by generation time descending
            query = query.order_by(desc(TradingSignal.generated_at))
            
            # Apply pagination
            signals = query.limit(limit).offset(offset).all()
            
            result = []
            for signal in signals:
                # Get feedback for this signal
                feedback = session.query(UserFeedback).filter_by(signal_id=signal.signal_id).all()
                feedback_data = []
                
                for fb in feedback:
                    feedback_data.append({
                        'user_id': fb.user_id,
                        'rating': fb.rating,
                        'comment': fb.comment,
                        'traded': fb.traded,
                        'user_outcome': fb.user_outcome,
                        'created_at': fb.created_at.isoformat() if fb.created_at else None
                    })
                
                # Format signal data
                result.append({
                    'signal_id': signal.signal_id,
                    'pair': signal.pair,
                    'timeframe': signal.timeframe,
                    'direction': signal.direction,
                    'confidence': signal.confidence,
                    'entry_price': signal.entry_price,
                    'exit_price': signal.exit_price,
                    'outcome': signal.outcome,
                    'profit_loss': signal.profit_loss,
                    'expiry_time': signal.expiry_time.isoformat() if signal.expiry_time else None,
                    'generated_at': signal.generated_at.isoformat() if signal.generated_at else None,
                    'indicators': json.loads(signal.indicators) if signal.indicators and signal.indicators.startswith('{') else signal.indicators,
                    'pattern': signal.pattern,
                    'regime': signal.regime,
                    'ml_prediction': signal.ml_prediction,
                    'ml_confidence': signal.ml_confidence,
                    'volume_signal': signal.volume_signal,
                    'volume_confidence': signal.volume_confidence,
                    'notes': signal.notes,
                    'feedback': feedback_data
                })
                
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {str(e)}")
            if session:
                session.close()
            return []
            
    def generate_performance_chart(self, chart_type: str = 'win_rate', 
                                  dimension: str = 'pair',
                                  top_n: int = 5) -> Optional[str]:
        """
        Generate a performance chart as base64 encoded image
        
        Args:
            chart_type: Type of chart ('win_rate', 'profit', 'volume')
            dimension: Dimension to group by ('pair', 'timeframe', 'daily', 'weekly', 'monthly')
            top_n: Number of top items to include
            
        Returns:
            Base64 encoded image or None if error
        """
        try:
            metrics = self.get_performance_by_dimension(dimension)
            
            if not metrics:
                logger.warning(f"No metrics available for {dimension} dimension")
                return None
                
            # Sort metrics based on chart type
            if chart_type == 'win_rate':
                metrics.sort(key=lambda x: x['win_rate'], reverse=True)
                y_key = 'win_rate'
                title = f'Win Rate by {dimension.capitalize()}'
                ylabel = 'Win Rate (%)'
            elif chart_type == 'profit':
                metrics.sort(key=lambda x: x['total_profit'], reverse=True)
                y_key = 'total_profit'
                title = f'Total Profit by {dimension.capitalize()}'
                ylabel = 'Profit (%)'
            elif chart_type == 'volume':
                metrics.sort(key=lambda x: x['total_signals'], reverse=True)
                y_key = 'total_signals'
                title = f'Signal Volume by {dimension.capitalize()}'
                ylabel = 'Number of Signals'
            else:
                logger.error(f"Invalid chart type: {chart_type}")
                return None
                
            # Limit to top N
            metrics = metrics[:top_n]
            
            # Create the plot
            plt.figure(figsize=(10, 6))
            x_values = [m['dimension_value'] for m in metrics]
            y_values = [m[y_key] for m in metrics]
            
            colors = ['green' if y > 0 else 'red' for y in y_values] if chart_type == 'profit' else 'skyblue'
            
            plt.bar(x_values, y_values, color=colors)
            plt.xlabel(dimension.capitalize())
            plt.ylabel(ylabel)
            plt.title(title)
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Convert plot to base64 image
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png')
            plt.close()
            
            buffer.seek(0)
            image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            return None
            
    def get_signal_timeline(self, days: int = 30, pair: Optional[str] = None) -> Dict:
        """
        Get signal timeline data for the specified period
        
        Args:
            days: Number of days to include
            pair: Optional filter for a specific pair
            
        Returns:
            Dictionary with timeline data
        """
        try:
            session = self.Session()
            
            # Calculate date range
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(days=days)
            
            # Build query with date filter
            query = session.query(
                func.date(TradingSignal.generated_at).label('date'),
                func.count(TradingSignal.id).label('total'),
                func.sum(case((TradingSignal.outcome == 'WIN', 1), else_=0)).label('wins'),
                func.sum(case((TradingSignal.outcome == 'LOSS', 1), else_=0)).label('losses')
            ).filter(TradingSignal.generated_at >= start_date)
            
            if pair:
                query = query.filter(TradingSignal.pair == pair)
                
            # Group by date
            query = query.group_by(func.date(TradingSignal.generated_at))
            
            # Order by date
            query = query.order_by(func.date(TradingSignal.generated_at))
            
            results = query.all()
            
            # Format results
            dates = []
            totals = []
            wins = []
            losses = []
            
            for row in results:
                dates.append(row.date.strftime('%Y-%m-%d'))
                totals.append(row.total)
                wins.append(row.wins or 0)  # Handle None values
                losses.append(row.losses or 0)  # Handle None values
                
            session.close()
            
            return {
                'dates': dates,
                'totals': totals,
                'wins': wins,
                'losses': losses
            }
            
        except Exception as e:
            logger.error(f"Error getting signal timeline: {str(e)}")
            if session:
                session.close()
            return {
                'dates': [],
                'totals': [],
                'wins': [],
                'losses': []
            }
            
    def get_top_performing_pairs(self, limit: int = 5) -> List[Dict]:
        """
        Get top performing pairs
        
        Args:
            limit: Maximum number of pairs to return
            
        Returns:
            List of pair performance dictionaries
        """
        try:
            session = self.Session()
            
            # Get pair metrics ordered by win rate
            metrics = session.query(PerformanceMetrics).filter(
                PerformanceMetrics.metric_type == 'pair'
            ).order_by(desc(PerformanceMetrics.win_rate)).limit(limit).all()
            
            result = []
            for metric in metrics:
                result.append({
                    'pair': metric.metric_value,
                    'win_rate': metric.win_rate,
                    'total_signals': metric.win_count + metric.loss_count + metric.draw_count,
                    'total_profit': metric.total_profit
                })
                
            session.close()
            return result
            
        except Exception as e:
            logger.error(f"Error getting top performing pairs: {str(e)}")
            if session:
                session.close()
            return []