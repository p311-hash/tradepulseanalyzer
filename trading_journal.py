"""
Trading journal PDF export module for binary options bot.

This module handles the creation and export of trading journals in PDF format,
including signal history, performance metrics, and visualizations.
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union, Any
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.figure import Figure
from fpdf import FPDF
import pandas as pd
import numpy as np
from io import BytesIO
import base64

# Import local modules
from continuous_learning import ContinuousLearningSystem

logger = logging.getLogger(__name__)

class TradingJournal:
    """Class for generating and exporting trading journals in PDF format."""
    
    def __init__(self, 
                 continuous_learner: ContinuousLearningSystem,
                 output_dir: str = './exports',
                 user_id: Optional[int] = None,
                 user_name: Optional[str] = None):
        """
        Initialize the trading journal generator.
        
        Args:
            continuous_learner: ContinuousLearningSystem instance with signal history
            output_dir: Directory to save generated PDFs
            user_id: Optional Telegram user ID for filtering signals
            user_name: Optional user name for the report
        """
        self.continuous_learner = continuous_learner
        self.output_dir = output_dir
        self.user_id = user_id
        self.user_name = user_name
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
    def _prepare_signal_data(self, 
                            days: int = 30,
                            pair: Optional[str] = None) -> pd.DataFrame:
        """
        Prepare signal data for the journal.
        
        Args:
            days: Number of days to include in the journal
            pair: Optional currency pair to filter by
            
        Returns:
            DataFrame with filtered and processed signal data
        """
        try:
            # Get signal history from continuous learner
            signals = self.continuous_learner.signal_history
            
            # Filter by user if specified
            if self.user_id:
                # Filter signals that have user feedback from this user
                user_signals = []
                for signal in signals:
                    for feedback in signal.get('user_feedback', []):
                        if feedback.get('user_id') == self.user_id:
                            user_signals.append(signal)
                            break
                signals = user_signals
            
            # Filter by date range
            cutoff_date = datetime.now() - timedelta(days=days)
            cutoff_timestamp = cutoff_date.timestamp()
            
            filtered_signals = []
            for signal in signals:
                timestamp = signal.get('timestamp', 0)
                if timestamp >= cutoff_timestamp:
                    filtered_signals.append(signal)
            
            # Filter by pair if specified
            if pair:
                filtered_signals = [s for s in filtered_signals if s.get('pair') == pair]
            
            # Convert to DataFrame for easier processing
            if not filtered_signals:
                return pd.DataFrame()
                
            # Extract relevant fields for the journal
            journal_data = []
            for signal in filtered_signals:
                # Get user specific feedback if available
                user_rating = None
                user_comment = None
                user_result = None
                
                if self.user_id:
                    for feedback in signal.get('user_feedback', []):
                        if feedback.get('user_id') == self.user_id:
                            user_rating = feedback.get('rating')
                            user_comment = feedback.get('comment')
                            user_result = feedback.get('result')
                            break
                
                entry = {
                    'date': datetime.fromtimestamp(signal.get('timestamp', 0)),
                    'pair': signal.get('pair', ''),
                    'timeframe': signal.get('timeframe', ''),
                    'direction': signal.get('prediction', 'NEUTRAL'),
                    'confidence': signal.get('confidence', 0),
                    'entry_price': signal.get('entry_price', 0),
                    'status': signal.get('status', 'pending'),
                    'outcome_price': signal.get('outcome_price', 0),
                    'profit_loss': signal.get('profit_loss', 0),
                    'user_rating': user_rating,
                    'user_comment': user_comment,
                    'user_result': user_result,
                    'patterns': ', '.join([k for k, v in signal.get('patterns', {}).items() if v])
                }
                journal_data.append(entry)
            
            df = pd.DataFrame(journal_data)
            
            # Sort by date (most recent first)
            if not df.empty and 'date' in df.columns:
                df = df.sort_values('date', ascending=False)
            
            return df
        
        except Exception as e:
            logger.error(f"Error preparing signal data: {str(e)}")
            return pd.DataFrame()
    
    def _generate_performance_charts(self, df: pd.DataFrame) -> Dict[str, BytesIO]:
        """
        Generate performance charts for the trading journal.
        
        Args:
            df: DataFrame with signal data
            
        Returns:
            Dictionary mapping chart name to BytesIO image data
        """
        charts = {}
        
        try:
            if df.empty:
                return charts
            
            # Daily performance chart
            if 'date' in df.columns and 'profit_loss' in df.columns:
                # Convert to datetime if not already
                df['date'] = pd.to_datetime(df['date'])
                
                # Group by date and calculate daily profit/loss
                daily_pnl = df.groupby(df['date'].dt.date)['profit_loss'].sum()
                
                if not daily_pnl.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Plot daily P&L
                    positive_pnl = daily_pnl[daily_pnl >= 0]
                    negative_pnl = daily_pnl[daily_pnl < 0]
                    
                    ax.bar(positive_pnl.index, positive_pnl, color='green', alpha=0.7)
                    ax.bar(negative_pnl.index, negative_pnl, color='red', alpha=0.7)
                    
                    ax.set_title('Daily Profit/Loss')
                    ax.set_xlabel('Date')
                    ax.set_ylabel('Profit/Loss')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Format x-axis with dates
                    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                    fig.autofmt_xdate()
                    
                    # Add cumulative line
                    ax2 = ax.twinx()
                    cumulative = daily_pnl.cumsum()
                    ax2.plot(cumulative.index, cumulative, 'b-', linewidth=2)
                    ax2.set_ylabel('Cumulative P&L', color='blue')
                    
                    plt.tight_layout()
                    
                    # Save to BytesIO
                    img_data = BytesIO()
                    fig.savefig(img_data, format='png')
                    img_data.seek(0)
                    charts['daily_pnl'] = img_data
                    plt.close(fig)
            
            # Win rate by currency pair
            if 'pair' in df.columns and 'status' in df.columns:
                pair_results = df.groupby('pair')['status'].apply(
                    lambda x: (x == 'correct').sum() / len(x) if len(x) > 0 else 0
                ).reset_index()
                pair_results.columns = ['pair', 'win_rate']
                pair_results['win_rate'] = pair_results['win_rate'] * 100  # Convert to percentage
                
                if not pair_results.empty:
                    fig, ax = plt.subplots(figsize=(10, 5))
                    
                    # Sort by win rate
                    pair_results = pair_results.sort_values('win_rate', ascending=False)
                    
                    # Plot win rate by pair
                    bars = ax.bar(pair_results['pair'], pair_results['win_rate'],
                           color=[plt.cm.RdYlGn(x/100) for x in pair_results['win_rate']])
                    
                    ax.set_title('Win Rate by Currency Pair')
                    ax.set_xlabel('Currency Pair')
                    ax.set_ylabel('Win Rate (%)')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    # Add win rate text on bars
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                f'{height:.1f}%', ha='center', va='bottom')
                    
                    plt.tight_layout()
                    
                    # Save to BytesIO
                    img_data = BytesIO()
                    fig.savefig(img_data, format='png')
                    img_data.seek(0)
                    charts['pair_win_rate'] = img_data
                    plt.close(fig)
            
            # Signal direction distribution
            if 'direction' in df.columns:
                direction_counts = df['direction'].value_counts()
                
                if not direction_counts.empty:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    # Create color map
                    colors = {'BUY': 'green', 'SELL': 'red', 'NEUTRAL': 'gray'}
                    pie_colors = [colors.get(dir, 'blue') for dir in direction_counts.index]
                    
                    # Plot pie chart
                    wedges, texts, autotexts = ax.pie(
                        direction_counts, 
                        labels=direction_counts.index,
                        autopct='%1.1f%%',
                        colors=pie_colors,
                        startangle=90
                    )
                    
                    # Style the chart
                    ax.set_title('Signal Direction Distribution')
                    
                    plt.tight_layout()
                    
                    # Save to BytesIO
                    img_data = BytesIO()
                    fig.savefig(img_data, format='png')
                    img_data.seek(0)
                    charts['direction_distribution'] = img_data
                    plt.close(fig)
            
            # Pattern effectiveness chart
            if 'patterns' in df.columns and 'status' in df.columns:
                # Expand patterns (comma-separated) into rows
                pattern_results = []
                
                for _, row in df.iterrows():
                    patterns = row['patterns'].split(', ') if row['patterns'] else []
                    for pattern in patterns:
                        if pattern:  # Skip empty patterns
                            pattern_results.append({
                                'pattern': pattern,
                                'correct': 1 if row['status'] == 'correct' else 0,
                                'incorrect': 1 if row['status'] == 'incorrect' else 0
                            })
                
                if pattern_results:
                    pattern_df = pd.DataFrame(pattern_results)
                    pattern_summary = pattern_df.groupby('pattern').agg({
                        'correct': 'sum',
                        'incorrect': 'sum'
                    }).reset_index()
                    
                    # Calculate win rate
                    pattern_summary['total'] = pattern_summary['correct'] + pattern_summary['incorrect']
                    pattern_summary['win_rate'] = pattern_summary['correct'] / pattern_summary['total'] * 100
                    
                    # Filter out patterns with too few occurrences
                    pattern_summary = pattern_summary[pattern_summary['total'] >= 3]
                    
                    if not pattern_summary.empty:
                        # Sort by win rate
                        pattern_summary = pattern_summary.sort_values('win_rate', ascending=False)
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        
                        # Plot pattern win rates
                        bars = ax.bar(
                            pattern_summary['pattern'], 
                            pattern_summary['win_rate'],
                            color=[plt.cm.RdYlGn(x/100) for x in pattern_summary['win_rate']]
                        )
                        
                        ax.set_title('Pattern Effectiveness')
                        ax.set_xlabel('Candlestick Pattern')
                        ax.set_ylabel('Win Rate (%)')
                        ax.grid(axis='y', linestyle='--', alpha=0.7)
                        
                        # Add count text on bars
                        for bar, count in zip(bars, pattern_summary['total']):
                            height = bar.get_height()
                            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                                    f'n={count}', ha='center', va='bottom', fontsize=8)
                        
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        
                        # Save to BytesIO
                        img_data = BytesIO()
                        fig.savefig(img_data, format='png')
                        img_data.seek(0)
                        charts['pattern_effectiveness'] = img_data
                        plt.close(fig)
        
        except Exception as e:
            logger.error(f"Error generating performance charts: {str(e)}")
        
        return charts
    
    def _calculate_performance_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate performance metrics from the signal data.
        
        Args:
            df: DataFrame with signal data
            
        Returns:
            Dictionary with performance metrics
        """
        metrics = {
            'total_signals': 0,
            'win_rate': 0,
            'profit_loss': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'win_loss_ratio': 0,
            'best_pair': '',
            'worst_pair': '',
            'best_pattern': '',
            'best_timeframe': '',
            'trades_by_day': {},
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
        
        try:
            if df.empty:
                return metrics
            
            # Filter to completed signals only
            completed = df[df['status'].isin(['correct', 'incorrect'])]
            
            if completed.empty:
                return metrics
            
            # Basic metrics
            total = len(completed)
            wins = (completed['status'] == 'correct').sum()
            losses = (completed['status'] == 'incorrect').sum()
            
            metrics['total_signals'] = total
            metrics['win_rate'] = (wins / total) * 100 if total > 0 else 0
            metrics['profit_loss'] = completed['profit_loss'].sum()
            
            # Average win/loss
            win_df = completed[completed['status'] == 'correct']
            loss_df = completed[completed['status'] == 'incorrect']
            
            metrics['avg_win'] = win_df['profit_loss'].mean() if not win_df.empty else 0
            metrics['avg_loss'] = loss_df['profit_loss'].mean() if not loss_df.empty else 0
            
            # Win/loss ratio
            metrics['win_loss_ratio'] = abs(metrics['avg_win'] / metrics['avg_loss']) if metrics['avg_loss'] != 0 else 0
            
            # Best and worst pairs
            if 'pair' in completed.columns:
                pair_performance = completed.groupby('pair').agg({
                    'status': lambda x: (x == 'correct').sum() / len(x) if len(x) > 0 else 0,
                    'profit_loss': 'sum'
                }).reset_index()
                
                if not pair_performance.empty:
                    # Filter pairs with at least 5 trades
                    pair_counts = completed['pair'].value_counts()
                    valid_pairs = pair_counts[pair_counts >= 5].index
                    filtered_pairs = pair_performance[pair_performance['pair'].isin(valid_pairs)]
                    
                    if not filtered_pairs.empty:
                        best_pair_idx = filtered_pairs['status'].argmax()
                        worst_pair_idx = filtered_pairs['status'].argmin()
                        
                        metrics['best_pair'] = f"{filtered_pairs.iloc[best_pair_idx]['pair']} " \
                                             f"({filtered_pairs.iloc[best_pair_idx]['status']*100:.1f}%)"
                        metrics['worst_pair'] = f"{filtered_pairs.iloc[worst_pair_idx]['pair']} " \
                                              f"({filtered_pairs.iloc[worst_pair_idx]['status']*100:.1f}%)"
            
            # Best pattern
            if 'patterns' in completed.columns:
                pattern_results = []
                
                for _, row in completed.iterrows():
                    patterns = row['patterns'].split(', ') if row['patterns'] else []
                    for pattern in patterns:
                        if pattern:  # Skip empty patterns
                            pattern_results.append({
                                'pattern': pattern,
                                'correct': row['status'] == 'correct'
                            })
                
                if pattern_results:
                    pattern_df = pd.DataFrame(pattern_results)
                    pattern_summary = pattern_df.groupby('pattern')['correct'].agg(['mean', 'count']).reset_index()
                    
                    # Filter patterns with at least 3 occurrences
                    pattern_summary = pattern_summary[pattern_summary['count'] >= 3]
                    
                    if not pattern_summary.empty:
                        best_pattern_idx = pattern_summary['mean'].argmax()
                        best_pattern = pattern_summary.iloc[best_pattern_idx]
                        metrics['best_pattern'] = f"{best_pattern['pattern']} " \
                                                f"({best_pattern['mean']*100:.1f}%, n={best_pattern['count']})"
            
            # Best timeframe
            if 'timeframe' in completed.columns:
                timeframe_performance = completed.groupby('timeframe').agg({
                    'status': lambda x: (x == 'correct').sum() / len(x) if len(x) > 0 else 0,
                    'profit_loss': 'sum'
                }).reset_index()
                
                if not timeframe_performance.empty:
                    # Filter timeframes with at least 5 trades
                    tf_counts = completed['timeframe'].value_counts()
                    valid_tfs = tf_counts[tf_counts >= 5].index
                    filtered_tfs = timeframe_performance[timeframe_performance['timeframe'].isin(valid_tfs)]
                    
                    if not filtered_tfs.empty:
                        best_tf_idx = filtered_tfs['status'].argmax()
                        metrics['best_timeframe'] = f"{filtered_tfs.iloc[best_tf_idx]['timeframe']} " \
                                                  f"({filtered_tfs.iloc[best_tf_idx]['status']*100:.1f}%)"
            
            # Trades by day
            if 'date' in completed.columns:
                # Convert to datetime if not already
                completed['date'] = pd.to_datetime(completed['date'])
                
                # Count trades by day of week
                day_map = {
                    0: 'Monday',
                    1: 'Tuesday',
                    2: 'Wednesday',
                    3: 'Thursday',
                    4: 'Friday',
                    5: 'Saturday',
                    6: 'Sunday'
                }
                
                completed['day_of_week'] = completed['date'].dt.dayofweek
                day_counts = completed.groupby('day_of_week').size()
                
                metrics['trades_by_day'] = {day_map[day]: count for day, count in day_counts.items()}
            
            # Consecutive wins/losses
            if not completed.empty and 'date' in completed.columns:
                # Sort by date
                sorted_results = completed.sort_values('date')
                
                # Calculate consecutive wins/losses
                max_consecutive_wins = 0
                max_consecutive_losses = 0
                current_wins = 0
                current_losses = 0
                
                for status in sorted_results['status']:
                    if status == 'correct':
                        current_wins += 1
                        current_losses = 0
                    else:
                        current_losses += 1
                        current_wins = 0
                    
                    max_consecutive_wins = max(max_consecutive_wins, current_wins)
                    max_consecutive_losses = max(max_consecutive_losses, current_losses)
                
                metrics['consecutive_wins'] = max_consecutive_wins
                metrics['consecutive_losses'] = max_consecutive_losses
        
        except Exception as e:
            logger.error(f"Error calculating performance metrics: {str(e)}")
        
        return metrics
    
    def generate_journal_pdf(self, 
                            days: int = 30, 
                            pair: Optional[str] = None,
                            include_signals: bool = True,
                            max_signals: int = 50) -> Optional[str]:
        """
        Generate a trading journal PDF.
        
        Args:
            days: Number of days to include in the journal
            pair: Optional currency pair to filter by
            include_signals: Whether to include individual signals
            max_signals: Maximum number of signals to include
            
        Returns:
            Path to the generated PDF file, or None if an error occurred
        """
        try:
            # Prepare data
            df = self._prepare_signal_data(days, pair)
            
            if df.empty:
                logger.warning("No signal data available for trading journal")
                return None
            
            # Calculate metrics
            metrics = self._calculate_performance_metrics(df)
            
            # Generate charts
            charts = self._generate_performance_charts(df)
            
            # Create PDF
            pdf = FPDF()
            pdf.add_page()
            
            # Title
            pdf.set_font('Arial', 'B', 16)
            title = f"Trading Journal - Binary Options"
            if self.user_name:
                title += f" - {self.user_name}"
            if pair:
                title += f" - {pair}"
            pdf.cell(0, 10, title, 0, 1, 'C')
            
            # Date range
            pdf.set_font('Arial', '', 10)
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
            pdf.cell(0, 10, f"Period: {start_date} to {end_date}", 0, 1, 'C')
            
            # Summary section
            pdf.set_font('Arial', 'B', 14)
            pdf.cell(0, 10, "Performance Summary", 0, 1, 'L')
            
            # Performance metrics
            pdf.set_font('Arial', '', 10)
            
            # First column
            col_width = pdf.w / 2 - 20
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Total Signals: {metrics['total_signals']}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Win Rate: {metrics['win_rate']:.1f}%", 0, 1)
            
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Net Profit/Loss: {metrics['profit_loss']:.2f}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Win/Loss Ratio: {metrics['win_loss_ratio']:.2f}", 0, 1)
            
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Avg Win: {metrics['avg_win']:.2f}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Avg Loss: {metrics['avg_loss']:.2f}", 0, 1)
            
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Best Pair: {metrics['best_pair']}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Worst Pair: {metrics['worst_pair']}", 0, 1)
            
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Best Pattern: {metrics['best_pattern']}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Best Timeframe: {metrics['best_timeframe']}", 0, 1)
            
            pdf.set_x(10)
            pdf.cell(col_width, 8, f"Max Consecutive Wins: {metrics['consecutive_wins']}", 0, 0)
            pdf.set_x(pdf.w / 2)
            pdf.cell(col_width, 8, f"Max Consecutive Losses: {metrics['consecutive_losses']}", 0, 1)
            
            pdf.ln(10)
            
            # Performance charts
            if charts:
                pdf.set_font('Arial', 'B', 14)
                pdf.cell(0, 10, "Performance Charts", 0, 1, 'L')
                
                # Daily PnL chart
                if 'daily_pnl' in charts:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Daily Profit/Loss", 0, 1, 'C')
                    
                    img_data = charts['daily_pnl']
                    img_data.seek(0)
                    
                    # Get temp file path
                    temp_path = os.path.join(self.output_dir, 'temp_daily_pnl.png')
                    with open(temp_path, 'wb') as f:
                        f.write(img_data.read())
                    
                    # Add image to PDF
                    pdf.image(temp_path, x=10, y=pdf.get_y(), w=pdf.w - 20)
                    pdf.ln(130)  # Space for the image
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Pair win rate chart
                if 'pair_win_rate' in charts:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Win Rate by Currency Pair", 0, 1, 'C')
                    
                    img_data = charts['pair_win_rate']
                    img_data.seek(0)
                    
                    # Get temp file path
                    temp_path = os.path.join(self.output_dir, 'temp_pair_win_rate.png')
                    with open(temp_path, 'wb') as f:
                        f.write(img_data.read())
                    
                    # Add image to PDF
                    pdf.image(temp_path, x=10, y=pdf.get_y(), w=pdf.w - 20)
                    pdf.ln(130)  # Space for the image
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Signal direction distribution chart
                if 'direction_distribution' in charts:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Signal Direction Distribution", 0, 1, 'C')
                    
                    img_data = charts['direction_distribution']
                    img_data.seek(0)
                    
                    # Get temp file path
                    temp_path = os.path.join(self.output_dir, 'temp_direction_dist.png')
                    with open(temp_path, 'wb') as f:
                        f.write(img_data.read())
                    
                    # Add image to PDF
                    pdf.image(temp_path, x=10, y=pdf.get_y(), w=pdf.w - 20)
                    pdf.ln(130)  # Space for the image
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
                
                # Pattern effectiveness chart
                if 'pattern_effectiveness' in charts:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 12)
                    pdf.cell(0, 10, "Pattern Effectiveness", 0, 1, 'C')
                    
                    img_data = charts['pattern_effectiveness']
                    img_data.seek(0)
                    
                    # Get temp file path
                    temp_path = os.path.join(self.output_dir, 'temp_pattern_effect.png')
                    with open(temp_path, 'wb') as f:
                        f.write(img_data.read())
                    
                    # Add image to PDF
                    pdf.image(temp_path, x=10, y=pdf.get_y(), w=pdf.w - 20)
                    pdf.ln(130)  # Space for the image
                    
                    # Clean up temp file
                    try:
                        os.remove(temp_path)
                    except:
                        pass
            
            # Recent Signals
            if include_signals:
                # Limit to max_signals
                signals_df = df.head(max_signals).copy()
                
                if not signals_df.empty:
                    pdf.add_page()
                    pdf.set_font('Arial', 'B', 14)
                    pdf.cell(0, 10, f"Recent Signals (Last {len(signals_df)})", 0, 1, 'L')
                    
                    # Table header
                    pdf.set_font('Arial', 'B', 8)
                    pdf.set_fill_color(240, 240, 240)
                    
                    # Set column widths
                    col_widths = {
                        'date': 25,
                        'pair': 20,
                        'timeframe': 15,
                        'direction': 15,
                        'confidence': 15,
                        'status': 15,
                        'profit_loss': 20,
                        'patterns': 60
                    }
                    
                    # Table headers (handle multi-page tables)
                    def print_table_header():
                        pdf.set_fill_color(240, 240, 240)
                        pdf.set_font('Arial', 'B', 8)
                        
                        pdf.cell(col_widths['date'], 6, "Date", 1, 0, 'C', True)
                        pdf.cell(col_widths['pair'], 6, "Pair", 1, 0, 'C', True)
                        pdf.cell(col_widths['timeframe'], 6, "Timeframe", 1, 0, 'C', True)
                        pdf.cell(col_widths['direction'], 6, "Direction", 1, 0, 'C', True)
                        pdf.cell(col_widths['confidence'], 6, "Confidence", 1, 0, 'C', True)
                        pdf.cell(col_widths['status'], 6, "Status", 1, 0, 'C', True)
                        pdf.cell(col_widths['profit_loss'], 6, "P/L", 1, 0, 'C', True)
                        pdf.cell(col_widths['patterns'], 6, "Patterns", 1, 1, 'C', True)
                    
                    print_table_header()
                    
                    # Table content
                    pdf.set_font('Arial', '', 8)
                    row_height = 6
                    
                    for i, row in signals_df.iterrows():
                        # Check if we need a new page
                        if pdf.get_y() + row_height > pdf.h - 15:
                            pdf.add_page()
                            print_table_header()
                        
                        # Format date
                        date_str = row['date'].strftime('%Y-%m-%d %H:%M') if isinstance(row['date'], datetime) else str(row['date'])
                        
                        # Set cell colors based on status
                        if row['status'] == 'correct':
                            pdf.set_fill_color(220, 255, 220)  # Light green
                            fill = True
                        elif row['status'] == 'incorrect':
                            pdf.set_fill_color(255, 220, 220)  # Light red
                            fill = True
                        else:
                            fill = False
                        
                        # Row data
                        pdf.cell(col_widths['date'], row_height, date_str, 1, 0, 'L', fill)
                        pdf.cell(col_widths['pair'], row_height, str(row['pair']), 1, 0, 'C', fill)
                        pdf.cell(col_widths['timeframe'], row_height, str(row['timeframe']), 1, 0, 'C', fill)
                        
                        # Direction cell with color
                        if row['direction'] == 'BUY':
                            pdf.set_text_color(0, 128, 0)  # Green text
                        elif row['direction'] == 'SELL':
                            pdf.set_text_color(255, 0, 0)  # Red text
                        else:
                            pdf.set_text_color(128, 128, 128)  # Gray text
                            
                        pdf.cell(col_widths['direction'], row_height, str(row['direction']), 1, 0, 'C', fill)
                        pdf.set_text_color(0, 0, 0)  # Reset text color
                        
                        pdf.cell(col_widths['confidence'], row_height, f"{row['confidence']:.1f}%", 1, 0, 'C', fill)
                        pdf.cell(col_widths['status'], row_height, str(row['status']), 1, 0, 'C', fill)
                        
                        # Format P/L with color
                        if row['profit_loss'] > 0:
                            pdf.set_text_color(0, 128, 0)  # Green text
                            pl_str = f"+{row['profit_loss']:.2f}"
                        elif row['profit_loss'] < 0:
                            pdf.set_text_color(255, 0, 0)  # Red text
                            pl_str = f"{row['profit_loss']:.2f}"
                        else:
                            pdf.set_text_color(0, 0, 0)  # Black text
                            pl_str = "0.00"
                            
                        pdf.cell(col_widths['profit_loss'], row_height, pl_str, 1, 0, 'R', fill)
                        pdf.set_text_color(0, 0, 0)  # Reset text color
                        
                        # Pattern column might need wrapping
                        patterns_str = str(row['patterns'])
                        if len(patterns_str) > 30:
                            patterns_str = patterns_str[:27] + "..."
                            
                        pdf.cell(col_widths['patterns'], row_height, patterns_str, 1, 1, 'L', fill)
            
            # Footer with timestamp
            pdf.set_y(pdf.h - 10)
            pdf.set_font('Arial', 'I', 8)
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            pdf.cell(0, 5, f"Generated on {now}", 0, 0, 'C')
            
            # Save PDF
            filename = f"trading_journal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            if self.user_id:
                filename = f"user_{self.user_id}_" + filename
            if pair:
                filename = f"{pair}_" + filename
                
            file_path = os.path.join(self.output_dir, filename)
            pdf.output(file_path)
            
            logger.info(f"Trading journal PDF generated: {file_path}")
            return file_path
        
        except Exception as e:
            logger.error(f"Error generating trading journal PDF: {str(e)}")
            return None