"""
Backtesting module for TradePulseAnalyzer
Tests trading signals against historical data to validate strategy performance
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import multiprocessing as mp
from datetime import datetime, timedelta
from signal_generator import SignalGenerator
from signal_analytics import SignalAnalytics
import logging

logger = logging.getLogger(__name__)

@dataclass
class TradeResult:
    entry_time: datetime
    exit_time: datetime
    entry_price: float
    exit_price: float
    position_size: float
    pnl: float
    trade_duration: timedelta
    max_drawdown: float
    risk_metrics: Dict

@dataclass
class BacktestMetrics:
    total_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    trades: List[TradeResult]
    monte_carlo_stats: Optional[Dict] = None

class EnterpriseBacktester:
    """Backtests trading signals against historical data"""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize backtester
        
        Args:
            risk_free_rate: Risk-free rate for Sharpe/Sortino ratio calculation
        """
        self.risk_free_rate = risk_free_rate
        self.num_monte_carlo_sims = 1000
        self.cpu_count = mp.cpu_count()
        
    def run_backtest(self, data: pd.DataFrame, 
                    timeframe: str = '5m',
                    stop_loss_pct: float = 1.0,
                    take_profit_pct: float = 2.0,
                    position_size_pct: float = 1.0) -> BacktestMetrics:
        """
        Run backtest on historical data
        
        Args:
            data: DataFrame with OHLCV data
            timeframe: Trading timeframe
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size as percentage of balance
            
        Returns:
            BacktestMetrics object with comprehensive backtest results
        """
        try:
            logger.info(f"Starting backtest with {len(data)} candles")
            
            trades = []
            capital = 100000  # Initial capital
            in_position = False
            max_capital = capital
            
            # Generate signals for each timeframe segment
            for i in range(50, len(data)):  # Start after warmup period
                # Get historical data up to current point
                historical_data = data.iloc[:i]
                
                # Generate signal
                generator = SignalGenerator(data=historical_data)
                signal = generator.generate_signal()
                
                current_price = data.iloc[i]['close']
                
                if not in_position and signal['direction'] != 'NEUTRAL' and signal['confidence'] > 60:  # Enter position
                    position_size = capital * (position_size_pct / 100)
                    entry_price = current_price
                    entry_time = data.index[i]
                    in_position = True
                    max_position_value = position_size * current_price
                    
                elif in_position and (
                    (signal['direction'] == 'BUY' and current_price - entry_price < -stop_loss_pct/100) or  # Stop loss
                    (signal['direction'] == 'SELL' and current_price - entry_price > stop_loss_pct/100)  # Stop loss
                ):
                    exit_price = current_price
                    pnl = (exit_price - entry_price) * position_size
                    capital += pnl
                    max_capital = max(max_capital, capital)
                    
                    trade = TradeResult(
                        entry_time=entry_time,
                        exit_time=data.index[i],
                        entry_price=entry_price,
                        exit_price=exit_price,
                        position_size=position_size,
                        pnl=pnl,
                        trade_duration=data.index[i] - entry_time,
                        max_drawdown=(max_position_value - min(position_size * data['low'].iloc[i-1:i+1])) / max_position_value,
                        risk_metrics={
                            'volatility': data['close'].iloc[i-20:i].std(),
                            'volume_profile': data['volume'].iloc[i-20:i].mean()
                        }
                    )
                    trades.append(trade)
                    in_position = False
                
            return self._calculate_metrics(trades, capital)
        
        except Exception as e:
            logger.error(f"Error in backtest: {str(e)}")
            return BacktestMetrics(
                total_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                trades=[]
            )
    
    def run_monte_carlo_simulation(self, trades: List[TradeResult], 
                                 initial_capital: float) -> Dict:
        """
        Run Monte Carlo simulation to estimate strategy robustness
        """
        def simulate_sequence(seed: int) -> Tuple[float, float]:
            np.random.seed(seed)
            shuffled_pnls = np.random.permutation([t.pnl for t in trades])
            capital = initial_capital
            equity_curve = [capital]
            
            for pnl in shuffled_pnls:
                capital += pnl
                equity_curve.append(capital)
                
            return (
                (capital - initial_capital) / initial_capital,  # Return
                min(0, min(np.diff(equity_curve)))  # Max drawdown
            )
        
        with mp.Pool(self.cpu_count) as pool:
            results = pool.map(simulate_sequence, range(self.num_monte_carlo_sims))
            
        returns, drawdowns = zip(*results)
        
        return {
            'return_mean': np.mean(returns),
            'return_std': np.std(returns),
            'return_95ci': (np.percentile(returns, 2.5),
                          np.percentile(returns, 97.5)),
            'drawdown_mean': np.mean(drawdowns),
            'drawdown_95ci': (np.percentile(drawdowns, 2.5),
                            np.percentile(drawdowns, 97.5)),
            'probability_profit': np.mean(np.array(returns) > 0)
        }
    
    def _calculate_metrics(self, trades: List[TradeResult], 
                         initial_capital: float) -> BacktestMetrics:
        """
        Calculate comprehensive trading metrics
        """
        if not trades:
            return BacktestMetrics(
                total_return=0,
                sharpe_ratio=0,
                sortino_ratio=0,
                max_drawdown=0,
                win_rate=0,
                profit_factor=0,
                trades=[]
            )
            
        # Calculate returns and drawdown
        pnls = [t.pnl for t in trades]
        cumulative_pnls = np.cumsum(pnls)
        total_return = cumulative_pnls[-1] / initial_capital
        
        # Calculate drawdown
        rolling_max = np.maximum.accumulate(cumulative_pnls)
        drawdowns = (rolling_max - cumulative_pnls) / rolling_max
        max_drawdown = np.max(drawdowns)
        
        # Calculate risk metrics
        returns = np.array(pnls) / initial_capital
        excess_returns = returns - self.risk_free_rate / 252  # Daily risk-free rate
        
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        downside_returns = returns[returns < 0]
        downside_vol = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-9
        
        sharpe_ratio = np.mean(excess_returns) / volatility if volatility != 0 else 0
        sortino_ratio = np.mean(excess_returns) / downside_vol if downside_vol != 0 else 0
        
        # Trading statistics
        winning_trades = [t for t in trades if t.pnl > 0]
        win_rate = len(winning_trades) / len(trades)
        
        gross_profits = sum(t.pnl for t in winning_trades)
        gross_losses = abs(sum(t.pnl for t in trades if t.pnl < 0))
        profit_factor = gross_profits / gross_losses if gross_losses != 0 else float('inf')
        
        # Run Monte Carlo simulation
        monte_carlo_stats = self.run_monte_carlo_simulation(trades, initial_capital)
        
        return BacktestMetrics(
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_drawdown,
            win_rate=win_rate,
            profit_factor=profit_factor,
            trades=trades,
            monte_carlo_stats=monte_carlo_stats
        )
    
    def generate_report(self, metrics: BacktestMetrics) -> Dict:
        """
        Generate comprehensive backtest report
        """
        # Basic performance metrics
        performance = {
            'total_return': f"{metrics.total_return * 100:.2f}%",
            'sharpe_ratio': f"{metrics.sharpe_ratio:.2f}",
            'sortino_ratio': f"{metrics.sortino_ratio:.2f}",
            'max_drawdown': f"{metrics.max_drawdown * 100:.2f}%",
            'win_rate': f"{metrics.win_rate * 100:.2f}%",
            'profit_factor': f"{metrics.profit_factor:.2f}",
            'number_of_trades': len(metrics.trades)
        }
        
        # Trade analysis
        trade_durations = [t.trade_duration.total_seconds() / 3600 for t in metrics.trades]
        trade_sizes = [t.position_size for t in metrics.trades]
        
        trade_stats = {
            'avg_trade_duration_hours': f"{np.mean(trade_durations):.2f}",
            'avg_position_size': f"{np.mean(trade_sizes):.2f}",
            'avg_profit_per_trade': f"{np.mean([t.pnl for t in metrics.trades]):.2f}",
            'largest_win': f"{max([t.pnl for t in metrics.trades]):.2f}",
            'largest_loss': f"{min([t.pnl for t in metrics.trades]):.2f}",
            'avg_max_drawdown': f"{np.mean([t.max_drawdown for t in metrics.trades]) * 100:.2f}%"
        }
        
        # Monte Carlo simulation results
        monte_carlo = {
            'expected_return': f"{metrics.monte_carlo_stats['return_mean'] * 100:.2f}%",
            'return_95ci': (
                f"{metrics.monte_carlo_stats['return_95ci'][0] * 100:.2f}%",
                f"{metrics.monte_carlo_stats['return_95ci'][1] * 100:.2f}%"
            ),
            'expected_max_drawdown': f"{abs(metrics.monte_carlo_stats['drawdown_mean']) * 100:.2f}%",
            'drawdown_95ci': (
                f"{abs(metrics.monte_carlo_stats['drawdown_95ci'][0]) * 100:.2f}%",
                f"{abs(metrics.monte_carlo_stats['drawdown_95ci'][1]) * 100:.2f}%"
            ),
            'probability_profit': f"{metrics.monte_carlo_stats['probability_profit'] * 100:.2f}%"
        }
        
        return {
            'performance_metrics': performance,
            'trade_statistics': trade_stats,
            'monte_carlo_analysis': monte_carlo,
            'timestamp': datetime.now().isoformat()
        }