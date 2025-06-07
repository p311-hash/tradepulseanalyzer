"""Enhanced risk management system with portfolio optimization and dynamic position sizing."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from scipy import stats
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PositionSizing:
    size: float
    confidence: float
    risk_factors: Dict[str, float]

@dataclass
class RiskMetrics:
    var_95: float
    cvar_95: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    correlation_risk: float

class RiskManager:
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.max_position_size = 0.02  # 2% of capital per trade
        self.max_portfolio_risk = 0.05  # 5% portfolio risk
        self.correlation_threshold = 0.7
        self.position_history = []
        self.risk_metrics_history = {}
        self.drawdown_threshold = 0.1  # 10% max drawdown
        self.volatility_lookback = 20
        self.confidence_level = 0.95
        
    def calculate_position_size(self, symbol: str, current_price: float,
                              atr: float, regime_multiplier: float = 1.0) -> PositionSizing:
        """Calculate dynamic position size based on multiple risk factors"""
        # Base position size using ATR for volatility adjustment
        volatility_factor = self._calculate_volatility_factor(atr)
        base_size = self.current_capital * self.max_position_size * volatility_factor
        
        # Calculate portfolio correlation risk
        correlation_risk = self._calculate_correlation_risk(symbol)
        
        # Calculate Value at Risk
        var_factor = self._calculate_var_factor(symbol, current_price)
        
        # Check drawdown risk
        drawdown_factor = self._check_drawdown_risk()
        
        # Combine risk factors
        risk_factors = {
            'volatility': volatility_factor,
            'correlation': correlation_risk,
            'var': var_factor,
            'drawdown': drawdown_factor,
            'regime': regime_multiplier
        }
        
        # Calculate final position size
        total_risk_factor = (
            0.3 * volatility_factor +
            0.2 * (1 - correlation_risk) +
            0.2 * var_factor +
            0.2 * drawdown_factor +
            0.1 * regime_multiplier
        )
        
        final_size = base_size * total_risk_factor
        
        # Calculate confidence in position size
        confidence = self._calculate_size_confidence(risk_factors)
        
        return PositionSizing(
            size=min(final_size, self.current_capital * self.max_position_size),
            confidence=confidence,
            risk_factors=risk_factors
        )
        
    def calculate_portfolio_risk_metrics(self, data: pd.DataFrame) -> Dict:
        """Calculate comprehensive risk metrics for the portfolio."""
        try:
            # Calculate returns from price data
            returns = data['close'].pct_change().dropna()
            
            # Calculate volatility
            volatility = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate Value at Risk (VaR)
            var_95 = np.percentile(returns, 5)
            
            # Calculate Expected Shortfall (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Calculate Sharpe Ratio (assuming risk-free rate of 2%)
            rf_rate = 0.02
            excess_returns = returns - (rf_rate / 252)
            sharpe = np.sqrt(252) * excess_returns.mean() / returns.std()
            
            # Calculate Sortino Ratio
            downside_returns = returns[returns < 0]
            sortino = np.sqrt(252) * excess_returns.mean() / downside_returns.std()
            
            # Calculate maximum drawdown
            cum_returns = (1 + returns).cumprod()
            rolling_max = cum_returns.expanding().max()
            drawdowns = (cum_returns - rolling_max) / rolling_max
            max_drawdown = drawdowns.min()
            
            return {
                'volatility': volatility,
                'value_at_risk_95': var_95,
                'expected_shortfall_95': cvar_95,
                'sharpe_ratio': sharpe,
                'sortino_ratio': sortino,
                'max_drawdown': max_drawdown
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {str(e)}")
            return {
                'volatility': 0.0,
                'value_at_risk_95': 0.0,
                'expected_shortfall_95': 0.0,
                'sharpe_ratio': 0.0,
                'sortino_ratio': 0.0,
                'max_drawdown': 0.0
            }
        
    def update_capital(self, pnl: float):
        """Update current capital and check risk limits"""
        self.current_capital += pnl
        
        # Calculate drawdown
        drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        
        if drawdown > self.drawdown_threshold:
            logger.warning(f"Drawdown threshold exceeded: {drawdown:.2%}")
            return False
            
        return True
        
    def initialize_settings(self, settings: Dict) -> None:
        """Initialize or update risk management settings.
        
        Args:
            settings: Dictionary containing risk management parameters:
                - max_position_size_pct: Maximum position size as percentage of capital
                - stop_loss_pct: Stop loss percentage per trade
                - take_profit_pct: Take profit percentage per trade
                - max_daily_trades: Maximum number of trades per day
                - max_daily_loss_pct: Maximum daily loss percentage
                - risk_per_trade_pct: Risk percentage per trade
        """
        try:
            # Convert percentage values to decimals
            self.max_position_size = settings.get('max_position_size_pct', 5.0) / 100
            self.max_portfolio_risk = settings.get('risk_per_trade_pct', 2.0) / 100
            self.drawdown_threshold = settings.get('max_daily_loss_pct', 15.0) / 100
            
            # Store additional settings
            self.max_daily_trades = settings.get('max_daily_trades', 10)
            self.stop_loss_pct = settings.get('stop_loss_pct', 2.0) / 100
            self.take_profit_pct = settings.get('take_profit_pct', 4.0) / 100
            
            logger.info("Risk management settings initialized successfully")
            logger.debug(f"Max position size: {self.max_position_size*100}%")
            logger.debug(f"Max portfolio risk: {self.max_portfolio_risk*100}%")
            logger.debug(f"Max drawdown: {self.drawdown_threshold*100}%")
            
        except Exception as e:
            logger.error(f"Error initializing risk management settings: {str(e)}")
            # Set safe default values if initialization fails
            self.max_position_size = 0.02  # 2%
            self.max_portfolio_risk = 0.02  # 2%
            self.drawdown_threshold = 0.10  # 10%
            self.max_daily_trades = 10
            self.stop_loss_pct = 0.02  # 2%
            self.take_profit_pct = 0.04  # 4%
        
    def _calculate_volatility_factor(self, atr: float) -> float:
        """Calculate position sizing factor based on volatility"""
        # Normalize ATR
        norm_atr = atr / self.current_capital
        
        # Inverse relationship with volatility
        return 1 / (1 + norm_atr * 100)
        
    def _calculate_correlation_risk(self, symbol: str) -> float:
        """Calculate correlation risk with existing positions"""
        if not self.position_history:
            return 0.0
            
        # Get returns for current symbol and existing positions
        correlations = []
        for pos in self.position_history[-self.volatility_lookback:]:
            if pos['symbol'] != symbol:
                corr = np.corrcoef(
                    pos['returns'],
                    pos['benchmark_returns']
                )[0,1]
                correlations.append(abs(corr))
                
        if not correlations:
            return 0.0
            
        return np.mean(correlations)
        
    def _calculate_var_factor(self, symbol: str, current_price: float) -> float:
        """Calculate position sizing factor based on Value at Risk"""
        if not self.position_history:
            return 1.0
            
        # Get historical returns
        returns = []
        for pos in self.position_history:
            if pos['symbol'] == symbol:
                returns.extend(pos['returns'])
                
        if not returns:
            return 1.0
            
        # Calculate VaR
        var_95 = np.percentile(returns, 5)  # 95% confidence level
        
        # Convert VaR to factor (higher VaR = lower factor)
        return 1 / (1 + abs(var_95) * 100)
        
    def _check_drawdown_risk(self) -> float:
        """Calculate position sizing factor based on drawdown risk"""
        current_drawdown = (self.initial_capital - self.current_capital) / self.initial_capital
        
        # Reduce position size as drawdown increases
        if current_drawdown > self.drawdown_threshold:
            return 0.5
        elif current_drawdown > self.drawdown_threshold / 2:
            return 0.75
        else:
            return 1.0
            
    def _calculate_size_confidence(self, risk_factors: Dict[str, float]) -> float:
        """Calculate confidence level in position sizing decision"""
        # Weight different factors
        weights = {
            'volatility': 0.3,
            'correlation': 0.2,
            'var': 0.2,
            'drawdown': 0.2,
            'regime': 0.1
        }
        
        confidence = sum(
            factor * weights[name]
            for name, factor in risk_factors.items()
        )
        
        return min(confidence, 1.0)
        
    def _calculate_portfolio_returns(self, positions: List[Dict]) -> np.ndarray:
        """Calculate historical portfolio returns"""
        if not positions:
            return np.array([])
            
        # Combine position returns weighted by size
        total_size = sum(pos['size'] for pos in positions)
        weighted_returns = []
        
        for pos in positions:
            weight = pos['size'] / total_size
            weighted_returns.append(pos['returns'] * weight)
            
        return np.sum(weighted_returns, axis=0)
        
    def _calculate_var(self, returns: np.ndarray) -> float:
        """Calculate Value at Risk"""
        if len(returns) == 0:
            return 0.0
            
        return np.percentile(returns, (1 - self.confidence_level) * 100)
        
    def _calculate_cvar(self, returns: np.ndarray) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) == 0:
            return 0.0
            
        var_95 = self._calculate_var(returns)
        return np.mean(returns[returns <= var_95])
        
    def _calculate_sharpe_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sharpe Ratio"""
        if len(returns) == 0:
            return 0.0
            
        return np.mean(returns) / (np.std(returns) + 1e-6)
        
    def _calculate_sortino_ratio(self, returns: np.ndarray) -> float:
        """Calculate Sortino Ratio"""
        if len(returns) == 0:
            return 0.0
            
        negative_returns = returns[returns < 0]
        downside_std = np.std(negative_returns) if len(negative_returns) > 0 else 1e-6
        
        return np.mean(returns) / (downside_std + 1e-6)
        
    def _calculate_max_drawdown(self, returns: np.ndarray) -> float:
        """Calculate Maximum Drawdown"""
        if len(returns) == 0:
            return 0.0
            
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        
        return np.max(drawdowns)
        
    def _calculate_portfolio_correlation(self, positions: List[Dict]) -> float:
        """Calculate average correlation between portfolio positions"""
        if len(positions) < 2:
            return 0.0
            
        correlations = []
        n = len(positions)
        
        for i in range(n):
            for j in range(i+1, n):
                corr = np.corrcoef(
                    positions[i]['returns'],
                    positions[j]['returns']
                )[0,1]
                correlations.append(abs(corr))
                
        return np.mean(correlations)
        
    def analyze_risk_metrics_history(self) -> Dict:
        """Analyze historical risk metrics for trends"""
        if not self.risk_metrics_history:
            return {}
            
        metrics_df = pd.DataFrame.from_dict(self.risk_metrics_history, orient='index')
        
        return {
            'var_trend': metrics_df['var_95'].diff().mean(),
            'sharpe_trend': metrics_df['sharpe_ratio'].diff().mean(),
            'drawdown_trend': metrics_df['max_drawdown'].diff().mean(),
            'correlation_trend': metrics_df['correlation_risk'].diff().mean()
        }
