from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import numpy as np

class TradingStyle(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"

class MarketCondition(Enum):
    TRENDING = "trending"
    RANGING = "ranging"
    VOLATILE = "volatile"

@dataclass
class StrategyParameters:
    name: str
    risk_level: TradingStyle
    market_condition: MarketCondition
    indicators: List[str]
    timeframes: List[str]
    stop_loss_multiplier: float
    take_profit_multiplier: float
    signal_confidence_threshold: int
    sentiment_weight: float
    description: str
    adaptations: List[str]

class StrategyTemplate:
    def __init__(self, pair: str):
        self.pair = pair
        self.parameters = self._get_default_parameters()

    def _get_default_parameters(self) -> Dict[str, StrategyParameters]:
        return {
            "swing_trader": StrategyParameters(
                name="Swing Trader",
                risk_level=TradingStyle.MODERATE,
                market_condition=MarketCondition.TRENDING,
                indicators=["MACD", "RSI", "EMA"],
                timeframes=["1h", "4h", "1d"],
                stop_loss_multiplier=1.5,
                take_profit_multiplier=2.0,
                signal_confidence_threshold=75,
                sentiment_weight=0.3,
                description="Captures medium to long-term price movements using trend-following indicators",
                adaptations=["Trail stops on strong trends", "Scale in during pullbacks"]
            ),
            "scalper": StrategyParameters(
                name="Scalper",
                risk_level=TradingStyle.AGGRESSIVE,
                market_condition=MarketCondition.RANGING,
                indicators=["Bollinger Bands", "Stochastic", "CCI"],
                timeframes=["1m", "5m", "15m"],
                stop_loss_multiplier=0.8,
                take_profit_multiplier=1.2,
                signal_confidence_threshold=85,
                sentiment_weight=0.1,
                description="Quick in-and-out trades capitalizing on small price movements",
                adaptations=["Tight stops", "Quick profit taking", "High-frequency trading"]
            ),
            "position_trader": StrategyParameters(
                name="Position Trader",
                risk_level=TradingStyle.CONSERVATIVE,
                market_condition=MarketCondition.TRENDING,
                indicators=["EMA", "ADX", "PSAR"],
                timeframes=["4h", "1d", "1w"],
                stop_loss_multiplier=2.0,
                take_profit_multiplier=3.0,
                signal_confidence_threshold=70,
                sentiment_weight=0.5,
                description="Long-term trades following major market trends",
                adaptations=["Wide stops for noise", "Multiple take-profit levels"]
            ),
            "breakout_trader": StrategyParameters(
                name="Breakout Trader",
                risk_level=TradingStyle.AGGRESSIVE,
                market_condition=MarketCondition.VOLATILE,
                indicators=["ATR", "Volume", "Support/Resistance"],
                timeframes=["15m", "1h", "4h"],
                stop_loss_multiplier=1.2,
                take_profit_multiplier=2.5,
                signal_confidence_threshold=80,
                sentiment_weight=0.4,
                description="Capitalizes on strong price movements after consolidation",
                adaptations=["Volume confirmation", "Momentum-based entries"]
            )
        }

    def get_strategy(self, name: str, risk_level: Optional[TradingStyle] = None) -> StrategyParameters:
        """Get strategy parameters with optional risk level customization"""
        params = self.parameters.get(name)
        if params and risk_level:
            # Adjust parameters based on risk level
            if risk_level == TradingStyle.CONSERVATIVE:
                params.stop_loss_multiplier *= 1.5
                params.take_profit_multiplier *= 0.8
                params.signal_confidence_threshold += 10
            elif risk_level == TradingStyle.AGGRESSIVE:
                params.stop_loss_multiplier *= 0.7
                params.take_profit_multiplier *= 1.5
                params.signal_confidence_threshold -= 10
        return params

    def optimize_strategy(self, params: StrategyParameters, sentiment_data: Dict, market_volatility: float) -> StrategyParameters:
        """Optimize strategy parameters based on market conditions and sentiment"""
        optimized = StrategyParameters(**params.__dict__)
        
        # Adjust based on market volatility
        if market_volatility > 0.8:  # High volatility
            optimized.stop_loss_multiplier *= 1.5
            optimized.take_profit_multiplier *= 1.3
            optimized.signal_confidence_threshold += 10
        elif market_volatility < 0.2:  # Low volatility
            optimized.stop_loss_multiplier *= 0.8
            optimized.take_profit_multiplier *= 0.9
            optimized.signal_confidence_threshold -= 5

        # Adjust based on sentiment
        sentiment_score = sentiment_data.get('composite_score', 0.5)
        if sentiment_score > 0.7:  # Strong positive sentiment
            optimized.take_profit_multiplier *= 1.2
            optimized.sentiment_weight *= 1.3
        elif sentiment_score < 0.3:  # Strong negative sentiment
            optimized.stop_loss_multiplier *= 1.2
            optimized.sentiment_weight *= 0.7

        return optimized

    def backtest_strategy(self, params: StrategyParameters, historical_data: pd.DataFrame) -> Dict:
        """Backtest strategy with given parameters on historical data"""
        if len(historical_data) < 100:  # Minimum data required
            return {
                'trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
                'max_drawdown': 0.0, 'sharpe_ratio': 0.0, 'avg_trade': 0.0
            }

        # Initialize tracking variables
        trades = []
        position = None
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        equity_curve = [100]  # Start with 100 units
        wins = 0
        losses = 0
        total_profit = 0
        total_loss = 0
        
        # Calculate technical indicators based on strategy
        signals = self._calculate_strategy_signals(historical_data, params)
        
        # Simulate trading
        for i in range(1, len(historical_data)):
            current_price = historical_data['close'].iloc[i]
            current_signal = signals.iloc[i]
            
            # Check for exit if in position
            if position:
                if (position == 'long' and 
                    (current_price <= stop_loss or current_price >= take_profit)):
                    # Close long position
                    profit = (current_price - entry_price) / entry_price
                    trades.append(profit)
                    equity_curve.append(equity_curve[-1] * (1 + profit))
                    
                    if profit > 0:
                        wins += 1
                        total_profit += profit
                    else:
                        losses += 1
                        total_loss += abs(profit)
                    
                    position = None
                    
                elif (position == 'short' and 
                      (current_price >= stop_loss or current_price <= take_profit)):
                    # Close short position
                    profit = (entry_price - current_price) / entry_price
                    trades.append(profit)
                    equity_curve.append(equity_curve[-1] * (1 + profit))
                    
                    if profit > 0:
                        wins += 1
                        total_profit += profit
                    else:
                        losses += 1
                        total_loss += abs(profit)
                    
                    position = None
            
            # Check for entry if not in position
            elif not position:
                if current_signal['signal'] == 'buy':
                    position = 'long'
                    entry_price = current_price
                    stop_loss = entry_price * (1 - params.stop_loss_multiplier)
                    take_profit = entry_price * (1 + params.take_profit_multiplier)
                    
                elif current_signal['signal'] == 'sell':
                    position = 'short'
                    entry_price = current_price
                    stop_loss = entry_price * (1 + params.stop_loss_multiplier)
                    take_profit = entry_price * (1 - params.take_profit_multiplier)
        
        # Calculate performance metrics
        total_trades = wins + losses
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        profit_factor = (total_profit / total_loss) if total_loss > 0 else 0
        
        # Calculate max drawdown
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        # Calculate Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
        returns = np.diff(equity_curve) / equity_curve[:-1]
        sharpe_ratio = np.sqrt(252) * (np.mean(returns) / np.std(returns)) if len(returns) > 0 else 0
        
        # Calculate average trade return
        avg_trade = np.mean(trades) if trades else 0
        
        return {
            'trades': total_trades,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'max_drawdown': max_drawdown * 100,  # Convert to percentage
            'sharpe_ratio': sharpe_ratio,
            'avg_trade': avg_trade * 100,  # Convert to percentage
            'equity_curve': equity_curve,
            'final_balance': equity_curve[-1]
        }

    def _calculate_strategy_signals(self, data: pd.DataFrame, params: StrategyParameters) -> pd.DataFrame:
        """Calculate trading signals based on strategy indicators"""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 'hold'  # Default to hold
        
        for indicator in params.indicators:
            if indicator == "MACD":
                # MACD signal logic
                ema12 = data['close'].ewm(span=12).mean()
                ema26 = data['close'].ewm(span=26).mean()
                macd = ema12 - ema26
                signal_line = macd.ewm(span=9).mean()
                signals.loc[macd > signal_line, 'signal'] = 'buy'
                signals.loc[macd < signal_line, 'signal'] = 'sell'
                
            elif indicator == "RSI":
                # RSI signal logic
                delta = data['close'].diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                signals.loc[rsi < 30, 'signal'] = 'buy'
                signals.loc[rsi > 70, 'signal'] = 'sell'
                
            elif indicator == "EMA":
                # EMA crossover logic
                ema20 = data['close'].ewm(span=20).mean()
                ema50 = data['close'].ewm(span=50).mean()
                signals.loc[ema20 > ema50, 'signal'] = 'buy'
                signals.loc[ema20 < ema50, 'signal'] = 'sell'
                
            elif indicator == "Bollinger Bands":
                # Bollinger Bands logic
                sma20 = data['close'].rolling(window=20).mean()
                std20 = data['close'].rolling(window=20).std()
                upper_band = sma20 + (std20 * 2)
                lower_band = sma20 - (std20 * 2)
                signals.loc[data['close'] < lower_band, 'signal'] = 'buy'
                signals.loc[data['close'] > upper_band, 'signal'] = 'sell'
                
            elif indicator == "PSAR":
                # Parabolic SAR logic (simplified)
                close = data['close']
                high = data.get('high', close)
                low = data.get('low', close)
                af = 0.02
                max_af = 0.2
                psar = close.copy()
                is_long = True
                
                for i in range(1, len(close)):
                    if is_long:
                        psar[i] = max(psar[i-1], low[i-1])
                        if close[i] < psar[i]:
                            is_long = False
                            signals.iloc[i]['signal'] = 'sell'
                    else:
                        psar[i] = min(psar[i-1], high[i-1])
                        if close[i] > psar[i]:
                            is_long = True
                            signals.iloc[i]['signal'] = 'buy'
        
        return signals
