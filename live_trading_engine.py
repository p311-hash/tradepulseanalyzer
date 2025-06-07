"""
Live Trading Engine with Comprehensive Risk Management
Handles order execution, position management, and real-time monitoring.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import json
from decimal import Decimal, ROUND_HALF_UP
import config
from enhanced_data_reliability import EnhancedDataReliabilityManager

logger = logging.getLogger(__name__)

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"

class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"

class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"

@dataclass
class Order:
    id: str
    symbol: str
    side: str  # 'buy' or 'sell'
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    status: OrderStatus = OrderStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    filled_quantity: float = 0.0
    filled_price: Optional[float] = None
    commission: float = 0.0

@dataclass
class Position:
    symbol: str
    side: PositionSide
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    opened_at: datetime = field(default_factory=datetime.now)

@dataclass
class RiskLimits:
    max_position_size: float = 0.02  # 2% of account
    max_daily_loss: float = 0.05     # 5% of account
    max_drawdown: float = 0.10       # 10% of account
    max_correlation: float = 0.7     # Maximum correlation between positions
    max_positions: int = 5           # Maximum concurrent positions

class LiveTradingEngine:
    """
    Comprehensive live trading engine with risk management.
    """

    def __init__(self, initial_balance: float = 10000.0):
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.trade_history: List[Dict] = []
        self.risk_limits = RiskLimits()
        self.data_manager = EnhancedDataReliabilityManager()

        # Performance tracking
        self.daily_pnl = 0.0
        self.total_pnl = 0.0
        self.max_drawdown = 0.0
        self.peak_balance = initial_balance

        # Risk monitoring
        self.is_trading_enabled = True
        self.last_risk_check = datetime.now()

        # Commission settings
        self.commission_rate = 0.001  # 0.1% commission

    async def execute_signal(self, signal: Dict) -> Optional[str]:
        """
        Execute a trading signal with comprehensive risk checks.

        Args:
            signal: Trading signal dictionary

        Returns:
            Order ID if successful, None if rejected
        """
        try:
            # Pre-execution risk checks
            if not self._pre_execution_risk_check(signal):
                logger.warning("Signal rejected by pre-execution risk check")
                return None

            # Calculate position size
            position_size = self._calculate_position_size(signal)
            if position_size <= 0:
                logger.warning("Position size calculation resulted in zero or negative size")
                return None

            # Create order
            order = self._create_order_from_signal(signal, position_size)

            # Execute order
            success = await self._execute_order(order)

            if success:
                logger.info(f"Successfully executed order {order.id} for {signal['symbol']}")
                return order.id
            else:
                logger.error(f"Failed to execute order for {signal['symbol']}")
                return None

        except Exception as e:
            logger.error(f"Error executing signal: {e}")
            return None

    def _pre_execution_risk_check(self, signal: Dict) -> bool:
        """Comprehensive pre-execution risk validation."""
        try:
            # Check if trading is enabled
            if not self.is_trading_enabled:
                logger.warning("Trading is disabled")
                return False

            # Check signal quality
            if signal.get('confidence', 0) < 0.6:  # Minimum 60% confidence
                logger.warning(f"Signal confidence too low: {signal.get('confidence', 0)}")
                return False

            # Check daily loss limit
            if self.daily_pnl < -self.risk_limits.max_daily_loss * self.initial_balance:
                logger.warning("Daily loss limit exceeded")
                self.is_trading_enabled = False
                return False

            # Check maximum drawdown
            current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            if current_drawdown > self.risk_limits.max_drawdown:
                logger.warning("Maximum drawdown exceeded")
                self.is_trading_enabled = False
                return False

            # Check maximum positions
            if len(self.positions) >= self.risk_limits.max_positions:
                logger.warning("Maximum number of positions reached")
                return False

            # Check for existing position in same symbol
            symbol = signal['symbol']
            if symbol in self.positions:
                logger.warning(f"Already have position in {symbol}")
                return False

            # Check correlation with existing positions
            if not self._check_correlation_risk(signal):
                logger.warning("Correlation risk too high")
                return False

            return True

        except Exception as e:
            logger.error(f"Error in pre-execution risk check: {e}")
            return False

    def _calculate_position_size(self, signal: Dict) -> float:
        """Calculate optimal position size based on risk management."""
        try:
            # Base position size as percentage of account
            base_size = self.risk_limits.max_position_size * self.current_balance

            # Adjust based on signal confidence
            confidence_multiplier = signal.get('confidence', 0.5)
            adjusted_size = base_size * confidence_multiplier

            # Adjust based on volatility (if available)
            if 'volatility' in signal:
                volatility = signal['volatility']
                # Reduce size for high volatility
                volatility_multiplier = max(0.5, 1.0 - volatility)
                adjusted_size *= volatility_multiplier

            # Ensure minimum and maximum limits
            min_size = 10.0  # Minimum $10 position
            max_size = self.risk_limits.max_position_size * self.current_balance

            final_size = max(min_size, min(adjusted_size, max_size))

            logger.info(f"Calculated position size: ${final_size:.2f} for {signal['symbol']}")
            return final_size

        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0

    def _create_order_from_signal(self, signal: Dict, position_size: float) -> Order:
        """Create an order object from a trading signal."""
        order_id = f"order_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{signal['symbol']}"

        # Determine order side
        side = 'buy' if signal['direction'].upper() in ['BUY', 'LONG'] else 'sell'

        # For now, use market orders for simplicity
        order = Order(
            id=order_id,
            symbol=signal['symbol'],
            side=side,
            order_type=OrderType.MARKET,
            quantity=position_size,
            price=signal.get('current_price')
        )

        self.orders[order_id] = order
        return order

    async def _execute_order(self, order: Order) -> bool:
        """Execute an order through the trading API."""
        try:
            # Get current market data
            current_data = await self.data_manager.get_reliable_data(
                order.symbol, '1m', min_sources=1
            )

            if current_data is None or current_data.empty:
                logger.error(f"No market data available for {order.symbol}")
                order.status = OrderStatus.REJECTED
                return False

            # Get current price
            current_price = float(current_data['close'].iloc[-1])

            # Simulate order execution (replace with actual API calls)
            execution_price = self._simulate_order_execution(order, current_price)

            if execution_price is None:
                order.status = OrderStatus.REJECTED
                return False

            # Fill the order
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_quantity = order.quantity
            order.filled_price = execution_price
            order.commission = order.quantity * self.commission_rate

            # Create position
            self._create_position_from_order(order)

            # Update balance
            self.current_balance -= order.commission

            # Record trade
            self._record_trade(order)

            logger.info(f"Order {order.id} executed at ${execution_price:.4f}")
            return True

        except Exception as e:
            logger.error(f"Error executing order {order.id}: {e}")
            order.status = OrderStatus.REJECTED
            return False

    def _simulate_order_execution(self, order: Order, market_price: float) -> Optional[float]:
        """Simulate order execution with realistic slippage."""
        try:
            # Add realistic slippage (0.01% to 0.05%)
            slippage_pct = np.random.uniform(0.0001, 0.0005)

            if order.side == 'buy':
                execution_price = market_price * (1 + slippage_pct)
            else:
                execution_price = market_price * (1 - slippage_pct)

            return execution_price

        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return None

    def _create_position_from_order(self, order: Order):
        """Create a position from a filled order."""
        try:
            side = PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT

            position = Position(
                symbol=order.symbol,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price
            )

            # Set stop loss and take profit based on risk management
            self._set_risk_management_levels(position)

            self.positions[order.symbol] = position
            logger.info(f"Created {side.value} position for {order.symbol}")

        except Exception as e:
            logger.error(f"Error creating position: {e}")

    def _set_risk_management_levels(self, position: Position):
        """Set stop loss and take profit levels."""
        try:
            # Default risk-reward ratio of 1:2
            risk_pct = 0.02  # 2% risk
            reward_pct = 0.04  # 4% reward

            if position.side == PositionSide.LONG:
                position.stop_loss = position.entry_price * (1 - risk_pct)
                position.take_profit = position.entry_price * (1 + reward_pct)
            else:
                position.stop_loss = position.entry_price * (1 + risk_pct)
                position.take_profit = position.entry_price * (1 - reward_pct)

            logger.info(f"Set SL: {position.stop_loss:.4f}, TP: {position.take_profit:.4f}")

        except Exception as e:
            logger.error(f"Error setting risk management levels: {e}")

    def _record_trade(self, order: Order):
        """Record trade in history."""
        trade_record = {
            'timestamp': order.filled_at.isoformat(),
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.filled_price,
            'commission': order.commission,
            'balance_after': self.current_balance
        }

        self.trade_history.append(trade_record)

    def _check_correlation_risk(self, signal: Dict) -> bool:
        """Check if new position would create excessive correlation risk."""
        # Simplified correlation check - in production, use actual correlation calculation
        return len(self.positions) < self.risk_limits.max_positions

    def _simulate_order_execution(self, order: Order, market_price: float) -> Optional[float]:
        """Simulate order execution with realistic slippage."""
        try:
            # Add realistic slippage (0.01% to 0.05%)
            slippage_pct = np.random.uniform(0.0001, 0.0005)

            if order.side == 'buy':
                execution_price = market_price * (1 + slippage_pct)
            else:
                execution_price = market_price * (1 - slippage_pct)

            return execution_price

        except Exception as e:
            logger.error(f"Error simulating order execution: {e}")
            return None

    def _create_position_from_order(self, order: Order):
        """Create a position from a filled order."""
        try:
            side = PositionSide.LONG if order.side == 'buy' else PositionSide.SHORT

            position = Position(
                symbol=order.symbol,
                side=side,
                quantity=order.filled_quantity,
                entry_price=order.filled_price,
                current_price=order.filled_price
            )

            # Set stop loss and take profit based on risk management
            self._set_risk_management_levels(position)

            self.positions[order.symbol] = position
            logger.info(f"Created {side.value} position for {order.symbol}")

        except Exception as e:
            logger.error(f"Error creating position: {e}")

    def _set_risk_management_levels(self, position: Position):
        """Set stop loss and take profit levels."""
        try:
            # Default risk-reward ratio of 1:2
            risk_pct = 0.02  # 2% risk
            reward_pct = 0.04  # 4% reward

            if position.side == PositionSide.LONG:
                position.stop_loss = position.entry_price * (1 - risk_pct)
                position.take_profit = position.entry_price * (1 + reward_pct)
            else:
                position.stop_loss = position.entry_price * (1 + risk_pct)
                position.take_profit = position.entry_price * (1 - reward_pct)

            logger.info(f"Set SL: {position.stop_loss:.4f}, TP: {position.take_profit:.4f}")

        except Exception as e:
            logger.error(f"Error setting risk management levels: {e}")

    def _record_trade(self, order: Order):
        """Record trade in history."""
        trade_record = {
            'timestamp': order.filled_at.isoformat(),
            'order_id': order.id,
            'symbol': order.symbol,
            'side': order.side,
            'quantity': order.filled_quantity,
            'price': order.filled_price,
            'commission': order.commission,
            'balance_after': self.current_balance
        }

        self.trade_history.append(trade_record)

    async def _monitor_positions(self):
        """Continuously monitor positions for stop loss and take profit."""
        while True:
            try:
                await self._update_positions()
                await self._check_risk_management()
                await asyncio.sleep(5)  # Check every 5 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in position monitoring: {e}")
                await asyncio.sleep(10)

    async def _update_positions(self):
        """Update current prices and P&L for all positions."""
        for symbol, position in list(self.positions.items()):
            try:
                # Get current market data
                current_data = await self.data_manager.get_reliable_data(
                    symbol, '1m', min_sources=1
                )

                if current_data is not None and not current_data.empty:
                    current_price = float(current_data['close'].iloc[-1])
                    position.current_price = current_price

                    # Calculate unrealized P&L
                    if position.side == PositionSide.LONG:
                        position.unrealized_pnl = (current_price - position.entry_price) * position.quantity
                    else:
                        position.unrealized_pnl = (position.entry_price - current_price) * position.quantity

            except Exception as e:
                logger.error(f"Error updating position {symbol}: {e}")

    async def _check_risk_management(self):
        """Check stop loss and take profit levels."""
        positions_to_close = []

        for symbol, position in self.positions.items():
            try:
                should_close = False
                close_reason = ""

                # Check stop loss
                if position.stop_loss:
                    if position.side == PositionSide.LONG and position.current_price <= position.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss"
                    elif position.side == PositionSide.SHORT and position.current_price >= position.stop_loss:
                        should_close = True
                        close_reason = "Stop Loss"

                # Check take profit
                if position.take_profit and not should_close:
                    if position.side == PositionSide.LONG and position.current_price >= position.take_profit:
                        should_close = True
                        close_reason = "Take Profit"
                    elif position.side == PositionSide.SHORT and position.current_price <= position.take_profit:
                        should_close = True
                        close_reason = "Take Profit"

                if should_close:
                    positions_to_close.append((symbol, close_reason))

            except Exception as e:
                logger.error(f"Error checking risk management for {symbol}: {e}")

        # Close positions that hit stop loss or take profit
        for symbol, reason in positions_to_close:
            await self._close_position(symbol, reason)

    async def _close_position(self, symbol: str, reason: str = "Manual"):
        """Close a position."""
        try:
            if symbol not in self.positions:
                logger.warning(f"No position found for {symbol}")
                return

            position = self.positions[symbol]

            # Create closing order
            close_side = 'sell' if position.side == PositionSide.LONG else 'buy'
            order_id = f"close_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{symbol}"

            close_order = Order(
                id=order_id,
                symbol=symbol,
                side=close_side,
                order_type=OrderType.MARKET,
                quantity=position.quantity,
                price=position.current_price
            )

            # Execute closing order
            success = await self._execute_order(close_order)

            if success:
                # Calculate realized P&L
                realized_pnl = position.unrealized_pnl - close_order.commission

                # Update account balance
                self.current_balance += realized_pnl
                self.total_pnl += realized_pnl
                self.daily_pnl += realized_pnl

                # Update peak balance
                if self.current_balance > self.peak_balance:
                    self.peak_balance = self.current_balance

                # Remove position
                del self.positions[symbol]

                logger.info(f"Closed position {symbol} - {reason} - P&L: ${realized_pnl:.2f}")

        except Exception as e:
            logger.error(f"Error closing position {symbol}: {e}")

    def get_account_summary(self) -> Dict:
        """Get comprehensive account summary."""
        try:
            total_unrealized_pnl = sum(pos.unrealized_pnl for pos in self.positions.values())
            equity = self.current_balance + total_unrealized_pnl

            return {
                'balance': self.current_balance,
                'equity': equity,
                'unrealized_pnl': total_unrealized_pnl,
                'realized_pnl': self.total_pnl,
                'daily_pnl': self.daily_pnl,
                'positions_count': len(self.positions),
                'max_drawdown': self.max_drawdown,
                'trading_enabled': self.is_trading_enabled,
                'positions': [
                    {
                        'symbol': pos.symbol,
                        'side': pos.side.value,
                        'quantity': pos.quantity,
                        'entry_price': pos.entry_price,
                        'current_price': pos.current_price,
                        'unrealized_pnl': pos.unrealized_pnl,
                        'stop_loss': pos.stop_loss,
                        'take_profit': pos.take_profit
                    }
                    for pos in self.positions.values()
                ]
            }

        except Exception as e:
            logger.error(f"Error getting account summary: {e}")
            return {}
