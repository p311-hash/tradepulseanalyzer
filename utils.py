"""
Utility functions for TradePulse Signals bot
"""
import os
import json
import logging
import pandas as pd
import numpy as np
import requests
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import time
import random
import config

logger = logging.getLogger(__name__)

def format_signal_message(signal: Dict[str, Any]) -> str:
    """
    Format a signal into a readable message for Telegram
    """
    # Add emoji based on signal direction
    direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴" if signal["direction"] == "SELL" else "⚪"

    message = (
        f"{direction_emoji} *{signal['asset']}*\n"
        f"Signal: *{signal['direction']}* ({signal['strength']})\n"
        f"Price: ${signal['price']:.2f}\n"
        f"Generated: {signal['timestamp']}"
    )

    return message

def get_signal_emoji(direction: str) -> str:
    """
    Get appropriate emoji for signal direction
    """
    if direction == "BUY":
        return "🟢"
    elif direction == "SELL":
        return "🔴"
    else:  # HOLD
        return "⚪"

def ensure_directories() -> None:
    """
    Ensure all required directories exist
    """
    directories = ["data"]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Ensured directory exists: {directory}")

def is_market_open() -> bool:
    """
    Check if markets are currently open
    This is a simplified implementation that can be expanded based on actual requirements
    """
    # For crypto (24/7 market)
    # Always return True
    return True

    # For traditional markets (stocks), we would implement actual market hours check
    # Example (commented out):
    # now = datetime.now()
    # # Check if it's a weekday (0 = Monday, 4 = Friday)
    # if now.weekday() > 4:
    #    return False
    # # Check if within market hours (9:30 AM to 4:00 PM EST)
    # market_open = now.replace(hour=9, minute=30, second=0)
    # market_close = now.replace(hour=16, minute=0, second=0)
    # return market_open <= now <= market_close

def validate_asset(asset: str) -> bool:
    """
    Validate if an asset string is properly formatted
    """
    # Crypto or forex pair format (e.g. BTC/USD)
    if '/' in asset:
        parts = asset.split('/')
        return len(parts) == 2 and all(p.isalpha() for p in parts)

    # Stock ticker format (e.g. AAPL)
    return asset.isalpha()

def setup_logging():
    """
    Configure logging for the application
    """
    log_level = getattr(logging, config.LOG_LEVEL.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=config.LOG_FORMAT,
        handlers=[
            logging.StreamHandler()
        ]
    )
    logger.info(f"Logging initialized at level {config.LOG_LEVEL}")

def is_using_real_data() -> bool:
    """
    Check if the bot is using real market data or simulated data
    """
    # Check environment variable for real data setting
    use_real_data = os.environ.get('USE_REAL_DATA', 'false').lower() == 'true'
    return use_real_data

def get_api_client():
    """
    Get the API client for trading platform integration
    Returns None if not configured
    """
    # This is a placeholder for actual API client implementation
    # In a real implementation, this would create and return a client object
    # for the trading platform (e.g., PocketOption, IQ Option, etc.)
    return None

def fetch_market_data(symbol: str, timeframe: str, periods: int = 100) -> pd.DataFrame:
    """Enhanced market data fetcher with real data integration."""
    try:
        # Try to use real market data fetcher first
        try:
            import asyncio
            from real_market_data_fetcher import ProfessionalMarketDataFetcher
            
            # Create fetcher instance
            fetcher = ProfessionalMarketDataFetcher()
            
            # Run async fetch in sync context
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            real_data = loop.run_until_complete(
                fetcher.fetch_real_market_data(symbol, timeframe, periods)
            )
            loop.close()
            
            if real_data is not None and not real_data.empty:
                logger.info(f"✅ Using REAL market data for {symbol} {timeframe}")
                return real_data
            else:
                logger.warning(f"⚠️ Real data unavailable, using enhanced simulation for {symbol}")
                
        except Exception as e:
            logger.error(f"Real data fetch failed: {e}")
        
        # Enhanced fallback simulation (much more realistic)
        logger.info(f"🔄 Using ENHANCED simulation for {symbol} {timeframe}")
        return _generate_professional_simulation(symbol, timeframe, periods)
        
    except Exception as e:
        logger.error(f"Critical error in fetch_market_data: {e}")
        return _generate_basic_fallback(symbol, timeframe, periods)

def _generate_professional_simulation(symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
    """Generate professional-grade market simulation."""
    try:
        # Professional base prices (updated to current market levels)
        base_prices = {
            'EURUSD_otc': 1.0847,
            'GBPUSD_otc': 1.2634,
            'USDJPY_otc': 149.82,
            'USDCAD_otc': 1.3592,
            'AUDUSD_otc': 0.6418,
            'NZDUSD_otc': 0.5887,
            'USDCHF_otc': 0.9124
        }
        
        # Professional volatility models
        volatility_profiles = {
            'EURUSD_otc': {'base': 0.0008, 'trend_strength': 0.6, 'mean_reversion': 0.3},
            'GBPUSD_otc': {'base': 0.0015, 'trend_strength': 0.8, 'mean_reversion': 0.2},
            'USDJPY_otc': {'base': 0.0012, 'trend_strength': 0.7, 'mean_reversion': 0.25},
            'USDCAD_otc': {'base': 0.0010, 'trend_strength': 0.5, 'mean_reversion': 0.4},
            'AUDUSD_otc': {'base': 0.0014, 'trend_strength': 0.6, 'mean_reversion': 0.3},
            'NZDUSD_otc': {'base': 0.0016, 'trend_strength': 0.7, 'mean_reversion': 0.2},
            'USDCHF_otc': {'base': 0.0009, 'trend_strength': 0.4, 'mean_reversion': 0.5}
        }
        
        base_price = base_prices.get(symbol, 1.1000)
        vol_profile = volatility_profiles.get(symbol, {'base': 0.0010, 'trend_strength': 0.5, 'mean_reversion': 0.3})
        
        # Timeframe adjustments
        timeframe_factors = {
            '5s': 0.2, '15s': 0.3, '30s': 0.4,
            '1m': 0.6, '2m': 0.7, '3m': 0.8,
            '5m': 1.0, '10m': 1.2, '15m': 1.4, '30m': 1.8, '1h': 2.2
        }
        
        vol_multiplier = timeframe_factors.get(timeframe, 1.0)
        volatility = vol_profile['base'] * vol_multiplier
        
        # Generate professional price series
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        # Market regime simulation
        regime = np.random.choice(['trending_up', 'trending_down', 'ranging', 'volatile'], 
                                p=[0.25, 0.25, 0.35, 0.15])
        
        prices = []
        current_price = base_price
        
        # Regime-specific parameters
        if regime == 'trending_up':
            trend_strength = 0.0002
            noise_level = 0.7
        elif regime == 'trending_down':
            trend_strength = -0.0002
            noise_level = 0.7
        elif regime == 'ranging':
            trend_strength = 0.0
            noise_level = 0.5
        else:  # volatile
            trend_strength = 0.0
            noise_level = 1.5
        
        for i in range(periods):
            # Trend component
            trend_component = trend_strength
            
            # Random walk with regime-specific noise
            random_component = np.random.normal(0, volatility * noise_level)
            
            # Mean reversion
            mean_reversion = (base_price - current_price) * vol_profile['mean_reversion'] * 0.01
            
            # Market microstructure noise
            microstructure_noise = np.random.normal(0, volatility * 0.1)
            
            # Combine all components
            total_change = trend_component + random_component + mean_reversion + microstructure_noise
            current_price += total_change
            prices.append(current_price)
        
        # Generate professional OHLCV data
        data = []
        for i, close_price in enumerate(prices):
            open_price = prices[i-1] if i > 0 else close_price
            
            # Professional spread modeling
            spread_factor = np.random.uniform(0.3, 1.2)
            max_spread = volatility * spread_factor
            
            # Generate high and low with realistic distribution
            high_offset = np.random.exponential(max_spread * 0.3)
            low_offset = np.random.exponential(max_spread * 0.3)
            
            high = max(open_price, close_price) + high_offset
            low = min(open_price, close_price) - low_offset
            
            # Ensure OHLC consistency
            high = max(high, open_price, close_price)
            low = min(low, open_price, close_price)
            
            # Professional volume modeling
            base_volume = 1500
            volume_volatility = 0.8
            time_factor = 1.0 + 0.3 * np.sin(2 * np.pi * i / 24)  # Daily volume pattern
            volume = int(base_volume * time_factor * np.random.lognormal(0, volume_volatility))
            
            data.append({
                'timestamp': dates[i],
                'open': round(open_price, 5),
                'high': round(high, 5),
                'low': round(low, 5),
                'close': round(close_price, 5),
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Generated professional simulation: {symbol} {timeframe} ({regime} regime)")
        return df
        
    except Exception as e:
        logger.error(f"Error in professional simulation: {e}")
        return _generate_basic_fallback(symbol, timeframe, periods)

def _generate_basic_fallback(symbol: str, timeframe: str, periods: int) -> pd.DataFrame:
    """Basic fallback when everything else fails."""
    try:
        base_price = 1.1000 if 'EUR' in symbol else 100.0
        dates = pd.date_range(end=datetime.now(), periods=periods, freq='1min')
        
        data = []
        for i, date in enumerate(dates):
            price = base_price * (1 + np.random.normal(0, 0.001))
            data.append({
                'timestamp': date,
                'open': price,
                'high': price * 1.001,
                'low': price * 0.999,
                'close': price,
                'volume': 1000
            })
        
        df = pd.DataFrame(data)
        df.set_index('timestamp', inplace=True)
        return df
        
    except Exception as e:
        logger.error(f"Critical fallback error: {e}")
        return pd.DataFrame()
    
def _generate_simulated_data(pair: str, timeframe: str, num_candles: int) -> pd.DataFrame:
    """
    Generate simulated market data for testing

    Args:
        pair: Currency pair
        timeframe: Timeframe string
        num_candles: Number of candles to generate

    Returns:
        DataFrame with simulated OHLCV data
    """
    # Get timeframe in seconds
    tf_seconds = config.TIMEFRAMES.get(timeframe, 60)  # Default to 1 minute

    # Generate timestamps
    end_time = int(time.time())
    timestamps = [end_time - (i * tf_seconds) for i in range(num_candles)]
    timestamps.reverse()  # Ascending order

    # Generate price data with some randomness but overall trend
    base_price = 1.0
    if 'EUR' in pair:
        base_price = 1.1
    elif 'GBP' in pair:
        base_price = 1.3
    elif 'JPY' in pair:
        base_price = 100.0

    # Add some randomness to starting price
    price = base_price * (1 + random.uniform(-0.02, 0.02))

    # Generate OHLCV data
    data = []
    for timestamp in timestamps:
        # Small random change
        change = random.uniform(-0.001, 0.001)

        # Larger trend component
        trend = 0.0002 * np.sin(timestamp / 10000)

        # Update price
        price = price * (1 + change + trend)

        # Generate candle
        open_price = price
        high_price = open_price * (1 + random.uniform(0, 0.001))
        low_price = open_price * (1 - random.uniform(0, 0.001))
        close_price = open_price * (1 + random.uniform(-0.0008, 0.0008))
        volume = random.randint(100, 1000)

        data.append({
            'timestamp': timestamp,
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': close_price,
            'volume': volume
        })

    # Create DataFrame
    df = pd.DataFrame(data)
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')

    return df
