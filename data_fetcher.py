"""
Data fetcher module for TradePulse Signals bot
Responsible for retrieving market data from various APIs
"""
import os
import logging
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from config import MARKET_DATA_API, FOREX_API, STOCK_API
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

logger = logging.getLogger(__name__)

def fetch_crypto_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch cryptocurrency market data from CoinGecko API
    """
    url = f"{MARKET_DATA_API}/coins/markets"
    params = {
        "vs_currency": "usd",
        "ids": symbol,
        "order": "market_cap_desc",
        "per_page": 1,
        "page": 1,
        "sparkline": False,
        "price_change_percentage": "24h"
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if data and isinstance(data, list) and len(data) > 0:
            return data[0]
        else:
            logger.warning(f"No data returned for crypto symbol: {symbol}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching crypto data for {symbol}: {e}")
        return None

def fetch_forex_data(base_currency: str) -> Optional[Dict[str, Any]]:
    """
    Fetch forex data from Exchange Rate API
    """
    try:
        response = requests.get(FOREX_API)
        response.raise_for_status()
        data = response.json()
        
        if "rates" not in data:
            logger.warning("No rates data in forex API response")
            return None
        
        # Get the exchange rate for the base currency
        if base_currency in data["rates"]:
            rate = data["rates"][base_currency]
            
            # Calculate inverse for USD pair (e.g., EUR/USD)
            inverse_rate = 1 / rate
            
            # Construct response similar to crypto data format for consistency
            return {
                "current_price": inverse_rate,
                "price_change_percentage_24h": 0,  # Would need historical data for this
                "total_volume": 0  # Would need additional API for volume data
            }
        else:
            logger.warning(f"Currency {base_currency} not found in forex rates")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching forex data for {base_currency}: {e}")
        return None

def fetch_stock_data(symbol: str) -> Optional[Dict[str, Any]]:
    """
    Fetch stock data from Alpha Vantage API
    """
    api_key = os.environ.get("ALPHA_VANTAGE_API_KEY", "demo")
    url = STOCK_API
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "apikey": api_key
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        if "Global Quote" in data and data["Global Quote"]:
            return data
        else:
            logger.warning(f"No data returned for stock symbol: {symbol}")
            return None
            
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching stock data for {symbol}: {e}")
        return None

class EnhancedDataFetcher:
    """Enhanced data fetcher with multiple data source support and validation."""
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_KEY')
        self.data_cache = {}
        self.trade_cache = {}
        self.order_book_cache = {}
        self.sources = {
            'forex': self._fetch_forex_data,
            'stocks': self._fetch_stock_data,
            'crypto': self._fetch_crypto_data
        }
        
    def fetch_market_data(self, 
                         symbol: str,
                         timeframe: str,
                         source: str = 'auto',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch market data from multiple sources with automatic fallback.
        
        Args:
            symbol: Trading pair or stock symbol
            timeframe: Timeframe for the data
            source: Data source ('auto', 'alpha_vantage', 'yahoo', 'crypto')
            start_date: Start date for historical data
            end_date: End date for historical data
        """
        try:
            if source == 'auto':
                source = self._determine_best_source(symbol)
            
            data = self.sources[source](symbol, timeframe, start_date, end_date)
            
            if data.empty:
                logger.warning(f"No data returned for {symbol} from {source}")
                return pd.DataFrame()
            
            # Validate and clean data
            data = self._validate_and_clean_data(data)
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {str(e)}")
            return pd.DataFrame()
    
    def fetch_data(self, symbol: str, timeframe: str = '1h', lookback_periods: int = 1000) -> pd.DataFrame:
        """Fetch OHLCV market data"""
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.data_cache:
            cached_data = self.data_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 60:  # 1-minute cache
                return cached_data['data']
                
        try:
            if '_' in symbol:  # Crypto pair
                data = self._fetch_crypto_data(symbol, timeframe, lookback_periods)
            else:  # Stock/Forex
                data = self._fetch_traditional_data(symbol, timeframe, lookback_periods)
                
            if data is not None and not data.empty:
                self.data_cache[cache_key] = {
                    'data': data,
                    'timestamp': datetime.now()
                }
                return data
                
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_trades(self, symbol: str, timeframe: str = '1h') -> pd.DataFrame:
        """Fetch individual trade data for deep market structure analysis"""
        cache_key = f"{symbol}_trades_{timeframe}"
        
        if cache_key in self.trade_cache:
            cached_data = self.trade_cache[cache_key]
            if (datetime.now() - cached_data['timestamp']).seconds < 30:  # 30-second cache
                return cached_data['data']
                
        try:
            # Convert timeframe to minutes for calculation
            minutes = self._timeframe_to_minutes(timeframe)
            end_time = datetime.now()
            start_time = end_time - timedelta(minutes=minutes)
            
            trades = self._fetch_trade_data(symbol, start_time, end_time)
            
            if trades is not None and not trades.empty:
                self.trade_cache[cache_key] = {
                    'data': trades,
                    'timestamp': datetime.now()
                }
                return trades
                
            return pd.DataFrame()
            
        except Exception as e:
            logger.error(f"Error fetching trade data for {symbol}: {e}")
            return pd.DataFrame()
            
    def fetch_order_book(self, symbol: str) -> Dict:
        """Fetch real-time order book data"""
        if symbol in self.order_book_cache:
            cached_data = self.order_book_cache[symbol]
            if (datetime.now() - cached_data['timestamp']).seconds < 5:  # 5-second cache
                return cached_data['data']
                
        try:
            order_book = self._fetch_order_book_data(symbol)
            
            if order_book:
                self.order_book_cache[symbol] = {
                    'data': order_book,
                    'timestamp': datetime.now()
                }
                return order_book
                
            return {'bids': [], 'asks': []}
            
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'bids': [], 'asks': []}
            
    def _fetch_forex_data(self, 
                         symbol: str,
                         timeframe: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch forex data from Alpha Vantage."""
        try:
            if not self.alpha_vantage_key:
                raise ValueError("Alpha Vantage API key not found")
                
            base, quote = symbol.split('_')[0][:3], symbol.split('_')[0][3:6]
            
            url = (
                f"https://www.alphavantage.co/query?"
                f"function=FX_INTRADAY&"
                f"from_symbol={base}&"
                f"to_symbol={quote}&"
                f"interval={timeframe}&"
                f"apikey={self.alpha_vantage_key}&"
                f"outputsize=full"
            )
            
            response = requests.get(url)
            data = response.json()
            
            if 'Time Series FX' not in data:
                logger.error(f"Invalid response from Alpha Vantage: {data}")
                return pd.DataFrame()
                
            df = pd.DataFrame(data['Time Series FX']).T
            df.columns = ['open', 'high', 'low', 'close', 'volume']
            df = df.astype(float)
            df.index = pd.to_datetime(df.index)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forex data: {str(e)}")
            return pd.DataFrame()
    
    def _fetch_stock_data(self,
                         symbol: str,
                         timeframe: str,
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance."""
        try:
            ticker = yf.Ticker(symbol)
            interval = self._convert_timeframe(timeframe)
            
            data = ticker.history(
                interval=interval,
                start=start_date,
                end=end_date
            )
            
            return data
            
        except Exception as e:
            logger.error(f"Error fetching stock data: {str(e)}")
            return pd.DataFrame()
    
    def _validate_and_clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean market data."""
        try:
            # Remove duplicates
            data = data[~data.index.duplicated(keep='last')]
            
            # Sort by timestamp
            data = data.sort_index()
            
            # Handle missing values
            data = data.fillna(method='ffill')
            
            # Remove outliers
            for col in ['open', 'high', 'low', 'close']:
                if col in data.columns:
                    mean = data[col].mean()
                    std = data[col].std()
                    data = data[
                        (data[col] > mean - 4*std) & 
                        (data[col] < mean + 4*std)
                    ]
            
            # Ensure required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in data.columns:
                    data[col] = 0
            
            return data
            
        except Exception as e:
            logger.error(f"Error validating data: {str(e)}")
            return data
    
    def _determine_best_source(self, symbol: str) -> str:
        """Determine the best data source for a symbol."""
        if '_' in symbol:  # Forex pairs typically use underscore
            return 'forex'
        elif any(crypto in symbol.upper() for crypto in ['BTC', 'ETH', 'USDT']):
            return 'crypto'
        else:
            return 'stocks'
    
    def _convert_timeframe(self, timeframe: str) -> str:
        """Convert internal timeframe format to provider-specific format."""
        conversions = {
            '1m': '1m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '4h': '4h',
            '1d': '1d'
        }
        return conversions.get(timeframe, '1m')
    
    def _fetch_trade_data(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch individual trades from exchange API"""
        try:
            if '_' in symbol:  # Crypto pair
                trades = self._fetch_crypto_trades(symbol, start_time, end_time)
            else:  # Stock/Forex
                trades = self._fetch_traditional_trades(symbol, start_time, end_time)
                
            return trades
            
        except Exception as e:
            logger.error(f"Error fetching trade data: {e}")
            return pd.DataFrame()
    
    def _timeframe_to_minutes(self, timeframe: str) -> int:
        """Convert timeframe string to minutes"""
        unit = timeframe[-1].lower()
        value = int(timeframe[:-1])
        
        if unit == 'm':
            return value
        elif unit == 'h':
            return value * 60
        elif unit == 'd':
            return value * 1440
        else:
            raise ValueError(f"Unsupported timeframe unit: {unit}")
    
    def _fetch_crypto_trades(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch crypto trades from exchange API"""
        # Implement exchange-specific API calls here
        # For demonstration, return simulated data
        n_trades = 1000
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_trades)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.normal(100, 1, n_trades),
            'volume': np.random.exponential(1, n_trades),
            'side': np.random.choice(['buy', 'sell'], n_trades)
        })
    
    def _fetch_traditional_trades(self, symbol: str, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetch stock/forex trades from market data provider"""
        # Implement broker-specific API calls here
        # For demonstration, return simulated data
        n_trades = 1000
        timestamps = pd.date_range(start=start_time, end=end_time, periods=n_trades)
        
        return pd.DataFrame({
            'timestamp': timestamps,
            'price': np.random.normal(100, 1, n_trades),
            'volume': np.random.exponential(1, n_trades),
            'side': np.random.choice(['buy', 'sell'], n_trades)
        })
