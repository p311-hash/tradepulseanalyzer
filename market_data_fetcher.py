"""
Real market data fetcher for binary options signals
"""
import ccxt
import pandas as pd
import logging
import random
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

class MarketDataFetcher:
    def __init__(self):
        try:
            self.exchange = ccxt.pro_binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        except Exception:
            self.exchange = ccxt.binance({
                'enableRateLimit': True,
                'options': {'defaultType': 'future'}
            })
        
        self.timeframe_map = {
            '1m': '1m',
            '2m': '2m',
            '3m': '3m',
            '5m': '5m',
            '15m': '15m',
            '30m': '30m',
            '1h': '1h',
            '2h': '2h',
            '4h': '4h',
            '6h': '6h',
            '8h': '8h',
            '12h': '12h',
            '1d': '1d'
        }

    async def fetch_market_data(self, symbol: str, timeframe: str, limit: int = 100) -> pd.DataFrame:
        """Fetch real market data for a given symbol and timeframe."""
        try:
            # Convert forex pairs to crypto format
            if '_otc' in symbol:
                symbol = symbol.replace('_otc', '')
            symbol = symbol.replace('/', '')
            if not symbol.endswith('USDT'):
                symbol += 'USDT'

            # Fetch OHLCV data
            ohlcv = await self.exchange.fetch_ohlcv(
                symbol,
                timeframe=self.timeframe_map.get(timeframe, '1m'),
                limit=limit
            )

            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            return df

        except Exception as e:
            logger.error(f"Error fetching market data: {str(e)}")
            return self._generate_mock_data(limit)

    def _generate_mock_data(self, limit: int = 100) -> pd.DataFrame:
        """Generate mock data only if real data fetch fails."""
        current_price = 1.1000
        timestamps = pd.date_range(end=datetime.now(), periods=limit, freq='1min')
        
        data = []
        for _ in range(limit):
            current_price *= (1 + (random.random() - 0.5) * 0.002)
            high_price = current_price * (1 + random.random() * 0.001)
            low_price = current_price * (1 - random.random() * 0.001)
            data.append([
                high_price,
                low_price,
                current_price * (1 + (random.random() - 0.5) * 0.001),  # open
                current_price,  # close
                random.random() * 100000  # volume
            ])
        
        df = pd.DataFrame(data, columns=['high', 'low', 'open', 'close', 'volume'], index=timestamps)
        return df
