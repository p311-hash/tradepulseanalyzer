"""
Data aggregator module for combining and validating data from multiple sources.
Implements cross-validation and data quality checks to reduce false signals.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
import logging
from datetime import datetime, timedelta
import aiohttp
import asyncio
from scipy import stats
from collections import defaultdict
import time
from data_fetcher import fetch_market_data
import config

logger = logging.getLogger(__name__)

# Rate limiting settings
RATE_LIMITS = {
    'crypto_compare': {
        'requests': 50,
        'per_minute': 1,
        'burst_limit': 10,  # Max requests in burst
        'burst_duration': 10  # Seconds
    },
    'alpha_vantage': {
        'requests': 5,
        'per_minute': 1,
        'burst_limit': 2,
        'burst_duration': 10
    },
    'yahoo_finance': {
        'requests': 100,
        'per_minute': 1,
        'burst_limit': 20,
        'burst_duration': 10
    }
}

# Timeframe normalization mapping
TIMEFRAME_MAP = {
    '1m': timedelta(minutes=1),
    '5m': timedelta(minutes=5),
    '15m': timedelta(minutes=15),
    '30m': timedelta(minutes=30),
    '1h': timedelta(hours=1),
    '4h': timedelta(hours=4),
    'D': timedelta(days=1)
}

class DataAggregator:
    """Aggregates and validates data from multiple sources."""

    def __init__(self):
        self.data_sources = {
            'primary': self._fetch_primary_data,
            'alternative': self._fetch_alternative_data,
            'crypto_compare': self._fetch_crypto_compare_data,
            'alpha_vantage': self._fetch_alpha_vantage_data
        }
        self.data_cache = {}
        self.cache_duration = timedelta(minutes=5)

        # Rate limiting tracking
        self.request_counts = defaultdict(lambda: {'count': 0, 'reset_time': time.time()})

    async def _fetch_primary_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from primary source."""
        try:
            return await fetch_market_data(pair=symbol, timeframe=timeframe)
        except Exception as e:
            logger.error(f"Error fetching primary data: {str(e)}")
            return pd.DataFrame()

    async def _fetch_alternative_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Yahoo Finance as an alternative source."""
        try:
            # Convert timeframe to Yahoo Finance interval
            interval_map = {
                '1m': '1m',
                '5m': '5m',
                '15m': '15m',
                '30m': '30m',
                '1h': '1h',
                '4h': '4h',
                'D': '1d'
            }

            yf_interval = interval_map.get(timeframe)
            if not yf_interval:
                logger.error(f"Unsupported timeframe for Yahoo Finance: {timeframe}")
                return pd.DataFrame()

            # Calculate period based on timeframe
            periods = {
                '1m': '7d',   # Yahoo limits 1m data to 7 days
                '5m': '60d',
                '15m': '60d',
                '30m': '60d',
                '1h': '730d', # 2 years
                '4h': '730d',
                'D': '1825d'  # 5 years
            }

            async with aiohttp.ClientSession() as session:
                # Format symbol for Yahoo Finance
                # For forex: EURUSD=X
                # For crypto: BTC-USD
                # For stocks: Use as is
                formatted_symbol = symbol
                if len(symbol) == 6 and all(c.isalpha() for c in symbol):  # Forex pair
                    formatted_symbol = f"{symbol}=X"
                elif symbol.endswith('USD') or symbol.endswith('USDT'):  # Crypto
                    formatted_symbol = f"{symbol[:-3]}-USD"

                url = f"https://query1.finance.yahoo.com/v8/finance/chart/{formatted_symbol}"
                params = {
                    "interval": yf_interval,
                    "period1": int((datetime.now() - timedelta(days=int(periods[timeframe][:3]))).timestamp()),
                    "period2": int(datetime.now().timestamp())
                }

                async with session.get(url, params=params) as response:
                    if response.status != 200:
                        logger.error(f"Yahoo Finance API error: {response.status}")
                        return pd.DataFrame()

                    data = await response.json()
                    if 'chart' not in data or not data['chart']['result']:
                        logger.error("No data returned from Yahoo Finance")
                        return pd.DataFrame()

                    result = data['chart']['result'][0]
                    quotes = result['indicators']['quote'][0]
                    timestamps = pd.to_datetime(result['timestamp'], unit='s')

                    df = pd.DataFrame({
                        'open': quotes.get('open', []),
                        'high': quotes.get('high', []),
                        'low': quotes.get('low', []),
                        'close': quotes.get('close', []),
                        'volume': quotes.get('volume', [])
                    }, index=timestamps)

                    # Clean up NaN values if any
                    df.fillna(method='ffill', inplace=True)
                    df = self._normalize_timeframe(df, timeframe)
                    return df

        except Exception as e:
            logger.error(f"Error fetching alternative data: {str(e)}")
            return pd.DataFrame()

    async def _fetch_crypto_compare_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from CryptoCompare."""
        try:
            if not await self._check_rate_limit('crypto_compare'):
                logger.warning("Rate limit exceeded for CryptoCompare")
                return pd.DataFrame()

            async with aiohttp.ClientSession() as session:
                url = f"https://min-api.cryptocompare.com/data/v2/histominute"
                params = {
                    "fsym": symbol[:3],
                    "tsym": symbol[3:],
                    "limit": 2000,
                    "api_key": config.CRYPTO_COMPARE_API_KEY
                }
                async with session.get(url, params=params) as response:
                    if response.status == 429:  # Too Many Requests
                        logger.warning("CryptoCompare rate limit hit")
                        return pd.DataFrame()

                    data = await response.json()
                    if data.get('Response') == 'Success':
                        df = pd.DataFrame(data['Data']['Data'])
                        df['timestamp'] = pd.to_datetime(df['time'], unit='s')
                        df.set_index('timestamp', inplace=True)
                        df = self._normalize_timeframe(df, timeframe)
                        return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching CryptoCompare data: {str(e)}")
            return pd.DataFrame()

    async def _fetch_alpha_vantage_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """Fetch data from Alpha Vantage."""
        try:
            if not await self._check_rate_limit('alpha_vantage'):
                logger.warning("Rate limit exceeded for Alpha Vantage")
                return pd.DataFrame()

            async with aiohttp.ClientSession() as session:
                url = "https://www.alphavantage.co/query"
                params = {
                    "function": "FX_INTRADAY",
                    "from_symbol": symbol[:3],
                    "to_symbol": symbol[3:],
                    "interval": timeframe,
                    "apikey": config.ALPHA_VANTAGE_API_KEY,
                    "outputsize": "full"
                }
                async with session.get(url, params=params) as response:
                    if response.status == 429:  # Too Many Requests
                        logger.warning("Alpha Vantage rate limit hit")
                        return pd.DataFrame()

                    data = await response.json()
                    if 'Time Series FX' in data:
                        df = pd.DataFrame.from_dict(data['Time Series FX'], orient='index')
                        df.index = pd.to_datetime(df.index)

                        # Rename columns to standard format
                        column_map = {
                            '1. open': 'open',
                            '2. high': 'high',
                            '3. low': 'low',
                            '4. close': 'close',
                            '5. volume': 'volume'
                        }
                        df.rename(columns=column_map, inplace=True)

                        # Convert string values to float
                        for col in ['open', 'high', 'low', 'close', 'volume']:
                            df[col] = pd.to_numeric(df[col], errors='coerce')

                        df = self._normalize_timeframe(df, timeframe)
                        return df
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error fetching Alpha Vantage data: {str(e)}")
            return pd.DataFrame()

    async def _fetch_with_retry(self, fetch_func, symbol: str, timeframe: str, max_retries: int = 3) -> pd.DataFrame:
        """
        Execute a fetch function with retry logic and exponential backoff.

        Args:
            fetch_func: The async function to fetch data
            symbol: Trading symbol
            timeframe: Data timeframe
            max_retries: Maximum number of retry attempts

        Returns:
            DataFrame with fetched data or empty DataFrame on failure
        """
        for attempt in range(max_retries):
            try:
                df = await fetch_func(symbol, timeframe)
                if not df.empty:
                    return df

                # If we got an empty DataFrame, wait before retry
                wait_time = (2 ** attempt) * 1.5  # Exponential backoff: 1.5s, 3s, 6s
                logger.warning(f"Attempt {attempt + 1} failed, retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

            except Exception as e:
                wait_time = (2 ** attempt) * 1.5
                logger.error(f"Error in attempt {attempt + 1}: {str(e)}")
                logger.warning(f"Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)

        logger.error(f"All {max_retries} attempts failed")
        return pd.DataFrame()

    async def _check_rate_limit(self, source: str) -> bool:
        """
        Enhanced rate limit checking with burst protection.

        Args:
            source: The data source to check rate limits for

        Returns:
            bool: True if request is allowed, False if rate limited
        """
        if source not in RATE_LIMITS:
            return True

        current_time = time.time()
        tracking = self.request_counts[source]
        limits = RATE_LIMITS[source]

        # Initialize burst tracking if not present
        if 'burst_count' not in tracking:
            tracking.update({
                'burst_count': 0,
                'burst_start': current_time,
                'minute_count': 0,
                'minute_start': current_time
            })

        # Reset minute counter if we're in a new minute
        if current_time - tracking['minute_start'] >= 60:
            tracking['minute_count'] = 0
            tracking['minute_start'] = current_time

        # Reset burst counter if burst duration has passed
        if current_time - tracking['burst_start'] >= limits['burst_duration']:
            tracking['burst_count'] = 0
            tracking['burst_start'] = current_time

        # Check both minute and burst limits
        if tracking['minute_count'] >= limits['requests']:
            logger.warning(f"Minute rate limit reached for {source}")
            return False

        if tracking['burst_count'] >= limits['burst_limit']:
            logger.warning(f"Burst rate limit reached for {source}")
            return False

        # Update counters
        tracking['minute_count'] += 1
        tracking['burst_count'] += 1

        return True

    def _validate_data_quality(self, data: pd.DataFrame) -> bool:
        """
        Validate data quality with comprehensive checks.

        Args:
            data: DataFrame to validate

        Returns:
            bool: True if data passes quality checks
        """
        try:
            if data.empty:
                logger.error("Empty dataset")
                return False

            # Check for required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(data.columns):
                logger.error(f"Missing required columns. Found: {data.columns}")
                return False

            # Check for sufficient data points
            min_data_points = 100
            if len(data) < min_data_points:
                logger.error(f"Insufficient data points: {len(data)} < {min_data_points}")
                return False

            # Check for NaN values
            nan_pct = data[list(required_columns)].isnull().mean()
            if (nan_pct > 0.01).any():  # Allow max 1% NaN values
                logger.error(f"Too many NaN values: {nan_pct}")
                return False

            # Check price continuity
            returns = data['close'].pct_change()
            # Detect gaps (extreme returns)
            extreme_returns = abs(returns) > 0.5  # 50% price change
            if extreme_returns.sum() > len(data) * 0.01:  # Allow max 1% extreme returns
                logger.error("Too many extreme price changes detected")
                return False

            # Check OHLC relationship consistency
            if ((data['high'] < data['low']) |
                (data['close'] > data['high']) |
                (data['close'] < data['low']) |
                (data['open'] > data['high']) |
                (data['open'] < data['low'])).any():
                logger.error("OHLC relationship violation detected")
                return False

            # Check timestamp continuity
            time_diffs = pd.Series(data.index).diff()
            expected_diff = pd.Timedelta(data.index[1] - data.index[0])
            irregular_intervals = abs(time_diffs - expected_diff) > pd.Timedelta('1min')
            if irregular_intervals.sum() > len(data) * 0.01:  # Allow max 1% irregular intervals
                logger.error("Too many irregular time intervals detected")
                return False

            # Check for zero/negative values where inappropriate
            if ((data[['high', 'low', 'close', 'volume']] <= 0).any()).any():
                logger.error("Invalid zero or negative values detected")
                return False

            # Price staleness check
            price_changes = (data['close'] != data['close'].shift()).mean()
            if price_changes < 0.001:  # Less than 0.1% of prices changing
                logger.warning("Potentially stale price data detected")
                return False

            # Volume activity check
            zero_volume_pct = (data['volume'] == 0).mean()
            if zero_volume_pct > 0.5:  # More than 50% zero volume
                logger.warning("Insufficient trading activity detected")
                return False

            # Basic statistical checks
            for col in ['open', 'high', 'low', 'close']:
                zscore = np.abs(stats.zscore(data[col]))
                if zscore[-1] > 4:  # Last value is an extreme outlier
                    logger.error(f"Extreme outlier detected in {col}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Error validating data quality: {str(e)}")
            return False

    async def get_aggregated_data(self, symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
        """
        Get aggregated and validated data from multiple sources.

        Args:
            symbol: Trading pair symbol
            timeframe: Timeframe string

        Returns:
            Validated and aggregated DataFrame or None if data is invalid
        """
        try:
            if timeframe not in TIMEFRAME_MAP:
                logger.error(f"Unsupported timeframe: {timeframe}")
                return None

            cache_key = f"{symbol}_{timeframe}"

            # Check cache
            if cache_key in self.data_cache:
                cached_time, cached_data = self.data_cache[cache_key]
                if datetime.now() - cached_time < self.cache_duration:
                    is_valid, reason = self._validate_consistency(cached_data)
                    if is_valid:
                        return cached_data
                    else:
                        logger.warning(f"Cached data validation failed: {reason}")

            # Fetch data from all sources concurrently with retry
            tasks = []
            for _, fetch_func in self.data_sources.items():
                tasks.append(self._fetch_with_retry(fetch_func, symbol, timeframe))

            # Wait for all data sources with timeout
            try:
                data_frames = await asyncio.wait_for(asyncio.gather(*tasks), timeout=30.0)
            except asyncio.TimeoutError:
                logger.error("Timeout while fetching data from sources")
                data_frames = [pd.DataFrame() for _ in self.data_sources]

            # Filter and validate frames
            validated_frames = []
            for df in data_frames:
                if df.empty:
                    continue

                # Validate data quality
                if not self._validate_data_quality(df):
                    continue

                # Validate data consistency
                is_valid, reason = self._validate_consistency(df)
                if not is_valid:
                    logger.warning(f"Data consistency check failed: {reason}")
                    continue

                validated_frames.append(df)

            if not validated_frames:
                logger.error("No valid data from any source")

            if not validated_frames:
                logger.error("No data passed validation")
                return None

            # Align timestamps and combine data
            aligned_frames = []
            for df in validated_frames:
                df = df.resample(timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                })
                aligned_frames.append(df)

            # Enhanced data combination with outlier detection and weighted average
            combined_data = self._combine_data_sources(aligned_frames)

            if combined_data is None:
                logger.error("Failed to combine data sources")
                return None

            # Verify final data quality
            is_valid, reason = self._validate_consistency(combined_data)
            if not is_valid:
                logger.error(f"Final combined data validation failed: {reason}")
                return None

            # Cache the validated result
            self.data_cache[cache_key] = (datetime.now(), combined_data)

            return combined_data

        except Exception as e:
            logger.error(f"Error aggregating data: {str(e)}")
            return None

    def _normalize_timeframe(self, data: pd.DataFrame, target_timeframe: str) -> pd.DataFrame:
        """Normalize data to target timeframe."""
        if data.empty:
            return data

        try:
            # Get target timedelta
            target_delta = TIMEFRAME_MAP.get(target_timeframe)
            if not target_delta:
                logger.error(f"Unsupported timeframe: {target_timeframe}")
                return data

            # Check current timeframe
            current_delta = pd.Timedelta(data.index[1] - data.index[0])

            # If current timeframe is smaller than target, resample
            if current_delta < target_delta:
                resampled = data.resample(target_timeframe).agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last',
                    'volume': 'sum'
                }).dropna()

                # Validate resampled data
                if len(resampled) < len(data) * 0.9:  # Less than 90% of expected data points
                    logger.warning("Significant data loss during resampling")
                    return data

                return resampled

            return data

        except Exception as e:
            logger.error(f"Error normalizing timeframe: {str(e)}")
            return data

    def _validate_consistency(self, data: pd.DataFrame) -> Tuple[bool, str]:
        """Validate data consistency and return reason if invalid."""
        try:
            if data.empty:
                return False, "Empty DataFrame"

            # Check for missing required columns
            required_columns = {'open', 'high', 'low', 'close', 'volume'}
            if not required_columns.issubset(data.columns):
                return False, f"Missing columns: {required_columns - set(data.columns)}"

            # Validate OHLC relationship
            if any(data['high'] < data['low']):
                return False, "High < Low detected"
            if any(data['close'] > data['high']) or any(data['close'] < data['low']):
                return False, "Close outside H/L range"
            if any(data['open'] > data['high']) or any(data['open'] < data['low']):
                return False, "Open outside H/L range"

            # Check for negative values
            if any(data[['open', 'high', 'low', 'close', 'volume']] < 0):
                return False, "Negative values detected"

            # Check for reasonable price changes
            returns = data['close'].pct_change()
            if any(abs(returns) > 0.2):  # 20% price change
                return False, "Unreasonable price change detected"

            return True, "Data valid"

        except Exception as e:
            return False, f"Validation error: {str(e)}"

    def _combine_data_sources(self, data_frames: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
        """
        Combine multiple data sources with intelligent weighting and outlier detection.

        Args:
            data_frames: List of DataFrames to combine

        Returns:
            Combined DataFrame or None if combination fails
        """
        try:
            if not data_frames:
                return None

            # Calculate weights based on data quality
            weights = []
            for df in data_frames:
                # Score based on multiple factors
                score = 0.0

                # Data completeness score
                completeness = 1 - df.isnull().sum().mean()
                score += completeness * 0.4

                # Timeliness score
                latest_time = pd.to_datetime(df.index[-1])
                time_diff = datetime.now() - latest_time
                timeliness = max(0, 1 - time_diff.total_seconds() / (60 * 60))  # 1 hour max
                score += timeliness * 0.3

                # Price consistency score
                returns = df['close'].pct_change()
                consistency = 1 - (np.abs(stats.zscore(returns)) > 3).mean()
                score += consistency * 0.3

                weights.append(score)

            # Normalize weights
            total_weight = sum(weights)
            if total_weight == 0:
                weights = [1/len(weights)] * len(weights)
            else:
                weights = [w/total_weight for w in weights]

            # Combine data with weighted average
            combined = pd.DataFrame()
            for df, weight in zip(data_frames, weights):
                if combined.empty:
                    combined = df * weight
                else:
                    combined = combined.add(df * weight, fill_value=0)

            # Round to appropriate decimals
            combined = combined.round({
                'open': 5,
                'high': 5,
                'low': 5,
                'close': 5,
                'volume': 2
            })

            return combined

        except Exception as e:
            logger.error(f"Error combining data sources: {str(e)}")
            return None

    def _detect_outliers(self, data: pd.DataFrame, window: int = 20) -> pd.DataFrame:
        """
        Detect and handle outliers in price data.

        Args:
            data: DataFrame to check for outliers
            window: Rolling window size for outlier detection

        Returns:
            DataFrame with outliers handled
        """
        try:
            df = data.copy()

            # Calculate rolling median and std
            for col in ['open', 'high', 'low', 'close']:
                rolling_median = df[col].rolling(window=window, center=True).median()
                rolling_std = df[col].rolling(window=window, center=True).std()

                # Detect outliers (3 standard deviations from rolling median)
                outliers = np.abs(df[col] - rolling_median) > (3 * rolling_std)

                # Replace outliers with rolling median
                df.loc[outliers, col] = rolling_median[outliers]

            # Handle volume outliers separately (5 standard deviations)
            vol_median = df['volume'].rolling(window=window, center=True).median()
            vol_std = df['volume'].rolling(window=window, center=True).std()
            vol_outliers = np.abs(df['volume'] - vol_median) > (5 * vol_std)
            df.loc[vol_outliers, 'volume'] = vol_median[vol_outliers]

            return df

        except Exception as e:
            logger.error(f"Error detecting outliers: {str(e)}")
            return data
