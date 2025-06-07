"""
Enhanced technical indicators with advanced analysis capabilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
import ta
import logging

logger = logging.getLogger(__name__)

class EnhancedIndicators:
    """Enhanced technical indicators with advanced analysis capabilities."""
    
    @staticmethod
    def calculate_williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R indicator."""
        try:
            return ta.momentum.WilliamsRIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            ).williams_r()
        except Exception as e:
            logger.error(f"Error calculating Williams %R: {str(e)}")
            return pd.Series()

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        try:
            adx_indicator = ta.trend.ADXIndicator(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            )
            
            return pd.DataFrame({
                'ADX': adx_indicator.adx(),
                '+DI': adx_indicator.adx_pos(),
                '-DI': adx_indicator.adx_neg()
            })
        except Exception as e:
            logger.error(f"Error calculating ADX: {str(e)}")
            return pd.DataFrame()

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR)."""
        try:
            return ta.volatility.AverageTrueRange(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                window=period
            ).average_true_range()
        except Exception as e:
            logger.error(f"Error calculating ATR: {str(e)}")
            return pd.Series()

    @staticmethod
    def calculate_fibonacci_levels(df: pd.DataFrame, period: int = 20) -> Dict[str, float]:
        """Calculate Fibonacci retracement levels."""
        try:
            # Get high and low for the period
            high = df['high'].rolling(window=period).max().iloc[-1]
            low = df['low'].rolling(window=period).min().iloc[-1]
            diff = high - low

            # Calculate Fibonacci levels
            levels = {
                'Level_0': low,
                'Level_0.236': low + 0.236 * diff,
                'Level_0.382': low + 0.382 * diff,
                'Level_0.5': low + 0.5 * diff,
                'Level_0.618': low + 0.618 * diff,
                'Level_0.786': low + 0.786 * diff,
                'Level_1': high
            }
            
            return levels
        except Exception as e:
            logger.error(f"Error calculating Fibonacci levels: {str(e)}")
            return {}

    @staticmethod
    def calculate_vwap(df: pd.DataFrame) -> pd.Series:
        """Calculate Volume Weighted Average Price (VWAP)."""
        try:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
            return vwap
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series()

    @staticmethod
    def calculate_ichimoku(df: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate Ichimoku Cloud components."""
        try:
            ichimoku = ta.trend.IchimokuIndicator(
                high=df['high'],
                low=df['low']
            )
            
            return {
                'tenkan_sen': ichimoku.ichimoku_conversion_line(),
                'kijun_sen': ichimoku.ichimoku_base_line(),
                'senkou_span_a': ichimoku.ichimoku_a(),
                'senkou_span_b': ichimoku.ichimoku_b(),
                'chikou_span': df['close'].shift(-26)
            }
        except Exception as e:
            logger.error(f"Error calculating Ichimoku Cloud: {str(e)}")
            return {}
