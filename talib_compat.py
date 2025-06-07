"""
TA-Lib compatibility layer using the 'ta' library.
This module provides the same interface as talib but uses the 'ta' library underneath.

IMPORTANT: This is a compatibility layer that provides similar functionality to TA-Lib
but may have slight differences in calculation methods and accuracy. For maximum
accuracy, install the real TA-Lib with Visual C++ Build Tools.

The implementations here are enhanced to be as close as possible to TA-Lib's behavior.
"""

import pandas as pd
import numpy as np
import ta
from ta.trend import SMAIndicator, EMAIndicator, MACD, ADXIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.others import DailyReturnIndicator

def SMA(close, timeperiod=30):
    """Simple Moving Average"""
    return SMAIndicator(close=close, window=timeperiod).sma_indicator()

def EMA(close, timeperiod=30):
    """Exponential Moving Average"""
    return EMAIndicator(close=close, window=timeperiod).ema_indicator()

def RSI(close, timeperiod=14):
    """Relative Strength Index"""
    return RSIIndicator(close=close, window=timeperiod).rsi()

def MACD(close, fastperiod=12, slowperiod=26, signalperiod=9):
    """MACD indicator"""
    macd_indicator = MACD(close=close, window_fast=fastperiod, window_slow=slowperiod, window_sign=signalperiod)
    macd_line = macd_indicator.macd()
    signal_line = macd_indicator.macd_signal()
    histogram = macd_indicator.macd_diff()
    return macd_line, signal_line, histogram

def STOCH(high, low, close, fastk_period=5, slowk_period=3, slowd_period=3):
    """Stochastic Oscillator"""
    stoch = StochasticOscillator(high=high, low=low, close=close, window=fastk_period, smooth_window=slowk_period)
    slowk = stoch.stoch()
    slowd = stoch.stoch_signal()
    return slowk, slowd

def ADX(high, low, close, timeperiod=14):
    """Average Directional Index"""
    return ADXIndicator(high=high, low=low, close=close, window=timeperiod).adx()

def BBANDS(close, timeperiod=5, nbdevup=2, nbdevdn=2, matype=0):
    """Bollinger Bands"""
    bb = BollingerBands(close=close, window=timeperiod, window_dev=nbdevup)
    upper = bb.bollinger_hband()
    middle = bb.bollinger_mavg()
    lower = bb.bollinger_lband()
    return upper, middle, lower

def ATR(high, low, close, timeperiod=14):
    """Average True Range"""
    return AverageTrueRange(high=high, low=low, close=close, window=timeperiod).average_true_range()

# Candlestick pattern functions (simplified implementations)
def CDLENGULFING(open_prices, high, low, close):
    """Engulfing Pattern"""
    # Simplified implementation
    result = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        # Bullish engulfing
        if (close.iloc[i-1] < open_prices.iloc[i-1] and  # Previous candle was bearish
            close.iloc[i] > open_prices.iloc[i] and      # Current candle is bullish
            open_prices.iloc[i] < close.iloc[i-1] and    # Current open below previous close
            close.iloc[i] > open_prices.iloc[i-1]):      # Current close above previous open
            result.iloc[i] = 100
        # Bearish engulfing
        elif (close.iloc[i-1] > open_prices.iloc[i-1] and  # Previous candle was bullish
              close.iloc[i] < open_prices.iloc[i] and      # Current candle is bearish
              open_prices.iloc[i] > close.iloc[i-1] and    # Current open above previous close
              close.iloc[i] < open_prices.iloc[i-1]):      # Current close below previous open
            result.iloc[i] = -100
    return result

def CDLHAMMER(open_prices, high, low, close):
    """Hammer Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(len(close)):
        body = abs(close.iloc[i] - open_prices.iloc[i])
        lower_shadow = min(close.iloc[i], open_prices.iloc[i]) - low.iloc[i]
        upper_shadow = high.iloc[i] - max(close.iloc[i], open_prices.iloc[i])

        if (lower_shadow > 2 * body and upper_shadow < body * 0.1):
            result.iloc[i] = 100
    return result

def CDLSHOOTINGSTAR(open_prices, high, low, close):
    """Shooting Star Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(len(close)):
        body = abs(close.iloc[i] - open_prices.iloc[i])
        lower_shadow = min(close.iloc[i], open_prices.iloc[i]) - low.iloc[i]
        upper_shadow = high.iloc[i] - max(close.iloc[i], open_prices.iloc[i])

        if (upper_shadow > 2 * body and lower_shadow < body * 0.1):
            result.iloc[i] = -100
    return result

def CDLMORNINGSTAR(open_prices, high, low, close):
    """Morning Star Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(2, len(close)):
        # Simplified 3-candle morning star pattern
        if (close.iloc[i-2] < open_prices.iloc[i-2] and  # First candle bearish
            abs(close.iloc[i-1] - open_prices.iloc[i-1]) < abs(close.iloc[i-2] - open_prices.iloc[i-2]) * 0.3 and  # Small middle candle
            close.iloc[i] > open_prices.iloc[i] and      # Third candle bullish
            close.iloc[i] > (open_prices.iloc[i-2] + close.iloc[i-2]) / 2):  # Third candle closes above midpoint of first
            result.iloc[i] = 100
    return result

def CDLEVENINGSTAR(open_prices, high, low, close):
    """Evening Star Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(2, len(close)):
        # Simplified 3-candle evening star pattern
        if (close.iloc[i-2] > open_prices.iloc[i-2] and  # First candle bullish
            abs(close.iloc[i-1] - open_prices.iloc[i-1]) < abs(close.iloc[i-2] - open_prices.iloc[i-2]) * 0.3 and  # Small middle candle
            close.iloc[i] < open_prices.iloc[i] and      # Third candle bearish
            close.iloc[i] < (open_prices.iloc[i-2] + close.iloc[i-2]) / 2):  # Third candle closes below midpoint of first
            result.iloc[i] = -100
    return result

# Additional functions needed by enhanced_feature_engineering
def ROC(close, timeperiod=10):
    """Rate of Change"""
    return close.pct_change(periods=timeperiod) * 100

def MFI(high, low, close, volume, timeperiod=14):
    """Money Flow Index"""
    from ta.volume import VolumeSMAIndicator, MFIIndicator
    return MFIIndicator(high=high, low=low, close=close, volume=volume, window=timeperiod).money_flow_index()

def OBV(close, volume):
    """On Balance Volume"""
    from ta.volume import OnBalanceVolumeIndicator
    return OnBalanceVolumeIndicator(close=close, volume=volume).on_balance_volume()

def AD(high, low, close, volume):
    """Accumulation/Distribution Line"""
    from ta.volume import AccDistIndexIndicator
    return AccDistIndexIndicator(high=high, low=low, close=close, volume=volume).acc_dist_index()

def ADOSC(high, low, close, volume, fastperiod=3, slowperiod=10):
    """Chaikin A/D Oscillator"""
    from ta.volume import ChaikinMoneyFlowIndicator
    return ChaikinMoneyFlowIndicator(high=high, low=low, close=close, volume=volume, window=slowperiod).chaikin_money_flow()

def PLUS_DI(high, low, close, timeperiod=14):
    """Plus Directional Indicator"""
    from ta.trend import ADXIndicator
    return ADXIndicator(high=high, low=low, close=close, window=timeperiod).adx_pos()

def MINUS_DI(high, low, close, timeperiod=14):
    """Minus Directional Indicator"""
    from ta.trend import ADXIndicator
    return ADXIndicator(high=high, low=low, close=close, window=timeperiod).adx_neg()

def AROON(high, low, timeperiod=14):
    """Aroon Indicator"""
    from ta.trend import AroonIndicator
    aroon = AroonIndicator(high=high, low=low, window=timeperiod)
    return aroon.aroon_up(), aroon.aroon_down()

# Additional candlestick patterns
def CDLHARAMI(open_prices, high, low, close):
    """Harami Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        # Simplified harami pattern
        prev_body = abs(close.iloc[i-1] - open_prices.iloc[i-1])
        curr_body = abs(close.iloc[i] - open_prices.iloc[i])
        if (prev_body > curr_body * 2 and  # Previous candle much larger
            min(close.iloc[i], open_prices.iloc[i]) > min(close.iloc[i-1], open_prices.iloc[i-1]) and
            max(close.iloc[i], open_prices.iloc[i]) < max(close.iloc[i-1], open_prices.iloc[i-1])):
            result.iloc[i] = 100 if close.iloc[i-1] < open_prices.iloc[i-1] else -100
    return result

def CDLPIERCING(open_prices, high, low, close):
    """Piercing Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if (close.iloc[i-1] < open_prices.iloc[i-1] and  # Previous bearish
            close.iloc[i] > open_prices.iloc[i] and      # Current bullish
            open_prices.iloc[i] < close.iloc[i-1] and    # Gap down
            close.iloc[i] > (open_prices.iloc[i-1] + close.iloc[i-1]) / 2):  # Close above midpoint
            result.iloc[i] = 100
    return result

def CDLDARKCLOUDCOVER(open_prices, high, low, close):
    """Dark Cloud Cover Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(1, len(close)):
        if (close.iloc[i-1] > open_prices.iloc[i-1] and  # Previous bullish
            close.iloc[i] < open_prices.iloc[i] and      # Current bearish
            open_prices.iloc[i] > close.iloc[i-1] and    # Gap up
            close.iloc[i] < (open_prices.iloc[i-1] + close.iloc[i-1]) / 2):  # Close below midpoint
            result.iloc[i] = -100
    return result

def CDLMARUBOZU(open_prices, high, low, close):
    """Marubozu Pattern"""
    result = pd.Series(0, index=close.index)
    for i in range(len(close)):
        body = abs(close.iloc[i] - open_prices.iloc[i])
        total_range = high.iloc[i] - low.iloc[i]
        if body / total_range > 0.95:  # Very little shadow
            result.iloc[i] = 100 if close.iloc[i] > open_prices.iloc[i] else -100
    return result
