"""
Enhanced feature engineering module for trading signal generation.
Implements advanced technical indicators and pattern recognition.
"""

import pandas as pd
import numpy as np
from typing import Dict, List
import talib
from sklearn.preprocessing import StandardScaler
import logging
import joblib
import os
from typing import Dict, List, Optional, Tuple, Union
import warnings
from enhanced_logging import logger as enhanced_logger

# Configure logging
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for trading signals."""    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_importance = {}
        self.feature_names = [
            # Price features
            'returns', 'log_returns', 'volatility',
            
            # Momentum indicators
            'rsi', 'rsi_divergence',
            'macd', 'macd_signal', 'macd_divergence',
            
            # Moving averages
            'ema_short', 'ema_medium', 'ema_long',
            
            # Volatility indicators
            'atr', 'normalized_atr', 'bb_width',
            'bbands_upper', 'bbands_middle', 'bbands_lower',
            
            # Volume indicators
            'volume_sma', 'volume_ratio', 'volume_vwap', 
            'obv', 'obv_sma',
            
            # Additional technical indicators
            'mfi', 'cci', 'adx',
            'dmi_plus', 'dmi_minus',
            
            # Oscillators
            'stoch_k', 'stoch_d',
            
            # Pattern recognition
            'engulfing', 'hammer', 'shooting_star',
            
            # Market structure
            'trend_strength', 'regime',
            'higher_highs', 'lower_lows', 'consolidation',
            'support_resistance',  # Added new feature
            
            # Additional pattern metrics
            'price_momentum', 'volume_momentum', 'trend_score'
        ]

    def save_state(self, directory: str = "data") -> None:
        """
        Save the current state of the feature engineer.

        Args:
            directory: Directory to save the state files
        """
        try:
            os.makedirs(directory, exist_ok=True)

            # Save StandardScaler state
            scaler_path = os.path.join(directory, "feature_scaler.joblib")
            joblib.dump(self.scaler, scaler_path)

            # Save feature importance dictionary
            importance_path = os.path.join(directory, "feature_importance.joblib")
            joblib.dump(self.feature_importance, importance_path)

            logger.info(f"Feature engineering state saved to {directory}")

        except Exception as e:
            logger.error(f"Error saving feature engineering state: {str(e)}")

    def load_state(self, directory: str = "data") -> bool:
        """
        Load a previously saved state of the feature engineer.

        Args:
            directory: Directory containing the state files

        Returns:
            bool: True if state was loaded successfully, False otherwise
        """
        try:
            # Load StandardScaler state
            scaler_path = os.path.join(directory, "feature_scaler.joblib")
            if os.path.exists(scaler_path):
                self.scaler = joblib.load(scaler_path)

            # Load feature importance dictionary
            importance_path = os.path.join(directory, "feature_importance.joblib")
            if os.path.exists(importance_path):
                self.feature_importance = joblib.load(importance_path)

            logger.info(f"Feature engineering state loaded from {directory}")
            return True

        except Exception as e:
            logger.error(f"Error loading feature engineering state: {str(e)}")
            return False    
    def engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate advanced technical features from price data.

        Args:
            data: DataFrame with OHLCV data

        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()
            component = "FeatureEngineering"
            
            # Log start of feature engineering
            enhanced_logger.logger.info(f"Starting feature engineering on data shape: {df.shape}")
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log1p(df['returns'])
            df['volatility'] = df['returns'].rolling(window=20).std()

            # RSI and RSI divergence
            df['rsi'] = talib.RSI(df['close'])
            df['rsi_divergence'] = self._calculate_divergence(df['close'], df['rsi'])

            # MACD
            df['macd'], df['macd_signal'], _ = talib.MACD(df['close'])
            df['macd_divergence'] = self._calculate_divergence(df['close'], df['macd'])

            # Moving averages
            df['ema_short'] = talib.EMA(df['close'], timeperiod=10)
            df['ema_medium'] = talib.EMA(df['close'], timeperiod=20)
            df['ema_long'] = talib.EMA(df['close'], timeperiod=50)

            # ATR and normalized ATR
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
            df['normalized_atr'] = df['atr'] / df['close']

            # Bollinger Bands
            df['bbands_upper'], df['bbands_middle'], df['bbands_lower'] = talib.BBANDS(df['close'])
            df['bb_width'] = (df['bbands_upper'] - df['bbands_lower']) / df['bbands_middle']

            # Volume indicators
            df['volume_sma'] = df['volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
            df['volume_vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['obv'] = talib.OBV(df['close'], df['volume'])
            df['obv_sma'] = df['obv'].rolling(window=20).mean()

            # Additional technical indicators
            df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])
            df['cci'] = talib.CCI(df['high'], df['low'], df['close'])
            df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
            df['dmi_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
            df['dmi_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'])

            # Stochastic
            df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])

            # Price patterns
            df['engulfing'] = talib.CDLENGULFING(df['open'], df['high'], df['low'], df['close'])
            df['hammer'] = talib.CDLHAMMER(df['open'], df['high'], df['low'], df['close'])
            df['shooting_star'] = talib.CDLSHOOTINGSTAR(df['open'], df['high'], df['low'], df['close'])            # Market structure features
            df['trend_strength'] = self._calculate_trend_strength(df)
            df['regime'] = self._detect_market_regime(df)
            df['higher_highs'] = self._detect_higher_highs(df)
            df['lower_lows'] = self._detect_lower_lows(df)
            df['consolidation'] = self._detect_consolidation(df)
            df['support_resistance'] = self._calculate_support_resistance(df)

            # Additional pattern metrics
            df['price_momentum'] = df['close'].diff(5) / df['close'].shift(5)
            df['volume_momentum'] = df['volume'].diff(5) / df['volume'].shift(5)
            df['trend_score'] = df['trend_strength'] * df['adx'] / 100.0

            # Replace NaN values with 0
            df = df.fillna(0)

            # Handle any missing columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"Missing required column: {col}")
                    raise ValueError(f"Input data must contain {col} column")

            # Validate data types
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in numeric_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    logger.warning(f"Converting {col} to numeric type")
                    df[col] = pd.to_numeric(df[col], errors='coerce')

            # Remove any remaining infinite values
            df = df.replace([np.inf, -np.inf], 0)

            # Normalize features
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = self.scaler.fit_transform(df[numeric_columns])

            # Verify all expected features are present
            missing_features = set(self.feature_names) - set(df.columns)
            if missing_features:
                error_msg = f"Failed to generate all required features. Missing: {missing_features}"
                enhanced_logger.log_error(
                    ValueError(error_msg),
                    component,
                    additional_data={
                        "expected_features": self.feature_names,
                        "generated_features": list(df.columns),
                        "data_shape": df.shape
                    }
                )
                raise ValueError(error_msg)
            
            # Debug logging for feature verification
            enhanced_logger.logger.debug(
                "Feature verification:",
                extra={
                    "generated_features": list(df.columns),
                    "expected_features": self.feature_names,
                    "feature_counts": {
                        "total": len(self.feature_names),
                        "generated": len(df.columns),
                        "matching": len(set(df.columns).intersection(self.feature_names))
                    }
                }
            )
            
            # Log successful feature generation
            enhanced_logger.logger.info(
                f"Successfully generated {len(self.feature_names)} features"
            )
            
            # Return only the features we declared in feature_names, in the correct order
            result = df[self.feature_names]
            
            # Final verification
            if len(result.columns) != len(self.feature_names):
                error_msg = f"Feature count mismatch: expected {len(self.feature_names)}, got {len(result.columns)}"
                enhanced_logger.log_error(
                    ValueError(error_msg),
                    component,
                    additional_data={
                        "expected_count": len(self.feature_names),
                        "actual_count": len(result.columns),
                        "missing": set(self.feature_names) - set(result.columns)
                    }
                )
                raise ValueError(error_msg)
            
            return result
        except Exception as e:
            enhanced_logger.log_error(
                e, 
                component,
                additional_data={
                    "data_shape": data.shape,
                    "feature_count": len(self.feature_names)
                }
            )
            raise

    def _calculate_divergence(self, price: pd.Series, indicator: pd.Series, window: int = 20) -> pd.Series:
        """Calculate divergence between price and indicator."""
        price_grad = np.gradient(price.rolling(window).mean())
        indicator_grad = np.gradient(indicator.rolling(window).mean())
        return np.where(price_grad * indicator_grad < 0, 1, 0)

    def _calculate_trend_strength(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Calculate trend strength using multiple indicators."""
        # Linear regression slope
        x = np.arange(window)
        slopes = pd.Series(index=data.index, dtype=float)

        for i in range(window - 1, len(data)):
            y = data['close'].iloc[i-window+1:i+1].values
            slope, _ = np.polyfit(x, y, 1)
            slopes.iloc[i] = slope

        return slopes

    def _detect_market_regime(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect market regime (trending, ranging, volatile)."""
        volatility = data['returns'].rolling(window).std()
        trend = self._calculate_trend_strength(data)
        adx = data['adx']

        regime = pd.Series(index=data.index, dtype=str)

        # Classify regimes using trend strength and ADX
        strong_trend = (adx > 25) & (abs(trend) > trend.quantile(0.7))
        regime.loc[strong_trend] = 'TRENDING'
        regime.loc[volatility > volatility.quantile(0.8)] = 'VOLATILE'
        regime.loc[(adx <= 25) & (volatility <= volatility.quantile(0.8))] = 'RANGING'

        return regime

    def _detect_higher_highs(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """Detect higher highs pattern."""
        highs = data['high'].rolling(window=window).max()
        return (highs > highs.shift(1)).astype(int)

    def _detect_lower_lows(self, data: pd.DataFrame, window: int = 10) -> pd.Series:
        """Detect lower lows pattern."""
        lows = data['low'].rolling(window=window).min()
        return (lows < lows.shift(1)).astype(int)

    def _detect_consolidation(self, data: pd.DataFrame, window: int = 20, threshold: float = 0.02) -> pd.Series:
        """Detect price consolidation periods."""
        rolling_std = data['close'].rolling(window).std()
        rolling_mean = data['close'].rolling(window).mean()
        return (rolling_std / rolling_mean < threshold).astype(int)

    def _calculate_support_resistance(self, data: pd.DataFrame, window: int = 20) -> pd.Series:
        """Detect support and resistance levels using price action analysis."""
        highs = data['high'].rolling(window=window).max()
        lows = data['low'].rolling(window=window).min()
        
        # Calculate distance from current price to support/resistance levels
        price = data['close']
        resistance_distance = (highs - price) / price
        support_distance = (price - lows) / price
        
        # Combine into a single metric
        support_resistance = pd.Series(0.0, index=data.index)
        
        # Identify strong levels when price is near support or resistance
        near_support = support_distance < 0.02  # Within 2% of support
        near_resistance = resistance_distance < 0.02  # Within 2% of resistance
        
        # Score based on proximity to levels
        support_resistance[near_support] = -support_distance[near_support]
        support_resistance[near_resistance] = resistance_distance[near_resistance]
        
        return support_resistance

    def get_feature_importance(self) -> Dict[str, float]:
        """Get the importance scores of features."""
        return self.feature_importance

    def get_feature_names(self) -> List[str]:
        """Return the list of feature names generated by engineer_features."""
        return self.feature_names.copy()
