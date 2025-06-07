"""
Enhanced feature engineering module for TradePulseAnalyzer.
Implements advanced technical indicators, pattern recognition, and adaptive feature selection.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
import talib_compat as talib
import logging

logger = logging.getLogger(__name__)

class EnhancedFeatureEngineer:
    """Advanced feature engineering with adaptive feature selection."""

    def __init__(self, use_pca: bool = True, pca_components: int = 10):
        """
        Initialize the feature engineer.

        Args:
            use_pca: Whether to use PCA for dimensionality reduction
            pca_components: Number of PCA components to use
        """
        self.standard_scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.pca = PCA(n_components=pca_components) if use_pca else None
        self.feature_importance: Dict[str, float] = {}
        self.selected_features: List[str] = []
        self.market_regime_cache: Dict[str, pd.Series] = {}

    def engineer_features(self, data: pd.DataFrame, target: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Generate advanced technical features with adaptive selection.

        Args:
            data: DataFrame with OHLCV data
            target: Optional target variable for supervised feature selection

        Returns:
            DataFrame with engineered features
        """
        try:
            df = data.copy()

            # Basic price and return features
            self._add_price_features(df)

            # Momentum indicators
            self._add_momentum_features(df)

            # Volatility indicators
            self._add_volatility_features(df)

            # Volume analysis
            self._add_volume_features(df)

            # Pattern recognition
            self._add_pattern_features(df)

            # Market regime features
            self._add_regime_features(df)

            # Adaptive feature selection
            if target is not None:
                self._update_feature_importance(df, target)
                df = self._select_best_features(df)

            # Normalize features
            df = self._normalize_features(df)

            return df

        except Exception as e:
            logger.error(f"Error engineering features: {str(e)}")
            return data

    def _add_price_features(self, df: pd.DataFrame) -> None:
        """Add price-based features."""
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log1p(df['returns'])
        df['high_low_ratio'] = df['high'] / df['low']
        df['close_open_ratio'] = df['close'] / df['open']

        # Price distances
        df['dist_from_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['dist_from_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']

        # Moving averages
        for period in [9, 20, 50, 200]:
            df[f'sma_{period}'] = talib.SMA(df['close'], timeperiod=period)
            df[f'ema_{period}'] = talib.EMA(df['close'], timeperiod=period)

        # Moving average crossovers
        df['sma_9_20_cross'] = np.where(df['sma_9'] > df['sma_20'], 1, -1)
        df['ema_9_20_cross'] = np.where(df['ema_9'] > df['ema_20'], 1, -1)

    def _add_momentum_features(self, df: pd.DataFrame) -> None:
        """Add momentum indicators."""
        # RSI with multiple periods
        for period in [7, 14, 21]:
            df[f'rsi_{period}'] = talib.RSI(df['close'], timeperiod=period)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'])

        # Stochastic
        df['stoch_k'], df['stoch_d'] = talib.STOCH(df['high'], df['low'], df['close'])

        # ROC
        for period in [5, 10, 20]:
            df[f'roc_{period}'] = talib.ROC(df['close'], timeperiod=period)

        # Money Flow Index
        df['mfi'] = talib.MFI(df['high'], df['low'], df['close'], df['volume'])

    def _add_volatility_features(self, df: pd.DataFrame) -> None:
        """Add volatility indicators."""
        # ATR and variants
        df['atr'] = talib.ATR(df['high'], df['low'], df['close'])
        df['atr_ratio'] = df['atr'] / df['close']

        # Bollinger Bands
        df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(df['close'])
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])

        # Historical volatility
        for period in [5, 21]:
            df[f'volatility_{period}'] = df['returns'].rolling(period).std()

        # Keltner Channels
        df['kc_middle'] = df['ema_20']
        df['kc_upper'] = df['kc_middle'] + (df['atr'] * 2)
        df['kc_lower'] = df['kc_middle'] - (df['atr'] * 2)

    def _add_volume_features(self, df: pd.DataFrame) -> None:
        """Add volume analysis features."""
        # Basic volume metrics
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma']
        df['volume_variance'] = df['volume'].rolling(20).var()

        # On-Balance Volume and variants
        df['obv'] = talib.OBV(df['close'], df['volume'])
        df['obv_sma'] = df['obv'].rolling(20).mean()
        df['obv_trend'] = np.where(df['obv'] > df['obv_sma'], 1, -1)

        # Volume-price relationship
        df['volume_price_trend'] = df['volume'] * df['returns']
        df['volume_price_trend_sma'] = df['volume_price_trend'].rolling(20).mean()

        # Accumulation/Distribution
        df['adl'] = talib.AD(df['high'], df['low'], df['close'], df['volume'])
        df['adl_sma'] = df['adl'].rolling(20).mean()
        df['chaikin_money_flow'] = talib.ADOSC(df['high'], df['low'], df['close'], df['volume'])

    def _add_pattern_features(self, df: pd.DataFrame) -> None:
        """Add candlestick pattern recognition features."""
        pattern_funcs = [
            talib.CDLENGULFING,
            talib.CDLHAMMER,
            talib.CDLHARAMI,
            talib.CDLMORNINGSTAR,
            talib.CDLEVENINGSTAR,
            talib.CDLPIERCING,
            talib.CDLDARKCLOUDCOVER,
            talib.CDLSHOOTINGSTAR,
            talib.CDLMARUBOZU
        ]

        for func in pattern_funcs:
            pattern_name = func.__name__.replace('CDL', '').lower()
            df[f'pattern_{pattern_name}'] = func(df['open'], df['high'], df['low'], df['close'])

        # Advanced pattern metrics
        df['doji'] = np.where(abs(df['close'] - df['open']) <= (df['high'] - df['low']) * 0.1, 1, 0)
        df['long_body'] = np.where(abs(df['close'] - df['open']) >= (df['high'] - df['low']) * 0.7, 1, 0)

    def _add_regime_features(self, df: pd.DataFrame) -> None:
        """Add market regime detection features."""
        # Trend strength
        df['adx'] = talib.ADX(df['high'], df['low'], df['close'])
        df['dmi_plus'] = talib.PLUS_DI(df['high'], df['low'], df['close'])
        df['dmi_minus'] = talib.MINUS_DI(df['high'], df['low'], df['close'])

        # Trend direction
        df['aroon_up'], df['aroon_down'] = talib.AROON(df['high'], df['low'])
        df['aroon_osc'] = df['aroon_up'] - df['aroon_down']

        # Regime classification
        df['regime_volatility'] = df['volatility_21'].rolling(20).mean()
        df['regime_trend'] = np.where(df['adx'] > 25, 1, 0)
        df['regime_momentum'] = np.where(abs(df['roc_20']) > df['roc_20'].rolling(100).std(), 1, 0)

    def _normalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize features using appropriate scalers."""
        try:
            # Separate features that need different scaling approaches
            standard_scale_cols = df.select_dtypes(include=[np.number]).columns.difference([
                'returns', 'log_returns', 'regime_trend', 'regime_momentum',
                'doji', 'long_body'
            ])

            # Apply standard scaling to most features
            if len(standard_scale_cols) > 0:
                df[standard_scale_cols] = self.standard_scaler.fit_transform(df[standard_scale_cols])

            # Apply dimensionality reduction if enabled
            if self.pca is not None and len(standard_scale_cols) > self.pca.n_components:
                pca_cols = [f'pca_{i}' for i in range(self.pca.n_components)]
                pca_features = self.pca.fit_transform(df[standard_scale_cols])
                df[pca_cols] = pca_features

            return df

        except Exception as e:
            logger.error(f"Error normalizing features: {str(e)}")
            return df

    def _update_feature_importance(self, df: pd.DataFrame, target: pd.Series) -> None:
        """Update feature importance scores."""
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            importance_scores = mutual_info_classif(df[numeric_cols].fillna(0), target)
            self.feature_importance = dict(zip(numeric_cols, importance_scores))

            # Sort features by importance
            self.selected_features = sorted(
                self.feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            )

        except Exception as e:
            logger.error(f"Error updating feature importance: {str(e)}")

    def _select_best_features(self, df: pd.DataFrame, top_n: int = 30) -> pd.DataFrame:
        """Select the most important features."""
        if not self.selected_features:
            return df

        top_features = [f[0] for f in self.selected_features[:top_n]]
        required_cols = ['open', 'high', 'low', 'close', 'volume']  # Always keep these
        selected_cols = list(set(top_features + required_cols))

        return df[selected_cols]

    def get_feature_importance(self) -> Dict[str, float]:
        """Get the current feature importance scores."""
        return dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
