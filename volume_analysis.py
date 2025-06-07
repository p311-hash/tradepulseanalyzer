"""
Volume analysis module for binary options trading signals.
This module adds volume-based indicators and analysis to improve signal quality.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

class VolumeAnalyzer:
    """Class for analyzing volume to improve signal quality."""
    
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with price and volume data.
        
        Args:
            data: DataFrame with OHLCV price data (must include volume)
        """
        self.data = data
        if 'volume' not in self.data.columns:
            raise ValueError("Data must include volume column")
        
        # Initialize results
        self._results = {}
        
    def calculate_vwap(self, period: int = 20) -> pd.Series:
        """
        Calculate Volume Weighted Average Price (VWAP).
        
        Args:
            period: Lookback period
            
        Returns:
            Series with VWAP values
        """
        try:
            # Calculate typical price
            typical_price = (self.data['high'] + self.data['low'] + self.data['close']) / 3
            
            # Calculate VWAP
            vwap = (typical_price * self.data['volume']).rolling(window=period).sum() / \
                   self.data['volume'].rolling(window=period).sum()
            
            self._results['vwap'] = vwap
            logger.debug(f"Calculated VWAP with period {period}")
            return vwap
            
        except Exception as e:
            logger.error(f"Error calculating VWAP: {str(e)}")
            return pd.Series(np.nan, index=self.data.index)
    
    def detect_volume_spikes(self, threshold: float = 2.0, period: int = 20) -> pd.Series:
        """
        Detect significant volume spikes.
        
        Args:
            threshold: Multiple of average volume to consider a spike
            period: Lookback period for average calculation
            
        Returns:
            Series with volume spike indicators (0=no spike, 1=spike)
        """
        try:
            # Calculate average volume
            avg_volume = self.data['volume'].rolling(window=period).mean()
            
            # Calculate standard deviation
            vol_std = self.data['volume'].rolling(window=period).std()
            
            # Detect spikes using z-score
            z_scores = (self.data['volume'] - avg_volume) / vol_std
            spikes = pd.Series(0, index=self.data.index)
            spikes[z_scores > threshold] = 1  # Mark volumes above threshold standard deviations as spikes
            spikes[self.data['volume'] > avg_volume * threshold] = 1
            
            self._results['volume_spikes'] = spikes
            spike_count = spikes.sum()
            logger.debug(f"Detected {spike_count} volume spikes with threshold {threshold}")
            return spikes
            
        except Exception as e:
            logger.error(f"Error detecting volume spikes: {str(e)}")
            return pd.Series(0, index=self.data.index)
    
    def calculate_volume_trend_correlation(self, period: int = 14) -> pd.Series:
        """
        Calculate correlation between price trend and volume.
        
        Args:
            period: Lookback period for correlation
            
        Returns:
            Series with correlation values (-1 to 1)
        """
        try:
            # Calculate price changes
            price_changes = self.data['close'].diff()
            
            # Calculate correlation using rolling window
            correlation = price_changes.rolling(period).corr(self.data['volume'])
            
            self._results['vol_price_corr'] = correlation
            logger.debug(f"Calculated volume-price correlation with period {period}")
            return correlation
            
        except Exception as e:
            logger.error(f"Error calculating volume-price correlation: {str(e)}")
            return pd.Series(np.nan, index=self.data.index)
    
    def calculate_obv(self) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        Returns:
            Series with OBV values
        """
        try:
            close_diff = self.data['close'].diff()
            obv = pd.Series(0.0, index=self.data.index)  # Initialize as float instead of int
            
            for i in range(1, len(self.data)):
                if close_diff.iloc[i] > 0:
                    obv.iloc[i] = obv.iloc[i-1] + float(self.data['volume'].iloc[i])  # Ensure float conversion
                elif close_diff.iloc[i] < 0:
                    obv.iloc[i] = obv.iloc[i-1] - float(self.data['volume'].iloc[i])  # Ensure float conversion
                else:
                    obv.iloc[i] = obv.iloc[i-1]
            
            self._results['obv'] = obv
            logger.debug("Calculated On-Balance Volume (OBV)")
            return obv
            
        except Exception as e:
            logger.error(f"Error calculating OBV: {str(e)}")
            return pd.Series(np.nan, index=self.data.index)
    
    def calculate_volume_profile(self, num_bins: int = 50) -> Dict:
        """
        Calculate volume profile metrics.
        
        Args:
            num_bins: Number of price levels for volume distribution
            
        Returns:
            Dictionary with volume profile metrics
        """
        try:
            # Calculate price range
            price_min = self.data['low'].min()
            price_max = self.data['high'].max()
            price_bins = np.linspace(price_min, price_max, num_bins)
            
            # Initialize volume profile
            volume_profile = np.zeros(num_bins - 1)
            
            # Calculate volume at each price level
            for i in range(len(self.data)):
                candle_low = self.data['low'].iloc[i]
                candle_high = self.data['high'].iloc[i]
                candle_volume = self.data['volume'].iloc[i]
                
                # Distribute volume across price levels
                for j in range(num_bins - 1):
                    if candle_low <= price_bins[j+1] and candle_high >= price_bins[j]:
                        volume_profile[j] += candle_volume
                        
            # Find point of control (price level with highest volume)
            poc_index = np.argmax(volume_profile)
            poc_price = (price_bins[poc_index] + price_bins[poc_index + 1]) / 2
            
            # Calculate value area (70% of volume)
            total_volume = np.sum(volume_profile)
            value_area_threshold = total_volume * 0.7
            
            # Find value area range
            volume_sum = 0
            va_high_idx = poc_index
            va_low_idx = poc_index
            
            while volume_sum < value_area_threshold and (va_high_idx < len(volume_profile) - 1 or va_low_idx > 0):
                high_vol = volume_profile[va_high_idx] if va_high_idx < len(volume_profile) else 0
                low_vol = volume_profile[va_low_idx] if va_low_idx > 0 else 0
                
                if high_vol >= low_vol and va_high_idx < len(volume_profile) - 1:
                    volume_sum += high_vol
                    va_high_idx += 1
                elif va_low_idx > 0:
                    volume_sum += low_vol
                    va_low_idx -= 1
                    
            value_area_high = price_bins[va_high_idx]
            value_area_low = price_bins[va_low_idx]
            
            # Calculate volume weighted average price (VWAP)
            vwap = np.sum(self.data['close'] * self.data['volume']) / np.sum(self.data['volume'])
            
            return {
                'volume_profile': volume_profile.tolist(),
                'price_levels': price_bins.tolist(),
                'point_of_control': float(poc_price),
                'value_area_high': float(value_area_high),
                'value_area_low': float(value_area_low),
                'vwap': float(vwap)
            }
            
        except Exception as e:
            logger.error(f"Error calculating volume profile: {str(e)}")
            return {}
    
    def analyze_volume_patterns(self) -> Dict:
        """
        Analyze volume patterns and anomalies.
        
        Returns:
            Dictionary with volume pattern analysis
        """
        try:
            # Calculate volume moving averages
            short_ma = self.data['volume'].rolling(window=5).mean()
            medium_ma = self.data['volume'].rolling(window=20).mean()
            long_ma = self.data['volume'].rolling(window=50).mean()
            
            # Detect volume anomalies
            volume_std = self.data['volume'].rolling(window=20).std()
            volume_upper = medium_ma + 2 * volume_std
            volume_lower = medium_ma - 2 * volume_std
            
            # Volume trend analysis
            volume_trend = np.where(short_ma > medium_ma, 1,
                                  np.where(short_ma < medium_ma, -1, 0))
            
            # Climax analysis
            is_climax = (self.data['volume'] > volume_upper) & \
                       (abs(self.data['close'] - self.data['open']) > \
                        abs(self.data['close'].shift(1) - self.data['open'].shift(1)))
            
            # Calculate volume momentum
            volume_momentum = self.data['volume'].diff() / self.data['volume'].shift(1)
            
            # Identify churning (high volume, small price change)
            typical_range = abs(self.data['high'] - self.data['low']).rolling(window=20).mean()
            current_range = abs(self.data['high'] - self.data['low'])
            is_churning = (self.data['volume'] > volume_upper) & (current_range < typical_range)
            
            return {
                'volume_trend': int(volume_trend[-1]),
                'is_climax': bool(is_climax.iloc[-1]),
                'volume_momentum': float(volume_momentum.iloc[-1]),
                'is_churning': bool(is_churning.iloc[-1]),
                'volume_anomaly': bool(self.data['volume'].iloc[-1] > volume_upper.iloc[-1]),
                'volume_stats': {
                    'current_volume': float(self.data['volume'].iloc[-1]),
                    'average_volume': float(medium_ma.iloc[-1]),
                    'volume_std': float(volume_std.iloc[-1])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume patterns: {str(e)}")
            return {}
    
    def volume_weighted_signals(self) -> Dict[str, float]:
        """
        Generate volume-weighted trading signals.
        
        Returns:
            Dictionary with signal indicators and confidence scores
        """
        signals = {}
        
        try:
            # Calculate all volume indicators if not already done
            if 'vwap' not in self._results:
                self.calculate_vwap()
            if 'volume_spikes' not in self._results:
                self.detect_volume_spikes()
            if 'vol_price_corr' not in self._results:
                self.calculate_volume_trend_correlation()
            if 'obv' not in self._results:
                self.calculate_obv()
            
            # Get latest values
            latest = self.data.iloc[-1]
            latest_vwap = self._results['vwap'].iloc[-1]
            latest_vol_spike = self._results['volume_spikes'].iloc[-1]
            latest_corr = self._results['vol_price_corr'].iloc[-1]
            latest_obv = self._results['obv'].iloc[-1]
            obv_trend = self._results['obv'].diff(3).iloc[-1]  # 3-period OBV change
            
            # VWAP signals
            if latest['close'] > latest_vwap:
                signals['vwap_signal'] = 'BUY'
                signals['vwap_strength'] = min(100, ((latest['close'] / latest_vwap - 1) * 1000))
            elif latest['close'] < latest_vwap:
                signals['vwap_signal'] = 'SELL'
                signals['vwap_strength'] = min(100, ((1 - latest['close'] / latest_vwap) * 1000))
            else:
                signals['vwap_signal'] = 'NEUTRAL'
                signals['vwap_strength'] = 0
                
            # Volume spike signals
            if latest_vol_spike == 1:
                if latest['close'] > latest['open']:
                    signals['spike_signal'] = 'BUY'
                    signals['spike_strength'] = 80  # High confidence for volume spike with price rise
                elif latest['close'] < latest['open']:
                    signals['spike_signal'] = 'SELL'
                    signals['spike_strength'] = 80  # High confidence for volume spike with price drop
                else:
                    signals['spike_signal'] = 'NEUTRAL'
                    signals['spike_strength'] = 40  # Medium confidence for volume spike with no price change
            else:
                signals['spike_signal'] = 'NEUTRAL'
                signals['spike_strength'] = 0
                
            # Volume-price correlation signals
            if not np.isnan(latest_corr):
                if latest_corr > 0.7:  # Strong positive correlation
                    if latest['close'] > latest['open']:
                        signals['corr_signal'] = 'BUY'
                        signals['corr_strength'] = min(100, latest_corr * 100)
                    else:
                        signals['corr_signal'] = 'SELL'
                        signals['corr_strength'] = min(100, latest_corr * 100)
                elif latest_corr < -0.7:  # Strong negative correlation
                    if latest['close'] < latest['open']:
                        signals['corr_signal'] = 'BUY'
                        signals['corr_strength'] = min(100, abs(latest_corr) * 100)
                    else:
                        signals['corr_signal'] = 'SELL'
                        signals['corr_strength'] = min(100, abs(latest_corr) * 100)
                else:
                    signals['corr_signal'] = 'NEUTRAL'
                    signals['corr_strength'] = 0
            else:
                signals['corr_signal'] = 'NEUTRAL'
                signals['corr_strength'] = 0
                
            # OBV signals
            if obv_trend > 0:
                signals['obv_signal'] = 'BUY'
                signals['obv_strength'] = min(100, abs(obv_trend) / latest_obv * 1000 if latest_obv != 0 else 50)
            elif obv_trend < 0:
                signals['obv_signal'] = 'SELL'
                signals['obv_strength'] = min(100, abs(obv_trend) / latest_obv * 1000 if latest_obv != 0 else 50)
            else:
                signals['obv_signal'] = 'NEUTRAL'
                signals['obv_strength'] = 0
                
            # Consolidated volume signal
            buy_count = sum(1 for key in signals if key.endswith('_signal') and signals[key] == 'BUY')
            sell_count = sum(1 for key in signals if key.endswith('_signal') and signals[key] == 'SELL')
            
            total_strength = sum(signals[key] for key in signals if key.endswith('_strength'))
            buy_strength = sum(signals[key] for key in signals if key.endswith('_strength') and 
                              signals[key.replace('_strength', '_signal')] == 'BUY')
            sell_strength = sum(signals[key] for key in signals if key.endswith('_strength') and 
                               signals[key.replace('_strength', '_signal')] == 'SELL')
            
            if buy_count > sell_count:
                signals['volume_signal'] = 'BUY'
                signals['volume_confidence'] = buy_strength / total_strength * 100 if total_strength > 0 else 0
            elif sell_count > buy_count:
                signals['volume_signal'] = 'SELL'
                signals['volume_confidence'] = sell_strength / total_strength * 100 if total_strength > 0 else 0
            else:
                if buy_strength > sell_strength:
                    signals['volume_signal'] = 'BUY'
                    signals['volume_confidence'] = buy_strength / total_strength * 100 if total_strength > 0 else 0
                elif sell_strength > buy_strength:
                    signals['volume_signal'] = 'SELL'
                    signals['volume_confidence'] = sell_strength / total_strength * 100 if total_strength > 0 else 0
                else:
                    signals['volume_signal'] = 'NEUTRAL'
                    signals['volume_confidence'] = 0
                    
            logger.info(f"Generated volume signals - Direction: {signals['volume_signal']}, Confidence: {signals['volume_confidence']:.1f}%")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating volume weighted signals: {str(e)}")
            return {
                'volume_signal': 'NEUTRAL',
                'volume_confidence': 0
            }
    
    def analyze_volume(self) -> Dict:
        """
        Run comprehensive volume analysis.
        
        Returns:
            Dictionary with all volume analysis results
        """
        try:
            # Calculate all indicators
            self.calculate_vwap()
            self.detect_volume_spikes()
            self.calculate_volume_trend_correlation()
            self.calculate_obv()
            
            # Generate signals
            signals = self.volume_weighted_signals()
            
            # Add indicators to signals
            signals['vwap'] = self._results['vwap'].iloc[-1]
            signals['obv'] = self._results['obv'].iloc[-1]
            signals['vol_price_corr'] = self._results['vol_price_corr'].iloc[-1]
            signals['volume_spike'] = bool(self._results['volume_spikes'].iloc[-1])
            
            logger.info("Volume analysis completed successfully")
            return signals
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {str(e)}")
            return {
                'volume_signal': 'NEUTRAL',
                'volume_confidence': 0
            }
    
    def analyze_volume_profile(self, num_bins: int = 50) -> Dict:
        """
        Calculate and analyze volume profile metrics.
        
        Args:
            num_bins: Number of price levels for volume distribution
            
        Returns:
            Dictionary with volume profile analysis
        """
        try:
            # Calculate price range and bins
            price_range = (self.data['high'].max() - self.data['low'].min())
            bin_size = price_range / num_bins
            price_levels = np.linspace(self.data['low'].min(), self.data['high'].max(), num_bins + 1)
            
            # Initialize volume profile
            volume_profile = np.zeros(num_bins)
            
            # Calculate volume distribution
            for i in range(len(self.data)):
                price_idx = int((self.data['close'].iloc[i] - self.data['low'].min()) / bin_size)
                if 0 <= price_idx < num_bins:
                    volume_profile[price_idx] += self.data['volume'].iloc[i]
            
            # Find point of control (price level with highest volume)
            poc_idx = np.argmax(volume_profile)
            poc_price = price_levels[poc_idx]
            
            # Calculate value area (70% of total volume)
            sorted_idx = np.argsort(volume_profile)[::-1]
            cumsum_volume = np.cumsum(volume_profile[sorted_idx])
            value_area_idx = cumsum_volume <= (np.sum(volume_profile) * 0.7)
            value_area_prices = price_levels[sorted_idx[value_area_idx]]
            
            # Calculate volume weighted average price
            vwap = np.sum(self.data['close'] * self.data['volume']) / np.sum(self.data['volume'])
            
            # Volume profile metrics
            profile_metrics = {
                'point_of_control': float(poc_price),
                'value_area_high': float(value_area_prices.max()),
                'value_area_low': float(value_area_prices.min()),
                'vwap': float(vwap),
                'volume_distribution': {
                    'price_levels': price_levels[:-1].tolist(),
                    'volumes': volume_profile.tolist()
                }
            }
            
            # Volume concentration analysis
            total_volume = np.sum(volume_profile)
            poc_concentration = volume_profile[poc_idx] / total_volume
            value_area_concentration = np.sum(volume_profile[sorted_idx[value_area_idx]]) / total_volume
            
            profile_metrics.update({
                'volume_concentration': {
                    'poc_concentration': float(poc_concentration),
                    'value_area_concentration': float(value_area_concentration)
                }
            })
            
            return profile_metrics
            
        except Exception as e:
            logger.error(f"Error analyzing volume profile: {str(e)}")
            return {}