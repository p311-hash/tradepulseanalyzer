import numpy as np
from scipy import signal
import pywt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pandas as pd
from datetime import datetime

@dataclass
class SignalComponents:
    trend: np.ndarray
    cycle: np.ndarray
    noise: np.ndarray

class AdvancedSignalProcessor:    
    def __init__(self):
        self.kalman_states = {}
        self.wavelet_levels = 4
        self.fourier_window = 128
        self.signal_history = []
        self.validation_window = 20
        self.adaptive_threshold = 0.75
        self.regime_specific_params = {
            'TRENDING_UP': {
                'kalman_process_noise': 0.001,
                'wavelet_levels': 3,
                'fourier_window': 96
            },
            'TRENDING_DOWN': {
                'kalman_process_noise': 0.001,
                'wavelet_levels': 3,
                'fourier_window': 96
            },
            'RANGING_LOW_VOL': {
                'kalman_process_noise': 0.0005,
                'wavelet_levels': 5,
                'fourier_window': 160
            },
            'RANGING_HIGH_VOL': {
                'kalman_process_noise': 0.002,
                'wavelet_levels': 4,
                'fourier_window': 128
            },
            'CHOPPY': {
                'kalman_process_noise': 0.003,
                'wavelet_levels': 6,
                'fourier_window': 192
            }
        }
        
    def apply_kalman_filter(self, price_data: np.ndarray,
                           measurement_noise: float = 0.1,
                           process_noise: float = 0.001) -> np.ndarray:
        """
        Apply Kalman filter for noise reduction and trend estimation
        """
        n = len(price_data)
        # State transition matrix
        A = np.array([[1, 1], [0, 1]])
        # Observation matrix
        H = np.array([[1, 0]])
        # Process noise covariance
        Q = np.array([[process_noise, 0], [0, process_noise]])
        # Measurement noise covariance
        R = np.array([[measurement_noise]])
        
        # Initial state
        x = np.array([[price_data[0]], [0]])
        P = np.array([[1, 0], [0, 1]])
        
        filtered_values = np.zeros(n)
        
        for i in range(n):
            # Predict
            x = A @ x
            P = A @ P @ A.T + Q
            
            # Update
            y = price_data[i] - H @ x
            S = H @ P @ H.T + R
            K = P @ H.T / S
            
            x = x + K * y
            P = (np.eye(2) - K @ H) @ P
            
            filtered_values[i] = x[0]
            
        return filtered_values
    
    def wavelet_decomposition(self, data: np.ndarray, 
                            wavelet: str = 'db4') -> Dict[str, np.ndarray]:
        """
        Perform wavelet decomposition for multi-scale analysis
        """
        coeffs = pywt.wavedec(data, wavelet, level=self.wavelet_levels)
        reconstructed = {}
        
        # Reconstruct each level
        for i in range(self.wavelet_levels + 1):
            coeff_list = [np.zeros_like(c) for c in coeffs]
            coeff_list[i] = coeffs[i]
            reconstructed[f'level_{i}'] = pywt.waverec(coeff_list, wavelet)
            
        # Calculate trend and noise components
        trend = reconstructed['level_0']  # Approximation
        noise = sum(reconstructed[f'level_{i}'] 
                   for i in range(1, self.wavelet_levels + 1))  # Details
        
        return {
            'trend': trend,
            'noise': noise,
            'levels': reconstructed
        }
    
    def fourier_analysis(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Perform Fourier analysis for frequency domain analysis
        """
        # Apply Hamming window
        window = signal.hamming(self.fourier_window)
        
        # Short-time Fourier transform
        f, t, Zxx = signal.stft(data, window=window,
                               nperseg=self.fourier_window,
                               noverlap=self.fourier_window // 2)
        
        # Calculate power spectrum
        power_spectrum = np.abs(Zxx)**2
        
        # Find dominant frequencies
        dominant_freqs = f[np.argmax(power_spectrum, axis=0)]
        
        return {
            'frequencies': f,
            'time_points': t,
            'spectrogram': Zxx,
            'power_spectrum': power_spectrum,
            'dominant_frequencies': dominant_freqs
        }
    
    def detect_chaos_indicators(self, data: np.ndarray, 
                              embedding_dim: int = 3,
                              delay: int = 1) -> Dict[str, float]:
        """
        Calculate chaos theory indicators (Lyapunov exponent and correlation dimension)
        """
        # Create time-delayed embedding
        n = len(data) - (embedding_dim - 1) * delay
        embedding = np.array([data[i:i + embedding_dim * delay:delay] 
                            for i in range(n)])
        
        # Calculate largest Lyapunov exponent (simplified)
        distances = np.zeros(n-1)
        for i in range(n-1):
            distances[i] = np.linalg.norm(embedding[i+1] - embedding[i])
        lyapunov = np.mean(np.log(distances[distances > 0]))
        
        # Calculate correlation dimension (simplified)
        r = np.std(data) * np.logspace(-2, 0, 20)
        correlation_sum = np.zeros_like(r)
        
        for i, radius in enumerate(r):
            distances = np.linalg.norm(embedding[:, None] - embedding, axis=2)
            correlation_sum[i] = np.sum(distances < radius) / (n * (n-1))
            
        # Estimate correlation dimension from slope
        correlation_dim = np.polyfit(np.log(r), np.log(correlation_sum), 1)[0]
        
        return {
            'lyapunov_exponent': lyapunov,
            'correlation_dimension': correlation_dim,
            'is_chaotic': lyapunov > 0
        }
    
    def process_signal(self, price_data: np.ndarray, deep_market_structure: Optional[Dict] = None, ml_feature_importance: Optional[Dict] = None) -> Dict:
        """
        Comprehensive signal processing combining all methods, with optional deep market structure and ML feedback integration
        """
        # Normalize data
        normalized_data = (price_data - np.mean(price_data)) / np.std(price_data)
        
        # Apply Kalman filter
        filtered_data = self.apply_kalman_filter(normalized_data)
        
        # Wavelet analysis
        wavelet_results = self.wavelet_decomposition(normalized_data)
        
        # Fourier analysis
        fourier_results = self.fourier_analysis(normalized_data)
        
        # Chaos analysis
        chaos_results = self.detect_chaos_indicators(normalized_data)
        
        # Extract main components
        components = SignalComponents(
            trend=wavelet_results['trend'],
            cycle=fourier_results['dominant_frequencies'],
            noise=wavelet_results['noise']
        )
        
        # Integrate deep market structure features if provided
        market_structure_features = {}
        if deep_market_structure is not None:
            # Example: integrate market delta, auction zones, price acceptance
            latest = deep_market_structure[-1] if isinstance(deep_market_structure, list) and deep_market_structure else None
            if latest:
                market_structure_features['market_delta'] = latest.get('delta', 0)
                market_structure_features['cumulative_delta'] = latest.get('cumulative_delta', 0)
                market_structure_features['auction_zones'] = latest.get('zones', {})
                market_structure_features['price_acceptance'] = latest.get('price_acceptance', {})
        
        # Integrate ML feature importance feedback if provided
        feature_importance_feedback = {}
        if ml_feature_importance is not None:
            # Example: pass through top features and their weights
            feature_importance_feedback = ml_feature_importance
        
        return {
            'filtered_data': filtered_data,
            'components': components,
            'wavelet_analysis': wavelet_results,
            'fourier_analysis': fourier_results,
            'chaos_indicators': chaos_results,
            'signal_quality': self._assess_signal_quality(components, chaos_results),
            'market_structure_features': market_structure_features,
            'ml_feature_importance': feature_importance_feedback
        }
    
    def _assess_signal_quality(self, components: SignalComponents,
                             chaos_results: Dict) -> Dict[str, float]:
        """
        Assess the quality and reliability of the processed signal
        """
        # Signal-to-noise ratio
        signal_power = np.var(components.trend)
        noise_power = np.var(components.noise)
        snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
        
        # Trend strength
        trend_strength = signal_power / (signal_power + noise_power)
        
        # Chaos level (normalized Lyapunov exponent)
        chaos_level = np.clip(chaos_results['lyapunov_exponent'], 0, 1)
        
        return {
            'signal_to_noise_ratio': snr,
            'trend_strength': trend_strength,
            'chaos_level': chaos_level,
            'signal_reliability': (0.4 * trend_strength + 
                                 0.4 * (1 - chaos_level) +
                                 0.2 * (np.clip(snr / 20, 0, 1)))
        }
    
    def validate_signal_quality(self, signal_components: SignalComponents,
                              market_regime: str = 'normal') -> Dict[str, float]:
        """
        Validate signal quality using adaptive thresholds and market regime
        """
        # Adjust thresholds based on market regime
        if market_regime == 'high_volatility':
            snr_threshold = 10.0
            trend_threshold = 0.6
            chaos_threshold = 0.7
        elif market_regime == 'low_volatility':
            snr_threshold = 15.0
            trend_threshold = 0.8
            chaos_threshold = 0.4
        else:  # normal regime
            snr_threshold = 12.0
            trend_threshold = 0.7
            chaos_threshold = 0.5
            
        # Calculate signal metrics
        signal_metrics = self._assess_signal_quality(
            signal_components,
            {'lyapunov_exponent': self._calculate_local_lyapunov(signal_components.trend)}
        )
        
        # Adaptive threshold adjustment based on historical performance
        if len(self.signal_history) >= self.validation_window:
            historical_metrics = pd.DataFrame(self.signal_history[-self.validation_window:])
            snr_threshold *= np.clip(
                1 + 0.2 * historical_metrics['signal_reliability'].mean(),
                0.8, 1.2
            )
            
        # Store current metrics for future adaptation
        self.signal_history.append(signal_metrics)
        if len(self.signal_history) > self.validation_window * 2:
            self.signal_history.pop(0)
            
        # Validation results
        validation = {
            'snr_valid': signal_metrics['signal_to_noise_ratio'] > snr_threshold,
            'trend_valid': signal_metrics['trend_strength'] > trend_threshold,
            'chaos_valid': signal_metrics['chaos_level'] < chaos_threshold,
            'overall_valid': signal_metrics['signal_reliability'] > self.adaptive_threshold
        }
        
        return {
            'metrics': signal_metrics,
            'validation': validation,
            'thresholds': {
                'snr': snr_threshold,
                'trend': trend_threshold,
                'chaos': chaos_threshold,
                'reliability': self.adaptive_threshold
            }
        }
    
    def optimize_signal_parameters(self, data: np.ndarray,
                                target_metric: str = 'signal_reliability',
                                optimize_kalman: bool = True,
                                optimize_wavelet: bool = True) -> Dict:
        """
        Optimize signal processing parameters in real-time
        """
        best_params = {}
        best_metric = -float('inf')
        
        if optimize_kalman:
            # Optimize Kalman filter parameters
            for measurement_noise in np.logspace(-3, 0, 10):
                for process_noise in np.logspace(-4, -1, 10):
                    filtered = self.apply_kalman_filter(
                        data,
                        measurement_noise=measurement_noise,
                        process_noise=process_noise
                    )
                    components = SignalComponents(
                        trend=filtered,
                        cycle=np.zeros_like(filtered),  # placeholder
                        noise=data - filtered
                    )
                    metrics = self._assess_signal_quality(
                        components,
                        {'lyapunov_exponent': self._calculate_local_lyapunov(filtered)}
                    )
                    
                    if metrics[target_metric] > best_metric:
                        best_metric = metrics[target_metric]
                        best_params['kalman'] = {
                            'measurement_noise': measurement_noise,
                            'process_noise': process_noise
                        }
                        
        if optimize_wavelet:
            # Optimize wavelet parameters
            wavelet_types = ['db4', 'db6', 'db8', 'sym4', 'sym6']
            for wavelet in wavelet_types:
                for level in range(3, 7):
                    self.wavelet_levels = level
                    decomp = self.wavelet_decomposition(data, wavelet=wavelet)
                    components = SignalComponents(
                        trend=decomp['trend'],
                        cycle=np.zeros_like(data),  # placeholder
                        noise=decomp['noise']
                    )
                    metrics = self._assess_signal_quality(
                        components,
                        {'lyapunov_exponent': self._calculate_local_lyapunov(decomp['trend'])}
                    )
                    
                    if metrics[target_metric] > best_metric:
                        best_metric = metrics[target_metric]
                        best_params['wavelet'] = {
                            'type': wavelet,
                            'levels': level
                        }
                        
        # Update instance parameters with optimized values
        if 'kalman' in best_params:
            self.kalman_measurement_noise = best_params['kalman']['measurement_noise']
            self.kalman_process_noise = best_params['kalman']['process_noise']
            
        if 'wavelet' in best_params:
            self.wavelet_type = best_params['wavelet']['type']
            self.wavelet_levels = best_params['wavelet']['levels']
            
        return {
            'optimized_parameters': best_params,
            'best_metric_value': best_metric,
            'optimization_time': datetime.now().isoformat()
        }
        
    def _calculate_local_lyapunov(self, data: np.ndarray,
                                window: int = 20) -> float:
        """
        Calculate local Lyapunov exponent for recent data
        """
        if len(data) < window:
            return 0.0
            
        recent_data = data[-window:]
        distances = np.diff(recent_data)
        positive_distances = distances[distances > 0]
        
        if len(positive_distances) == 0:
            return 0.0
            
        return np.mean(np.log(positive_distances))
    
    def adapt_to_regime(self, regime: str):
        """Adapt signal processing parameters to current market regime"""
        if regime in self.regime_specific_params:
            params = self.regime_specific_params[regime]
            self.wavelet_levels = params['wavelet_levels']
            self.fourier_window = params['fourier_window']
            self.kalman_process_noise = params['kalman_process_noise']
            
    def process_signal_with_regime(self, price_data: np.ndarray, 
                                 regime_analysis: Dict,
                                 deep_structure: Optional[Dict] = None,
                                 ml_feature_importance: Optional[Dict] = None) -> Dict:
        """Process signal with regime-specific adaptations"""
        # Adapt parameters to current regime
        self.adapt_to_regime(regime_analysis['regime'].value)
        
        # Process signal with adapted parameters
        signal_analysis = self.process_signal(
            price_data, deep_structure, ml_feature_importance
        )
        
        # Adjust signal quality metrics based on regime stability
        stability = regime_analysis.get('stability', 0.5)
        signal_quality = signal_analysis['signal_quality']
        
        # Adjust signal reliability based on regime stability
        adjusted_reliability = signal_quality['signal_reliability'] * (0.5 + 0.5 * stability)
        
        signal_analysis['signal_quality']['signal_reliability'] = adjusted_reliability
        signal_analysis['regime_adaptation'] = {
            'applied_regime': regime_analysis['regime'].value,
            'stability': stability,
            'adapted_parameters': {
                'wavelet_levels': self.wavelet_levels,
                'fourier_window': self.fourier_window,
                'kalman_process_noise': self.kalman_process_noise
            }
        }
        
        return signal_analysis
