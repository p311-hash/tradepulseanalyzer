"""
Test script for signal enhancement features.
This tests volume analysis, market regime detection, sequence patterns, and correlation analysis.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import enhancement modules
from volume_analysis import VolumeAnalyzer
from market_regime import MarketRegimeDetector, MarketRegime
from sequence_patterns import SequencePatternRecognizer
from signal_correlation import SignalCorrelationAnalyzer

# Import platform access for data
from BinaryOptionsTools.platforms.pocketoption.stable_api import PocketOption

def create_test_data():
    """Create test data for signal enhancements."""
    # Generate timestamps
    timestamps = [(datetime.now() - timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S") 
                 for i in range(200, 0, -1)]
    
    # Starting price
    price = 1.10
    
    # Generate OHLCV data
    data = []
    for i in range(len(timestamps)):
        # Add some randomness to price movement
        change = np.random.normal(0, 0.0005)
        price = price * (1 + change)
        
        # Create OHLCV data
        open_price = price * (1 + np.random.normal(0, 0.0002))
        high_price = max(open_price, price) * (1 + abs(np.random.normal(0, 0.0003)))
        low_price = min(open_price, price) * (1 - abs(np.random.normal(0, 0.0003)))
        volume = abs(np.random.normal(500, 300))
        
        data.append({
            'timestamp': timestamps[i],
            'open': open_price,
            'high': high_price,
            'low': low_price,
            'close': price,
            'volume': volume
        })
    
    # Create DataFrame
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    logger.info(f"Created test data with {len(df)} candles")
    return df

def test_volume_analysis():
    """Test volume analysis functionality."""
    logger.info("=== Testing Volume Analysis ===")
    
    # Create test data
    df = create_test_data()
    
    # Initialize volume analyzer
    volume_analyzer = VolumeAnalyzer(df)
    
    # Test VWAP calculation
    vwap = volume_analyzer.calculate_vwap()
    logger.info(f"VWAP calculated with shape {vwap.shape}")
    logger.info(f"Latest VWAP: {vwap.iloc[-1]:.6f}")
    
    # Test volume spike detection
    spikes = volume_analyzer.detect_volume_spikes()
    spike_count = spikes.sum()
    logger.info(f"Detected {spike_count} volume spikes")
    
    # Test volume-price correlation
    corr = volume_analyzer.calculate_volume_trend_correlation()
    logger.info(f"Volume-price correlation: {corr.iloc[-1]:.3f}")
    
    # Test OBV calculation
    obv = volume_analyzer.calculate_obv()
    logger.info(f"On-Balance Volume trend: {'Positive' if obv.iloc[-1] > obv.iloc[-10] else 'Negative'}")
    
    # Test signal generation
    signals = volume_analyzer.volume_weighted_signals()
    logger.info(f"Volume signals: {signals['volume_signal']} with {signals['volume_confidence']:.1f}% confidence")
    
    # Test comprehensive analysis
    analysis = volume_analyzer.analyze_volume()
    logger.info(f"Volume analysis results: Signal={analysis['volume_signal']}, Confidence={analysis['volume_confidence']:.1f}%")
    
    return analysis

def test_market_regime_detection():
    """Test market regime detection functionality."""
    logger.info("=== Testing Market Regime Detection ===")
    
    # Create test data
    df = create_test_data()
    
    # Initialize regime detector
    regime_detector = MarketRegimeDetector(df)
    
    # Test ADX calculation
    adx, plus_di, minus_di = regime_detector.calculate_adx()
    logger.info(f"ADX calculated with shape {adx.shape}")
    logger.info(f"Latest ADX: {adx[-1]:.2f}, +DI: {plus_di[-1]:.2f}, -DI: {minus_di[-1]:.2f}")
    
    # Test volatility calculation
    volatility = regime_detector.calculate_volatility()
    logger.info(f"Latest volatility: {volatility[-1]:.6f}")
    
    # Test Bollinger width calculation
    bb_width = regime_detector.calculate_bollinger_width()
    logger.info(f"Latest Bollinger width: {bb_width[-1]:.6f}")
    
    # Test support/resistance test
    sr_test = regime_detector.detect_support_resistance_test()
    logger.info(f"S/R test: Testing support={sr_test['is_testing_support']}, Testing resistance={sr_test['is_testing_resistance']}")
    
    # Test regime detection
    regime = regime_detector.detect_regime(df)
    logger.info(f"Detected regime: {regime['regime']} with {regime['confidence']:.1f}% confidence")
    if 'explanations' in regime:
        logger.info(f"Regime explanations: {', '.join(regime['explanations'])}")
    
    # Test optimal strategy
    strategy = regime_detector.get_optimal_strategy()
    logger.info(f"Optimal strategy for {regime['regime']}: {strategy['description']}")
    if 'recommended_signal' in strategy and 'signal_confidence_boost' in strategy:
        logger.info(f"Signal parameters: {strategy['recommended_signal']} with {strategy['signal_confidence_boost']}% boost")
    
    return regime

def test_sequence_pattern_recognition():
    """Test sequence pattern recognition functionality."""
    logger.info("=== Testing Sequence Pattern Recognition ===")
    
    # Create test data with specific pattern
    df = create_test_data()
    
    # Force a three white soldiers pattern in the last 3 candles
    # First bullish candle
    df.iloc[-3, df.columns.get_indexer(['open'])[0]] = 1.09
    df.iloc[-3, df.columns.get_indexer(['close'])[0]] = 1.095
    df.iloc[-3, df.columns.get_indexer(['high'])[0]] = 1.096
    df.iloc[-3, df.columns.get_indexer(['low'])[0]] = 1.089
    
    # Second bullish candle
    df.iloc[-2, df.columns.get_indexer(['open'])[0]] = 1.095
    df.iloc[-2, df.columns.get_indexer(['close'])[0]] = 1.10
    df.iloc[-2, df.columns.get_indexer(['high'])[0]] = 1.101
    df.iloc[-2, df.columns.get_indexer(['low'])[0]] = 1.094
    
    # Third bullish candle
    df.iloc[-1, df.columns.get_indexer(['open'])[0]] = 1.10
    df.iloc[-1, df.columns.get_indexer(['close'])[0]] = 1.105
    df.iloc[-1, df.columns.get_indexer(['high'])[0]] = 1.106
    df.iloc[-1, df.columns.get_indexer(['low'])[0]] = 1.099
    
    # Initialize pattern recognizer
    pattern_recognizer = SequencePatternRecognizer(df)
    
    # Test pattern recognition
    patterns = pattern_recognizer.recognize_patterns()
    if patterns:
        logger.info(f"Detected sequence patterns: {', '.join(patterns.keys())}")
        for pattern in patterns:
            description = pattern_recognizer.get_pattern_description(pattern)
            logger.info(f"Pattern description: {description}")
            
            # Get pattern image path
            image_path = pattern_recognizer.get_pattern_svg_path(pattern)
            if image_path and os.path.exists(image_path):
                logger.info(f"Pattern image available at: {image_path}")
    else:
        logger.info("No sequence patterns detected")
    
    # Test pattern signal generation
    signal = pattern_recognizer.generate_pattern_signal()
    logger.info(f"Pattern signal: {signal['pattern_signal']} with {signal['pattern_confidence']:.1f}% confidence")
    
    return patterns

def test_signal_correlation():
    """Test signal correlation functionality."""
    logger.info("=== Testing Signal Correlation ===")
    
    # Create sample signal history
    signal_history = {
        'signal1': {
            'pair': 'EURUSD_otc',
            'signal': 'BUY',
            'timestamp': (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': 'win'
        },
        'signal2': {
            'pair': 'GBPUSD_otc',
            'signal': 'BUY',
            'timestamp': (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': 'win'
        },
        'signal3': {
            'pair': 'USDCHF_otc',
            'signal': 'SELL',
            'timestamp': (datetime.now() - timedelta(hours=3)).strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': 'win'
        },
        'signal4': {
            'pair': 'EURUSD_otc',
            'signal': 'SELL',
            'timestamp': (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': 'loss'
        },
        'signal5': {
            'pair': 'GBPUSD_otc',
            'signal': 'SELL',
            'timestamp': (datetime.now() - timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            'outcome': 'loss'
        }
    }
    
    # Save to temporary file
    with open('temp_signal_history.json', 'w') as f:
        json.dump(signal_history, f)
    
    # Initialize correlation analyzer
    correlation_analyzer = SignalCorrelationAnalyzer('temp_signal_history.json')
    
    # Test dynamic correlation calculation
    correlations = correlation_analyzer.calculate_dynamic_correlations()
    logger.info(f"Dynamic correlations calculated for {len(correlations)} pairs")
    
    # Test correlated pairs
    eurusd_correlated = correlation_analyzer.get_correlated_pairs('EURUSD_otc')
    if eurusd_correlated:
        logger.info(f"EURUSD_otc correlations: {eurusd_correlated}")
    
    # Test correlated signal analysis
    recent_signals = {
        'EURUSD_otc': {'signal': 'BUY', 'confidence': 70},
        'GBPUSD_otc': {'signal': 'BUY', 'confidence': 65},
        'USDCHF_otc': {'signal': 'SELL', 'confidence': 75},
    }
    
    analysis = correlation_analyzer.analyze_correlated_signals('EURUSD_otc', 'BUY', recent_signals)
    logger.info(f"Correlation analysis: confirmation={analysis['confirmation_level']:.1f}%, boost={analysis['correlation_boost']:.1f}%")
    
    # Test divergence detection
    df = create_test_data()
    recent_signals['EURUSD_otc']['price_change_pct'] = 0.5
    recent_signals['GBPUSD_otc']['price_change_pct'] = 0.4
    recent_signals['USDCHF_otc']['price_change_pct'] = -0.4
    
    divergence = correlation_analyzer.get_divergence('EURUSD_otc', df, recent_signals)
    logger.info(f"Divergence detected: {divergence['divergence_detected']}")
    if divergence['divergence_detected']:
        logger.info(f"Divergence potential: {divergence['divergence_potential']:.2f}")
    
    # Clean up temporary file
    if os.path.exists('temp_signal_history.json'):
        os.remove('temp_signal_history.json')
    
    return analysis

def test_combined_signal_enhancements():
    """Test all signal enhancement features together."""
    logger.info("=== Testing Combined Signal Enhancements ===")
    
    # Create test data
    df = create_test_data()
    
    # Run each enhancement
    volume_results = VolumeAnalyzer(df).analyze_volume()
    regime_results = MarketRegimeDetector(df).detect_regime()
    pattern_results = SequencePatternRecognizer(df).generate_pattern_signal()
    
    # Create sample recent signals for correlation analysis
    recent_signals = {
        'EURUSD_otc': {'signal': 'BUY', 'confidence': 70, 'price_change_pct': 0.5},
        'GBPUSD_otc': {'signal': 'BUY', 'confidence': 65, 'price_change_pct': 0.4},
        'USDCHF_otc': {'signal': 'SELL', 'confidence': 75, 'price_change_pct': -0.4},
    }
    
    correlation_results = SignalCorrelationAnalyzer().analyze_correlated_signals('EURUSD_otc', 'BUY', recent_signals)
    
    # Combine results for enhanced signal
    signal = 'NEUTRAL'
    confidence = 0
    explanations = []
    
    # Consider volume signal
    if volume_results['volume_signal'] != 'NEUTRAL':
        signal = volume_results['volume_signal']
        confidence = volume_results['volume_confidence']
        explanations.append(f"Volume analysis: {signal} ({confidence:.1f}%)")
    
    # Apply market regime
    strategy = MarketRegimeDetector(df).get_optimal_strategy()
    if strategy['recommended_signal'] != 'NEUTRAL':
        if signal == 'NEUTRAL':
            signal = strategy['recommended_signal']
        
        # Apply confidence boost if signals agree
        if signal == strategy['recommended_signal']:
            confidence += strategy['signal_confidence_boost']
            explanations.append(f"Market regime ({strategy['regime']}): +{strategy['signal_confidence_boost']}% confidence")
    
    # Apply pattern signal
    if pattern_results['pattern_signal'] != 'NEUTRAL':
        if signal == 'NEUTRAL':
            signal = pattern_results['pattern_signal']
            confidence = pattern_results['pattern_confidence']
            explanations.append(f"Pattern signal: {signal} ({confidence:.1f}%)")
        elif signal == pattern_results['pattern_signal']:
            confidence = (confidence + pattern_results['pattern_confidence']) / 2
            confidence += 10  # Bonus for pattern confirmation
            explanations.append(f"Pattern confirmation: +10% confidence")
    
    # Apply correlation analysis
    if correlation_results['correlated_confirmation']:
        confidence += correlation_results['correlation_boost']
        explanations.append(f"Correlation boost: +{correlation_results['correlation_boost']:.1f}%")
    
    # Cap confidence at 100%
    confidence = min(confidence, 100)
    
    # Generate final signal
    enhanced_signal = {
        'signal': signal,
        'confidence': confidence,
        'explanations': explanations,
        'volume_analysis': volume_results,
        'market_regime': regime_results,
        'pattern_signal': pattern_results,
        'correlation_analysis': correlation_results
    }
    
    logger.info(f"Enhanced signal: {signal} with {confidence:.1f}% confidence")
    for explanation in explanations:
        logger.info(f"- {explanation}")
    
    return enhanced_signal

def test_with_real_data():
    """Test enhancement features with real market data if available."""
    logger.info("=== Testing with Real Market Data ===")
    
    try:
        # Initialize PocketOption API in demo mode
        api = PocketOption(demo=True)
        
        # Get EURUSD_otc data
        candles = api.get_candles('EURUSD', 60, 200)  # 1-minute candles, 200 candles
        
        # Check if candles is a DataFrame and not empty
        if isinstance(candles, pd.DataFrame):
            if candles.empty or len(candles) < 10:
                logger.warning("Not enough candle data available for testing")
                return None
        else:
            logger.warning("Invalid candle data format received")
            return None
            
        # The data should already be a DataFrame from the API
        df = candles
        
        # Check if we need to rename columns (only if they don't match our expected format)
        if 'at' in df.columns and 'open' not in df.columns:
            # Rename columns to match our expected format
            df.rename(columns={
                'at': 'timestamp',
                'o': 'open',
                'h': 'high', 
                'l': 'low',
                'c': 'close',
                'v': 'volume'
            }, inplace=True)
            
            # Set timestamp as index if it exists as a column
            if 'timestamp' in df.columns:
                df.set_index('timestamp', inplace=True)
        
        logger.info(f"Retrieved {len(df)} candles from PocketOption API")
        
        # Run each enhancement test with real data
        volume_results = VolumeAnalyzer(df).analyze_volume()
        logger.info(f"Volume signal with real data: {volume_results['volume_signal']} ({volume_results['volume_confidence']:.1f}%)")
        
        regime_results = MarketRegimeDetector(df).detect_regime()
        logger.info(f"Market regime with real data: {regime_results['regime']} ({regime_results['confidence']:.1f}%)")
        
        pattern_results = SequencePatternRecognizer(df).generate_pattern_signal()
        logger.info(f"Pattern signal with real data: {pattern_results['pattern_signal']} ({pattern_results['pattern_confidence']:.1f}%)")
        
        return {
            'volume_results': volume_results,
            'regime_results': regime_results,
            'pattern_results': pattern_results
        }
        
    except Exception as e:
        logger.error(f"Error testing with real data: {str(e)}", exc_info=True)
        return None
        
def main():
    """Run all tests."""
    logger.info("Starting signal enhancement tests")
    
    # Test each enhancement feature individually
    volume_analysis = test_volume_analysis()
    market_regime = test_market_regime_detection()
    sequence_patterns = test_sequence_pattern_recognition()
    correlation = test_signal_correlation()
    
    # Test combined enhancements
    combined = test_combined_signal_enhancements()
    
    # Test with real data if available
    real_data_results = test_with_real_data()
    
    logger.info("Signal enhancement tests completed")
    
if __name__ == "__main__":
    main()