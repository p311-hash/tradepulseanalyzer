"""
Test script for fractal indicators and strategy

This script tests the implementation of Bill Williams' Fractal indicator
and the corresponding strategy with increased confidence.
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta

from technical_analysis import calculate_fractals, fractal_strategy, calculate_indicators

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create test data with known fractal patterns"""
    # Create datetime index - last 100 minutes
    now = datetime.now()
    dates = [now - timedelta(minutes=i) for i in range(100)]
    dates.reverse()  # Ascending order
    
    # Create price data with known fractal patterns
    # We'll create a wave pattern that will have clear bullish and bearish fractals
    np.random.seed(42)  # For reproducibility
    
    # Start with a sine wave
    base_price = 1.1000
    x = np.linspace(0, 4*np.pi, 100)
    wave = np.sin(x) * 0.0050
    
    # Add some noise
    noise = np.random.normal(0, 0.0005, 100)
    
    # Create OHLC data
    opens = base_price + wave + noise
    closes = base_price + wave + np.random.normal(0, 0.0005, 100)
    highs = np.maximum(opens, closes) + np.random.uniform(0.0001, 0.0010, 100)
    lows = np.minimum(opens, closes) - np.random.uniform(0.0001, 0.0010, 100)
    
    # Create DataFrame
    data = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': np.random.normal(500, 100, 100)
    }, index=dates)
    
    return data

def test_fractal_calculation():
    """Test the calculation of Bill Williams' Fractals"""
    logger.info("=== Testing Fractal Calculation ===")
    
    # Create test data
    data = create_test_data()
    logger.info(f"Created test data with {len(data)} candles")
    
    # Calculate fractals
    bullish_fractals, bearish_fractals = calculate_fractals(data)
    
    # Count fractal signals
    bullish_count = (bullish_fractals > 0).sum()
    bearish_count = (bearish_fractals > 0).sum()
    
    logger.info(f"Detected {bullish_count} bullish fractals and {bearish_count} bearish fractals")
    
    # Verify we have a reasonable number of fractals (typically 10-20% of candles should be fractals)
    total_fractals = bullish_count + bearish_count
    fractal_percentage = total_fractals / len(data) * 100
    
    logger.info(f"Fractal percentage: {fractal_percentage:.2f}%")
    assert 5 <= fractal_percentage <= 40, "Fractal percentage outside reasonable range"
    
    # Verify fractal logic
    # Pick a bullish fractal and verify it's a local minimum
    if bullish_count > 0:
        idx = np.where(bullish_fractals > 0)[0][0]
        if idx > 2 and idx < len(data) - 2:
            bullish_idx = data.index[idx]
            bullish_value = bullish_fractals[idx]
            
            # Check actual fractal criteria (low with two higher lows on each side)
            current_low = data.loc[bullish_idx, 'low']
            prev_lows = [data.iloc[idx-i]['low'] for i in range(1, 3)]
            next_lows = [data.iloc[idx+i]['low'] for i in range(1, 3)]
            
            all_higher = all(current_low < l for l in prev_lows + next_lows)
            logger.info(f"Bullish fractal at index {idx} (value: {bullish_value:.6f}) - All surrounding lows higher: {all_higher}")
            assert all_higher, "Bullish fractal criteria not met"
            logger.info("✓ Bullish fractal logic verified")
    
    # Pick a bearish fractal and verify it's a local maximum
    if bearish_count > 0:
        idx = np.where(bearish_fractals > 0)[0][0]
        if idx > 2 and idx < len(data) - 2:
            bearish_idx = data.index[idx]
            bearish_value = bearish_fractals[idx]
            
            # Check actual fractal criteria (high with two lower highs on each side)
            current_high = data.loc[bearish_idx, 'high']
            prev_highs = [data.iloc[idx-i]['high'] for i in range(1, 3)]
            next_highs = [data.iloc[idx+i]['high'] for i in range(1, 3)]
            
            all_lower = all(current_high > h for h in prev_highs + next_highs)
            logger.info(f"Bearish fractal at index {idx} (value: {bearish_value:.6f}) - All surrounding highs lower: {all_lower}")
            assert all_lower, "Bearish fractal criteria not met"
            logger.info("✓ Bearish fractal logic verified")
    
    return data, bullish_fractals, bearish_fractals

def test_fractal_strategy():
    """Test the fractal strategy with confirmation indicators"""
    logger.info("\n=== Testing Fractal Strategy ===")
    
    # Create test data
    data = create_test_data()
    
    # Calculate all indicators
    data_with_indicators = calculate_indicators(data)
    
    # Run the fractal strategy
    signal = fractal_strategy(data_with_indicators)
    
    logger.info(f"Fractal strategy signal: {signal['direction']} with {signal['confidence']:.2f}% confidence")
    
    # Verify confidence level is elevated (should be at least 70%)
    assert signal['confidence'] >= 70 or signal['confidence'] == 0, "Confidence not elevated to at least 70%"
    logger.info(f"✓ Confidence level verified: {signal['confidence']:.2f}%")
    
    # Ensure signal direction is valid
    assert signal['direction'] in ['BUY', 'SELL', 'NEUTRAL'], f"Invalid signal direction: {signal['direction']}"
    logger.info(f"✓ Signal direction verified: {signal['direction']}")
    
    # Test with modified data to force different signals
    # Create bullish fractal scenario
    bullish_data = data_with_indicators.copy()
    
    # Set a bullish fractal in recent data
    bullish_data.iloc[-2, bullish_data.columns.get_loc('bullish_fractal')] = bullish_data.iloc[-2]['low']
    bullish_data.iloc[-2, bullish_data.columns.get_loc('fractal_signal')] = 1
    
    # Set confirmation indicators to bullish
    bullish_data.iloc[-1, bullish_data.columns.get_loc('ma_20')] = 1.1000
    bullish_data.iloc[-1, bullish_data.columns.get_loc('ma_50')] = 1.0980
    bullish_data.iloc[-1, bullish_data.columns.get_loc('rsi')] = 55
    bullish_data.iloc[-1, bullish_data.columns.get_loc('psar')] = 1.0970
    
    # Run strategy
    bullish_signal = fractal_strategy(bullish_data)
    logger.info(f"Forced BULLISH scenario - Signal: {bullish_signal['direction']} with {bullish_signal['confidence']:.2f}% confidence")
    
    # Create bearish fractal scenario
    bearish_data = data_with_indicators.copy()
    
    # Set a bearish fractal in recent data
    bearish_data.iloc[-2, bearish_data.columns.get_loc('bearish_fractal')] = bearish_data.iloc[-2]['high']
    bearish_data.iloc[-2, bearish_data.columns.get_loc('fractal_signal')] = -1
    
    # Set confirmation indicators to bearish
    bearish_data.iloc[-1, bearish_data.columns.get_loc('ma_20')] = 1.0980
    bearish_data.iloc[-1, bearish_data.columns.get_loc('ma_50')] = 1.1000
    bearish_data.iloc[-1, bearish_data.columns.get_loc('rsi')] = 45
    bearish_data.iloc[-1, bearish_data.columns.get_loc('psar')] = 1.1020
    
    # Run strategy
    bearish_signal = fractal_strategy(bearish_data)
    logger.info(f"Forced BEARISH scenario - Signal: {bearish_signal['direction']} with {bearish_signal['confidence']:.2f}% confidence")
    
    # Verify signals
    assert bullish_signal['direction'] == 'BUY', "Failed to generate BUY signal in bullish scenario"
    assert bearish_signal['direction'] == 'SELL', "Failed to generate SELL signal in bearish scenario"
    
    logger.info("✓ Strategy signals verified for both bullish and bearish scenarios")
    logger.info("✓ All fractal strategy tests passed!")
    
    return bullish_signal, bearish_signal

def test_strategy_confidence():
    """Test that confidence levels are appropriately increased"""
    logger.info("\n=== Testing Strategy Confidence ===")
    
    # Create test data
    data = create_test_data()
    data_with_indicators = calculate_indicators(data)
    
    # 1. Test with no confirmations
    basic_data = data_with_indicators.copy()
    basic_data.iloc[-2, basic_data.columns.get_loc('bullish_fractal')] = basic_data.iloc[-2]['low']
    basic_data.iloc[-2, basic_data.columns.get_loc('fractal_signal')] = 1
    
    basic_signal = fractal_strategy(basic_data)
    
    # 2. Test with one confirmation
    one_confirm_data = basic_data.copy()
    one_confirm_data.iloc[-1, one_confirm_data.columns.get_loc('ma_20')] = 1.1000
    one_confirm_data.iloc[-1, one_confirm_data.columns.get_loc('ma_50')] = 1.0980
    
    one_confirm_signal = fractal_strategy(one_confirm_data)
    
    # 3. Test with all confirmations
    all_confirm_data = one_confirm_data.copy()
    all_confirm_data.iloc[-1, all_confirm_data.columns.get_loc('rsi')] = 55
    all_confirm_data.iloc[-1, all_confirm_data.columns.get_loc('psar')] = 1.0970
    
    all_confirm_signal = fractal_strategy(all_confirm_data)
    
    # Log and verify results
    logger.info(f"Basic signal confidence: {basic_signal['confidence']:.2f}%")
    logger.info(f"One confirmation signal confidence: {one_confirm_signal['confidence']:.2f}%")
    logger.info(f"All confirmations signal confidence: {all_confirm_signal['confidence']:.2f}%")
    
    # Verify confidence increases with more confirmations (or maxes out at 100%)
    assert all_confirm_signal['confidence'] >= one_confirm_signal['confidence'], "Confidence decreased with more confirmations"
    assert one_confirm_signal['confidence'] >= basic_signal['confidence'], "Confidence not increasing with confirmation"
    
    # Check if we've hit the maximum confidence cap of 100%
    if one_confirm_signal['confidence'] == 100.0 and all_confirm_signal['confidence'] == 100.0:
        logger.info("Maximum confidence cap of 100% reached with only one confirmation")
    
    # Verify base confidence is at least 70%
    assert basic_signal['confidence'] >= 70, "Base confidence not at least 70%"
    
    logger.info("✓ Confidence levels increasing with more confirmations")
    logger.info("✓ Base confidence level verified at 70% or higher")
    logger.info("✓ All confidence tests passed!")
    
    return basic_signal, one_confirm_signal, all_confirm_signal

def main():
    """Run all tests"""
    logger.info("Starting tests for Bill Williams' Fractal indicator and strategy")
    
    # Test fractal calculation
    data, bullish_fractals, bearish_fractals = test_fractal_calculation()
    
    # Test fractal strategy
    bullish_signal, bearish_signal = test_fractal_strategy()
    
    # Test strategy confidence
    basic_signal, one_confirm_signal, all_confirm_signal = test_strategy_confidence()
    
    logger.info("\n========================================")
    logger.info("All tests completed successfully!")
    logger.info("Bill Williams' Fractal indicator and strategy fully operational")
    logger.info("========================================")

if __name__ == "__main__":
    main()