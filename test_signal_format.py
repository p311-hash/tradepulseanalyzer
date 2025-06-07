#!/usr/bin/env python3
"""
Test script to verify that indicators and patterns are removed from signal format.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import format_signal_message

def test_signal_format():
    """Test that signal format no longer includes indicators."""
    
    # Create a sample signal
    sample_signal = {
        'asset': 'EURUSD_otc',
        'direction': 'BUY',
        'price': 1.0850,
        'timestamp': '2024-01-15 14:30:00',
        'strength': '85.5%'
    }
    
    # Format the signal message
    formatted_message = format_signal_message(sample_signal)
    
    print("=== Signal Format Test ===")
    print("Sample signal data:")
    for key, value in sample_signal.items():
        print(f"  {key}: {value}")
    
    print("\nFormatted message:")
    print(formatted_message)
    
    # Check that indicators are not in the message
    indicators_keywords = ['indicators', 'rsi', 'macd', 'ema', 'sma', 'bollinger', 'pattern']
    
    message_lower = formatted_message.lower()
    found_indicators = []
    
    for keyword in indicators_keywords:
        if keyword in message_lower:
            found_indicators.append(keyword)
    
    print(f"\n=== Test Results ===")
    if found_indicators:
        print(f"❌ FAILED: Found indicator keywords: {found_indicators}")
        return False
    else:
        print("✅ PASSED: No indicator keywords found in signal format")
        return True

if __name__ == "__main__":
    success = test_signal_format()
    sys.exit(0 if success else 1)
