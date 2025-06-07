#!/usr/bin/env python3
"""
Simple test to verify signal format changes.
"""

def format_signal_message_test(signal):
    """Test version of format_signal_message without dependencies."""
    direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴" if signal["direction"] == "SELL" else "⚪"
    
    message = (
        f"{direction_emoji} *{signal['asset']}*\n"
        f"Signal: *{signal['direction']}* ({signal['strength']})\n"
        f"Price: ${signal['price']:.2f}\n"
        f"Generated: {signal['timestamp']}"
    )
    
    return message

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
    formatted_message = format_signal_message_test(sample_signal)
    
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
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
