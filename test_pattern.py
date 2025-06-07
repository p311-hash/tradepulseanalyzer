"""Test script for pattern recognition functionality."""
import logging
from BinaryOptionsTools import PocketOption
from pattern_recognition import EnhancedPatternRecognizer
import pandas as pd
import numpy as np

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pattern_recognition():
    """Test pattern recognition functionality."""
    try:
        # Create test data with precisely defined patterns
        current_time = pd.Timestamp.now()
        
        # Create test data frame with known patterns
        test_candles = pd.DataFrame([
            # Create pattern for is_inverted_hammer: upper_shadow > (2 * body) and lower_shadow < body and body > 0
            {
                'open': 1.1000,
                'close': 1.0999, # Body is 0.0001 (small)
                'high': 1.1020,  # Upper shadow is 0.0021 (21x body size)
                'low': 1.0999,   # Lower shadow is 0.0000 (zero)
                'volume': 100
            },
            # Hammer pattern (small body, small upper shadow, long lower shadow)
            {
                'open': 1.1000,  
                'close': 1.1001, # Body is 0.0001 (small)
                'high': 1.1001,  # Upper shadow is 0.0000 (negligible)
                'low': 1.0980,   # Lower shadow is 0.0020 (long, 20x body size)
                'volume': 100
            }
        ], index=[current_time - pd.Timedelta(minutes=1), current_time - pd.Timedelta(minutes=2)])
        
        logger.info("Created test candle data with predefined patterns")
        logger.debug("Latest candle data (Inverted Hammer):\n%s", test_candles.iloc[0])
        logger.debug("Second latest candle data (Hammer):\n%s", test_candles.iloc[1])

        # Initialize pattern recognizer with test data
        recognizer = EnhancedPatternRecognizer(test_candles)
        logger.info("Pattern recognizer initialized")

        # Test pattern recognition
        patterns = recognizer.recognize_patterns()
        logger.info("Pattern recognition results:")
        found_patterns = 0
        for pattern, found in patterns.items():
            status = 'Found' if found else 'Not found'
            logger.info(f"  {pattern}: {status}")
            if found:
                found_patterns += 1

        # Verify specific patterns in test data
        inverted_hammer_candle = test_candles.iloc[0]
        hammer_candle = test_candles.iloc[1]

        # Test inverted hammer pattern (should be in latest candle)
        is_inverted = recognizer.is_inverted_hammer(inverted_hammer_candle)
        logger.info(f"Inverted hammer pattern test: {'Found' if is_inverted else 'Not found'}")
        assert is_inverted, "Inverted hammer pattern not detected in latest candle"
        logger.info("✓ Inverted hammer pattern verified in latest candle")

        # Test hammer pattern (should be in second latest candle)
        is_hammer = recognizer.is_hammer(hammer_candle)
        logger.info(f"Hammer pattern test: {'Found' if is_hammer else 'Not found'}")
        assert is_hammer, "Hammer pattern not detected in second latest candle"
        logger.info("✓ Hammer pattern verified in second latest candle")

        # Verify measurements
        def verify_candle_measurements(candle, pattern_type):
            """Verify candle measurements for pattern validation."""
            body = abs(candle['open'] - candle['close'])
            upper_shadow = candle['high'] - max(candle['open'], candle['close'])
            lower_shadow = min(candle['open'], candle['close']) - candle['low']
            logger.debug(f"{pattern_type} measurements - Body: {body:.5f}, Upper: {upper_shadow:.5f}, Lower: {lower_shadow:.5f}")

            # Add assertions for pattern criteria
            if pattern_type == "Hammer":
                assert lower_shadow > (2 * body), "Hammer lower shadow not long enough"
                assert upper_shadow < body, "Hammer upper shadow too long"
                assert body > 0, "Hammer body should be positive"
            elif pattern_type == "Inverted Hammer":
                assert upper_shadow > (2 * body), "Inverted hammer upper shadow not long enough"
                assert lower_shadow < body, "Inverted hammer lower shadow too long"
                assert body > 0, "Inverted hammer body should be positive"

            return body, upper_shadow, lower_shadow

        inverted_measurements = verify_candle_measurements(inverted_hammer_candle, "Inverted Hammer")
        logger.info("✓ Inverted hammer measurements verified")

        hammer_measurements = verify_candle_measurements(hammer_candle, "Hammer")
        logger.info("✓ Hammer measurements verified")

        logger.info(f"Total patterns found: {found_patterns}")
        # We don't expect to find both patterns in a single call to recognize_patterns()
        # since recognize_patterns() only looks at the latest candle
        
        # Verify basic pattern consistency
        assert not (patterns['tweezer_top'] and patterns['tweezer_bottom']), \
            "Cannot have both tweezer top and bottom in same candle pair"
        logger.info("✓ Pattern combinations are valid")

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    test_pattern_recognition()