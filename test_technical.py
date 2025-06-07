"""Test script for technical analysis functionality."""
import logging
import pandas as pd
import numpy as np
from BinaryOptionsTools import PocketOption, TechnicalAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_technical_analysis():
    """Test technical analysis functionality."""
    try:
        # Initialize API with mock data
        api = PocketOption(ssid=None, demo=True)
        logger.info("Initialized PocketOption API")

        # Get candle data
        candles = api.get_candles("EURUSD", period=60, count=100)
        if candles.empty:
            logger.error("Failed to get candles")
            return

        logger.info(f"Retrieved {len(candles)} candles")
        logger.debug("Latest candle data:\n%s", candles.iloc[-1])
        logger.debug("Second latest candle data:\n%s", candles.iloc[-2])

        # Initialize technical analyzer
        analyzer = TechnicalAnalyzer(candles)
        logger.info("Technical analyzer initialized")

        # Test Vortex Indicator
        vi_pos, vi_neg = analyzer.calculate_vortex()
        logger.info("Vortex Indicator calculation completed")
        if not (vi_pos.empty or vi_neg.empty):
            logger.debug("Latest VI+ value: %.4f", vi_pos.iloc[-1])
            logger.debug("Latest VI- value: %.4f", vi_neg.iloc[-1])

            # Validate Vortex values are within expected range [0, 1]
            assert 0 <= vi_pos.iloc[-1] <= 1, "VI+ value out of range"
            assert 0 <= vi_neg.iloc[-1] <= 1, "VI- value out of range"
            logger.info("✓ Vortex values within valid range")

            # Verify Vortex crossovers make sense
            assert abs(vi_pos.iloc[-1] - vi_neg.iloc[-1]) < 1, "Unrealistic Vortex separation"
            logger.info("✓ Vortex crossover values realistic")
        else:
            logger.error("Failed to calculate Vortex indicator")

        # Test Parabolic SAR
        psar = analyzer.calculate_parabolic_sar()
        logger.info("Parabolic SAR calculation completed")
        if not psar.empty:
            logger.debug("Latest PSAR value: %.4f", psar.iloc[-1])
            latest_close = candles['close'].iloc[-1]
            logger.debug("PSAR relative to price: %.4f vs %.4f", 
                        psar.iloc[-1], latest_close)

            # Verify PSAR is not too far from price
            price_diff_pct = abs(psar.iloc[-1] - latest_close) / latest_close
            assert price_diff_pct < 0.05, "PSAR too far from current price"
            logger.info("✓ PSAR within reasonable range of price")

            # Verify PSAR trend changes
            psar_changes = (psar.diff() != 0).sum()
            assert psar_changes > 0, "PSAR shows no trend changes"
            logger.info("✓ PSAR trend changes detected")
        else:
            logger.error("Failed to calculate PSAR")

        # Test Support/Resistance levels
        levels = analyzer.calculate_support_resistance()
        logger.info("Support and Resistance levels calculated")
        if levels['support'] > 0 and levels['resistance'] > 0:
            logger.debug("Support level: %.4f", levels['support'])
            logger.debug("Resistance level: %.4f", levels['resistance'])

            # Verify S/R level relationships
            assert levels['support'] <= levels['resistance'], "Invalid S/R levels"
            current_price = candles['close'].iloc[-1]
            assert levels['support'] <= current_price <= levels['resistance'], \
                "Current price outside S/R range"

            # Verify reasonable S/R spread
            sr_spread = (levels['resistance'] - levels['support']) / current_price
            assert 0.001 <= sr_spread <= 0.05, "Unrealistic S/R spread"
            logger.info("✓ Support/Resistance levels are valid")
        else:
            logger.error("Failed to calculate valid S/R levels")

        # Test trend analysis
        trend = analyzer.analyze_trend()
        logger.info("Trend analysis result: %s", trend)
        assert trend in ["BULLISH", "BEARISH", "NEUTRAL"], "Invalid trend value"

        # Verify trend matches indicators
        if trend != "NEUTRAL":
            if trend == "BULLISH":
                assert vi_pos.iloc[-1] > vi_neg.iloc[-1], "Bullish trend doesn't match Vortex"
                assert current_price > psar.iloc[-1], "Bullish trend doesn't match PSAR"
                assert current_price > candles['close'].iloc[-5:].mean(), "Price not above recent average"
            else:  # BEARISH
                assert vi_pos.iloc[-1] < vi_neg.iloc[-1], "Bearish trend doesn't match Vortex"
                assert current_price < psar.iloc[-1], "Bearish trend doesn't match PSAR"
                assert current_price < candles['close'].iloc[-5:].mean(), "Price not below recent average"
        logger.info("✓ Trend analysis validation completed")
        
        # Test Parabolic SAR + CCI Strategy
        logger.info("Testing Parabolic SAR + CCI Strategy")
        from technical_analysis import parabolic_sar_cci_strategy
        
        # Create test data with specific PSAR and CCI values for testing
        test_data = candles.copy()
        
        # Add CCI indicator to test data (>100 for bullish signal)
        test_data['cci'] = 120.0  # Strongly bullish CCI
        test_data['psar'] = 1.0990  # PSAR below price for buy signal
        
        # Test buy signal condition (PSAR below price, CCI > 100)
        current_price = test_data['close'].iloc[-1]
        psar_value = test_data['psar'].iloc[-1]
        cci_value = test_data['cci'].iloc[-1]
        
        logger.info(f"Testing buy signal - Price: {current_price:.4f}, PSAR: {psar_value:.4f}, CCI: {cci_value:.1f}")
        assert psar_value < current_price, "PSAR should be below price for buy signal test"
        assert cci_value > 100, "CCI should be above 100 for buy signal test"
        
        buy_signal = parabolic_sar_cci_strategy(test_data)
        logger.info(f"Buy signal test result: {buy_signal}")
        assert buy_signal['direction'] == 'BUY', "Failed to generate buy signal with PSAR below price and CCI > 100"
        assert buy_signal['confidence'] > 60, "Buy signal confidence should be significant"
        logger.info("✓ PSAR+CCI Buy signal successfully verified")
        
        # Test sell signal condition (PSAR above price, CCI < -100)
        test_data['cci'] = -120.0  # Strongly bearish CCI
        test_data['psar'] = 1.1020  # PSAR above price for sell signal
        psar_value = test_data['psar'].iloc[-1]
        cci_value = test_data['cci'].iloc[-1]
        
        logger.info(f"Testing sell signal - Price: {current_price:.4f}, PSAR: {psar_value:.4f}, CCI: {cci_value:.1f}")
        assert psar_value > current_price, "PSAR should be above price for sell signal test"
        assert cci_value < -100, "CCI should be below -100 for sell signal test"
        
        sell_signal = parabolic_sar_cci_strategy(test_data)
        logger.info(f"Sell signal test result: {sell_signal}")
        assert sell_signal['direction'] == 'SELL', "Failed to generate sell signal with PSAR above price and CCI < -100"
        assert sell_signal['confidence'] > 60, "Sell signal confidence should be significant"
        logger.info("✓ PSAR+CCI Sell signal successfully verified")
        
        # Test neutral condition (when conditions don't match either buy or sell)
        test_data['cci'] = 50.0  # Neutral CCI (between -100 and 100)
        test_data['psar'] = 1.1020  # PSAR above price
        psar_value = test_data['psar'].iloc[-1]
        cci_value = test_data['cci'].iloc[-1]
        
        logger.info(f"Testing neutral condition - Price: {current_price:.4f}, PSAR: {psar_value:.4f}, CCI: {cci_value:.1f}")
        neutral_signal = parabolic_sar_cci_strategy(test_data)
        logger.info(f"Neutral condition test result: {neutral_signal}")
        assert neutral_signal['direction'] == 'NEUTRAL', "Should generate neutral signal when CCI is between -100 and 100"
        logger.info("✓ PSAR+CCI Neutral condition successfully verified")
        
        logger.info("✓ Parabolic SAR + CCI Strategy tests completed successfully")

    except Exception as e:
        logger.error("Test failed: %s", str(e), exc_info=True)
        raise

if __name__ == "__main__":
    test_technical_analysis()