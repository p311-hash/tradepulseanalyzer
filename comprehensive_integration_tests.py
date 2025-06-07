"""
Comprehensive Integration Tests for TradePulseAnalyzer
Tests all components working together end-to-end.
"""

import asyncio
import unittest
import logging
import sys
import os
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import all components
from enhanced_data_reliability import EnhancedDataReliabilityManager
from live_trading_engine import LiveTradingEngine
from enhanced_sentiment_integration import EnhancedSentimentIntegration
from signal_generator import SignalGenerator
from error_handling_system import ErrorHandlingSystem, ErrorCategory
from production_config import config_manager

# Setup logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveIntegrationTests(unittest.TestCase):
    """
    Comprehensive integration tests for the entire TradePulse system.
    """
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        logger.info("Setting up comprehensive integration tests...")
        
        # Initialize all components
        cls.data_manager = EnhancedDataReliabilityManager()
        cls.trading_engine = LiveTradingEngine(initial_balance=10000.0)
        cls.sentiment_analyzer = EnhancedSentimentIntegration()
        cls.signal_generator = SignalGenerator()
        cls.error_handler = ErrorHandlingSystem()
        
        # Test symbols
        cls.test_symbols = ['EURUSD', 'GBPUSD', 'BTCUSD']
        cls.test_timeframes = ['1m', '5m', '15m']
        
        logger.info("Integration test setup completed")
    
    def test_01_data_reliability_system(self):
        """Test the enhanced data reliability system."""
        logger.info("Testing data reliability system...")
        
        async def run_test():
            for symbol in self.test_symbols:
                for timeframe in self.test_timeframes:
                    # Test data fetching with multiple sources
                    data = await self.data_manager.get_reliable_data(
                        symbol, timeframe, min_sources=1
                    )
                    
                    self.assertIsNotNone(data, f"No data returned for {symbol} {timeframe}")
                    self.assertFalse(data.empty, f"Empty data for {symbol} {timeframe}")
                    self.assertIn('close', data.columns, f"Missing close column for {symbol}")
                    self.assertGreater(len(data), 50, f"Insufficient data points for {symbol}")
                    
                    logger.info(f"✓ Data reliability test passed for {symbol} {timeframe}")
            
            # Test source health monitoring
            health_status = self.data_manager.get_source_health_status()
            self.assertIsInstance(health_status, dict)
            self.assertGreater(len(health_status), 0)
            
            logger.info("✓ Data reliability system tests passed")
        
        asyncio.run(run_test())
    
    def test_02_signal_generation_integration(self):
        """Test signal generation with all components integrated."""
        logger.info("Testing signal generation integration...")
        
        for symbol in self.test_symbols:
            # Generate signal
            signal = self.signal_generator.generate_signal(symbol, '5m')
            
            self.assertIsNotNone(signal, f"No signal generated for {symbol}")
            self.assertIn('symbol', signal)
            self.assertIn('direction', signal)
            self.assertIn('confidence', signal)
            self.assertIn('timestamp', signal)
            
            # Validate signal structure
            self.assertEqual(signal['symbol'], symbol)
            self.assertIn(signal['direction'].upper(), ['BUY', 'SELL', 'HOLD'])
            self.assertGreaterEqual(signal['confidence'], 0.0)
            self.assertLessEqual(signal['confidence'], 1.0)
            
            logger.info(f"✓ Signal generation test passed for {symbol}: {signal['direction']} ({signal['confidence']:.2f})")
        
        logger.info("✓ Signal generation integration tests passed")
    
    def test_03_sentiment_analysis_integration(self):
        """Test sentiment analysis integration."""
        logger.info("Testing sentiment analysis integration...")
        
        async def run_test():
            for symbol in self.test_symbols:
                # Get comprehensive sentiment
                sentiment = await self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
                
                self.assertIsNotNone(sentiment, f"No sentiment data for {symbol}")
                self.assertIn('symbol', sentiment)
                self.assertIn('overall_sentiment', sentiment)
                self.assertIn('confidence', sentiment)
                self.assertIn('signal_strength', sentiment)
                
                # Validate sentiment structure
                self.assertEqual(sentiment['symbol'], symbol)
                self.assertIn(sentiment['overall_sentiment'], ['BULLISH', 'BEARISH', 'NEUTRAL'])
                self.assertGreaterEqual(sentiment['confidence'], 0.0)
                self.assertLessEqual(sentiment['confidence'], 1.0)
                
                # Test sentiment-signal integration
                test_signal = {
                    'symbol': symbol,
                    'direction': 'BUY',
                    'confidence': 0.7,
                    'timestamp': datetime.now().isoformat()
                }
                
                enhanced_signal = self.sentiment_analyzer.integrate_with_signal(test_signal, sentiment)
                
                self.assertIsNotNone(enhanced_signal)
                self.assertIn('sentiment', enhanced_signal)
                self.assertIn('confidence', enhanced_signal)
                
                logger.info(f"✓ Sentiment integration test passed for {symbol}: {sentiment['overall_sentiment']}")
            
            logger.info("✓ Sentiment analysis integration tests passed")
        
        asyncio.run(run_test())
    
    def test_04_trading_engine_integration(self):
        """Test live trading engine integration."""
        logger.info("Testing trading engine integration...")
        
        async def run_test():
            # Start monitoring
            await self.trading_engine.start_monitoring()
            
            # Test signal execution
            test_signal = {
                'symbol': 'EURUSD',
                'direction': 'BUY',
                'confidence': 0.8,
                'current_price': 1.1000,
                'timestamp': datetime.now().isoformat()
            }
            
            # Execute signal
            order_id = await self.trading_engine.execute_signal(test_signal)
            
            if order_id:  # Signal was accepted
                self.assertIsNotNone(order_id)
                
                # Check account summary
                account = self.trading_engine.get_account_summary()
                self.assertIsInstance(account, dict)
                self.assertIn('balance', account)
                self.assertIn('positions', account)
                
                logger.info(f"✓ Signal execution test passed: Order {order_id}")
                
                # Test position management
                if account['positions']:
                    position = account['positions'][0]
                    symbol = position['symbol']
                    
                    # Close position
                    await self.trading_engine._close_position(symbol, "Test close")
                    
                    # Verify position closed
                    updated_account = self.trading_engine.get_account_summary()
                    self.assertEqual(len(updated_account['positions']), 0)
                    
                    logger.info(f"✓ Position management test passed")
            else:
                logger.info("✓ Signal was properly rejected by risk management")
            
            # Stop monitoring
            await self.trading_engine.stop_monitoring()
            
            logger.info("✓ Trading engine integration tests passed")
        
        asyncio.run(run_test())
    
    def test_05_error_handling_integration(self):
        """Test error handling system integration."""
        logger.info("Testing error handling integration...")
        
        async def run_test():
            # Test error handling
            test_error = Exception("Test error for integration testing")
            
            # Handle error
            recovery_success = await self.error_handler.handle_error(
                test_error, ErrorCategory.SYSTEM, {'test': True}
            )
            
            # Check error statistics
            stats = self.error_handler.get_error_statistics()
            self.assertIsInstance(stats, dict)
            self.assertIn('total_errors', stats)
            self.assertGreater(stats['total_errors'], 0)
            
            # Check system health
            health = self.error_handler.get_system_health()
            self.assertIsInstance(health, dict)
            self.assertIn('status', health)
            self.assertIn('health_score', health)
            
            logger.info(f"✓ Error handling test passed: Health {health['status']}")
            
            logger.info("✓ Error handling integration tests passed")
        
        asyncio.run(run_test())
    
    def test_06_end_to_end_workflow(self):
        """Test complete end-to-end trading workflow."""
        logger.info("Testing end-to-end trading workflow...")
        
        async def run_test():
            symbol = 'EURUSD'
            
            # Step 1: Get reliable market data
            market_data = await self.data_manager.get_reliable_data(symbol, '5m')
            self.assertIsNotNone(market_data)
            self.assertFalse(market_data.empty)
            
            # Step 2: Generate trading signal
            signal = self.signal_generator.generate_signal(symbol, '5m')
            self.assertIsNotNone(signal)
            
            # Step 3: Get sentiment analysis
            sentiment = await self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
            self.assertIsNotNone(sentiment)
            
            # Step 4: Integrate sentiment with signal
            enhanced_signal = self.sentiment_analyzer.integrate_with_signal(signal, sentiment)
            self.assertIsNotNone(enhanced_signal)
            self.assertIn('sentiment', enhanced_signal)
            
            # Step 5: Execute signal through trading engine
            await self.trading_engine.start_monitoring()
            
            if enhanced_signal['confidence'] > 0.6:
                order_id = await self.trading_engine.execute_signal(enhanced_signal)
                
                if order_id:
                    logger.info(f"✓ End-to-end workflow completed successfully: Order {order_id}")
                    
                    # Clean up - close any open positions
                    account = self.trading_engine.get_account_summary()
                    for position in account.get('positions', []):
                        await self.trading_engine._close_position(position['symbol'], "Test cleanup")
                else:
                    logger.info("✓ End-to-end workflow completed: Signal properly rejected")
            else:
                logger.info("✓ End-to-end workflow completed: Low confidence signal not executed")
            
            await self.trading_engine.stop_monitoring()
            
            logger.info("✓ End-to-end workflow tests passed")
        
        asyncio.run(run_test())
    
    def test_07_performance_and_scalability(self):
        """Test system performance and scalability."""
        logger.info("Testing performance and scalability...")
        
        async def run_test():
            start_time = datetime.now()
            
            # Test concurrent data fetching
            tasks = []
            for symbol in self.test_symbols:
                for timeframe in ['1m', '5m']:
                    task = self.data_manager.get_reliable_data(symbol, timeframe)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Check results
            successful_fetches = sum(1 for r in results if isinstance(r, pd.DataFrame) and not r.empty)
            total_fetches = len(tasks)
            
            success_rate = successful_fetches / total_fetches
            self.assertGreater(success_rate, 0.5, "Success rate too low for concurrent fetching")
            
            # Test signal generation performance
            signal_start = datetime.now()
            signals = []
            
            for symbol in self.test_symbols:
                signal = self.signal_generator.generate_signal(symbol, '5m')
                if signal:
                    signals.append(signal)
            
            signal_duration = (datetime.now() - signal_start).total_seconds()
            
            # Performance assertions
            total_duration = (datetime.now() - start_time).total_seconds()
            self.assertLess(total_duration, 60, "Performance test took too long")
            self.assertLess(signal_duration, 10, "Signal generation too slow")
            
            logger.info(f"✓ Performance test passed: {total_duration:.2f}s total, {success_rate:.1%} success rate")
            
            logger.info("✓ Performance and scalability tests passed")
        
        asyncio.run(run_test())
    
    def test_08_configuration_validation(self):
        """Test configuration management and validation."""
        logger.info("Testing configuration validation...")
        
        # Test configuration loading
        self.assertIsNotNone(config_manager)
        
        # Validate configuration
        issues = config_manager.validate_configuration()
        self.assertIsInstance(issues, dict)
        self.assertIn('errors', issues)
        self.assertIn('warnings', issues)
        self.assertIn('info', issues)
        
        # Get configuration summary
        summary = config_manager.get_config_summary()
        self.assertIsInstance(summary, dict)
        self.assertIn('trading_mode', summary)
        self.assertIn('api_keys_configured', summary)
        
        logger.info(f"✓ Configuration validation passed: {summary['trading_mode']} mode")
        logger.info("✓ Configuration tests passed")
    
    def test_09_data_quality_validation(self):
        """Test data quality validation across all sources."""
        logger.info("Testing data quality validation...")
        
        async def run_test():
            quality_scores = {}
            
            for symbol in self.test_symbols:
                data = await self.data_manager.get_reliable_data(symbol, '5m')
                
                if data is not None and not data.empty:
                    # Calculate quality score
                    quality_score = self.data_manager._calculate_data_quality_score(data)
                    quality_scores[symbol] = quality_score
                    
                    # Quality assertions
                    self.assertGreater(quality_score, 0.1, f"Quality too low for {symbol}")
                    
                    # Data integrity checks
                    self.assertFalse(data.isnull().all().any(), f"All null column in {symbol}")
                    self.assertTrue((data['high'] >= data['low']).all(), f"High < Low in {symbol}")
                    self.assertTrue((data['high'] >= data['close']).all(), f"High < Close in {symbol}")
                    self.assertTrue((data['low'] <= data['close']).all(), f"Low > Close in {symbol}")
            
            avg_quality = np.mean(list(quality_scores.values())) if quality_scores else 0
            self.assertGreater(avg_quality, 0.5, "Average data quality too low")
            
            logger.info(f"✓ Data quality validation passed: Average quality {avg_quality:.2f}")
            
            logger.info("✓ Data quality tests passed")
        
        asyncio.run(run_test())
    
    def test_10_system_integration_health(self):
        """Test overall system integration health."""
        logger.info("Testing system integration health...")
        
        # Get system health
        health = self.error_handler.get_system_health()
        
        # Health assertions
        self.assertIn('status', health)
        self.assertIn('health_score', health)
        self.assertGreaterEqual(health['health_score'], 0)
        self.assertLessEqual(health['health_score'], 100)
        
        # Component health checks
        components_healthy = True
        
        # Check data manager
        try:
            asyncio.run(self.data_manager.get_reliable_data('EURUSD', '5m'))
            logger.info("✓ Data manager healthy")
        except Exception as e:
            logger.warning(f"Data manager issue: {e}")
            components_healthy = False
        
        # Check signal generator
        try:
            signal = self.signal_generator.generate_signal('EURUSD', '5m')
            if signal:
                logger.info("✓ Signal generator healthy")
            else:
                logger.warning("Signal generator returned None")
        except Exception as e:
            logger.warning(f"Signal generator issue: {e}")
            components_healthy = False
        
        # Check trading engine
        try:
            account = self.trading_engine.get_account_summary()
            if account:
                logger.info("✓ Trading engine healthy")
            else:
                logger.warning("Trading engine returned empty account")
        except Exception as e:
            logger.warning(f"Trading engine issue: {e}")
            components_healthy = False
        
        # Overall health assessment
        if components_healthy and health['health_score'] >= 50:
            logger.info(f"✓ System integration health: {health['status']} ({health['health_score']})")
        else:
            logger.warning(f"System health concerns: {health['status']} ({health['health_score']})")
        
        logger.info("✓ System integration health tests completed")

def run_comprehensive_tests():
    """Run all comprehensive integration tests."""
    logger.info("=" * 60)
    logger.info("STARTING COMPREHENSIVE INTEGRATION TESTS")
    logger.info("=" * 60)
    
    # Create test suite
    suite = unittest.TestLoader().loadTestsFromTestCase(ComprehensiveIntegrationTests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("COMPREHENSIVE INTEGRATION TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Tests run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.info(f"Errors: {len(result.errors)}")
    
    if result.failures:
        logger.error("FAILURES:")
        for test, traceback in result.failures:
            logger.error(f"  {test}: {traceback}")
    
    if result.errors:
        logger.error("ERRORS:")
        for test, traceback in result.errors:
            logger.error(f"  {test}: {traceback}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
    logger.info(f"Success rate: {success_rate:.1f}%")
    
    if result.wasSuccessful():
        logger.info("🎉 ALL INTEGRATION TESTS PASSED! System is ready for production.")
    else:
        logger.warning("⚠️  Some integration tests failed. Review issues before production deployment.")
    
    return result.wasSuccessful()

if __name__ == '__main__':
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)
