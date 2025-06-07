#!/usr/bin/env python3
"""
Test script to verify bot startup and basic functionality.
"""

import os
import sys
import logging
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
logger = logging.getLogger(__name__)

def test_environment():
    """Test environment configuration."""
    logger.info("Testing environment configuration...")
    
    # Load environment variables
    load_dotenv()
    
    # Check required variables
    token = os.getenv('TELEGRAM_TOKEN')
    if not token:
        logger.error("TELEGRAM_TOKEN not found in environment")
        return False
    
    logger.info(f"Token found: {token[:10]}...")
    return True

def test_imports():
    """Test required imports."""
    logger.info("Testing imports...")
    
    try:
        import telegram
        logger.info("✅ telegram imported successfully")
        
        from telegram.ext import Application, CommandHandler
        logger.info("✅ telegram.ext imported successfully")
        
        from dotenv import load_dotenv
        logger.info("✅ dotenv imported successfully")
        
        import pandas
        logger.info("✅ pandas imported successfully")
        
        import numpy
        logger.info("✅ numpy imported successfully")
        
        return True
        
    except ImportError as e:
        logger.error(f"❌ Import error: {e}")
        return False

def test_bot_creation():
    """Test bot creation."""
    logger.info("Testing bot creation...")
    
    try:
        from telegram.ext import Application
        
        token = os.getenv('TELEGRAM_TOKEN')
        application = Application.builder().token(token).build()
        
        logger.info("✅ Bot application created successfully")
        return True
        
    except Exception as e:
        logger.error(f"❌ Bot creation error: {e}")
        return False

def main():
    """Run all tests."""
    logger.info("🚀 Starting TradePulseAnalyzer Bot Startup Test")
    
    tests = [
        ("Environment", test_environment),
        ("Imports", test_imports),
        ("Bot Creation", test_bot_creation),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Testing {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name}: {status}")
        except Exception as e:
            logger.error(f"{test_name}: ❌ FAILED with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Bot should be ready to run.")
        return True
    else:
        logger.error("❌ Some tests failed. Please fix the issues before running the bot.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
