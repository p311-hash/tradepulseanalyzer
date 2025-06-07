#!/usr/bin/env python3
"""
Test script to check if all required imports are working.
"""

import sys
import traceback

def test_import(module_name, description=""):
    try:
        __import__(module_name)
        print(f"✅ {module_name} - {description}")
        return True
    except ImportError as e:
        print(f"❌ {module_name} - {description}: {str(e)}")
        return False
    except Exception as e:
        print(f"⚠️ {module_name} - {description}: {str(e)}")
        return False

def main():
    print("Testing imports for TradeMind Bot...")
    print("=" * 50)
    
    # Basic Python modules
    test_import("os", "Operating system interface")
    test_import("logging", "Logging facility")
    test_import("json", "JSON encoder/decoder")
    test_import("datetime", "Date and time handling")
    
    # Third-party dependencies
    test_import("dotenv", "Environment variables (.env file support)")
    test_import("pandas", "Data analysis library")
    test_import("numpy", "Numerical computing")
    
    # Telegram bot dependencies
    test_import("telegram", "Python Telegram Bot library")
    test_import("telegram.ext", "Telegram bot extensions")
    test_import("telegram.constants", "Telegram constants")
    
    # Plotting dependencies
    test_import("matplotlib", "Plotting library")
    test_import("matplotlib.pyplot", "Matplotlib pyplot interface")
    
    # Technical analysis
    test_import("ta", "Technical analysis library")
    test_import("ta.momentum", "TA momentum indicators")
    test_import("ta.trend", "TA trend indicators")
    test_import("ta.volatility", "TA volatility indicators")
    
    # Project modules
    print("\n" + "=" * 50)
    print("Testing project modules...")
    test_import("config", "Configuration module")
    test_import("utils", "Utility functions")
    test_import("signal_generator", "Signal generation")
    test_import("signal_analytics", "Signal analytics")
    test_import("technical_analysis", "Technical analysis")
    test_import("pattern_recognition", "Pattern recognition")
    test_import("sentiment_analysis", "Sentiment analysis")
    
    # Enhanced modules
    print("\n" + "=" * 50)
    print("Testing enhanced modules...")
    test_import("enhanced_telegram_handler", "Enhanced Telegram handler")
    
    print("\n" + "=" * 50)
    print("Import testing complete!")

if __name__ == "__main__":
    main()
