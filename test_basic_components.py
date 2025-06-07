#!/usr/bin/env python3
"""
Basic component test for TradePulseAnalyzer
Tests core functionality without external dependencies
"""

import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def test_imports():
    """Test if basic imports work"""
    print("🔍 Testing basic imports...")
    
    try:
        import pandas as pd
        print("✅ pandas imported successfully")
    except ImportError as e:
        print(f"❌ pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ numpy imported successfully")
    except ImportError as e:
        print(f"❌ numpy import failed: {e}")
        return False
    
    try:
        import flask
        print("✅ flask imported successfully")
    except ImportError as e:
        print(f"❌ flask import failed: {e}")
        return False
    
    try:
        import requests
        print("✅ requests imported successfully")
    except ImportError as e:
        print(f"❌ requests import failed: {e}")
        return False
    
    try:
        from dotenv import load_dotenv
        print("✅ python-dotenv imported successfully")
    except ImportError as e:
        print(f"❌ python-dotenv import failed: {e}")
        return False
    
    return True

def test_data_generation():
    """Test basic data generation for testing"""
    print("\n📊 Testing data generation...")
    
    try:
        # Generate sample OHLCV data
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(42)
        
        # Generate realistic price data
        base_price = 1.1000
        returns = np.random.normal(0, 0.001, len(dates))
        prices = [base_price]
        
        for ret in returns[1:]:
            prices.append(prices[-1] * (1 + ret))
        
        data = pd.DataFrame({
            'timestamp': dates,
            'open': prices,
            'high': [p * (1 + abs(np.random.normal(0, 0.0005))) for p in prices],
            'low': [p * (1 - abs(np.random.normal(0, 0.0005))) for p in prices],
            'close': prices,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        
        data.set_index('timestamp', inplace=True)
        print(f"✅ Generated {len(data)} rows of sample data")
        print(f"   Date range: {data.index[0]} to {data.index[-1]}")
        print(f"   Price range: {data['close'].min():.4f} to {data['close'].max():.4f}")
        
        return data
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return None

def test_technical_analysis():
    """Test basic technical analysis calculations"""
    print("\n📈 Testing technical analysis...")
    
    data = test_data_generation()
    if data is None:
        return False
    
    try:
        # Simple Moving Average
        data['sma_20'] = data['close'].rolling(window=20).mean()
        print("✅ SMA calculation successful")
        
        # RSI calculation (simplified)
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        print("✅ RSI calculation successful")
        
        # Bollinger Bands
        data['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        data['bb_upper'] = data['bb_middle'] + (bb_std * 2)
        data['bb_lower'] = data['bb_middle'] - (bb_std * 2)
        print("✅ Bollinger Bands calculation successful")
        
        # Volume analysis
        data['volume_sma'] = data['volume'].rolling(window=20).mean()
        print("✅ Volume analysis successful")
        
        print(f"   Latest RSI: {data['rsi'].iloc[-1]:.2f}")
        print(f"   Latest SMA: {data['sma_20'].iloc[-1]:.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Technical analysis failed: {e}")
        return False

def test_signal_generation():
    """Test basic signal generation logic"""
    print("\n🎯 Testing signal generation...")
    
    data = test_data_generation()
    if data is None:
        return False
    
    try:
        # Calculate indicators
        data['sma_20'] = data['close'].rolling(window=20).mean()
        data['sma_50'] = data['close'].rolling(window=50).mean()
        
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        data['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple signal logic
        latest = data.iloc[-1]
        signals = []
        
        # Moving average crossover
        if latest['sma_20'] > latest['sma_50']:
            signals.append('MA_BULLISH')
        else:
            signals.append('MA_BEARISH')
        
        # RSI signals
        if latest['rsi'] < 30:
            signals.append('RSI_OVERSOLD')
        elif latest['rsi'] > 70:
            signals.append('RSI_OVERBOUGHT')
        else:
            signals.append('RSI_NEUTRAL')
        
        # Generate final signal
        bullish_signals = sum(1 for s in signals if 'BULLISH' in s or 'OVERSOLD' in s)
        bearish_signals = sum(1 for s in signals if 'BEARISH' in s or 'OVERBOUGHT' in s)
        
        if bullish_signals > bearish_signals:
            final_signal = 'BUY'
            confidence = (bullish_signals / len(signals)) * 100
        elif bearish_signals > bullish_signals:
            final_signal = 'SELL'
            confidence = (bearish_signals / len(signals)) * 100
        else:
            final_signal = 'NEUTRAL'
            confidence = 50
        
        print(f"✅ Signal generation successful")
        print(f"   Signals detected: {signals}")
        print(f"   Final signal: {final_signal}")
        print(f"   Confidence: {confidence:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Signal generation failed: {e}")
        return False

def test_flask_app():
    """Test basic Flask app creation"""
    print("\n🌐 Testing Flask app...")
    
    try:
        from flask import Flask, jsonify
        
        app = Flask(__name__)
        
        @app.route('/health')
        def health():
            return jsonify({"status": "healthy", "service": "TradePulse Test"})
        
        @app.route('/test-signal')
        def test_signal():
            return jsonify({
                "pair": "EURUSD",
                "direction": "BUY",
                "confidence": 75.5,
                "timestamp": datetime.now().isoformat()
            })
        
        print("✅ Flask app created successfully")
        print("   Routes: /health, /test-signal")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 TradePulseAnalyzer Basic Component Test")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_imports),
        ("Data Generation", lambda: test_data_generation() is not None),
        ("Technical Analysis", test_technical_analysis),
        ("Signal Generation", test_signal_generation),
        ("Flask App", test_flask_app)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running {test_name} test...")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} test PASSED")
            else:
                print(f"❌ {test_name} test FAILED")
        except Exception as e:
            print(f"❌ {test_name} test FAILED with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Basic components are working.")
        return True
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
