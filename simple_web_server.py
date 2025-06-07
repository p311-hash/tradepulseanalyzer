#!/usr/bin/env python3
"""
Simplified web server for TradePulseAnalyzer
This version runs without complex dependencies for testing
"""

import os
import sys
import logging
from datetime import datetime
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from flask import Flask, jsonify, render_template_string
    from dotenv import load_dotenv
    import pandas as pd
    import numpy as np
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install flask python-dotenv pandas numpy")
    sys.exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create Flask app
app = Flask(__name__)

# Simple data generator for testing
def generate_sample_data():
    """Generate sample trading data for testing"""
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    
    # Generate realistic price data
    base_price = 1.1000
    returns = np.random.normal(0, 0.001, len(dates))
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return {
        'timestamps': [d.isoformat() for d in dates],
        'prices': prices,
        'latest_price': prices[-1],
        'change': (prices[-1] - prices[0]) / prices[0] * 100
    }

def generate_sample_signal():
    """Generate a sample trading signal"""
    pairs = ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'AUDUSD_otc']
    directions = ['BUY', 'SELL', 'NEUTRAL']
    
    np.random.seed(int(datetime.now().timestamp()) % 1000)
    
    return {
        'pair': np.random.choice(pairs),
        'direction': np.random.choice(directions),
        'confidence': round(np.random.uniform(60, 95), 1),
        'timestamp': datetime.now().isoformat(),
        'indicators': 'RSI, MACD, Bollinger Bands',
        'entry_price': round(np.random.uniform(1.0500, 1.1500), 4),
        'stop_loss': round(np.random.uniform(1.0400, 1.0600), 4),
        'take_profit': round(np.random.uniform(1.1400, 1.1600), 4)
    }

# Web interface template
HOME_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>TradePulse Analyzer - Test Mode</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body { background-color: #1a1a1a; color: #ffffff; }
        .card { background-color: #2d2d2d; border: 1px solid #444; }
        .signal-buy { border-left: 4px solid #28a745; }
        .signal-sell { border-left: 4px solid #dc3545; }
        .signal-neutral { border-left: 4px solid #ffc107; }
        .status-indicator { 
            display: inline-block; 
            width: 12px; 
            height: 12px; 
            border-radius: 50%; 
            background-color: #28a745; 
            margin-right: 8px; 
        }
    </style>
</head>
<body>
    <div class="container mt-4">
        <div class="row">
            <div class="col-12">
                <h1 class="text-center mb-4">
                    📈 TradePulse Analyzer
                    <small class="text-muted">(Test Mode)</small>
                </h1>
            </div>
        </div>
        
        <div class="row">
            <div class="col-md-6">
                <div class="card mb-4">
                    <div class="card-header">
                        <h5 class="mb-0">
                            <span class="status-indicator"></span>
                            System Status
                        </h5>
                    </div>
                    <div class="card-body">
                        <p><strong>Status:</strong> <span class="text-success">Running</span></p>
                        <p><strong>Mode:</strong> Test/Development</p>
                        <p><strong>Last Update:</strong> {{ current_time }}</p>
                        <p><strong>Components:</strong></p>
                        <ul>
                            <li>✅ Web Server</li>
                            <li>✅ Data Generator</li>
                            <li>✅ Signal Generator</li>
                            <li>⚠️ Telegram Bot (Disabled in test mode)</li>
                            <li>⚠️ Real Market Data (Using simulated data)</li>
                        </ul>
                    </div>
                </div>
            </div>
            
            <div class="col-md-6">
                <div class="card mb-4 signal-{{ signal.direction.lower() }}">
                    <div class="card-header">
                        <h5 class="mb-0">Latest Signal</h5>
                    </div>
                    <div class="card-body">
                        <h6>{{ signal.pair }}</h6>
                        <p class="mb-1">
                            <strong>Direction:</strong> 
                            <span class="badge bg-{{ 'success' if signal.direction == 'BUY' else 'danger' if signal.direction == 'SELL' else 'warning' }}">
                                {{ signal.direction }}
                            </span>
                        </p>
                        <p class="mb-1"><strong>Confidence:</strong> {{ signal.confidence }}%</p>
                        <p class="mb-1"><strong>Entry Price:</strong> {{ signal.entry_price }}</p>
                        <p class="mb-1"><strong>Stop Loss:</strong> {{ signal.stop_loss }}</p>
                        <p class="mb-1"><strong>Take Profit:</strong> {{ signal.take_profit }}</p>
                        <small class="text-muted">{{ signal.timestamp }}</small>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">Quick Actions</h5>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-3 mb-2">
                                <a href="/api/signals" class="btn btn-primary w-100">View API Signals</a>
                            </div>
                            <div class="col-md-3 mb-2">
                                <a href="/api/data" class="btn btn-info w-100">Sample Data</a>
                            </div>
                            <div class="col-md-3 mb-2">
                                <a href="/health" class="btn btn-success w-100">Health Check</a>
                            </div>
                            <div class="col-md-3 mb-2">
                                <button onclick="location.reload()" class="btn btn-secondary w-100">Refresh</button>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <div class="row mt-4">
            <div class="col-12">
                <div class="card">
                    <div class="card-header">
                        <h5 class="mb-0">About This Test Mode</h5>
                    </div>
                    <div class="card-body">
                        <p>This is a simplified version of TradePulse Analyzer running in test mode. It demonstrates:</p>
                        <ul>
                            <li>Basic web interface functionality</li>
                            <li>Sample signal generation</li>
                            <li>API endpoints</li>
                            <li>System status monitoring</li>
                        </ul>
                        <p class="text-warning">
                            <strong>Note:</strong> This test mode uses simulated data and signals. 
                            For production use, configure real API keys and enable live data feeds.
                        </p>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
"""

# Routes
@app.route('/')
def home():
    """Home page with system status and latest signal"""
    signal = generate_sample_signal()
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return render_template_string(HOME_TEMPLATE, signal=signal, current_time=current_time)

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "TradePulse Analyzer (Test Mode)",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0-test"
    })

@app.route('/api/signals')
def api_signals():
    """API endpoint for trading signals"""
    signals = {}
    pairs = ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'AUDUSD_otc']
    
    for pair in pairs:
        np.random.seed(hash(pair) % 1000)
        signals[pair] = generate_sample_signal()
        signals[pair]['pair'] = pair
    
    return jsonify(signals)

@app.route('/api/data')
def api_data():
    """API endpoint for sample market data"""
    return jsonify(generate_sample_data())

@app.route('/api/signal/<pair>')
def api_signal_pair(pair):
    """Get signal for specific trading pair"""
    signal = generate_sample_signal()
    signal['pair'] = pair.upper()
    return jsonify(signal)

if __name__ == '__main__':
    logger.info("Starting TradePulse Analyzer Test Server...")
    logger.info("This is a simplified test version with simulated data")
    
    # Get port from environment or use default
    port = int(os.environ.get('PORT', 5000))
    
    try:
        app.run(
            host='0.0.0.0',
            port=port,
            debug=True
        )
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)
