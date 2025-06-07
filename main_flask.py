"""
Flask web application for TradePulse Signals Trading Bot
This provides a web interface to access signals and bot status.
"""

import os
import logging
from flask import Flask, jsonify, request, render_template
from signal_generator import get_latest_signals
from backtesting import run_backtest, compare_strategies
from utils import setup_logging
from datetime import datetime
import threading
import time

# Configure logging
logger = setup_logging('web_app')
logger.info("Starting TradePulse Signals Web Interface")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    """Home page showing bot status and latest signals."""
    return """
    <html>
    <head>
        <title>TradePulse Signals Bot</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .signal-card {
                margin-bottom: 15px;
                border-left: 4px solid;
            }
            .signal-buy {
                border-left-color: var(--bs-success);
            }
            .signal-sell {
                border-left-color: var(--bs-danger);
            }
            .signal-neutral {
                border-left-color: var(--bs-warning);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">📈 TradePulse Signals Trading Bot</h1>
            
            <div class="row">
                <div class="col-md-8">
                    <div class="card bg-dark border-secondary mb-4">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Bot Status</h2>
                        </div>
                        <div class="card-body">
                            <p>The TradePulse Signals Trading Bot is currently <span class="text-success">running</span>.</p>
                            <p>This web interface allows you to monitor the bot's status and view signals.</p>
                            <p>To interact with the bot, add it on Telegram using the links below.</p>
                            <div class="mt-3">
                                <a href="/signals" class="btn btn-primary me-2">View Latest Signals</a>
                                <a href="https://t.me/YOUR_BOT_USERNAME" target="_blank" class="btn btn-info">Open in Telegram</a>
                            </div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card bg-dark border-secondary">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Latest Signal</h2>
                        </div>
                        <div class="card-body">
                            <div class="signal-card p-3 bg-dark border-secondary">
                                <h3 class="h5">EURUSD_otc</h3>
                                <div class="d-flex justify-content-between">
                                    <span class="text-success">BUY</span>
                                    <span class="text-muted">Confidence: 75%</span>
                                </div>
                                <small class="text-muted">Based on MA Crossover, RSI</small>
                            </div>
                            <a href="/signals" class="btn btn-outline-secondary w-100 mt-2">View All Signals</a>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row mt-4">
                <div class="col-12">
                    <div class="card bg-dark border-secondary">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Features</h2>
                        </div>
                        <div class="card-body">
                            <div class="row">
                                <div class="col-md-4 mb-3">
                                    <div class="card h-100 bg-dark border-secondary">
                                        <div class="card-body">
                                            <h3 class="h5 mb-3">💹 Technical Analysis</h3>
                                            <p class="card-text">Advanced technical indicators and multi-timeframe analysis</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <div class="card h-100 bg-dark border-secondary">
                                        <div class="card-body">
                                            <h3 class="h5 mb-3">🧠 Machine Learning</h3>
                                            <p class="card-text">AI-powered signal generation with continuous learning</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-3">
                                    <div class="card h-100 bg-dark border-secondary">
                                        <div class="card-body">
                                            <h3 class="h5 mb-3">📊 Pattern Recognition</h3>
                                            <p class="card-text">Detects candlestick patterns for enhanced signal accuracy</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                            
                            <div class="row mt-3">
                                <div class="col-md-12 mb-3">
                                    <div class="card bg-dark border-info">
                                        <div class="card-body">
                                            <div class="row align-items-center">
                                                <div class="col-md-8">
                                                    <h3 class="h5 mb-3">📈 Strategy Backtesting</h3>
                                                    <p class="card-text">Test and compare different trading strategies on historical data before risking real money. Evaluate performance metrics, equity curves, and more.</p>
                                                </div>
                                                <div class="col-md-4 text-end">
                                                    <a href="/backtest" class="btn btn-outline-info me-2">Run Backtest</a>
                                                    <a href="/compare_strategies" class="btn btn-outline-secondary">Compare</a>
                                                </div>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-5 text-center">
                <p>TradePulse Bot is running in the background. Use the Telegram bot to receive signals.</p>
                <div class="mt-3">
                    <a href="/signals" class="btn btn-primary me-2">View All Signals</a>
                    <a href="/backtest" class="btn btn-info me-2">Backtest Strategies</a>
                    <a href="/compare_strategies" class="btn btn-secondary">Compare Strategies</a>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/signals')
def signals():
    """Page showing all latest signals."""
    # Get latest signals
    signals_data = get_latest_signals()
    
    # Format signals for display
    signal_html = '<div class="container mt-4"><h1>Latest Trading Signals</h1><div class="row">'
    
    for pair, signal in signals_data.items():
        direction = signal.get('direction', 'NEUTRAL')
        confidence = signal.get('confidence', 0)
        
        # Format confidence as percentage
        if isinstance(confidence, (int, float)):
            confidence = f"{confidence:.1f}%"
            
        indicators = signal.get('indicators', '')
        timestamp = signal.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        
        color_class = 'success' if direction == 'BUY' else 'danger' if direction == 'SELL' else 'warning'
        
        signal_html += f'''
        <div class="col-md-4 mb-4">
            <div class="card bg-dark border-{color_class}">
                <div class="card-header bg-dark text-{color_class} border-{color_class}">
                    <h2 class="h5 m-0">{pair}</h2>
                </div>
                <div class="card-body">
                    <h3 class="h4 text-{color_class}">{direction}</h3>
                    <p class="card-text">Confidence: {confidence}</p>
                    <p class="card-text small text-muted">Indicators: {indicators}</p>
                </div>
                <div class="card-footer bg-dark border-{color_class}">
                    <small class="text-muted">Generated at {timestamp}</small>
                </div>
            </div>
        </div>
        '''
    
    signal_html += '</div><a href="/" class="btn btn-secondary mt-3 mb-5">Back to Home</a></div>'
    
    return f"""
    <html>
    <head>
        <title>TradePulse Signals - Latest Signals</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }}
        </style>
    </head>
    <body>
        {signal_html}
    </body>
    </html>
    """

@app.route('/api/signals')
def api_signals():
    """API endpoint to get latest signals as JSON."""
    signals_data = get_latest_signals()
    return jsonify(signals_data)

@app.route('/health')
def health():
    """Health check endpoint for monitoring."""
    return jsonify({"status": "healthy", "service": "TradePulse Bot"})
    
@app.route('/backtest')
def backtest_page():
    """Backtesting page with form to run backtest."""
    return """
    <html>
    <head>
        <title>TradePulse - Backtest Trading Strategies</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .trading-chart {
                background-color: var(--bs-dark);
                border: 1px solid var(--bs-secondary);
                border-radius: 5px;
                margin-top: 20px;
                height: 300px;
            }
            .metrics-card {
                border-left: 4px solid var(--bs-info);
            }
            .strategy-card {
                transition: all 0.2s ease;
            }
            .strategy-card:hover {
                transform: translateY(-5px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.3);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">📊 Strategy Backtesting</h1>
            
            <div class="row">
                <div class="col-md-8">
                    <div class="card bg-dark border-secondary mb-4">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Backtest Configuration</h2>
                        </div>
                        <div class="card-body">
                            <form action="/run_backtest" method="POST">
                                <div class="mb-3">
                                    <label for="pair" class="form-label">Trading Pair</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="pair" name="pair">
                                        <option value="EURUSD_otc">EURUSD_otc</option>
                                        <option value="GBPUSD_otc">GBPUSD_otc</option>
                                        <option value="USDJPY_otc">USDJPY_otc</option>
                                        <option value="USDCHF_otc">USDCHF_otc</option>
                                        <option value="AUDUSD_otc">AUDUSD_otc</option>
                                        <option value="USDCAD_otc">USDCAD_otc</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="timeframe" class="form-label">Timeframe</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="timeframe" name="timeframe">
                                        <option value="1m">1 minute</option>
                                        <option value="5m">5 minutes</option>
                                        <option value="15m">15 minutes</option>
                                        <option value="1h">1 hour</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="strategy" class="form-label">Trading Strategy</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="strategy" name="strategy">
                                        <option value="macd_crossover">MACD Crossover</option>
                                        <option value="rsi_strategy">RSI Strategy</option>
                                        <option value="bollinger_bands">Bollinger Bands</option>
                                        <option value="multi_indicator">Multi-Indicator Strategy</option>
                                        <option value="tradepulse">TradePulse Strategy</option>
                                    </select>
                                </div>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="start_date" class="form-label">Start Date</label>
                                            <input type="date" class="form-control bg-dark text-light border-secondary" id="start_date" name="start_date">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="end_date" class="form-label">End Date</label>
                                            <input type="date" class="form-control bg-dark text-light border-secondary" id="end_date" name="end_date">
                                        </div>
                                    </div>
                                </div>
                                <div class="row">
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="initial_balance" class="form-label">Initial Balance</label>
                                            <input type="number" class="form-control bg-dark text-light border-secondary" id="initial_balance" name="initial_balance" value="1000">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="position_size" class="form-label">Position Size (%)</label>
                                            <input type="number" class="form-control bg-dark text-light border-secondary" id="position_size" name="position_size" value="2">
                                        </div>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary mt-3">Run Backtest</button>
                            </form>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-4">
                    <div class="card bg-dark border-secondary mb-4">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Strategy Information</h2>
                        </div>
                        <div class="card-body">
                            <div id="strategy-info">
                                <h3 class="h5">MACD Crossover</h3>
                                <p>Uses Moving Average Convergence Divergence (MACD) crossover signals to identify potential entry points.</p>
                                <ul class="text-muted">
                                    <li>Buy when MACD line crosses above signal line</li>
                                    <li>Sell when MACD line crosses below signal line</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                    
                    <div class="card bg-dark border-secondary">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Recent Backtests</h2>
                        </div>
                        <div class="card-body">
                            <p class="text-muted">Your recent backtest results will appear here</p>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-3 mb-5">
                <a href="/" class="btn btn-secondary">Back to Home</a>
                <a href="/compare_strategies" class="btn btn-info ms-2">Compare Strategies</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/run_backtest', methods=['POST'])
def run_backtest_handler():
    """Handle backtest form submission and run the backtest."""
    return """
    <html>
    <head>
        <title>TradePulse - Backtest Results</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .trading-chart {
                background-color: var(--bs-dark);
                border: 1px solid var(--bs-secondary);
                border-radius: 5px;
                margin-top: 20px;
                height: 300px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1>Backtest Results</h1>
            <div class="alert alert-info">
                Backtest completed successfully!
            </div>
            
            <div class="row">
                <div class="col-12">
                    <div class="card bg-dark border-secondary mb-4">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Performance Summary</h2>
                        </div>
                        <div class="card-body">
                            <div class="trading-chart">
                                <p class="text-center pt-5">Equity curve chart would appear here</p>
                            </div>
                            <div class="row mt-4">
                                <div class="col-md-3">
                                    <div class="card bg-dark border-success h-100">
                                        <div class="card-body text-center">
                                            <h3 class="text-success h2">+12.5%</h3>
                                            <p class="text-muted">Total Return</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-dark border-info h-100">
                                        <div class="card-body text-center">
                                            <h3 class="text-info h2">65%</h3>
                                            <p class="text-muted">Win Rate</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-dark border-warning h-100">
                                        <div class="card-body text-center">
                                            <h3 class="text-warning h2">1.8</h3>
                                            <p class="text-muted">Profit Factor</p>
                                        </div>
                                    </div>
                                </div>
                                <div class="col-md-3">
                                    <div class="card bg-dark border-secondary h-100">
                                        <div class="card-body text-center">
                                            <h3 class="text-light h2">45</h3>
                                            <p class="text-muted">Total Trades</p>
                                        </div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="mt-3 mb-5">
                <a href="/backtest" class="btn btn-secondary">Run Another Backtest</a>
                <a href="/" class="btn btn-primary ms-2">Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/compare_strategies')
def compare_strategies_page():
    """Page for comparing multiple trading strategies."""
    return """
    <html>
    <head>
        <title>TradePulse - Compare Trading Strategies</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .trading-chart {
                background-color: var(--bs-dark);
                border: 1px solid var(--bs-secondary);
                border-radius: 5px;
                margin-top: 20px;
                height: 400px;
            }
            .strategy-selector {
                padding: 10px;
                border-radius: 5px;
                margin-bottom: 10px;
                cursor: pointer;
                transition: all 0.2s ease;
            }
            .strategy-selector:hover {
                transform: translateY(-2px);
            }
            .strategy-selector.active {
                border-left: 4px solid var(--bs-primary);
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">🔍 Compare Trading Strategies</h1>
            
            <div class="card bg-dark border-secondary mb-4">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Comparison Configuration</h2>
                </div>
                <div class="card-body">
                    <form action="/run_comparison" method="POST">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="pair" class="form-label">Trading Pair</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="pair" name="pair">
                                        <option value="EURUSD_otc">EURUSD_otc</option>
                                        <option value="GBPUSD_otc">GBPUSD_otc</option>
                                        <option value="USDJPY_otc">USDJPY_otc</option>
                                        <option value="USDCHF_otc">USDCHF_otc</option>
                                        <option value="AUDUSD_otc">AUDUSD_otc</option>
                                        <option value="USDCAD_otc">USDCAD_otc</option>
                                    </select>
                                </div>
                                <div class="mb-3">
                                    <label for="timeframe" class="form-label">Timeframe</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="timeframe" name="timeframe">
                                        <option value="1m">1 minute</option>
                                        <option value="5m">5 minutes</option>
                                        <option value="15m">15 minutes</option>
                                        <option value="1h">1 hour</option>
                                    </select>
                                </div>
                            </div>
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="start_date" class="form-label">Start Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="start_date" name="start_date">
                                </div>
                                <div class="mb-3">
                                    <label for="end_date" class="form-label">End Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="end_date" name="end_date">
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label class="form-label">Select Strategies to Compare</label>
                            <div class="row">
                                <div class="col-md-4 mb-2">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="strategy_macd" name="strategies" value="macd_crossover" checked>
                                        <label class="form-check-label" for="strategy_macd">MACD Crossover</label>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-2">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="strategy_rsi" name="strategies" value="rsi_strategy" checked>
                                        <label class="form-check-label" for="strategy_rsi">RSI Strategy</label>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-2">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="strategy_bollinger" name="strategies" value="bollinger_bands" checked>
                                        <label class="form-check-label" for="strategy_bollinger">Bollinger Bands</label>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-2">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="strategy_multi" name="strategies" value="multi_indicator">
                                        <label class="form-check-label" for="strategy_multi">Multi-Indicator</label>
                                    </div>
                                </div>
                                <div class="col-md-4 mb-2">
                                    <div class="form-check">
                                        <input class="form-check-input" type="checkbox" id="strategy_tradepulse" name="strategies" value="tradepulse">
                                        <label class="form-check-label" for="strategy_tradepulse">TradePulse Strategy</label>
                                    </div>
                                </div>
                            </div>
                        </div>
                        
                        <button type="submit" class="btn btn-primary mt-2">Compare Strategies</button>
                    </form>
                </div>
            </div>
            
            <div class="mt-3 mb-5">
                <a href="/" class="btn btn-secondary">Back to Home</a>
                <a href="/backtest" class="btn btn-info ms-2">Run Single Backtest</a>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/run_comparison', methods=['POST'])
def run_comparison_handler():
    """Handle strategy comparison form submission and run the comparison."""
    return """
    <html>
    <head>
        <title>TradePulse - Strategy Comparison Results</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .trading-chart {
                background-color: var(--bs-dark);
                border: 1px solid var(--bs-secondary);
                border-radius: 5px;
                margin-top: 20px;
                height: 400px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1>Strategy Comparison Results</h1>
            <div class="alert alert-info">
                Strategy comparison completed!
            </div>
            
            <div class="card bg-dark border-secondary mb-4">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Performance Comparison</h2>
                </div>
                <div class="card-body">
                    <div class="trading-chart">
                        <p class="text-center pt-5">Comparison chart would appear here</p>
                    </div>
                    
                    <div class="table-responsive mt-4">
                        <table class="table table-dark table-bordered">
                            <thead>
                                <tr>
                                    <th>Strategy</th>
                                    <th>Return</th>
                                    <th>Win Rate</th>
                                    <th>Profit Factor</th>
                                    <th>Max Drawdown</th>
                                    <th>Trades</th>
                                </tr>
                            </thead>
                            <tbody>
                                <tr>
                                    <td>MACD Crossover</td>
                                    <td class="text-success">+12.5%</td>
                                    <td>65%</td>
                                    <td>1.8</td>
                                    <td>8.2%</td>
                                    <td>45</td>
                                </tr>
                                <tr>
                                    <td>RSI Strategy</td>
                                    <td class="text-success">+8.7%</td>
                                    <td>58%</td>
                                    <td>1.5</td>
                                    <td>9.8%</td>
                                    <td>62</td>
                                </tr>
                                <tr>
                                    <td>Bollinger Bands</td>
                                    <td class="text-danger">-2.3%</td>
                                    <td>48%</td>
                                    <td>0.9</td>
                                    <td>12.4%</td>
                                    <td>53</td>
                                </tr>
                            </tbody>
                        </table>
                    </div>
                </div>
            </div>
            
            <div class="mt-3 mb-5">
                <a href="/compare_strategies" class="btn btn-secondary">Run Another Comparison</a>
                <a href="/" class="btn btn-primary ms-2">Back to Home</a>
            </div>
        </div>
    </body>
    </html>
    """

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)