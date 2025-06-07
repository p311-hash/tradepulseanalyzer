import logging
import os
from dotenv import load_dotenv
from telegram_handler import TelegramHandler
from utils import setup_logging, fetch_market_data
from signal_generator import SignalGenerator, get_latest_signals
import schedule
import time
import threading
from datetime import datetime, timezone
import config
import pandas as pd
from typing import Dict
from flask import Flask, jsonify, request

# Create Flask app for web interface
app = Flask(__name__)

# Web interface routes
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
                                            <input type="number" class="form-control bg-dark text-light border-secondary" id="initial_balance" name="initial_balance" value="1000" min="100">
                                        </div>
                                    </div>
                                    <div class="col-md-6">
                                        <div class="mb-3">
                                            <label for="position_size" class="form-label">Position Size (%)</label>
                                            <input type="number" class="form-control bg-dark text-light border-secondary" id="position_size" name="position_size" value="2" min="0.1" max="100" step="0.1">
                                        </div>
                                    </div>
                                </div>
                                <button type="submit" class="btn btn-primary">Run Backtest</button>
                                <a href="/compare_strategies" class="btn btn-info ms-2">Compare Strategies</a>
                                <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                            </form>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card bg-dark border-secondary">
                        <div class="card-header bg-dark border-secondary">
                            <h2 class="h4 m-0">Strategies</h2>
                        </div>
                        <div class="card-body">
                            <div class="strategy-card card bg-dark border-secondary mb-3">
                                <div class="card-body">
                                    <h3 class="h5">MACD Crossover</h3>
                                    <p class="small text-muted">Buy when MACD line crosses above signal line, sell when it crosses below.</p>
                                </div>
                            </div>
                            <div class="strategy-card card bg-dark border-secondary mb-3">
                                <div class="card-body">
                                    <h3 class="h5">RSI Strategy</h3>
                                    <p class="small text-muted">Buy when RSI crosses above oversold threshold, sell when it crosses below overbought threshold.</p>
                                </div>
                            </div>
                            <div class="strategy-card card bg-dark border-secondary mb-3">
                                <div class="card-body">
                                    <h3 class="h5">Bollinger Bands</h3>
                                    <p class="small text-muted">Buy when price touches lower band, sell when it touches upper band.</p>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Set default dates (1 month ago to today)
            document.addEventListener('DOMContentLoaded', function() {
                const today = new Date();
                const oneMonthAgo = new Date();
                oneMonthAgo.setMonth(oneMonthAgo.getMonth() - 1);

                document.getElementById('end_date').valueAsDate = today;
                document.getElementById('start_date').valueAsDate = oneMonthAgo;
            });
        </script>
    </body>
    </html>
    """

@app.route('/run_backtest', methods=['POST'])
def run_backtest_handler():
    """Handle backtest form submission and run the backtest."""
    from backtesting import run_backtest
    from datetime import datetime

    # Get form data
    pair = request.form.get('pair')
    timeframe = request.form.get('timeframe')
    strategy = request.form.get('strategy')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    initial_balance = float(request.form.get('initial_balance', 1000))
    position_size = float(request.form.get('position_size', 2))

    # Run backtest
    try:
        result = run_backtest(
            strategy_name=strategy,
            pair=pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            position_size_pct=position_size
        )

        # Generate chart images
        equity_curve = result.generate_equity_curve()
        trade_distribution = result.generate_trade_distribution()
        monthly_returns = result.generate_monthly_returns()

        # Format metrics for display
        metrics = result.to_dict()

        return f"""
        <html>
        <head>
            <title>TradePulse - Backtest Results</title>
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }}
                .metrics-card {{
                    border-left: 4px solid var(--bs-info);
                }}
                .chart-container {{
                    margin-bottom: 20px;
                    background-color: var(--bs-dark);
                    border-radius: 5px;
                    padding: 10px;
                    border: 1px solid var(--bs-secondary);
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mt-4 mb-4">📈 Backtest Results</h1>

                <div class="row">
                    <div class="col-md-4">
                        <div class="card bg-dark border-info mb-4 metrics-card">
                            <div class="card-header bg-dark border-info">
                                <h2 class="h4 m-0">Backtest Summary</h2>
                            </div>
                            <div class="card-body">
                                <p><strong>Pair:</strong> {pair}</p>
                                <p><strong>Timeframe:</strong> {timeframe}</p>
                                <p><strong>Strategy:</strong> {strategy}</p>
                                <p><strong>Period:</strong> {start_date} to {end_date}</p>
                                <p><strong>Initial Balance:</strong> ${initial_balance:.2f}</p>
                                <p><strong>Final Balance:</strong> ${metrics.get('final_balance', 0):.2f}</p>
                                <p><strong>Net Profit:</strong> <span class="{'text-success' if metrics.get('net_profit_percent', 0) >= 0 else 'text-danger'}">${metrics.get('net_profit', 0):.2f} ({metrics.get('net_profit_percent', 0):.2f}%)</span></p>
                                <a href="/backtest" class="btn btn-primary mt-3">Run Another Backtest</a>
                                <a href="/" class="btn btn-secondary mt-3">Back to Home</a>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-8">
                        <div class="card bg-dark border-secondary mb-4">
                            <div class="card-header bg-dark border-secondary">
                                <h2 class="h4 m-0">Performance Metrics</h2>
                            </div>
                            <div class="card-body">
                                <div class="row">
                                    <div class="col-md-6">
                                        <p><strong>Total Trades:</strong> {metrics.get('total_trades', 0)}</p>
                                        <p><strong>Win Rate:</strong> {metrics.get('win_rate', 0):.2f}%</p>
                                        <p><strong>Profit Factor:</strong> {metrics.get('profit_factor', 0):.2f}</p>
                                        <p><strong>Max Drawdown:</strong> {metrics.get('max_drawdown_percent', 0):.2f}%</p>
                                    </div>
                                    <div class="col-md-6">
                                        <p><strong>Sharpe Ratio:</strong> {metrics.get('sharpe_ratio', 0):.2f}</p>
                                        <p><strong>Avg Profit/Trade:</strong> ${metrics.get('avg_profit_per_trade', 0):.2f}</p>
                                        <p><strong>Avg Profit %/Trade:</strong> {metrics.get('avg_profit_percent_per_trade', 0):.2f}%</p>
                                        <p><strong>Max Consecutive Losses:</strong> {metrics.get('max_consecutive_losses', 0)}</p>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-12">
                        <div class="card bg-dark border-secondary mb-4">
                            <div class="card-header bg-dark border-secondary">
                                <h2 class="h4 m-0">Equity Curve</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{equity_curve}" alt="Equity Curve">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>

                <div class="row">
                    <div class="col-md-6">
                        <div class="card bg-dark border-secondary mb-4">
                            <div class="card-header bg-dark border-secondary">
                                <h2 class="h4 m-0">Trade Distribution</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{trade_distribution}" alt="Trade Distribution">
                                </div>
                            </div>
                        </div>
                    </div>

                    <div class="col-md-6">
                        <div class="card bg-dark border-secondary mb-4">
                            <div class="card-header bg-dark border-secondary">
                                <h2 class="h4 m-0">Monthly Returns</h2>
                            </div>
                            <div class="card-body">
                                <div class="chart-container">
                                    <img src="data:image/png;base64,{monthly_returns}" alt="Monthly Returns">
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <head>
            <title>TradePulse - Backtest Error</title>
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
            <div class="container">
                <h1 class="mt-4 mb-4">❌ Backtest Error</h1>

                <div class="card bg-dark border-danger mb-4">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 m-0">Error Running Backtest</h2>
                    </div>
                    <div class="card-body">
                        <p>An error occurred while running the backtest:</p>
                        <div class="alert alert-danger">{str(e)}</div>
                        <a href="/backtest" class="btn btn-primary mt-3">Back to Backtest Form</a>
                    </div>
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
        <title>TradePulse - Compare Strategies</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .strategy-checkbox {
                margin-right: 10px;
            }
            .chart-container {
                margin-top: 20px;
                background-color: var(--bs-dark);
                border-radius: 5px;
                padding: 10px;
                border: 1px solid var(--bs-secondary);
            }
            .chart-container img {
                max-width: 100%;
                height: auto;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">🔄 Compare Trading Strategies</h1>

            <div class="card bg-dark border-secondary mb-4">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Comparison Configuration</h2>
                </div>
                <div class="card-body">
                    <form action="/run_comparison" method="POST">
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
                            <label class="form-label">Select Strategies to Compare</label>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="macd_crossover" id="macd_strategy" name="strategies" checked>
                                <label class="form-check-label" for="macd_strategy">
                                    MACD Crossover
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="rsi_strategy" id="rsi_strategy" name="strategies" checked>
                                <label class="form-check-label" for="rsi_strategy">
                                    RSI Strategy
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="bollinger_bands" id="bb_strategy" name="strategies" checked>
                                <label class="form-check-label" for="bb_strategy">
                                    Bollinger Bands
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="multi_indicator" id="multi_strategy" name="strategies" checked>
                                <label class="form-check-label" for="multi_strategy">
                                    Multi-Indicator Strategy
                                </label>
                            </div>
                            <div class="form-check">
                                <input class="form-check-input" type="checkbox" value="tradepulse" id="tradepulse_strategy" name="strategies" checked>
                                <label class="form-check-label" for="tradepulse_strategy">
                                    TradePulse Strategy
                                </label>
                            </div>
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

                        <div class="mb-3">
                            <label for="initial_balance" class="form-label">Initial Balance</label>
                            <input type="number" class="form-control bg-dark text-light border-secondary" id="initial_balance" name="initial_balance" value="1000" min="100">
                        </div>

                        <button type="submit" class="btn btn-primary">Compare Strategies</button>
                        <a href="/backtest" class="btn btn-secondary ms-2">Back to Backtest</a>
                        <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                    </form>
                </div>
            </div>
        </div>

        <script>
            // Set default dates (3 months ago to today)
            document.addEventListener('DOMContentLoaded', function() {
                const today = new Date();
                const threeMonthsAgo = new Date();
                threeMonthsAgo.setMonth(threeMonthsAgo.getMonth() - 3);

                document.getElementById('end_date').valueAsDate = today;
                document.getElementById('start_date').valueAsDate = threeMonthsAgo;
            });
        </script>
    </body>
    </html>
    """

@app.route('/run_comparison', methods=['POST'])
def run_comparison_handler():
    """Handle strategy comparison form submission and run the comparison."""
    from backtesting import compare_strategies, generate_comparison_chart

    # Get form data
    pair = request.form.get('pair')
    timeframe = request.form.get('timeframe')
    strategies = request.form.getlist('strategies')
    start_date = request.form.get('start_date')
    end_date = request.form.get('end_date')
    initial_balance = float(request.form.get('initial_balance', 1000))

    if not strategies:
        return """
        <html>
        <head>
            <title>TradePulse - Comparison Error</title>
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mt-4 mb-4">❌ Comparison Error</h1>

                <div class="card bg-dark border-danger mb-4">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 m-0">Error Running Comparison</h2>
                    </div>
                    <div class="card-body">
                        <p>You must select at least one strategy to compare.</p>
                        <a href="/compare_strategies" class="btn btn-primary mt-3">Back to Comparison Form</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

    try:
        from backtesting import run_backtest

        # Run backtest for each selected strategy
        results = {}
        for strategy in strategies:
            result = run_backtest(
                strategy_name=strategy,
                pair=pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date,
                initial_balance=initial_balance
            )
            results[strategy] = result

        # Generate comparison chart
        comparison_chart = generate_comparison_chart(results)

        # Build comparison table
        comparison_table = '<table class="table table-dark table-hover"><thead><tr>'
        comparison_table += '<th>Strategy</th><th>Net Profit</th><th>Win Rate</th><th>Profit Factor</th><th>Max Drawdown</th><th>Sharpe Ratio</th>'
        comparison_table += '</tr></thead><tbody>'

        for strategy, result in results.items():
            metrics = result.to_dict()
            comparison_table += f'<tr>'
            comparison_table += f'<td>{strategy}</td>'
            comparison_table += f'<td class="{"text-success" if metrics.get("net_profit", 0) >= 0 else "text-danger"}">${metrics.get("net_profit", 0):.2f} ({metrics.get("net_profit_percent", 0):.2f}%)</td>'
            comparison_table += f'<td>{metrics.get("win_rate", 0):.2f}%</td>'
            comparison_table += f'<td>{metrics.get("profit_factor", 0):.2f}</td>'
            comparison_table += f'<td>{metrics.get("max_drawdown_percent", 0):.2f}%</td>'
            comparison_table += f'<td>{metrics.get("sharpe_ratio", 0):.2f}</td>'
            comparison_table += f'</tr>'

        comparison_table += '</tbody></table>'

        return f"""
        <html>
        <head>
            <title>TradePulse - Strategy Comparison Results</title>
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }}
                .chart-container {{
                    margin-top: 20px;
                    background-color: var(--bs-dark);
                    border-radius: 5px;
                    padding: 10px;
                    border: 1px solid var(--bs-secondary);
                }}
                .chart-container img {{
                    max-width: 100%;
                    height: auto;
                }}
                .strategy-winner {{
                    background-color: rgba(40, 167, 69, 0.2);
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 class="mt-4 mb-4">📊 Strategy Comparison Results</h1>

                <div class="card bg-dark border-secondary mb-4">
                    <div class="card-header bg-dark border-secondary">
                        <h2 class="h4 m-0">Comparison Summary</h2>
                    </div>
                    <div class="card-body">
                        <p><strong>Pair:</strong> {pair}</p>
                        <p><strong>Timeframe:</strong> {timeframe}</p>
                        <p><strong>Period:</strong> {start_date} to {end_date}</p>
                        <p><strong>Initial Balance:</strong> ${initial_balance:.2f}</p>
                        <p><strong>Strategies Compared:</strong> {", ".join(strategies)}</p>
                    </div>
                </div>

                <div class="card bg-dark border-secondary mb-4">
                    <div class="card-header bg-dark border-secondary">
                        <h2 class="h4 m-0">Performance Comparison</h2>
                    </div>
                    <div class="card-body">
                        <div class="table-responsive">
                            {comparison_table}
                        </div>
                    </div>
                </div>

                <div class="card bg-dark border-secondary mb-4">
                    <div class="card-header bg-dark border-secondary">
                        <h2 class="h4 m-0">Equity Curves Comparison</h2>
                    </div>
                    <div class="card-body">
                        <div class="chart-container">
                            <img src="data:image/png;base64,{comparison_chart}" alt="Strategy Comparison">
                        </div>
                    </div>
                </div>

                <div class="mt-4 mb-5">
                    <a href="/compare_strategies" class="btn btn-primary me-2">Run Another Comparison</a>
                    <a href="/backtest" class="btn btn-secondary me-2">Back to Backtest</a>
                    <a href="/" class="btn btn-secondary">Back to Home</a>
                </div>
            </div>
        </body>
        </html>
        """
    except Exception as e:
        return f"""
        <html>
        <head>
            <title>TradePulse - Comparison Error</title>
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
            <div class="container">
                <h1 class="mt-4 mb-4">❌ Comparison Error</h1>

                <div class="card bg-dark border-danger mb-4">
                    <div class="card-header bg-danger text-white">
                        <h2 class="h4 m-0">Error Running Comparison</h2>
                    </div>
                    <div class="card-body">
                        <p>An error occurred while comparing strategies:</p>
                        <div class="alert alert-danger">{str(e)}</div>
                        <a href="/compare_strategies" class="btn btn-primary mt-3">Back to Comparison Form</a>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """

# Initialize the keep_alive server
from keep_alive import start_server

# Import continuous learning module if available
try:
    from continuous_learning import ContinuousLearningSystem
    CONTINUOUS_LEARNING_ENABLED = True
except ImportError:
    CONTINUOUS_LEARNING_ENABLED = False

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Store signals for automated signals
last_signals = {}

def generate_scheduled_signals():
    """Generate and store signals for all pairs and timeframes on schedule."""
    try:
        logger.info("Generating scheduled signals")

        # Generate signals for each pair and timeframe
        for pair in config.CURRENCY_PAIRS:
            pair_signals = {}

            # Create multi-timeframe data for this pair
            timeframe_data = {}
            for tf in config.TIMEFRAMES:
                data = fetch_market_data(pair=pair, timeframe=tf)
                if not data.empty:
                    timeframe_data[tf] = data

            # Generate consolidated signal across timeframes
            if timeframe_data:
                multi_tf_signal = SignalGenerator.generate_multi_timeframe_signal(timeframe_data, pair)
                last_signals[pair] = multi_tf_signal
                logger.info(f"Generated {pair} signal: {multi_tf_signal['consolidated']['signal']} ({multi_tf_signal['consolidated']['confidence']:.1f}%)")
            else:
                logger.error(f"Failed to fetch data for {pair}")

    except Exception as e:
        logger.error(f"Error generating scheduled signals: {str(e)}", exc_info=True)

def setup_signal_scheduler():
    """Setup scheduler for regular signal generation."""
    # Generate signals every 15 minutes
    schedule.every(15).minutes.do(generate_scheduled_signals)

    # Generate initial signals
    generate_scheduled_signals()

    logger.info("Signal scheduler initialized")

def scheduler_thread():
    """Run scheduler in a separate thread."""
    while True:
        schedule.run_pending()
        time.sleep(1)

# Initialize continuous learning system
continuous_learner = None
if CONTINUOUS_LEARNING_ENABLED:
    try:
        continuous_learner = ContinuousLearningSystem()
        logger.info("Initialized continuous learning system")
    except Exception as e:
        logger.error(f"Error initializing continuous learning system: {str(e)}", exc_info=True)
        CONTINUOUS_LEARNING_ENABLED = False

def update_signal_outcomes():
    """Update outcomes for expired signals in the continuous learning system."""
    if not CONTINUOUS_LEARNING_ENABLED or continuous_learner is None:
        return

    try:
        logger.info("Updating signal outcomes for continuous learning")

        # Get historical signals that need outcome updates
        signals_to_update = continuous_learner.get_signals_pending_outcome()

        if not signals_to_update:
            logger.info("No signals pending outcome updates")
            return

        logger.info(f"Found {len(signals_to_update)} signals needing outcome updates")

        # Update each signal with its actual outcome
        for signal_id, signal_data in signals_to_update.items():
            try:
                # Get the current price for the signal's asset
                pair = signal_data.get('pair', 'unknown')
                if pair == 'unknown':
                    continue

                # Fetch latest price data
                current_data = fetch_market_data(pair=pair, timeframe='1m')
                if current_data.empty:
                    logger.warning(f"Could not fetch current price data for {pair}")
                    continue

                # Get current price
                current_price = current_data['close'].iloc[-1]

                # Update the signal outcome
                continuous_learner.update_signal_outcome(signal_id, current_price)
                logger.info(f"Updated outcome for signal {signal_id} - Current price: {current_price}")

            except Exception as e:
                logger.error(f"Error updating outcome for signal {signal_id}: {str(e)}")

        # Save learning data
        continuous_learner.save_data()

    except Exception as e:
        logger.error(f"Error updating signal outcomes: {str(e)}", exc_info=True)

def main():
    # Setup logging
    setup_logging()

    try:
        # Start the keep-alive server to prevent Replit from sleeping
        keep_alive_thread = start_server()
        logger.info("Started keep-alive server to prevent Replit from sleeping")

        # Initialize continuous learning if enabled
        if CONTINUOUS_LEARNING_ENABLED and continuous_learner is not None:
            # Start the continuous learning thread
            continuous_learner.start_learning_thread()
            logger.info("Started continuous learning background thread")

            # Schedule regular updates of signal outcomes
            schedule.every(5).minutes.do(update_signal_outcomes)
            logger.info("Scheduled signal outcome updates every 5 minutes")

        # Setup signal generation scheduler
        setup_signal_scheduler()

        # Start scheduler in a separate thread
        scheduler = threading.Thread(target=scheduler_thread)
        scheduler.daemon = True
        scheduler.start()
        logger.info("Signal scheduler thread started")

        # Check if running via gunicorn, in web server mode, or if "Start application" workflow
        # Note: When running via the tradepulse-bot workflow, we'll start the Telegram bot
        # When running as a web server via gunicorn or other web server mode, we'll skip the Telegram bot to avoid conflicts
        web_server_mode = (
            os.environ.get('GUNICORN_WORKER', '0') == '1' or
            os.environ.get('WEB_SERVER_MODE', '0') == '1' or
            'FLASK_APP' in os.environ
        )

        # Check if we're in bot mode (explicitly set BOT_MODE=1)
        bot_mode = os.environ.get('BOT_MODE', '0') == '1'

        # Look for a running bot using a file lock
        lock_file_path = '.bot_lock'
        try:
            if os.path.exists(lock_file_path):
                with open(lock_file_path, 'r') as f:
                    lock_time_str = f.read().strip()
                    try:
                        lock_time = float(lock_time_str)
                        # If lock file is older than 5 minutes, consider the bot crashed
                        if time.time() - lock_time > 300:  # 5 minutes
                            os.remove(lock_file_path)
                            logger.info("Detected stale bot lock - removed")
                        else:
                            logger.info(f"Bot lock found from {time.time() - lock_time:.1f} seconds ago")
                    except ValueError:
                        os.remove(lock_file_path)
                        logger.info("Invalid bot lock format - removed")
        except Exception as e:
            logger.error(f"Error checking bot lock: {str(e)}")

        # Set lock file if starting bot
        if (bot_mode or not web_server_mode) and not os.path.exists(lock_file_path):
            try:
                with open(lock_file_path, 'w') as f:
                    f.write(str(time.time()))

                # Start the Telegram bot
                os.environ['TELEGRAM_BOT_RUNNING'] = '1'
                # Initialize Telegram bot with token from config
                telegram_handler = TelegramHandler(config.TELEGRAM_TOKEN)

                # Start the bot
                logger.info("Starting Binary Trading Signals Bot...")
                telegram_handler.run()
            except Exception as e:
                logger.error(f"Failed to start bot: {str(e)}")
                if os.path.exists(lock_file_path):
                    try:
                        os.remove(lock_file_path)
                    except:
                        pass
        else:
            logger.info("Running in web server mode or another bot instance is already running - Telegram bot not started to avoid conflicts")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise

if __name__ == "__main__":
    main()
