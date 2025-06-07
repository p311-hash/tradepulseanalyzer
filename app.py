"""
Flask web application for TradePulse Signals Trading Bot.
This provides a web interface to access signals and bot status.
"""

import os
import logging
from datetime import datetime, timedelta
from flask import Flask, render_template, jsonify, request, redirect, url_for
from dotenv import load_dotenv
import config
from signal_generator import get_latest_signals
from keep_alive import app as keep_alive_app
from backtesting import run_backtest, compare_strategies

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('app')

# Use keep_alive app
app = keep_alive_app

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
                                <a href="/backtest" class="btn btn-success me-2">Backtest Strategies</a>
                                <a href="https://t.me/your_bot_username" target="_blank" class="btn btn-info">Open in Telegram</a>
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
                        </div>
                    </div>
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
    signals = get_latest_signals()
    
    # Format signals for display
    signal_html = '<div class="container mt-4"><h1>Latest Trading Signals</h1><div class="row">'
    
    for pair, signal in signals.items():
        direction = signal.get('direction', 'NEUTRAL')
        confidence = signal.get('strength', '0%')
        indicators = signal.get('indicators', '')
        
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
                    <small class="text-muted">Generated at {signal.get('timestamp', 'N/A')}</small>
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
    signals = get_latest_signals()
    return jsonify(signals)

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
        <title>TradePulse Signals - Backtest Strategies</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .card {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="mb-4">Strategy Backtesting</h1>
            
            <div class="card bg-dark border-secondary">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Backtest Trading Strategies</h2>
                </div>
                <div class="card-body">
                    <p class="card-text">
                        Test trading strategies against historical data to evaluate their performance without risking real money.
                    </p>
                    
                    <form action="/run_backtest" method="post" class="mt-4">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="pair" class="form-label">Trading Pair</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="pair" name="pair" required>
                                        <option value="EURUSD_otc">EURUSD_otc</option>
                                        <option value="GBPUSD_otc">GBPUSD_otc</option>
                                        <option value="USDJPY_otc">USDJPY_otc</option>
                                        <option value="AUDUSD_otc">AUDUSD_otc</option>
                                        <option value="EURGBP_otc">EURGBP_otc</option>
                                        <option value="GBPAUD_otc">GBPAUD_otc (New)</option>
                                        <option value="CADJPY_otc">CADJPY_otc (New)</option>
                                        <option value="EURCAD_otc">EURCAD_otc</option>
                                        <option value="USDCAD_otc">USDCAD_otc</option>
                                        <option value="GBPJPY_otc">GBPJPY_otc</option>
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="timeframe" class="form-label">Timeframe</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="timeframe" name="timeframe" required>
                                        <option value="1m">1 minute</option>
                                        <option value="5m">5 minutes</option>
                                        <option value="15m">15 minutes</option>
                                        <option value="30m">30 minutes</option>
                                        <option value="1h">1 hour</option>
                                        <option value="4h">4 hours</option>
                                        <option value="1d">1 day</option>
                                    </select>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="strategy" class="form-label">Strategy</label>
                                    <select class="form-select bg-dark text-light border-secondary" id="strategy" name="strategy" required>
                                        <option value="macd_crossover">MACD Crossover</option>
                                        <option value="rsi">RSI Strategy</option>
                                        <option value="bollinger_bands">Bollinger Bands</option>
                                        <option value="multi_indicator">Multi-Indicator Strategy</option>
                                        <option value="tradepulse">TradePulse Strategy</option>
                                        <option value="psar_cci" selected>Parabolic SAR + CCI Scalping (NEW)</option>
                                    </select>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="start_date" class="form-label">Start Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="start_date" name="start_date" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="end_date" class="form-label">End Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="end_date" name="end_date" required>
                                </div>
                                
                                <div class="mb-3">
                                    <label for="initial_balance" class="form-label">Initial Balance ($)</label>
                                    <input type="number" class="form-control bg-dark text-light border-secondary" id="initial_balance" name="initial_balance" value="1000" min="100" required>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="use_stop_loss" name="use_stop_loss">
                                <label class="form-check-label" for="use_stop_loss">Use Stop Loss</label>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <div class="form-check form-switch">
                                <input class="form-check-input" type="checkbox" id="use_risk_reward" name="use_risk_reward">
                                <label class="form-check-label" for="use_risk_reward">Use Risk/Reward Ratio</label>
                            </div>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="stop_loss_pct" class="form-label">Stop Loss (%)</label>
                                    <input type="number" class="form-control bg-dark text-light border-secondary" id="stop_loss_pct" name="stop_loss_pct" value="5" min="1" max="50" step="0.1">
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="risk_reward_ratio" class="form-label">Risk/Reward Ratio</label>
                                    <input type="number" class="form-control bg-dark text-light border-secondary" id="risk_reward_ratio" name="risk_reward_ratio" value="2" min="0.1" max="10" step="0.1">
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3">
                            <label for="position_size_pct" class="form-label">Position Size (%)</label>
                            <input type="number" class="form-control bg-dark text-light border-secondary" id="position_size_pct" name="position_size_pct" value="2" min="0.1" max="100" step="0.1">
                        </div>
                        
                        <div class="mb-3">
                            <label for="expiry_minutes" class="form-label">Option Expiry (minutes)</label>
                            <input type="number" class="form-control bg-dark text-light border-secondary" id="expiry_minutes" name="expiry_minutes" value="5" min="1" max="1440">
                        </div>
                        
                        <div class="mb-3 text-center">
                            <button type="submit" class="btn btn-primary">Run Backtest</button>
                            <a href="/compare_strategies" class="btn btn-info ms-2">Compare Strategies</a>
                            <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <script>
            // Set default dates (last 30 days)
            document.addEventListener('DOMContentLoaded', function() {
                const today = new Date();
                const thirtyDaysAgo = new Date();
                thirtyDaysAgo.setDate(today.getDate() - 30);
                
                document.getElementById('end_date').value = today.toISOString().split('T')[0];
                document.getElementById('start_date').value = thirtyDaysAgo.toISOString().split('T')[0];
            });
        </script>
    </body>
    </html>
    """
    
@app.route('/run_backtest', methods=['POST'])
def run_backtest_handler():
    """Handle backtest form submission and run the backtest."""
    try:
        # Get form data
        pair = request.form.get('pair', 'EURUSD_otc')
        timeframe = request.form.get('timeframe', '1m')
        strategy_name = request.form.get('strategy', 'macd_crossover')
        
        # Parse dates
        start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
        
        # Get numeric parameters
        initial_balance = float(request.form.get('initial_balance', 1000))
        position_size_pct = float(request.form.get('position_size_pct', 2))
        stop_loss_pct = float(request.form.get('stop_loss_pct', 5))
        risk_reward_ratio = float(request.form.get('risk_reward_ratio', 2))
        expiry_minutes = int(request.form.get('expiry_minutes', 5))
        
        # Get boolean parameters
        use_stop_loss = 'use_stop_loss' in request.form
        use_risk_reward = 'use_risk_reward' in request.form
        
        # Run the backtest
        logger.info(f"Running backtest for {pair} {timeframe} with {strategy_name} strategy")
        result = run_backtest(
            strategy_name=strategy_name,
            pair=pair,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            initial_balance=initial_balance,
            position_size_pct=position_size_pct,
            use_stop_loss=use_stop_loss,
            stop_loss_pct=stop_loss_pct,
            use_risk_to_reward=use_risk_reward,
            risk_reward_ratio=risk_reward_ratio,
            expiry_minutes=expiry_minutes
        )
        
        # Generate charts
        equity_curve = result.generate_equity_curve()
        trade_distribution = result.generate_trade_distribution()
        monthly_returns = result.generate_monthly_returns()
        
        # Format result data for display
        result_dict = result.to_dict()
        
        # Format dates for display
        result_dict['start_date'] = start_date.strftime('%Y-%m-%d')
        result_dict['end_date'] = end_date.strftime('%Y-%m-%d')
        
        # Build HTML for the results page
        html = f"""
        <html>
        <head>
            <title>TradePulse Signals - Backtest Results</title>
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }}
                .card {{
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .metrics-box {{
                    border-left: 4px solid var(--bs-primary);
                    padding-left: 15px;
                    margin-bottom: 15px;
                }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="mb-4">Backtest Results</h1>
                
                <div class="card bg-dark border-secondary">
                    <div class="card-header bg-dark border-secondary">
                        <div class="d-flex justify-content-between align-items-center">
                            <h2 class="h4 m-0">{result_dict['pair']} - {result_dict['strategy_name']} Strategy</h2>
                            <span class="badge bg-info">{result_dict['timeframe']} Timeframe</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <div class="row">
                            <div class="col-md-6">
                                <div class="metrics-box">
                                    <h3 class="h5">Summary</h3>
                                    <p>Period: {result_dict['start_date']} to {result_dict['end_date']}</p>
                                    <p>Initial Balance: ${result_dict['initial_balance']:.2f}</p>
                                    <p>Final Balance: ${result_dict['final_balance']:.2f}</p>
                                    <p>Total Profit: ${result_dict['total_profit']:.2f} ({result_dict['total_profit_pct']:.2f}%)</p>
                                </div>
                                
                                <div class="metrics-box">
                                    <h3 class="h5">Trade Statistics</h3>
                                    <p>Total Trades: {result_dict['total_trades']}</p>
                                    <p>Win Rate: {result_dict['win_rate']:.2f}%</p>
                                    <p>Winning Trades: {result_dict['winning_trades']}</p>
                                    <p>Losing Trades: {result_dict['losing_trades']}</p>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="metrics-box">
                                    <h3 class="h5">Risk Metrics</h3>
                                    <p>Profit Factor: {result_dict['profit_factor']:.2f}</p>
                                    <p>Max Drawdown: ${result_dict['max_drawdown']:.2f} ({result_dict['max_drawdown_pct']:.2f}%)</p>
                                    <p>Sharpe Ratio: {result_dict['sharpe_ratio']:.2f}</p>
                                </div>
                                
                                <div class="metrics-box">
                                    <h3 class="h5">Average Metrics</h3>
                                    <p>Avg Profit per Trade: ${result_dict['avg_profit_per_trade']:.2f}</p>
                                    <p>Avg Win: ${result_dict['avg_win']:.2f}</p>
                                    <p>Avg Loss: ${result_dict['avg_loss']:.2f}</p>
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-4">
                            <div class="col-12">
                                <div class="chart-container">
                                    <h3 class="h5">Equity Curve</h3>
                                    {f'<img src="data:image/png;base64,{equity_curve}" class="img-fluid" alt="Equity Curve">' if equity_curve else '<p class="text-muted">No equity curve data available</p>'}
                                </div>
                            </div>
                        </div>
                        
                        <div class="row mt-2">
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h3 class="h5">Trade Distribution</h3>
                                    {f'<img src="data:image/png;base64,{trade_distribution}" class="img-fluid" alt="Trade Distribution">' if trade_distribution else '<p class="text-muted">No trade distribution data available</p>'}
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="chart-container">
                                    <h3 class="h5">Monthly Returns</h3>
                                    {f'<img src="data:image/png;base64,{monthly_returns}" class="img-fluid" alt="Monthly Returns">' if monthly_returns else '<p class="text-muted">No monthly returns data available</p>'}
                                </div>
                            </div>
                        </div>
                        
                        <div class="text-center mt-3">
                            <a href="/backtest" class="btn btn-primary">Run Another Backtest</a>
                            <a href="/compare_strategies" class="btn btn-info ms-2">Compare Strategies</a>
                            <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                        </div>
                    </div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error running backtest: {str(e)}")
        return f"""
        <html>
        <head>
            <title>TradePulse Signals - Backtest Error</title>
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
            <div class="container mt-4">
                <h1 class="mb-4">Backtest Error</h1>
                
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error running backtest!</h4>
                    <p>{str(e)}</p>
                </div>
                
                <a href="/backtest" class="btn btn-primary">Back to Backtest</a>
                <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
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
        <title>TradePulse Signals - Compare Strategies</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {
                padding: 20px;
                background-color: var(--bs-dark);
                color: var(--bs-light);
            }
            .card {
                margin-bottom: 20px;
            }
        </style>
    </head>
    <body>
        <div class="container mt-4">
            <h1 class="mb-4">Compare Trading Strategies</h1>
            
            <div class="card bg-dark border-secondary">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Strategy Comparison</h2>
                </div>
                <div class="card-body">
                    <p class="card-text">
                        Compare multiple trading strategies against the same historical data to evaluate their relative performance.
                    </p>
                    
                    <form action="/run_comparison" method="post" class="mt-4">
                        <div class="mb-3">
                            <label for="pair" class="form-label">Trading Pair</label>
                            <select class="form-select bg-dark text-light border-secondary" id="pair" name="pair" required>
                                <option value="EURUSD_otc">EURUSD_otc</option>
                                <option value="GBPUSD_otc">GBPUSD_otc</option>
                                <option value="USDJPY_otc">USDJPY_otc</option>
                                <option value="AUDUSD_otc">AUDUSD_otc</option>
                                <option value="EURGBP_otc">EURGBP_otc</option>
                            </select>
                        </div>
                        
                        <div class="mb-3">
                            <label for="timeframe" class="form-label">Timeframe</label>
                            <select class="form-select bg-dark text-light border-secondary" id="timeframe" name="timeframe" required>
                                <option value="1m">1 minute</option>
                                <option value="5m">5 minutes</option>
                                <option value="15m">15 minutes</option>
                                <option value="30m">30 minutes</option>
                                <option value="1h">1 hour</option>
                                <option value="4h">4 hours</option>
                                <option value="1d">1 day</option>
                            </select>
                        </div>
                        
                        <div class="row">
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="start_date" class="form-label">Start Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="start_date" name="start_date" required>
                                </div>
                            </div>
                            
                            <div class="col-md-6">
                                <div class="mb-3">
                                    <label for="end_date" class="form-label">End Date</label>
                                    <input type="date" class="form-control bg-dark text-light border-secondary" id="end_date" name="end_date" required>
                                </div>
                            </div>
                        </div>
                        
                        <div class="card bg-dark border-secondary mb-3">
                            <div class="card-header bg-dark border-secondary">
                                <h3 class="h5 m-0">Strategies to Compare</h3>
                            </div>
                            <div class="card-body">
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="macd_crossover" name="strategies" value="macd_crossover" checked>
                                    <label class="form-check-label" for="macd_crossover">MACD Crossover</label>
                                </div>
                                
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="rsi" name="strategies" value="rsi" checked>
                                    <label class="form-check-label" for="rsi">RSI Strategy</label>
                                </div>
                                
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="bollinger_bands" name="strategies" value="bollinger_bands" checked>
                                    <label class="form-check-label" for="bollinger_bands">Bollinger Bands</label>
                                </div>
                                
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="multi_indicator" name="strategies" value="multi_indicator" checked>
                                    <label class="form-check-label" for="multi_indicator">Multi-Indicator Strategy</label>
                                </div>
                                
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="tradepulse" name="strategies" value="tradepulse" checked>
                                    <label class="form-check-label" for="tradepulse">TradePulse Strategy</label>
                                </div>
                                
                                <div class="form-check">
                                    <input class="form-check-input" type="checkbox" id="psar_cci" name="strategies" value="psar_cci" checked>
                                    <label class="form-check-label" for="psar_cci">Parabolic SAR + CCI Scalping (NEW)</label>
                                </div>
                            </div>
                        </div>
                        
                        <div class="mb-3 text-center">
                            <button type="submit" class="btn btn-primary">Compare Strategies</button>
                            <a href="/backtest" class="btn btn-info ms-2">Individual Backtest</a>
                            <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                        </div>
                    </form>
                </div>
            </div>
        </div>
        
        <script>
            // Set default dates (last 30 days)
            document.addEventListener('DOMContentLoaded', function() {
                const today = new Date();
                const thirtyDaysAgo = new Date();
                thirtyDaysAgo.setDate(today.getDate() - 30);
                
                document.getElementById('end_date').value = today.toISOString().split('T')[0];
                document.getElementById('start_date').value = thirtyDaysAgo.toISOString().split('T')[0];
            });
        </script>
    </body>
    </html>
    """

@app.route('/run_comparison', methods=['POST'])
def run_comparison_handler():
    """Handle strategy comparison form submission and run the comparison."""
    try:
        # Get form data
        pair = request.form.get('pair', 'EURUSD_otc')
        timeframe = request.form.get('timeframe', '1m')
        
        # Get selected strategies
        strategies = request.form.getlist('strategies')
        if not strategies:
            strategies = ['macd_crossover', 'rsi', 'bollinger_bands', 'multi_indicator', 'tradepulse', 'psar_cci']
        
        # Parse dates
        start_date = datetime.strptime(request.form.get('start_date'), '%Y-%m-%d')
        end_date = datetime.strptime(request.form.get('end_date'), '%Y-%m-%d')
        
        # Run comparison for selected strategies
        logger.info(f"Comparing strategies for {pair} {timeframe}")
        
        results = {}
        for strategy in strategies:
            result = run_backtest(
                strategy_name=strategy,
                pair=pair,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )
            results[strategy] = result
        
        # Generate comparison chart
        from backtesting import generate_comparison_chart
        comparison_chart = generate_comparison_chart(results)
        
        # Build HTML table of results
        table_rows = ""
        for strategy, result in results.items():
            strategy_display = strategy.replace('_', ' ').title()
            profit_color = 'success' if result.final_balance > result.initial_balance else 'danger'
            
            table_rows += f"""
            <tr>
                <td>{strategy_display}</td>
                <td>${result.initial_balance:.2f}</td>
                <td class="text-{profit_color}">${result.final_balance:.2f}</td>
                <td class="text-{profit_color}">${result.final_balance - result.initial_balance:.2f} ({((result.final_balance / result.initial_balance) - 1) * 100:.2f}%)</td>
                <td>{result.total_trades}</td>
                <td>{result.win_rate:.2f}%</td>
                <td>{result.profit_factor:.2f}</td>
                <td>{result.max_drawdown_pct:.2f}%</td>
                <td>{result.sharpe_ratio:.2f}</td>
            </tr>
            """
        
        # Build HTML for comparison results page
        html = f"""
        <html>
        <head>
            <title>TradePulse Signals - Strategy Comparison</title>
            <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                body {{
                    padding: 20px;
                    background-color: var(--bs-dark);
                    color: var(--bs-light);
                }}
                .card {{
                    margin-bottom: 20px;
                }}
                .chart-container {{
                    text-align: center;
                    margin-bottom: 20px;
                }}
                .strategy-winner {{
                    background-color: rgba(25, 135, 84, 0.15);
                }}
            </style>
        </head>
        <body>
            <div class="container mt-4">
                <h1 class="mb-4">Strategy Comparison Results</h1>
                
                <div class="card bg-dark border-secondary">
                    <div class="card-header bg-dark border-secondary">
                        <div class="d-flex justify-content-between align-items-center">
                            <h2 class="h4 m-0">Comparing Strategies: {pair}</h2>
                            <span class="badge bg-info">{timeframe} Timeframe</span>
                        </div>
                    </div>
                    <div class="card-body">
                        <p>Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}</p>
                        
                        <div class="chart-container mt-4">
                            <h3 class="h5">Strategy Performance Comparison</h3>
                            {f'<img src="data:image/png;base64,{comparison_chart}" class="img-fluid" alt="Strategy Comparison">' if comparison_chart else '<p class="text-muted">No comparison chart available</p>'}
                        </div>
                        
                        <div class="table-responsive mt-4">
                            <table class="table table-dark table-striped">
                                <thead>
                                    <tr>
                                        <th>Strategy</th>
                                        <th>Initial Balance</th>
                                        <th>Final Balance</th>
                                        <th>Profit/Loss</th>
                                        <th>Total Trades</th>
                                        <th>Win Rate</th>
                                        <th>Profit Factor</th>
                                        <th>Max Drawdown</th>
                                        <th>Sharpe Ratio</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    {table_rows}
                                </tbody>
                            </table>
                        </div>
                        
                        <div class="text-center mt-3">
                            <a href="/compare_strategies" class="btn btn-primary">Compare Different Strategies</a>
                            <a href="/backtest" class="btn btn-info ms-2">Individual Backtest</a>
                            <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
                        </div>
                    </div>
                </div>
            </div>
            
            <script>
                // Highlight the best performing strategy
                document.addEventListener('DOMContentLoaded', function() {
                    const rows = document.querySelectorAll('tbody tr');
                    let bestProfit = -Infinity;
                    let bestRow = null;
                    
                    rows.forEach(row => {
                        const profitCell = row.cells[3];
                        const profitText = profitCell.textContent;
                        const profit = parseFloat(profitText.match(/[-+]?[0-9]*\.?[0-9]+/)[0]);
                        
                        if (profit > bestProfit) {
                            bestProfit = profit;
                            bestRow = row;
                        }
                    });
                    
                    if (bestRow && bestProfit > 0) {
                        bestRow.classList.add('strategy-winner');
                    }
                });
            </script>
        </body>
        </html>
        """
        
        return html
        
    except Exception as e:
        logger.error(f"Error comparing strategies: {str(e)}")
        return f"""
        <html>
        <head>
            <title>TradePulse Signals - Comparison Error</title>
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
            <div class="container mt-4">
                <h1 class="mb-4">Strategy Comparison Error</h1>
                
                <div class="alert alert-danger">
                    <h4 class="alert-heading">Error comparing strategies!</h4>
                    <p>{str(e)}</p>
                </div>
                
                <a href="/compare_strategies" class="btn btn-primary">Back to Compare</a>
                <a href="/" class="btn btn-secondary ms-2">Back to Home</a>
            </div>
        </body>
        </html>
        """

# Add the main function to make this file runnable
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)