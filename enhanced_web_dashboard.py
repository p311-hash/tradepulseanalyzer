"""
Enhanced Web Dashboard with Real-time Updates and Advanced Features
Provides comprehensive web interface with authentication, real-time data, and advanced charting.
"""

from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import asyncio
import threading
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import plotly.graph_objects as go
import plotly.utils
from enhanced_data_reliability import EnhancedDataReliabilityManager
from live_trading_engine import LiveTradingEngine
from enhanced_sentiment_integration import EnhancedSentimentIntegration
from signal_generator import SignalGenerator
from error_handling_system import ErrorHandlingSystem
import config

logger = logging.getLogger(__name__)

# Flask app setup
app = Flask(__name__)
app.config['SECRET_KEY'] = getattr(config, 'SECRET_KEY', 'your-secret-key-here')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User class for authentication
class User(UserMixin):
    def __init__(self, user_id, username, password_hash, role='user'):
        self.id = user_id
        self.username = username
        self.password_hash = password_hash
        self.role = role

# Simple user store (in production, use a database)
users = {
    'admin': User('admin', 'admin', generate_password_hash('admin123'), 'admin'),
    'trader': User('trader', 'trader', generate_password_hash('trader123'), 'trader')
}

@login_manager.user_loader
def load_user(user_id):
    return users.get(user_id)

class EnhancedWebDashboard:
    """
    Enhanced web dashboard with real-time capabilities.
    """
    
    def __init__(self):
        self.data_manager = EnhancedDataReliabilityManager()
        self.trading_engine = LiveTradingEngine()
        self.sentiment_analyzer = EnhancedSentimentIntegration()
        self.signal_generator = SignalGenerator()
        self.error_handler = ErrorHandlingSystem()
        
        # Real-time data storage
        self.live_data = {}
        self.active_signals = {}
        self.connected_clients = set()
        
        # Background tasks
        self.background_tasks = []
        self.is_running = False
        
        # Setup routes
        self._setup_routes()
        self._setup_socketio_events()
        
    def _setup_routes(self):
        """Setup Flask routes."""
        
        @app.route('/login', methods=['GET', 'POST'])
        def login():
            if request.method == 'POST':
                username = request.form['username']
                password = request.form['password']
                
                user = users.get(username)
                if user and check_password_hash(user.password_hash, password):
                    login_user(user)
                    return redirect(url_for('dashboard'))
                else:
                    flash('Invalid username or password')
            
            return render_template('login.html')
        
        @app.route('/logout')
        @login_required
        def logout():
            logout_user()
            return redirect(url_for('login'))
        
        @app.route('/')
        @login_required
        def dashboard():
            return render_template('dashboard.html', user=current_user)
        
        @app.route('/api/signals')
        @login_required
        def get_signals():
            """Get current trading signals."""
            try:
                signals = []
                for symbol, signal_data in self.active_signals.items():
                    signals.append({
                        'symbol': symbol,
                        'direction': signal_data.get('direction'),
                        'confidence': signal_data.get('confidence'),
                        'timestamp': signal_data.get('timestamp'),
                        'current_price': signal_data.get('current_price'),
                        'sentiment': signal_data.get('sentiment', {}).get('overall_sentiment', 'NEUTRAL')
                    })
                
                return jsonify({'signals': signals, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting signals: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/account')
        @login_required
        def get_account():
            """Get account information."""
            try:
                account_data = self.trading_engine.get_account_summary()
                return jsonify({'account': account_data, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting account data: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/chart/<symbol>')
        @login_required
        def get_chart_data(symbol):
            """Get chart data for a symbol."""
            try:
                timeframe = request.args.get('timeframe', '5m')
                
                # Get market data
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                data = loop.run_until_complete(
                    self.data_manager.get_reliable_data(symbol, timeframe)
                )
                loop.close()
                
                if data is None or data.empty:
                    return jsonify({'error': 'No data available', 'status': 'error'}), 404
                
                # Create candlestick chart
                fig = go.Figure(data=go.Candlestick(
                    x=data.index,
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name=symbol
                ))
                
                fig.update_layout(
                    title=f'{symbol} - {timeframe}',
                    xaxis_title='Time',
                    yaxis_title='Price',
                    template='plotly_dark'
                )
                
                chart_json = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
                return jsonify({'chart': chart_json, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting chart data: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/sentiment/<symbol>')
        @login_required
        def get_sentiment(symbol):
            """Get sentiment analysis for a symbol."""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                sentiment_data = loop.run_until_complete(
                    self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
                )
                loop.close()
                
                return jsonify({'sentiment': sentiment_data, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting sentiment: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/system/health')
        @login_required
        def get_system_health():
            """Get system health status."""
            try:
                health_data = self.error_handler.get_system_health()
                return jsonify({'health': health_data, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/execute_signal', methods=['POST'])
        @login_required
        def execute_signal():
            """Execute a trading signal."""
            try:
                if current_user.role not in ['admin', 'trader']:
                    return jsonify({'error': 'Insufficient permissions', 'status': 'error'}), 403
                
                signal_data = request.json
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                order_id = loop.run_until_complete(
                    self.trading_engine.execute_signal(signal_data)
                )
                loop.close()
                
                if order_id:
                    return jsonify({'order_id': order_id, 'status': 'success'})
                else:
                    return jsonify({'error': 'Failed to execute signal', 'status': 'error'}), 400
                
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/positions')
        @login_required
        def get_positions():
            """Get current positions."""
            try:
                account_data = self.trading_engine.get_account_summary()
                positions = account_data.get('positions', [])
                return jsonify({'positions': positions, 'status': 'success'})
                
            except Exception as e:
                logger.error(f"Error getting positions: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
        
        @app.route('/api/close_position', methods=['POST'])
        @login_required
        def close_position():
            """Close a position."""
            try:
                if current_user.role not in ['admin', 'trader']:
                    return jsonify({'error': 'Insufficient permissions', 'status': 'error'}), 403
                
                data = request.json
                symbol = data.get('symbol')
                
                if not symbol:
                    return jsonify({'error': 'Symbol required', 'status': 'error'}), 400
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(
                    self.trading_engine._close_position(symbol, "Manual close")
                )
                loop.close()
                
                return jsonify({'status': 'success', 'message': f'Position {symbol} closed'})
                
            except Exception as e:
                logger.error(f"Error closing position: {e}")
                return jsonify({'error': str(e), 'status': 'error'}), 500
    
    def _setup_socketio_events(self):
        """Setup SocketIO events for real-time communication."""
        
        @socketio.on('connect')
        def handle_connect():
            if current_user.is_authenticated:
                self.connected_clients.add(request.sid)
                join_room('dashboard')
                emit('status', {'message': 'Connected to TradePulse Dashboard'})
                logger.info(f"Client {request.sid} connected")
            else:
                return False
        
        @socketio.on('disconnect')
        def handle_disconnect():
            self.connected_clients.discard(request.sid)
            leave_room('dashboard')
            logger.info(f"Client {request.sid} disconnected")
        
        @socketio.on('subscribe_symbol')
        def handle_subscribe(data):
            symbol = data.get('symbol')
            if symbol:
                join_room(f'symbol_{symbol}')
                emit('status', {'message': f'Subscribed to {symbol}'})
        
        @socketio.on('unsubscribe_symbol')
        def handle_unsubscribe(data):
            symbol = data.get('symbol')
            if symbol:
                leave_room(f'symbol_{symbol}')
                emit('status', {'message': f'Unsubscribed from {symbol}'})
    
    async def start_background_tasks(self):
        """Start background tasks for real-time updates."""
        self.is_running = True
        
        # Start trading engine monitoring
        await self.trading_engine.start_monitoring()
        
        # Start real-time data updates
        task1 = asyncio.create_task(self._update_live_data())
        task2 = asyncio.create_task(self._update_signals())
        task3 = asyncio.create_task(self._monitor_system_health())
        
        self.background_tasks = [task1, task2, task3]
        
        # Wait for all tasks
        await asyncio.gather(*self.background_tasks)
    
    async def stop_background_tasks(self):
        """Stop background tasks."""
        self.is_running = False
        
        # Stop trading engine monitoring
        await self.trading_engine.stop_monitoring()
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
    
    async def _update_live_data(self):
        """Update live market data."""
        symbols = ['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHUSD']
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # Get latest data
                    data = await self.data_manager.get_reliable_data(symbol, '1m')
                    
                    if data is not None and not data.empty:
                        latest = data.iloc[-1]
                        
                        price_data = {
                            'symbol': symbol,
                            'price': float(latest['close']),
                            'change': float(latest['close'] - data.iloc[-2]['close']) if len(data) > 1 else 0,
                            'timestamp': datetime.now().isoformat()
                        }
                        
                        self.live_data[symbol] = price_data
                        
                        # Emit to connected clients
                        socketio.emit('price_update', price_data, room=f'symbol_{symbol}')
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                await self.error_handler.handle_error(e, 'data_source')
                await asyncio.sleep(10)
    
    async def _update_signals(self):
        """Update trading signals."""
        symbols = ['EURUSD', 'GBPUSD', 'BTCUSD', 'ETHUSD']
        
        while self.is_running:
            try:
                for symbol in symbols:
                    # Generate signal
                    signal = self.signal_generator.generate_signal(symbol)
                    
                    if signal and signal.get('confidence', 0) > 0.6:
                        # Get sentiment analysis
                        sentiment = await self.sentiment_analyzer.get_comprehensive_sentiment(symbol)
                        
                        # Integrate sentiment with signal
                        enhanced_signal = self.sentiment_analyzer.integrate_with_signal(signal, sentiment)
                        
                        self.active_signals[symbol] = enhanced_signal
                        
                        # Emit to connected clients
                        socketio.emit('signal_update', {
                            'symbol': symbol,
                            'signal': enhanced_signal
                        }, room='dashboard')
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                await self.error_handler.handle_error(e, 'signal_generation')
                await asyncio.sleep(60)
    
    async def _monitor_system_health(self):
        """Monitor system health and emit updates."""
        while self.is_running:
            try:
                health_data = self.error_handler.get_system_health()
                
                # Emit health update to connected clients
                socketio.emit('health_update', health_data, room='dashboard')
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                logger.error(f"Error monitoring system health: {e}")
                await asyncio.sleep(120)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the enhanced web dashboard."""
        # Start background tasks in a separate thread
        def run_background():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self.start_background_tasks())
        
        background_thread = threading.Thread(target=run_background, daemon=True)
        background_thread.start()
        
        # Run Flask-SocketIO app
        socketio.run(app, host=host, port=port, debug=debug)

# Global dashboard instance
dashboard = EnhancedWebDashboard()

if __name__ == '__main__':
    dashboard.run(debug=True)
