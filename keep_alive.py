"""
Enhanced keep-alive mechanism to ensure 24/7 operation for TradePulse Bot.
This creates a robust HTTP server that can be pinged regularly and monitors the bot's health.
"""

import threading
import logging
import time
import os
import json
from datetime import datetime
from flask import Flask, jsonify, send_from_directory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('keep_alive')

# Initialize Flask app
app = Flask(__name__)

# Global variables to track bot status
BOT_START_TIME = datetime.now().isoformat()
HEARTBEAT_TIME = time.time()
BOT_STATUS = "running"

@app.route('/')
def home():
    """Return a simple message to confirm the server is running."""
    return f"""
    <html>
    <head>
        <title>TradePulse Bot - 24/7 Operation</title>
        <link href="https://cdn.replit.com/agent/bootstrap-agent-dark-theme.min.css" rel="stylesheet">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            body {{ padding: 20px; background-color: #212529; color: #f8f9fa; }}
            .status-card {{ border-left: 4px solid #198754; padding: 15px; margin-bottom: 15px; }}
            .uptime {{ font-weight: bold; color: #20c997; }}
        </style>
        <meta http-equiv="refresh" content="60">
    </head>
    <body>
        <div class="container">
            <h1 class="mt-4 mb-4">📈 TradePulse Bot - 24/7 Operation</h1>
            
            <div class="card bg-dark border-secondary mb-4">
                <div class="card-header bg-dark border-secondary">
                    <h2 class="h4 m-0">Bot Status Monitor</h2>
                </div>
                <div class="card-body">
                    <div class="status-card">
                        <h3 class="h5">Status: <span class="text-success">ONLINE</span></h3>
                        <p>Started: {BOT_START_TIME}</p>
                        <p>Last heartbeat: {datetime.fromtimestamp(HEARTBEAT_TIME).isoformat()}</p>
                        <p>Bot mode: {os.environ.get('BOT_MODE', '0')}</p>
                    </div>
                    <p>This monitoring page automatically refreshes every 60 seconds.</p>
                    <div class="mt-3">
                        <a href="/health" class="btn btn-info me-2">Health Check API</a>
                        <a href="/bot-info" class="btn btn-primary">Bot Status API</a>
                    </div>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/health')
def health():
    """Enhanced health check endpoint for monitoring services."""
    # Update heartbeat
    global HEARTBEAT_TIME
    HEARTBEAT_TIME = time.time()
    
    # Check if bot lock file exists and is recent
    bot_running = False
    try:
        if os.path.exists('.bot_lock'):
            with open('.bot_lock', 'r') as f:
                lock_time = float(f.read().strip())
                # If lock is less than 5 minutes old, bot is running
                if time.time() - lock_time < 300:
                    bot_running = True
    except Exception as e:
        logger.error(f"Error checking bot lock: {str(e)}")
    
    status = "healthy" if bot_running else "bot-inactive"
    
    return jsonify({
        "service": "TradePulse Bot",
        "status": status,
        "uptime": int(time.time() - time.mktime(datetime.fromisoformat(BOT_START_TIME).timetuple())),
        "timestamp": datetime.now().isoformat()
    })

@app.route('/bot-info')
def bot_info():
    """Provides detailed bot information in JSON format."""
    return jsonify({
        "name": "TradePulse Trading Signals Bot",
        "version": "1.0.0",
        "status": BOT_STATUS,
        "start_time": BOT_START_TIME,
        "heartbeat_time": datetime.fromtimestamp(HEARTBEAT_TIME).isoformat(),
        "uptime_seconds": int(time.time() - time.mktime(datetime.fromisoformat(BOT_START_TIME).timetuple())),
        "environment": {
            "bot_mode": os.environ.get("BOT_MODE", "0"),
            "web_server_mode": os.environ.get("WEB_SERVER_MODE", "0")
        }
    })

@app.route('/images/<path:filename>')
def serve_image(filename):
    """Serve candlestick pattern images."""
    return send_from_directory('assets/images', filename)

@app.route('/ping')
def ping():
    """Simple ping endpoint for uptime monitoring services."""
    global HEARTBEAT_TIME
    HEARTBEAT_TIME = time.time()
    return "pong", 200

def update_heartbeat():
    """Periodically update the heartbeat file to indicate the service is running."""
    while True:
        try:
            global HEARTBEAT_TIME
            HEARTBEAT_TIME = time.time()
            
            # Update bot lock file if it exists
            if os.path.exists('.bot_lock'):
                # Only update if the process is running as the bot
                if os.environ.get('BOT_MODE', '0') == '1':
                    with open('.bot_lock', 'w') as f:
                        f.write(str(time.time()))
            
            time.sleep(60)  # Update every minute
        except Exception as e:
            logger.error(f"Error in heartbeat update: {str(e)}")
            time.sleep(60)  # Wait and try again

def run_server():
    """Run the Flask server in a separate thread."""
    try:
        # Use 0.0.0.0 to make the server publicly accessible
        app.run(host='0.0.0.0', port=5000)
    except Exception as e:
        logger.error(f"Error starting keep-alive server: {str(e)}")
        global BOT_STATUS
        BOT_STATUS = "error"

def start_server():
    """Start the keep-alive server in a background thread."""
    server_thread = threading.Thread(target=run_server)
    server_thread.daemon = True  # Thread will exit when main program exits
    server_thread.start()
    logger.info("Keep-alive server started at http://0.0.0.0:5000")
    
    # Start heartbeat thread
    heartbeat_thread = threading.Thread(target=update_heartbeat)
    heartbeat_thread.daemon = True
    heartbeat_thread.start()
    logger.info("Heartbeat monitor started")
    
    return server_thread