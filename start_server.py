"""
Flask application entry point for TradePulse Signals Trading Bot.
This script starts the web server on port 8080 for compatibility
with Replit's environment.
"""

import os
import sys
import logging
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('server')

# Set environment variable to prevent Telegram bot from starting
os.environ['GUNICORN_WORKER'] = '1'
logger.info("Web server mode enabled - Telegram bot will not be started")

# Import main app
sys.path.append('.')
from main import app

if __name__ == "__main__":
    # Run the Flask app
    port = 8080  # Replit standard port
    logger.info(f"Starting server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=True)