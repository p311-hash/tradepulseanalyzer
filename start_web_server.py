"""
Start the web application server for TradePulse

This script starts the Flask web application defined in web_app.py
"""

import os
from web_app import app

if __name__ == "__main__":
    # Set environment variable to indicate web server mode
    os.environ['WEB_SERVER_MODE'] = '1'
    app.run(host='0.0.0.0', port=5001, debug=True)