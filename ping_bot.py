"""
Simple script to ping the keep-alive server.
This can be run locally or hosted on a service like PythonAnywhere.
"""

import requests
import time
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ping_bot')

# Your Replit URL - Replace with your actual URL
REPLIT_URL = "https://YOUR-REPL-NAME.YOUR-USERNAME.repl.co"

def ping_server():
    """Ping the keep-alive server to prevent it from sleeping."""
    try:
        response = requests.get(f"{REPLIT_URL}/health", timeout=10)
        if response.status_code == 200:
            logger.info(f"Successfully pinged bot at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - Response: {response.text}")
            return True
        else:
            logger.error(f"Ping failed with status code: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Error pinging server: {str(e)}")
        return False

def main():
    """Main function to ping the server at regular intervals."""
    logger.info(f"Starting ping service for {REPLIT_URL}")
    
    # Ping every 5 minutes
    interval = 5 * 60  # 5 minutes in seconds
    
    while True:
        ping_server()
        time.sleep(interval)

if __name__ == "__main__":
    main()