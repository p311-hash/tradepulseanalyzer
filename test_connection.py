"""Test script for PocketOption API connection and basic functionality."""
import os
import logging
import pandas as pd
from dotenv import load_dotenv
import sys
import time

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from BinaryOptionsTools.platforms.pocketoption.api import PocketOptionAPI

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def test_functionality():
    """Test PocketOption API functionality."""
    # Load environment variables
    load_dotenv()
    
    # Get API credentials from environment
    ssid = os.environ.get('POCKETOPTION_SSID')
    use_demo = os.environ.get('POCKETOPTION_DEMO', 'true').lower() == 'true'
    
    if not ssid:
        logger.warning("POCKETOPTION_SSID not found in environment variables. Using simulated data mode.")
        ssid = None
    
    # Initialize API
    logger.info(f"Initializing PocketOption API (demo mode: {use_demo})")
    api = PocketOptionAPI(ssid=ssid, demo=use_demo)
    
    # Test connection
    logger.info("Testing API connection...")
    connected, error = api.connect()
    if not connected:
        logger.error(f"Failed to connect to PocketOption API: {error}")
        logger.info("Connection test failed - but this is expected when using simulated data")
    else:
        logger.info("Successfully connected to PocketOption API")
    
    # Test getting balance
    try:
        logger.info("Testing balance retrieval...")
        balance = api.get_balance()
        logger.info(f"Current balance: {balance}")
    except Exception as e:
        logger.error(f"Error getting balance: {str(e)}")
    
    # Test getting candles
    try:
        logger.info("Testing candle data retrieval...")
        for pair in ['EURUSD_otc', 'GBPUSD_otc']:
            logger.info(f"Fetching {pair} candles (1-minute timeframe)")
            candles = api.get_candles(pair, 60, count=20)
            
            if candles is not None and not candles.empty:
                logger.info(f"Successfully fetched {len(candles)} candles for {pair}")
                logger.info(f"Candle data sample:\n{candles.head(3)}")
            else:
                logger.warning(f"No candle data received for {pair}")
                
            # Small delay between requests
            time.sleep(1)
    except Exception as e:
        logger.error(f"Error fetching candles: {str(e)}")
    
    logger.info("PocketOption API test completed")
    logger.info("Note: If you want to use real data, set USE_REAL_DATA=true in your .env file")
    logger.info("      and provide your PocketOption SSID in POCKETOPTION_SSID")

if __name__ == "__main__":
    logger.info("="*50)
    logger.info("Starting PocketOption API test")
    logger.info("="*50)
    test_functionality()