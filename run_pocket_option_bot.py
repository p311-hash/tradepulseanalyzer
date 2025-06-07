"""
Main script for running the PocketOption trading bot
"""

import asyncio
import logging
import os
from datetime import datetime
import json
from pocket_option_trader import PocketOptionTrader

# Ensure required directories exist
REQUIRED_DIRS = ['logs', 'data']
for dir_name in REQUIRED_DIRS:
    dir_path = os.path.join(os.path.dirname(__file__), dir_name)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)  # Create the directory if it does not exist
        logging.info(f"Created directory: {dir_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'pocket_option_trades.log')),
        logging.StreamHandler()
    ]
)

logging.info(f"Created directory: {dir_path}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join('logs', 'pocket_option_trades.log')),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

# Trading symbols (Pocket Option OTC assets for 24/7 trading)
TRADING_SYMBOLS = [
    'EURUSD_otc',
    'GBPUSD_otc',
    'USDJPY_otc',
    'EURJPY_otc',
    'AUDUSD_otc',
    'GBPJPY_otc'
]

class PocketOptionBot:
    def __init__(self, ssid: str, demo: bool = True):
        """Initialize the trading bot."""
        self.trader = PocketOptionTrader(ssid, demo)
        self.active_trades = {}
        self.trade_history = []
        self.running = False
        
    async def monitor_trade(self, trade_id: str):
        """Monitor an individual trade until completion."""
        while True:
            if trade_id not in self.active_trades:
                return
                
            result = self.trader.check_trade_result(trade_id)
            if result['status'] in ['win', 'loss', 'draw']:
                self.trade_history.append(result)
                self.active_trades.pop(trade_id, None)
                self.save_trade_history()
                logger.info(f"Trade completed: {result}")
                return
                
            await asyncio.sleep(1)
    
    def save_trade_history(self):
        """Save trade history to file."""
        try:
            with open('pocket_option_history.json', 'w') as f:
                json.dump(self.trade_history, f)
        except Exception as e:
            logger.error(f"Error saving trade history: {str(e)}")
    
    async def analyze_symbol(self, symbol: str):
        """Analyze a single trading symbol."""
        try:
            opportunity = await self.trader.analyze_trading_opportunity(symbol)
            if opportunity:
                # Validate trade conditions
                success, trade_id = self.trader.place_trade(
                    symbol=opportunity['symbol'],
                    direction=opportunity['direction'],
                    expiry=opportunity['expiry']
                )
                
                if success and trade_id:
                    self.active_trades[trade_id] = opportunity
                    asyncio.create_task(self.monitor_trade(trade_id))
                    
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {str(e)}")
    
    async def run(self):
        """Main bot loop."""
        self.running = True
        logger.info("Starting PocketOption trading bot...")
        
        while self.running:
            try:
                # Analyze all symbols concurrently
                analysis_tasks = [
                    self.analyze_symbol(symbol) 
                    for symbol in TRADING_SYMBOLS
                ]
                await asyncio.gather(*analysis_tasks)
                
                # Brief pause between analysis cycles
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in main loop: {str(e)}")
                await asyncio.sleep(5)
    
    def stop(self):
        """Stop the trading bot."""
        self.running = False
        self.trader.close()
        self.save_trade_history()

async def main():
    # Get session ID from environment or input
    ssid = os.getenv('POCKETOPTION_SSID')
    if not ssid:
        ssid = input("Enter your PocketOption session ID: ")
    
    # Initialize and run bot
    bot = PocketOptionBot(ssid=ssid, demo=True)
    
    try:
        await bot.run()
    except KeyboardInterrupt:
        logger.info("Stopping bot...")
        bot.stop()
    except Exception as e:
        logger.error(f"Bot error: {str(e)}")
        bot.stop()

if __name__ == "__main__":
    asyncio.run(main())
