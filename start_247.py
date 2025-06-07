#!/usr/bin/env python
"""
Simple starter script for 24/7 operation of TradePulse Bot.
This script sets the appropriate environment variables and starts the bot
with the forever process manager.
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('start_247')

def main():
    """
    Start the bot in 24/7 mode with the forever process manager.
    """
    logger.info("Starting TradePulse Bot in 24/7 mode")
    
    # Set environment variables for bot mode
    os.environ['BOT_MODE'] = '1'
    
    # Record start time
    start_time = datetime.now().isoformat()
    with open('.bot_start_time', 'w') as f:
        f.write(start_time)
    
    # Get Python executable path
    python_path = sys.executable
    
    try:
        # Start the forever process manager
        logger.info("Starting forever process manager...")
        result = subprocess.run(
            [python_path, 'forever.py'],
            check=True
        )
        return result.returncode
    except subprocess.CalledProcessError as e:
        logger.error(f"Error starting forever process: {str(e)}")
        return e.returncode
    except KeyboardInterrupt:
        logger.info("Process interrupted by user")
        return 0
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main())