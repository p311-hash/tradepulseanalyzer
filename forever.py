#!/usr/bin/env python
"""
Forever script for TradePulse Bot - Ensures 24/7 operation

This script monitors and automatically restarts the TradePulse bot if it crashes.
It implements:
1. Process monitoring
2. Automatic restart
3. Crash log recording
4. Status notifications
"""

import os
import sys
import time
import subprocess
import logging
import signal
import datetime
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forever.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('forever')

# Constants
MAX_RESTART_ATTEMPTS = 5  # Maximum number of restart attempts within RESTART_WINDOW
RESTART_WINDOW = 3600  # 1 hour in seconds
RESTART_DELAY = 5  # Seconds to wait before restart
HEALTH_CHECK_INTERVAL = 60  # Seconds between health checks

class ForeverBot:
    """Bot process manager that ensures 24/7 operation."""
    
    def __init__(self, command='python main.py', env_vars=None):
        """
        Initialize the bot process manager.
        
        Args:
            command: Command to start the bot
            env_vars: Additional environment variables to set
        """
        self.command = command
        self.process = None
        self.restart_times = []
        self.running = True
        
        # Setup environment variables
        self.env = os.environ.copy()
        self.env['BOT_MODE'] = '1'  # Explicitly set bot mode
        
        if env_vars:
            self.env.update(env_vars)
        
        # Handle termination signals
        signal.signal(signal.SIGTERM, self.handle_signal)
        signal.signal(signal.SIGINT, self.handle_signal)
    
    def handle_signal(self, sig, frame):
        """Handle termination signals."""
        logger.info(f"Received signal {sig}, shutting down...")
        self.running = False
        if self.process:
            logger.info("Terminating bot process...")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                logger.warning("Bot process did not terminate, force killing...")
                self.process.kill()
        
        logger.info("Forever process manager shutting down.")
        sys.exit(0)
    
    def start_bot(self):
        """Start the bot process."""
        logger.info(f"Starting bot with command: {self.command}")
        
        # Record restart time
        self.restart_times.append(time.time())
        # Remove restart times older than RESTART_WINDOW
        self.restart_times = [t for t in self.restart_times 
                              if time.time() - t < RESTART_WINDOW]
        
        # Start the process
        try:
            # Split command into args for subprocess
            command_args = self.command.split()
            self.process = subprocess.Popen(
                command_args,
                env=self.env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            logger.info(f"Bot started with PID: {self.process.pid}")
            
            # Create a heartbeat file
            with open('.bot_heartbeat', 'w') as f:
                f.write(f"{time.time()}:{self.process.pid}")
                
            return True
        except Exception as e:
            logger.error(f"Failed to start bot: {str(e)}")
            return False
    
    def check_bot_health(self):
        """Check if bot is still running and healthy."""
        # Check if process is still running
        if self.process.poll() is not None:
            exit_code = self.process.poll()
            logger.warning(f"Bot process exited with code {exit_code}")
            return False
        
        # Check for lock file freshness
        try:
            if os.path.exists('.bot_lock'):
                with open('.bot_lock', 'r') as f:
                    lock_time = float(f.read().strip())
                    if time.time() - lock_time > 300:  # 5 minutes
                        logger.warning("Bot lock file is stale")
                        return False
        except Exception as e:
            logger.error(f"Error checking bot lock: {str(e)}")
        
        # Bot appears to be running normally
        return True
    
    def run_forever(self):
        """Main loop to keep the bot running forever."""
        logger.info("Starting Forever process manager for TradePulse Bot")
        
        while self.running:
            # Check for too many restarts
            if len(self.restart_times) >= MAX_RESTART_ATTEMPTS:
                logger.error(
                    f"Too many restart attempts ({len(self.restart_times)}) "
                    f"within {RESTART_WINDOW//60} minutes. Cooling down..."
                )
                # Wait until oldest restart is outside window
                oldest = min(self.restart_times)
                cooldown = RESTART_WINDOW - (time.time() - oldest) + 10
                logger.info(f"Cooling down for {int(cooldown//60)} minutes")
                time.sleep(cooldown)
                # Clear restart history
                self.restart_times = []
            
            # Start bot if not running
            if not self.process or self.process.poll() is not None:
                # Delay before restart
                if self.process:  # Only delay if this is a restart
                    logger.info(f"Waiting {RESTART_DELAY} seconds before restart...")
                    time.sleep(RESTART_DELAY)
                
                # Start the bot
                success = self.start_bot()
                if not success:
                    logger.error("Failed to start bot, trying again in 30 seconds...")
                    time.sleep(30)
                    continue
            
            # Monitor bot health
            while self.running and self.check_bot_health():
                time.sleep(HEALTH_CHECK_INTERVAL)
                
                # Read and log any output
                self._process_output()
                
                # Update heartbeat
                with open('.bot_heartbeat', 'w') as f:
                    f.write(f"{time.time()}:{self.process.pid}")
            
            # If we're still running but the bot is not healthy, restart it
            if self.running:
                logger.warning("Bot appears to be unhealthy, restarting...")
                if self.process:
                    # Try to terminate gracefully first
                    try:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            logger.warning("Process did not terminate, force killing...")
                            self.process.kill()
                    except Exception as e:
                        logger.error(f"Error terminating process: {str(e)}")
                
                # Process any remaining output before restarting
                self._process_output()
    
    def _process_output(self):
        """Process and log output from the bot."""
        if not self.process:
            return
            
        # Non-blocking read from stdout
        try:
            while True:
                output_line = self.process.stdout.readline()
                if not output_line and self.process.poll() is not None:
                    break
                if output_line:
                    # Log bot output at debug level
                    logger.debug(f"Bot: {output_line.strip()}")
        except Exception as e:
            logger.error(f"Error reading process output: {str(e)}")

def main():
    """Run the forever process manager."""
    # Get command from arguments or use default
    command = 'python main.py'
    if len(sys.argv) > 1:
        command = ' '.join(sys.argv[1:])
    
    # Set up additional environment variables
    env_vars = {
        'BOT_MODE': '1',
        'PYTHONUNBUFFERED': '1'  # Ensure Python output is unbuffered
    }
    
    # Start the forever process
    forever = ForeverBot(command=command, env_vars=env_vars)
    forever.run_forever()

if __name__ == "__main__":
    main()