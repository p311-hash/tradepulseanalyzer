"""
Script to start the web server on port 5001 (avoids conflict with the bot)
"""

import os
import subprocess
import sys

def main():
    """Start the web server on port 5001"""
    print("Starting TradePulse Web Server on port 5001...")
    
    # Set the environment variable for web server mode
    os.environ["WEB_SERVER_MODE"] = "1"
    
    try:
        # Use gunicorn to start the web server on port 5001
        subprocess.run(
            ["gunicorn", "--bind", "0.0.0.0:5001", "--reuse-port", "--reload", "main:app"],
            check=True,
        )
    except KeyboardInterrupt:
        print("\nWeb server stopped.")
    except Exception as e:
        print(f"Error starting web server: {e}", file=sys.stderr)
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())