#!/usr/bin/env python3
"""
Simple script to verify the Flask web server is running correctly.
"""
import requests
import sys
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def check_endpoint(url, name="endpoint"):
    """Check if an endpoint responds correctly."""
    try:
        logger.info(f"Checking {name} at {url}...")
        # Use subprocess to call curl for better reliability
        import subprocess
        result = subprocess.run(
            ['curl', '-s', '-o', '/dev/null', '-w', '%{http_code}', url, '--max-time', '5'],
            capture_output=True,
            text=True
        )
        status_code = int(result.stdout.strip())
        
        if status_code == 200:
            logger.info(f"✓ {name} is available (Status code: {status_code})")
            return True
        else:
            logger.error(f"✗ {name} returned unexpected status code: {status_code}")
            return False
    except Exception as e:
        logger.error(f"✗ Could not connect to {name}: {str(e)}")
        return False

def main():
    """Check server endpoints."""
    base_url = "http://0.0.0.0:5000"
    
    # Check if the server is responding
    if not check_endpoint(base_url, "server"):
        logger.error("Server is not responding. Please check that it's running.")
        return False
    
    # Check individual endpoints
    endpoints = {
        "/": "home page",
        "/signals": "signals page",
        "/api/signals": "signals API",
        "/health": "health check",
        "/backtest": "backtest page",
        "/compare_strategies": "strategy comparison page"
    }
    
    results = []
    for path, name in endpoints.items():
        results.append(check_endpoint(f"{base_url}{path}", name))
    
    success_rate = sum(results) / len(results) * 100
    logger.info(f"\nEndpoint check complete: {success_rate:.1f}% success rate ({sum(results)}/{len(results)} endpoints available)")
    
    return all(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)