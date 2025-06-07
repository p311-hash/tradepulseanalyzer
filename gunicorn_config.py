import os
import sys

# Set environment variable to indicate we're running under gunicorn
os.environ['GUNICORN_WORKER'] = '1'

# Print to stderr (for debugging)
print("GUNICORN_CONFIG LOADED - Worker mode enabled", file=sys.stderr)

# Gunicorn server configuration
bind = "0.0.0.0:5001"  # Using port 5001 to avoid conflict with the bot
workers = 1
threads = 2
timeout = 120
keepalive = 5
accesslog = "-"  # Log to stdout
errorlog = "-"   # Log errors to stdout
loglevel = "info"

# This config file will be loaded by gunicorn
# Any pre-loading or configuration can be done here before 
# the app is loaded