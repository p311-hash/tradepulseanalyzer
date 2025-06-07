"""
Simplified Flask application for testing the web interface
"""

from flask import Flask

app = Flask(__name__)

@app.route('/')
def home():
    """Simple home page"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>TradePulse Bot - Test Page</title>
        <meta name="viewport" content="width=device-width, initial-scale=1">
    </head>
    <body>
        <h1>TradePulse Bot is Running</h1>
        <p>This is a test page to verify the web server is accessible.</p>
    </body>
    </html>
    """

@app.route('/health')
def health():
    """Health check endpoint"""
    return {"status": "healthy"}

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)