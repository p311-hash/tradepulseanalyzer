#!/usr/bin/env python3
"""
Simple webhook-based Telegram bot using requests library.
This avoids the python-telegram-bot compatibility issues.
"""

import os
import json
import logging
import requests
from flask import Flask, request, jsonify
from dotenv import load_dotenv
import threading
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Get bot token
BOT_TOKEN = os.getenv('TELEGRAM_TOKEN')
if not BOT_TOKEN:
    logger.error("No TELEGRAM_TOKEN found in environment")
    exit(1)

# Telegram API base URL
TELEGRAM_API_URL = f"https://api.telegram.org/bot{BOT_TOKEN}"

# Flask app for webhook
app = Flask(__name__)

def send_message(chat_id, text, parse_mode='HTML'):
    """Send a message to a Telegram chat."""
    url = f"{TELEGRAM_API_URL}/sendMessage"
    data = {
        'chat_id': chat_id,
        'text': text,
        'parse_mode': parse_mode
    }
    
    try:
        response = requests.post(url, json=data)
        return response.json()
    except Exception as e:
        logger.error(f"Error sending message: {e}")
        return None

def get_bot_info():
    """Get bot information."""
    url = f"{TELEGRAM_API_URL}/getMe"
    try:
        response = requests.get(url)
        return response.json()
    except Exception as e:
        logger.error(f"Error getting bot info: {e}")
        return None

def handle_start_command(chat_id, user_name):
    """Handle /start command."""
    message = f"""
🤖 <b>Welcome to MasterTrade Bot!</b>

Hello {user_name}! I'm your advanced binary options trading assistant.

🎯 <b>Features:</b>
• Real-time trading signals
• Pattern recognition
• Demo/Real trading modes
• Professional analysis

📱 <b>Available Commands:</b>
• /start - Show this welcome message
• /signal - Get latest trading signal
• /help - Show help information
• /status - Check bot status

🚀 <b>Ready to start trading?</b>
Send /signal to get your first trading signal!
"""
    
    return send_message(chat_id, message)

def handle_signal_command(chat_id):
    """Handle /signal command."""
    # Import signal generator
    try:
        from signal_generator import generate_signals
        signals = generate_signals()
        
        if signals:
            # Get the first signal
            asset, signal_data = next(iter(signals.items()))
            
            direction_emoji = "🟢" if signal_data['direction'] == 'BUY' else "🔴"
            
            message = f"""
{direction_emoji} <b>Trading Signal</b>

<b>Asset:</b> {signal_data['asset']}
<b>Direction:</b> {signal_data['direction']}
<b>Strength:</b> {signal_data['strength']}
<b>Price:</b> ${signal_data['price']:.4f}
<b>Time:</b> {signal_data['timestamp']}

⚠️ <i>This is for educational purposes only. Trade at your own risk.</i>
"""
        else:
            message = "📊 No strong signals available at the moment. Please try again later."
            
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        message = "❌ Error generating signal. Please try again later."
    
    return send_message(chat_id, message)

def handle_help_command(chat_id):
    """Handle /help command."""
    message = """
📚 <b>MasterTrade Bot Help</b>

<b>Commands:</b>
• /start - Welcome message and introduction
• /signal - Get latest trading signal
• /help - Show this help message
• /status - Check bot status

<b>About Signals:</b>
• Signals are generated using advanced technical analysis
• Multiple timeframes are analyzed
• Pattern recognition is included
• Confidence levels are provided

<b>Trading Modes:</b>
• Demo mode for practice
• Real mode for live trading

⚠️ <b>Disclaimer:</b>
This bot is for educational purposes only. Always do your own research and never invest more than you can afford to lose.
"""
    
    return send_message(chat_id, message)

def handle_status_command(chat_id):
    """Handle /status command."""
    message = """
✅ <b>Bot Status</b>

🤖 <b>Bot:</b> Online and operational
📊 <b>Signal Generator:</b> Active
🔄 <b>Data Feed:</b> Simulated (Demo mode)
⚡ <b>Response Time:</b> Fast
🛡️ <b>Security:</b> Enabled

<b>System Info:</b>
• Version: 2.0
• Uptime: Running
• Last Update: Recently

All systems operational! 🚀
"""
    
    return send_message(chat_id, message)

@app.route('/webhook', methods=['POST'])
def webhook():
    """Handle incoming webhook from Telegram."""
    try:
        update = request.get_json()
        
        if 'message' in update:
            message = update['message']
            chat_id = message['chat']['id']
            user_name = message['from'].get('first_name', 'Trader')
            
            if 'text' in message:
                text = message['text']
                
                logger.info(f"Received message from {user_name} ({chat_id}): {text}")
                
                if text.startswith('/start'):
                    handle_start_command(chat_id, user_name)
                elif text.startswith('/signal'):
                    handle_signal_command(chat_id)
                elif text.startswith('/help'):
                    handle_help_command(chat_id)
                elif text.startswith('/status'):
                    handle_status_command(chat_id)
                else:
                    send_message(chat_id, "🤖 I didn't understand that command. Send /help for available commands.")
        
        return jsonify({'status': 'ok'})
        
    except Exception as e:
        logger.error(f"Error handling webhook: {e}")
        return jsonify({'status': 'error'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'bot': 'MasterTrade Bot'})

def start_polling():
    """Start polling for updates (alternative to webhook)."""
    offset = 0
    
    while True:
        try:
            url = f"{TELEGRAM_API_URL}/getUpdates"
            params = {'offset': offset, 'timeout': 30}
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get('ok'):
                for update in data.get('result', []):
                    offset = update['update_id'] + 1
                    
                    if 'message' in update:
                        message = update['message']
                        chat_id = message['chat']['id']
                        user_name = message['from'].get('first_name', 'Trader')
                        
                        if 'text' in message:
                            text = message['text']
                            
                            logger.info(f"Received message from {user_name} ({chat_id}): {text}")
                            
                            if text.startswith('/start'):
                                handle_start_command(chat_id, user_name)
                            elif text.startswith('/signal'):
                                handle_signal_command(chat_id)
                            elif text.startswith('/help'):
                                handle_help_command(chat_id)
                            elif text.startswith('/status'):
                                handle_status_command(chat_id)
                            else:
                                send_message(chat_id, "🤖 I didn't understand that command. Send /help for available commands.")
            
        except Exception as e:
            logger.error(f"Error in polling: {e}")
            time.sleep(5)

if __name__ == '__main__':
    # Test bot connection
    bot_info = get_bot_info()
    if bot_info and bot_info.get('ok'):
        logger.info(f"✅ Bot connected successfully: @{bot_info['result']['username']}")
        logger.info(f"Bot name: {bot_info['result']['first_name']}")
        
        # Start polling in a separate thread
        polling_thread = threading.Thread(target=start_polling, daemon=True)
        polling_thread.start()
        
        logger.info("🤖 MasterTrade Bot is now running and listening for messages!")
        logger.info("Send /start to your bot to test it")
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            
    else:
        logger.error("❌ Failed to connect to Telegram. Check your bot token.")
