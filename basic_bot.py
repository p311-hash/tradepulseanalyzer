#!/usr/bin/env python3
"""
Basic Telegram bot that definitely shows output
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv

print("=" * 60)
print("🚀 STARTING TELEGRAM BOT")
print("=" * 60)
print(f"Time: {datetime.now()}")

# Load environment
load_dotenv()
print("✅ Environment loaded")

# Check token
token = os.getenv('TELEGRAM_TOKEN')
if not token:
    print("❌ NO TOKEN FOUND!")
    exit(1)

print(f"✅ Token found: {token[:10]}...")

# Import telegram
try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode
    print("✅ Telegram imports successful")
except Exception as e:
    print(f"❌ Import error: {e}")
    exit(1)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)
print("✅ Logging setup complete")

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    user_id = update.effective_user.id
    username = update.effective_user.username or "Unknown"
    
    print(f"📨 /start from user {user_id} (@{username})")
    
    message = """
🤖 *TradePulse Bot is WORKING!*

✅ Connection successful
✅ Commands working
✅ Ready to trade

*Commands:*
/start - This message
/test - Test signal
/ping - Check status

Bot is operational! 🚀
"""
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    print("✅ Start message sent")

async def test_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /test command"""
    user_id = update.effective_user.id
    print(f"📨 /test from user {user_id}")
    
    import random
    directions = ["BUY 🟢", "SELL 🔴"]
    direction = random.choice(directions)
    confidence = random.randint(70, 95)
    
    message = f"""
🎯 *TEST SIGNAL*

*Pair:* EURUSD
*Direction:* {direction}
*Confidence:* {confidence}%
*Time:* {datetime.now().strftime("%H:%M:%S")}

✅ *Signal generated successfully!*
"""
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    print(f"✅ Test signal sent: {direction}")

async def ping_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ping command"""
    user_id = update.effective_user.id
    print(f"📨 /ping from user {user_id}")
    
    message = f"""
🏓 *PONG!*

Bot Status: ✅ ONLINE
Time: {datetime.now().strftime("%H:%M:%S")}
User ID: {user_id}

Everything is working! 🚀
"""
    
    await update.message.reply_text(message, parse_mode=ParseMode.MARKDOWN)
    print("✅ Pong sent")

def main():
    """Main function"""
    print("🔧 Creating application...")
    
    try:
        # Create app
        app = Application.builder().token(token).build()
        print("✅ Application created")
        
        # Add handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("test", test_command))
        app.add_handler(CommandHandler("ping", ping_command))
        print("✅ Handlers added")
        
        print("=" * 60)
        print("🚀 BOT IS STARTING...")
        print("📱 Bot Username: @MS0531_bot")
        print("💬 Send /start to test the bot")
        print("=" * 60)
        
        # Start polling
        app.run_polling(drop_pending_updates=True)
        
    except KeyboardInterrupt:
        print("\n🛑 Bot stopped by user")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
