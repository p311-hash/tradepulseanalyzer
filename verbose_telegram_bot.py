#!/usr/bin/env python3
"""
Verbose Telegram bot for TradePulseAnalyzer with detailed logging
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

# Force output to be flushed immediately
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

print("🚀 Starting Telegram Bot...")
print(f"📅 Time: {datetime.now()}")

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode
    print("✅ Telegram imports successful")
except ImportError as e:
    print(f"❌ Telegram import failed: {e}")
    exit(1)

# Load environment variables
load_dotenv()
print("✅ Environment variables loaded")

# Configure logging with more verbose output
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)

print("✅ Logging configured")

def generate_test_signal(pair="EURUSD", timeframe="1m"):
    """Generate a test trading signal"""
    import random

    directions = ["BUY", "SELL", "NEUTRAL"]
    direction = random.choice(directions)
    confidence = round(random.uniform(60, 95), 1)

    return {
        "pair": pair,
        "timeframe": timeframe,
        "direction": direction,
        "confidence": confidence,
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "entry_price": round(random.uniform(1.0500, 1.1500), 4),
        "indicators": "RSI, MACD, EMA"
    }

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /start command."""
    print(f"📨 Received /start command from user {update.effective_user.id}")
    logger.info(f"Start command from user {update.effective_user.id}")

    welcome_message = """
🤖 *Welcome to TradePulse Analyzer Bot!*

✅ Bot is working correctly!

*Available Commands:*
/start - Show this welcome message
/signal - Get a test trading signal
/pairs - List available trading pairs
/status - Check bot status

*Example:* `/signal EURUSD 5m`

⚠️ *Note:* This is a test version with simulated data.
"""

    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN
    )
    print("✅ Welcome message sent")

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /signal command."""
    print(f"📨 Received /signal command from user {update.effective_user.id}")
    logger.info(f"Signal command from user {update.effective_user.id}")

    try:
        # Parse arguments
        pair = "EURUSD"
        timeframe = "1m"

        if len(context.args) >= 1:
            pair = context.args[0].upper()
        if len(context.args) >= 2:
            timeframe = context.args[1].lower()

        # Generate test signal
        signal = generate_test_signal(pair, timeframe)

        # Format signal message
        direction_emoji = "🟢" if signal["direction"] == "BUY" else "🔴" if signal["direction"] == "SELL" else "🟡"

        message = f"""
🎯 *Trading Signal*

*Pair:* {signal['pair']}
*Timeframe:* {signal['timeframe']}
*Direction:* {direction_emoji} {signal['direction']}
*Confidence:* {signal['confidence']}%
*Entry Price:* {signal['entry_price']}
*Time:* {signal['timestamp']}

✅ *Signal generated successfully!*
⚠️ *Test data only*
"""

        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN
        )
        print(f"✅ Signal sent: {signal['direction']} for {pair}")

    except Exception as e:
        logger.error(f"Error in signal command: {e}")
        print(f"❌ Error in signal command: {e}")
        await update.message.reply_text("❌ Error generating signal. Please try again.")

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /status command."""
    print(f"📨 Received /status command from user {update.effective_user.id}")

    status_message = f"""
🤖 *Bot Status*

*Status:* ✅ Online and Working
*Time:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
*Version:* 1.0.0-verbose

✅ *All systems operational!*
"""

    await update.message.reply_text(
        status_message,
        parse_mode=ParseMode.MARKDOWN
    )
    print("✅ Status message sent")

async def pairs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /pairs command."""
    print(f"📨 Received /pairs command from user {update.effective_user.id}")

    pairs_message = """
📊 *Available Trading Pairs:*

• EURUSD - Euro / US Dollar
• GBPUSD - British Pound / US Dollar
• USDJPY - US Dollar / Japanese Yen
• AUDUSD - Australian Dollar / US Dollar

*Usage:* `/signal EURUSD 5m`
"""

    await update.message.reply_text(
        pairs_message,
        parse_mode=ParseMode.MARKDOWN
    )
    print("✅ Pairs list sent")

def main():
    """Start the bot."""
    print("🔧 Starting main function...")

    try:
        # Get token from environment
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            print("❌ TELEGRAM_TOKEN not found in environment variables")
            return

        print(f"🔑 Token found: {token[:10]}...")

        # Create application
        print("🏗️ Creating application...")
        application = Application.builder().token(token).build()
        print("✅ Application created")

        # Add command handlers
        print("🔧 Adding command handlers...")
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("signal", signal_command))
        application.add_handler(CommandHandler("status", status_command))
        application.add_handler(CommandHandler("pairs", pairs_command))
        print("✅ Command handlers added")

        # Start the bot
        print("🚀 Starting bot polling...")
        print("📱 Bot is now ready to receive messages!")
        print("💬 Send /start to @MS0531_bot to test it")
        print("=" * 50)

        logger.info("Bot started successfully")

        application.run_polling(drop_pending_updates=True)

    except Exception as e:
        print(f"❌ Error starting bot: {e}")
        logger.error(f"Error starting bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
