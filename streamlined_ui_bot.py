#!/usr/bin/env python3
"""
Simple Telegram bot for TradePulseAnalyzer
This version has minimal dependencies and focuses on basic functionality
"""

import os
import logging
import asyncio
from datetime import datetime
from dotenv import load_dotenv

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, ContextTypes
    from telegram.constants import ParseMode
except ImportError as e:
    print(f"Missing telegram package: {e}")
    print("Please install: pip install python-telegram-bot")
    exit(1)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Simple signal generator for testing
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
    welcome_message = """
🤖 *Welcome to TradePulse Analyzer Bot!*

I'm a simplified version of the trading signal bot for testing purposes.

*Available Commands:*
/start - Show this welcome message
/signal - Get a test trading signal
/signal EURUSD 5m - Get signal for specific pair and timeframe
/pairs - List available trading pairs
/timeframes - List available timeframes
/help - Show help information
/status - Check bot status

*Example Usage:*
• `/signal` - Get random signal
• `/signal GBPUSD 1m` - Get GBPUSD 1-minute signal

⚠️ *Note:* This is a test version with simulated data.
"""

    await update.message.reply_text(
        welcome_message,
        parse_mode=ParseMode.MARKDOWN
    )

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /help command."""
    await start_command(update, context)

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /signal command."""
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

⚠️ *Disclaimer:* This is test data for demonstration purposes only.
"""

        await update.message.reply_text(
            message,
            parse_mode=ParseMode.MARKDOWN
        )

    except Exception as e:
        logger.error(f"Error in signal command: {e}")
        await update.message.reply_text(
            "❌ Error generating signal. Please try again."
        )

async def pairs_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /pairs command."""
    pairs_message = """
📊 *Available Trading Pairs:*

*Major Pairs:*
• EURUSD - Euro / US Dollar
• GBPUSD - British Pound / US Dollar
• USDJPY - US Dollar / Japanese Yen
• USDCHF - US Dollar / Swiss Franc
• AUDUSD - Australian Dollar / US Dollar
• USDCAD - US Dollar / Canadian Dollar

*Cross Pairs:*
• EURGBP - Euro / British Pound
• EURJPY - Euro / Japanese Yen
• GBPJPY - British Pound / Japanese Yen

*Usage:* `/signal EURUSD 5m`
"""

    await update.message.reply_text(
        pairs_message,
        parse_mode=ParseMode.MARKDOWN
    )

async def timeframes_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /timeframes command."""
    timeframes_message = """
⏰ *Available Timeframes:*

*Ultra Short:*
• 5s - 5 seconds
• 15s - 15 seconds
• 30s - 30 seconds

*Short Term:*
• 1m - 1 minute
• 3m - 3 minutes
• 5m - 5 minutes
• 10m - 10 minutes
• 15m - 15 minutes

*Usage:* `/signal EURUSD 5m`
"""

    await update.message.reply_text(
        timeframes_message,
        parse_mode=ParseMode.MARKDOWN
    )

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle the /status command."""
    status_message = f"""
🤖 *Bot Status*

*Status:* ✅ Online
*Mode:* Test/Development
*Version:* 1.0.0-simple
*Uptime:* Running
*Last Update:* {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

*Features:*
• ✅ Basic Commands
• ✅ Test Signal Generation
• ✅ Pair/Timeframe Support
• ⚠️ Real Market Data (Disabled)
• ⚠️ Advanced Analysis (Disabled)

*Note:* This is a simplified test version.
"""

    await update.message.reply_text(
        status_message,
        parse_mode=ParseMode.MARKDOWN
    )

async def unknown_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unknown commands."""
    message = """
❓ *Unknown Command*

Use /help to see available commands.

*Quick Commands:*
• /start - Welcome message
• /signal - Get trading signal
• /pairs - List trading pairs
• /timeframes - List timeframes
• /status - Bot status
"""

    await update.message.reply_text(
        message,
        parse_mode=ParseMode.MARKDOWN
    )

def main():
    """Start the bot."""
    try:
        # Get token from environment
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            logger.error("TELEGRAM_TOKEN not found in environment variables")
            logger.error("Please set TELEGRAM_TOKEN in your .env file")
            return

        logger.info(f"Starting bot with token: {token[:10]}...")

        # Create application
        application = Application.builder().token(token).build()

        # Add command handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("signal", signal_command))
        application.add_handler(CommandHandler("pairs", pairs_command))
        application.add_handler(CommandHandler("timeframes", timeframes_command))
        application.add_handler(CommandHandler("status", status_command))

        # Start the bot
        logger.info("Bot is starting...")
        logger.info("Send /start to your bot to test it!")
        logger.info(f"Bot username: @{application.bot.username if hasattr(application.bot, 'username') else 'Unknown'}")
        logger.info("Bot is now running and waiting for messages...")

        application.run_polling(drop_pending_updates=True)

    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
