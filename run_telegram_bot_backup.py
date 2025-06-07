#!/usr/bin/env python3
"""
MasterTrade Bot - Professional Binary Options Trading Signal Bot
Enhanced with candlestick patterns, trading modes, and Pocket Option integration.
"""

import os
import logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ContextTypes
from signal_analytics import SignalAnalytics

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=os.getenv('LOG_LEVEL', 'INFO')
)
logger = logging.getLogger(__name__)

# Initialize handlers (will be done in main function)
telegram_handler = None
signal_analytics = SignalAnalytics()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with TradeMind Bot branding."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "Trader"

    # Initialize user settings with enhanced features
    telegram_handler.user_settings[user_id] = {
        'pair': 'EURUSD_otc',
        'timeframe': '2m',
        'category': 'Currencies',
        'trading_mode': 'demo'  # Default to demo mode for safety
    }

    welcome_message = (
        "🤖 *Welcome to MasterTrade Bot | Successful trading*\n\n"
        f"Hello {user_name}! I'm your advanced binary options trading assistant.\n\n"
        "🎯 *Enhanced Features:*\n"
        "• 9 Key candlestick pattern recognition\n"
        "• Demo/Real trading mode toggle\n"
        "• Extended timeframes (15s, 30s, 10m, 15m)\n"
        "• Professional signal analysis\n"
        "• Pocket Option integration\n"
        "• Advanced risk management\n\n"
        "� *Pattern Recognition:*\n"
        "• Morning Star, Evening Star patterns\n"
        "• Shooting Star, Doji analysis\n"
        "• Engulfing patterns detection\n"
        "• And 6 more key patterns\n\n"
        "🎮 *Trading Modes:*\n"
        "• Demo Mode - Safe practice environment\n"
        "• Real Mode - Live trading (with warnings)\n\n"
        "🚀 *Ready to start trading?*\n"
        "Use the enhanced menu below to begin:"
    )

    await telegram_handler.show_main_menu(update, context, welcome_message)

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send help information."""
    await telegram_handler._show_help(update, context)

async def admin_login_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin login command."""
    await telegram_handler.admin_login(update, context)

async def admin_logout_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin logout command."""
    await telegram_handler.admin_logout(update, context)

async def admin_panel_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Admin panel command."""
    await telegram_handler.admin_panel(update, context)

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate trading signal - redirect to enhanced interface."""
    await telegram_handler._generate_professional_signal(update, context)

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the main menu."""
    await telegram_handler.show_main_menu(update, context)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from inline keyboards."""
    await telegram_handler.handle_callback(update, context)

def main() -> None:
    """Start the enhanced TradeMind Bot."""
    global telegram_handler
    try:
        # Get bot token from environment
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            logger.error("Telegram token not found in environment variables")
            logger.info("Please set TELEGRAM_TOKEN environment variable")
            return

        # Initialize MasterTrade Bot handler
        global telegram_handler
        from enhanced_telegram_handler import MasterTradeBotHandler
        telegram_handler = MasterTradeBotHandler(token)

        # Create application
        from telegram.ext import Application, CommandHandler, CallbackQueryHandler
        application = Application.builder().token(token).build()

        # Add command handlers
        application.add_handler(CommandHandler("start", start))
        application.add_handler(CommandHandler("help", help_command))
        application.add_handler(CommandHandler("signal", signal_command))

        # Add admin command handlers
        application.add_handler(CommandHandler("admin_login", admin_login_command))
        application.add_handler(CommandHandler("admin_logout", admin_logout_command))
        application.add_handler(CommandHandler("admin_panel", admin_panel_command))
        application.add_handler(CommandHandler("admin", admin_panel_command))  # Shortcut

        # Add callback query handler for button interactions
        application.add_handler(CallbackQueryHandler(telegram_handler.handle_callback))

        # Start the bot
        logger.info("🤖 Starting TradeMind Bot...")
        logger.info("Bot is ready to receive commands!")
        application.run_polling()

    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()