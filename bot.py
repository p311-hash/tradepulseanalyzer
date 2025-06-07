"""
Telegram bot implementation for TradePulse Signals
"""
import os
import logging
from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
)
from apscheduler.schedulers.background import BackgroundScheduler
from signal_generator import generate_signals, get_latest_signals
from user_manager import (
    load_users,
    save_user,
    get_user_preferences,
    set_user_preferences,
)
from config import SUPPORTED_ASSETS

# State definitions for conversation
SELECTING_ASSETS = 1

# Initialize logger
logger = logging.getLogger(__name__)

# Initialize scheduler
scheduler = BackgroundScheduler()

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Start command handler - Introduction to the bot"""
    user_id = update.effective_user.id
    
    # Save user if new
    user_data = {
        "id": user_id,
        "username": update.effective_user.username,
        "first_name": update.effective_user.first_name,
        "preferences": {"assets": ["BTC/USD", "ETH/USD"]}  # Default assets
    }
    save_user(user_data)
    
    welcome_message = (
        f"Welcome to TradePulse Signals Bot, {update.effective_user.first_name}!\n\n"
        "I provide trading signals for various financial instruments based on technical analysis.\n\n"
        "Available commands:\n"
        "/start - Show this message\n"
        "/signals - View latest trading signals\n"
        "/preferences - Set your asset preferences\n"
        "/status - Check bot status\n"
        "/help - Show help information"
    )
    
    await update.message.reply_text(welcome_message)
    return ConversationHandler.END

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Display help information"""
    help_text = (
        "TradePulse Signals Bot Commands:\n\n"
        "/start - Initialize the bot and get welcome message\n"
        "/signals - View latest trading signals for your preferred assets\n"
        "/preferences - Set which assets you want to receive signals for\n"
        "/status - Check if the bot is working properly\n"
        "/help - Show this help message\n\n"
        "How it works:\n"
        "1. The bot analyzes market data every hour\n"
        "2. When a trading signal is generated, you'll receive a notification\n"
        "3. You can customize which assets you want to track"
    )
    await update.message.reply_text(help_text)

async def signals_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the latest signals for user's preferred assets"""
    user_id = update.effective_user.id
    preferences = get_user_preferences(user_id)
    user_assets = preferences.get("assets", [])
    
    if not user_assets:
        await update.message.reply_text(
            "You haven't selected any assets to track. Use /preferences to set your preferences."
        )
        return
    
    signals = get_latest_signals(user_assets)
    
    if not signals:
        await update.message.reply_text(
            "No signals available for your selected assets at the moment. "
            "Signals are generated regularly - please check back later."
        )
        return
    
    # Format and send signals
    message = "📊 Latest Trading Signals 📊\n\n"
    
    for asset, signal_data in signals.items():
        message += f"*{asset}*\n"
        message += f"Signal: {signal_data['direction']} ({signal_data['strength']})\n"
        message += f"Price: ${signal_data['price']:.2f}\n"
        message += f"Generated: {signal_data['timestamp']}\n"
        message += f"Indicators: {signal_data['indicators']}\n\n"
    
    message += "⚠️ Trading involves risk. Always do your own research before making decisions."
    
    await update.message.reply_text(message, parse_mode="Markdown")

async def preferences(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle preferences command - allows users to select assets"""
    user_id = update.effective_user.id
    preferences = get_user_preferences(user_id)
    selected_assets = preferences.get("assets", [])
    
    # Create keyboard with asset options
    keyboard = []
    for asset in SUPPORTED_ASSETS:
        # Mark selected assets with a check mark
        text = f"✅ {asset}" if asset in selected_assets else asset
        keyboard.append([InlineKeyboardButton(text, callback_data=f"asset_{asset}")])
    
    keyboard.append([InlineKeyboardButton("Done", callback_data="asset_done")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "Select the assets you want to receive signals for:",
        reply_markup=reply_markup
    )
    return SELECTING_ASSETS

async def asset_selection(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Handle asset selection callbacks"""
    query = update.callback_query
    await query.answer()
    user_id = update.effective_user.id
    
    if query.data == "asset_done":
        await query.edit_message_text("Preferences saved! Use /signals to check latest trading signals.")
        return ConversationHandler.END
    
    # Extract asset from callback data
    asset = query.data.replace("asset_", "")
    
    # Toggle asset selection
    preferences = get_user_preferences(user_id)
    selected_assets = preferences.get("assets", [])
    
    if asset in selected_assets:
        selected_assets.remove(asset)
    else:
        selected_assets.append(asset)
    
    # Update user preferences
    preferences["assets"] = selected_assets
    set_user_preferences(user_id, preferences)
    
    # Update keyboard
    keyboard = []
    for supported_asset in SUPPORTED_ASSETS:
        text = f"✅ {supported_asset}" if supported_asset in selected_assets else supported_asset
        keyboard.append([InlineKeyboardButton(text, callback_data=f"asset_{supported_asset}")])
    
    keyboard.append([InlineKeyboardButton("Done", callback_data="asset_done")])
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await query.edit_message_text(
        "Select the assets you want to receive signals for:",
        reply_markup=reply_markup
    )
    return SELECTING_ASSETS

async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show bot status and information"""
    users = load_users()
    total_users = len(users)
    signals = get_latest_signals(SUPPORTED_ASSETS)
    
    status_text = (
        "🤖 *TradePulse Signals Bot Status*\n\n"
        f"Bot is operational ✅\n"
        f"Total users: {total_users}\n"
        f"Available assets: {len(SUPPORTED_ASSETS)}\n"
        f"Latest signal generation: {signals[list(signals.keys())[0]]['timestamp'] if signals else 'N/A'}\n\n"
        "Signal generation occurs hourly. Notifications are sent when significant trading opportunities are detected."
    )
    
    await update.message.reply_text(status_text, parse_mode="Markdown")

async def send_signal_notification(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send notifications about new signals to subscribed users"""
    users = load_users()
    new_signals = generate_signals()
    
    if not new_signals:
        logger.info("No new signals generated for notification")
        return
    
    signal_count = 0
    
    for user_id, user_data in users.items():
        user_assets = user_data.get("preferences", {}).get("assets", [])
        relevant_signals = {asset: signal for asset, signal in new_signals.items() if asset in user_assets}
        
        if not relevant_signals:
            continue
        
        # Format notification message
        message = "🚨 *New Trading Signals Alert* 🚨\n\n"
        
        for asset, signal_data in relevant_signals.items():
            message += f"*{asset}*: {signal_data['direction']} ({signal_data['strength']})\n"
            message += f"Price: ${signal_data['price']:.2f}\n"
            message += f"Indicators: {signal_data['indicators']}\n\n"
        
        message += "Use /signals for more details"
        
        try:
            await context.bot.send_message(
                chat_id=int(user_id),
                text=message,
                parse_mode="Markdown"
            )
            signal_count += 1
        except Exception as e:
            logger.error(f"Failed to send notification to user {user_id}: {e}")
    
    logger.info(f"Sent signal notifications to {signal_count} users")

def schedule_signal_generation(app):
    """Set up scheduled signal generation and notifications"""
    job = scheduler.add_job(
        lambda: app.create_task(send_signal_notification(app)),
        'interval',
        hours=1,
        id='signal_generation'
    )
    scheduler.start()
    logger.info("Scheduled signal generation job started")

def start_bot():
    """Initialize and start the Telegram bot"""
    # Get token from environment
    telegram_token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not telegram_token:
        logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
        raise ValueError("Telegram bot token not provided")
    
    # Build application
    application = Application.builder().token(telegram_token).build()
    
    # Define conversation handler for preferences
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("preferences", preferences)],
        states={
            SELECTING_ASSETS: [CallbackQueryHandler(asset_selection)],
        },
        fallbacks=[CommandHandler("start", start)],
    )
    
    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("signals", signals_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(conv_handler)
    
    # Schedule signal generation
    schedule_signal_generation(application)
    
    # Start the bot
    application.run_polling()
    
    return application
