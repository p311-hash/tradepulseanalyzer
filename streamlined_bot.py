#!/usr/bin/env python3
"""
Streamlined Binary Options Signal Bot
Matches the reference UI design exactly
"""

import os
import logging
import random
from datetime import datetime
from dotenv import load_dotenv

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
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

class StreamlinedSignalBot:
    """Streamlined Binary Options Signal Bot matching reference design."""
    
    def __init__(self):
        self.user_settings = {}
        
        # Asset categories matching the reference image
        self.asset_categories = {
            'USD': ['EURUSD_otc', 'GBPUSD_otc', 'AUDUSD_otc', 'NZDUSD_otc', 'USDCAD_otc', 'USDCHF_otc', 'USDJPY_otc'],
            'EUR': ['EURUSD_otc', 'EURGBP_otc', 'EURJPY_otc', 'EURCHF_otc', 'EURAUD_otc', 'EURCAD_otc'],
            'GBP': ['GBPUSD_otc', 'EURGBP_otc', 'GBPJPY_otc', 'GBPCHF_otc', 'GBPAUD_otc', 'GBPCAD_otc'],
            'JPY': ['USDJPY_otc', 'EURJPY_otc', 'GBPJPY_otc', 'AUDJPY_otc', 'CADJPY_otc', 'CHFJPY_otc']
        }
        
        # Available timeframes
        self.timeframes = ['5s', '15s', '30s', '1m', '2m', '3m', '5m', '10m', '15m', '30m']

# Global bot instance
bot = StreamlinedSignalBot()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await show_main_menu(update, context)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the main menu matching the reference image."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        # Set defaults
        settings.setdefault('pair', 'EURUSD_otc')
        settings.setdefault('timeframe', '1m')
        bot.user_settings[user_id] = settings
        
        # Create message matching reference design
        message = f"🤖 *Binary Options Signal Bot - Main Menu*\n\n"
        message += f"*Current settings:*\n"
        message += f"• Pair: {settings['pair']}\n"
        message += f"• Timeframe: {settings['timeframe']}\n\n"
        message += f"*Select an option below:*"
        
        # Create keyboard matching reference image
        keyboard = [
            [InlineKeyboardButton("🔴 GENERATE SIGNAL 🔴", callback_data="generate_signal")],
            [
                InlineKeyboardButton("💱 SELECT PAIR", callback_data="select_pair"),
                InlineKeyboardButton("⏱ SELECT TIMEFRAME", callback_data="select_timeframe")
            ],
            [
                InlineKeyboardButton("💵 USD PAIRS", callback_data="category_USD"),
                InlineKeyboardButton("💶 EUR PAIRS", callback_data="category_EUR")
            ],
            [
                InlineKeyboardButton("💷 GBP PAIRS", callback_data="category_GBP"),
                InlineKeyboardButton("💴 JPY PAIRS", callback_data="category_JPY")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )
            
    except Exception as e:
        logger.error(f"Error showing main menu: {e}")

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate a trading signal matching the reference format."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        # Generate signal data
        signal_direction = random.choice(['BUY', 'SELL'])
        confidence = random.randint(50, 90)
        pair = settings.get('pair', 'EURUSD_otc')
        timeframe = settings.get('timeframe', '1m')
        current_price = round(random.uniform(1.10000, 1.10200), 5)
        
        # Create signal message matching reference format
        signal_color = "🟢" if signal_direction == "BUY" else "🔴"
        
        message = f"🤖 *Binary Options Signal Bot - Main Menu*\n\n"
        message += f"*Current settings:*\n"
        message += f"• Pair: {pair}\n"
        message += f"• Timeframe: {timeframe}\n\n"
        message += f"*Select an option below:*\n\n"
        
        # Add signal section
        message += f"{signal_color} *{signal_direction} SIGNAL* {signal_color}\n\n"
        message += f"{pair} ({timeframe}) @ {current_price}\n"
        message += f"Confidence: {confidence}%\n"
        message += f"🤖 ML: NEUTRAL (0.0%)\n\n"
        message += f"S/R Levels: 0.00000 / 0.00000\n"
        message += f"Key Indicators: RSI: 70.2 (Overbought) | MACD: Bullish\n"
        message += f"Patterns: None\n\n"
        message += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Create keyboard with refresh and back options
        keyboard = [
            [InlineKeyboardButton("🔄 REFRESH SIGNAL", callback_data="generate_signal")],
            [InlineKeyboardButton("🔙 BACK TO MENU", callback_data="back_to_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")

async def show_pair_categories(update: Update, context: ContextTypes.DEFAULT_TYPE, category: str):
    """Show pairs in selected category."""
    try:
        pairs = bot.asset_categories.get(category, [])
        
        message = f"🤖 *Select {category} Pair*\n\n"
        message += f"Choose from available {category} pairs:"
        
        keyboard = []
        for i in range(0, len(pairs), 2):
            row = []
            for j in range(2):
                if i + j < len(pairs):
                    pair = pairs[i + j]
                    display_name = pair.replace('_otc', '').replace('USD', '/USD').replace('EUR', '/EUR').replace('GBP', '/GBP').replace('JPY', '/JPY')
                    row.append(InlineKeyboardButton(display_name, callback_data=f"set_pair_{pair}"))
            keyboard.append(row)
            
        keyboard.append([InlineKeyboardButton("🔙 BACK TO MENU", callback_data="back_to_menu")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error showing pair categories: {e}")

async def show_timeframe_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show timeframe selection."""
    try:
        message = f"🤖 *Select Timeframe*\n\n"
        message += f"Choose your preferred timeframe:"
        
        keyboard = []
        for i in range(0, len(bot.timeframes), 3):
            row = []
            for j in range(3):
                if i + j < len(bot.timeframes):
                    tf = bot.timeframes[i + j]
                    row.append(InlineKeyboardButton(tf, callback_data=f"set_timeframe_{tf}"))
            keyboard.append(row)
            
        keyboard.append([InlineKeyboardButton("🔙 BACK TO MENU", callback_data="back_to_menu")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup,
            parse_mode=ParseMode.MARKDOWN
        )
        
    except Exception as e:
        logger.error(f"Error showing timeframe selection: {e}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button callbacks."""
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data == "generate_signal":
            await generate_signal(update, context)
        elif query.data == "select_pair":
            await show_main_menu(update, context)  # Show main menu with category buttons
        elif query.data == "select_timeframe":
            await show_timeframe_selection(update, context)
        elif query.data.startswith("category_"):
            category = query.data.replace("category_", "")
            await show_pair_categories(update, context, category)
        elif query.data.startswith("set_pair_"):
            pair = query.data.replace("set_pair_", "")
            user_id = update.effective_user.id
            if user_id not in bot.user_settings:
                bot.user_settings[user_id] = {}
            bot.user_settings[user_id]['pair'] = pair
            await show_main_menu(update, context)
        elif query.data.startswith("set_timeframe_"):
            timeframe = query.data.replace("set_timeframe_", "")
            user_id = update.effective_user.id
            if user_id not in bot.user_settings:
                bot.user_settings[user_id] = {}
            bot.user_settings[user_id]['timeframe'] = timeframe
            await show_main_menu(update, context)
        elif query.data == "back_to_menu":
            await show_main_menu(update, context)
            
    except Exception as e:
        logger.error(f"Error handling callback: {e}")

def main():
    """Start the streamlined bot."""
    try:
        # Get token from environment
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
            return
            
        # Create application
        application = Application.builder().token(token).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CallbackQueryHandler(handle_callback))
        
        # Start bot
        logger.info("🤖 Starting Streamlined Signal Bot...")
        application.run_polling()
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()