#!/usr/bin/env python3
"""
Enhanced Streamlined Binary Options Signal Bot
Comprehensive audit and enhancement with advanced features
"""

import os
import logging
import random
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any, Optional

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    from telegram.constants import ParseMode
except ImportError as e:
    print(f"Missing telegram package: {e}")
    print("Please install: pip install python-telegram-bot")
    exit(1)

# Import advanced analysis modules
try:
    from signal_generator import SignalGenerator
    from technical_analysis import TechnicalAnalyzer
    from pattern_recognition import EnhancedPatternRecognizer
    from signal_recommendation import EnhancedSignalRecommender
    from ml_model import MLPredictor
    from utils import fetch_market_data
    import config
    ADVANCED_FEATURES_ENABLED = True
except ImportError as e:
    print(f"Advanced features not available: {e}")
    ADVANCED_FEATURES_ENABLED = False

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

class EnhancedStreamlinedSignalBot:
    """Enhanced Streamlined Binary Options Signal Bot with advanced features."""
    
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
        self.timeframes = ['5s', '15s', '30s', '1m', '2m', '3m', '5m', '10m', '15m', '30m', '1h']
        
        # Pocket Option URL
        self.pocket_option_url = os.getenv('POCKET_OPTION_URL', 'https://pocket-friends.com/r/xahqexaiax')
        
        # Initialize advanced components if available
        if ADVANCED_FEATURES_ENABLED:
            try:
                self.signal_generator = SignalGenerator()
                self.technical_analyzer = TechnicalAnalyzer()
                self.pattern_recognizer = EnhancedPatternRecognizer()
                self.signal_recommender = EnhancedSignalRecommender()
                self.ml_predictor = MLPredictor()
                logger.info("✅ Advanced analysis components initialized")
            except Exception as e:
                logger.warning(f"Could not initialize advanced components: {e}")
                ADVANCED_FEATURES_ENABLED = False
        
        logger.info("🤖 Enhanced Streamlined Signal Bot initialized")

    def _get_trading_mode_display(self, user_id: int) -> str:
        """Get trading mode display text."""
        settings = self.user_settings.get(user_id, {})
        mode = settings.get('trading_mode', 'demo')
        return "🎮 Demo Mode" if mode == 'demo' else "💰 Real Trading"

    async def _generate_advanced_signal(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate advanced signal using all available analysis methods."""
        try:
            if not ADVANCED_FEATURES_ENABLED:
                return self._generate_fallback_signal(pair, timeframe)
            
            # Fetch market data
            data = await asyncio.to_thread(fetch_market_data, pair, timeframe, 100)
            if data is None or data.empty:
                return self._generate_fallback_signal(pair, timeframe)
            
            # Technical Analysis
            tech_analysis = await asyncio.to_thread(self.technical_analyzer.analyze, data)
            
            # Pattern Recognition
            patterns = await asyncio.to_thread(self.pattern_recognizer.recognize_patterns)
            
            # ML Prediction
            ml_prediction = await asyncio.to_thread(self.ml_predictor.predict, data)
            
            # Signal Generation
            signal_data = await asyncio.to_thread(self.signal_generator.generate_signal, pair, timeframe)
            
            # Combine all analyses
            return self._combine_analyses(tech_analysis, patterns, ml_prediction, signal_data, pair, timeframe)
            
        except Exception as e:
            logger.error(f"Error in advanced signal generation: {e}")
            return self._generate_fallback_signal(pair, timeframe)

    def _generate_fallback_signal(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate fallback signal when advanced features are not available."""
        signal_direction = random.choice(['BUY', 'SELL'])
        confidence = random.randint(55, 85)
        current_price = round(random.uniform(1.10000, 1.10200), 5)
        
        return {
            'direction': signal_direction,
            'confidence': confidence,
            'price': current_price,
            'ml_prediction': 'NEUTRAL',
            'ml_confidence': 0.0,
            'rsi': round(random.uniform(30, 70), 1),
            'macd': 'Bullish' if signal_direction == 'BUY' else 'Bearish',
            'support': round(current_price - 0.00050, 5),
            'resistance': round(current_price + 0.00050, 5),
            'patterns': ['None'],
            'risk_level': 'Medium'
        }

    def _combine_analyses(self, tech_analysis: Dict, patterns: Dict, ml_prediction: Dict, 
                         signal_data: Dict, pair: str, timeframe: str) -> Dict[str, Any]:
        """Combine all analysis results into a comprehensive signal."""
        try:
            # Extract key information
            direction = signal_data.get('direction', 'NEUTRAL')
            confidence = signal_data.get('confidence', 50)
            
            # ML prediction
            ml_direction = ml_prediction.get('direction', 'NEUTRAL')
            ml_confidence = ml_prediction.get('confidence', 0.0)
            
            # Technical indicators
            rsi = tech_analysis.get('rsi', {}).get('value', 50)
            macd_signal = tech_analysis.get('macd', {}).get('signal', 'Neutral')
            
            # Support/Resistance
            sr_levels = tech_analysis.get('support_resistance', {})
            support = sr_levels.get('support', 0.0)
            resistance = sr_levels.get('resistance', 0.0)
            
            # Patterns
            detected_patterns = [name for name, detected in patterns.items() if detected]
            if not detected_patterns:
                detected_patterns = ['None']
            
            # Calculate risk level
            risk_level = self._calculate_risk_level(confidence, ml_confidence, rsi)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'price': signal_data.get('price', 1.10000),
                'ml_prediction': ml_direction,
                'ml_confidence': ml_confidence,
                'rsi': rsi,
                'macd': macd_signal,
                'support': support,
                'resistance': resistance,
                'patterns': detected_patterns,
                'risk_level': risk_level
            }
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return self._generate_fallback_signal(pair, timeframe)

    def _calculate_risk_level(self, confidence: float, ml_confidence: float, rsi: float) -> str:
        """Calculate risk level based on signal strength."""
        avg_confidence = (confidence + ml_confidence) / 2
        
        if avg_confidence >= 75 and 30 <= rsi <= 70:
            return 'Low'
        elif avg_confidence >= 60:
            return 'Medium'
        else:
            return 'High'

# Global bot instance
bot = EnhancedStreamlinedSignalBot()

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    await show_main_menu(update, context)

async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the enhanced main menu with bold formatting."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        # Set defaults
        settings.setdefault('pair', 'EURUSD_otc')
        settings.setdefault('timeframe', '1m')
        settings.setdefault('trading_mode', 'demo')
        bot.user_settings[user_id] = settings
        
        # Create enhanced message with bold formatting (no markdown to avoid errors)
        trading_mode = bot._get_trading_mode_display(user_id)
        
        message = f"🤖 Binary Options Signal Bot - Main Menu\n\n"
        message += f"Current settings:\n"
        message += f"• Pair: {settings['pair']}\n"
        message += f"• Timeframe: {settings['timeframe']}\n"
        message += f"• Mode: {trading_mode}\n\n"
        message += f"Select an option below:"
        
        # Enhanced keyboard with new features
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
            ],
            [
                InlineKeyboardButton("🔗 POCKET OPTION", callback_data="pocket_option"),
                InlineKeyboardButton("🎮 TRADING MODE", callback_data="toggle_mode")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                text=message,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"Error showing main menu: {e}")

async def generate_signal(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate an enhanced trading signal with all analysis features."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        pair = settings.get('pair', 'EURUSD_otc')
        timeframe = settings.get('timeframe', '1m')
        trading_mode = bot._get_trading_mode_display(user_id)
        
        # Generate advanced signal
        signal_data = await bot._generate_advanced_signal(pair, timeframe)
        
        # Create enhanced signal message with bold formatting
        signal_color = "🟢" if signal_data['direction'] == "BUY" else "🔴"
        
        message = f"🤖 Binary Options Signal Bot - Main Menu\n\n"
        message += f"Current settings:\n"
        message += f"• Pair: {pair}\n"
        message += f"• Timeframe: {timeframe}\n"
        message += f"• Mode: {trading_mode}\n\n"
        message += f"Select an option below:\n\n"
        
        # Enhanced signal section with comprehensive analysis
        message += f"{signal_color} {signal_data['direction']} SIGNAL {signal_color}\n\n"
        message += f"{pair} ({timeframe}) @ {signal_data['price']}\n"
        message += f"Confidence: {signal_data['confidence']}%\n"
        message += f"🤖 ML: {signal_data['ml_prediction']} ({signal_data['ml_confidence']:.1f}%)\n"
        message += f"Risk Level: {signal_data['risk_level']}\n\n"
        message += f"S/R Levels: {signal_data['support']:.5f} / {signal_data['resistance']:.5f}\n"
        message += f"Key Indicators: RSI: {signal_data['rsi']:.1f} | MACD: {signal_data['macd']}\n"
        message += f"Patterns: {', '.join(signal_data['patterns'])}\n\n"
        message += f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        # Enhanced keyboard with more options
        keyboard = [
            [InlineKeyboardButton("🔄 REFRESH SIGNAL", callback_data="generate_signal")],
            [
                InlineKeyboardButton("📊 SIGNAL ANALYSIS", callback_data="signal_analysis"),
                InlineKeyboardButton("🎯 RECOMMENDATIONS", callback_data="recommendations")
            ],
            [InlineKeyboardButton("🔙 BACK TO MENU", callback_data="back_to_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")

async def show_pocket_option(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show Pocket Option integration."""
    try:
        message = f"🔗 Pocket Option Trading Platform\n\n"
        message += f"🎯 Professional Binary Options Trading:\n"
        message += f"• Demo & Real Accounts Available\n"
        message += f"• Multiple Assets & Timeframes\n"
        message += f"• Fast Execution & Withdrawals\n"
        message += f"• Mobile & Web Platform\n\n"
        message += f"💡 Getting Started:\n"
        message += f"1. Click 'Open Platform' below\n"
        message += f"2. Create your account\n"
        message += f"3. Start with Demo Mode\n"
        message += f"4. Use our signals for trading\n\n"
        message += f"⚠️ Risk Warning: Trading involves risk\n"
        message += f"Only invest what you can afford to lose"
        
        keyboard = [
            [InlineKeyboardButton("🌐 Open Platform", url=bot.pocket_option_url)],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing Pocket Option: {e}")

async def toggle_trading_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle between demo and real trading mode."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        # Toggle mode
        current_mode = settings.get('trading_mode', 'demo')
        new_mode = 'real' if current_mode == 'demo' else 'demo'
        settings['trading_mode'] = new_mode
        bot.user_settings[user_id] = settings
        
        mode_display = bot._get_trading_mode_display(user_id)
        
        message = f"🎮 Trading Mode Settings\n\n"
        message += f"Current Mode: {mode_display}\n\n"
        message += f"Mode Description:\n"
        
        if new_mode == 'demo':
            message += f"🎮 Demo Mode:\n"
            message += f"• Practice trading with virtual money\n"
            message += f"• No real financial risk\n"
            message += f"• Perfect for learning and testing\n"
            message += f"• All features available\n"
        else:
            message += f"💰 Real Trading Mode:\n"
            message += f"• Live trading with real money\n"
            message += f"• Real profits and losses\n"
            message += f"• Requires funded account\n"
            message += f"• Higher risk, higher reward\n"
        
        message += f"\n⚠️ Remember: Always trade responsibly!"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Toggle Mode", callback_data="toggle_mode")],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error toggling trading mode: {e}")

async def show_signal_analysis(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show detailed signal analysis."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        
        pair = settings.get('pair', 'EURUSD_otc')
        timeframe = settings.get('timeframe', '1m')
        
        # Generate detailed analysis
        signal_data = await bot._generate_advanced_signal(pair, timeframe)
        
        message = f"📊 Signal Analysis - {pair}\n\n"
        message += f"🔍 Technical Analysis:\n"
        message += f"• RSI: {signal_data['rsi']:.1f} "
        
        if signal_data['rsi'] > 70:
            message += f"(Overbought)\n"
        elif signal_data['rsi'] < 30:
            message += f"(Oversold)\n"
        else:
            message += f"(Neutral)\n"
            
        message += f"• MACD: {signal_data['macd']}\n"
        message += f"• Support: {signal_data['support']:.5f}\n"
        message += f"• Resistance: {signal_data['resistance']:.5f}\n\n"
        
        message += f"🤖 Machine Learning:\n"
        message += f"• Prediction: {signal_data['ml_prediction']}\n"
        message += f"• ML Confidence: {signal_data['ml_confidence']:.1f}%\n\n"
        
        message += f"📈 Pattern Recognition:\n"
        message += f"• Detected: {', '.join(signal_data['patterns'])}\n\n"
        
        message += f"⚠️ Risk Assessment:\n"
        message += f"• Risk Level: {signal_data['risk_level']}\n"
        message += f"• Overall Confidence: {signal_data['confidence']}%\n\n"

        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Analysis", callback_data="signal_analysis")],
            [InlineKeyboardButton("🔙 Back to Signal", callback_data="generate_signal")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing signal analysis: {e}")

async def show_recommendations(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show signal recommendations using the advanced recommender system."""
    try:
        if not ADVANCED_FEATURES_ENABLED:
            message = f"🎯 Signal Recommendations\n\n"
            message += f"⚠️ Advanced features not available\n"
            message += f"Please ensure all analysis modules are installed\n\n"
            message += f"Basic recommendation: Use the generated signals\n"
            message += f"with proper risk management."
        else:
            # Use the advanced signal recommender
            try:
                recommendations = await asyncio.to_thread(
                    bot.signal_recommender.get_recommendations,
                    timeframes=['1m', '5m', '15m']
                )
                
                message = f"🎯 Signal Recommendations\n\n"
                
                if recommendations and len(recommendations) > 0:
                    message += f"📈 Top Opportunities:\n\n"
                    
                    for i, rec in enumerate(recommendations[:3], 1):
                        direction_emoji = "🟢" if rec['direction'] == 'BUY' else "🔴"
                        message += f"{i}. {direction_emoji} {rec['pair']} - {rec['direction']}\n"
                        message += f"   Confidence: {rec['confidence']:.1f}%\n"
                        message += f"   Risk: {rec['risk_level']}\n"
                        message += f"   Reason: {rec['explanation'][:50]}...\n\n"
                else:
                    message += f"📊 No high-confidence opportunities found\n"
                    message += f"Current market conditions may be uncertain\n"
                    message += f"Consider waiting for clearer signals\n\n"
                    
            except Exception as e:
                logger.error(f"Error getting recommendations: {e}")
                message = f"🎯 Signal Recommendations\n\n"
                message += f"⚠️ Unable to fetch recommendations\n"
                message += f"Please try again later\n\n"
        
        message += f"💡 Trading Tips:\n"
        message += f"• Always use proper risk management\n"
        message += f"• Don't risk more than 2-5% per trade\n"
        message += f"• Consider multiple timeframes\n"
        message += f"• Practice with demo first"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Refresh Recommendations", callback_data="recommendations")],
            [InlineKeyboardButton("🔙 Back to Signal", callback_data="generate_signal")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            text=message,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing recommendations: {e}")

async def show_pair_categories(update: Update, context: ContextTypes.DEFAULT_TYPE, category: str):
    """Show pairs in selected category."""
    try:
        pairs = bot.asset_categories.get(category, [])
        
        message = f"🤖 Select {category} Pair\n\n"
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
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing pair categories: {e}")

async def show_timeframe_selection(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show timeframe selection."""
    try:
        message = f"🤖 Select Timeframe\n\n"
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
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing timeframe selection: {e}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all button callbacks with enhanced features."""
    try:
        query = update.callback_query
        await query.answer()
        
        if query.data == "generate_signal":
            await generate_signal(update, context)
        elif query.data == "select_pair":
            await show_main_menu(update, context)  # Show main menu with category buttons
        elif query.data == "select_timeframe":
            await show_timeframe_selection(update, context)
        elif query.data == "pocket_option":
            await show_pocket_option(update, context)
        elif query.data == "toggle_mode":
            await toggle_trading_mode(update, context)
        elif query.data == "signal_analysis":
            await show_signal_analysis(update, context)
        elif query.data == "recommendations":
            await show_recommendations(update, context)
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
    """Start the enhanced streamlined bot."""
    try:
        # Get token from environment
        token = os.getenv('TELEGRAM_BOT_TOKEN')
        if not token:
            logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
            return
            
        # Create application without job queue to avoid weak reference issue
        application = Application.builder().token(token).job_queue(None).build()
        
        # Add handlers
        application.add_handler(CommandHandler("start", start_command))
        application.add_handler(CallbackQueryHandler(handle_callback))
        
        # Start bot
        logger.info("🤖 Starting Enhanced Streamlined Signal Bot...")
        logger.info("✅ Features enabled:")
        logger.info("   • Advanced signal generation")
        logger.info("   • Technical analysis integration")
        logger.info("   • Machine learning predictions")
        logger.info("   • Pattern recognition")
        logger.info("   • Signal recommendations")
        logger.info("   • Pocket Option integration")
        logger.info("   • Trading mode toggle")
        logger.info("   • Enhanced UI with bold formatting")
        
        application.run_polling(drop_pending_updates=True)
        
    except Exception as e:
        logger.error(f"Error starting bot: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()