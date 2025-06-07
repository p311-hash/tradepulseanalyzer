from telegram.ext import Application, CommandHandler, ContextTypes, CallbackQueryHandler
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
import logging
import json
from utils import fetch_market_data
from enhanced_main import EnhancedTradePulseAnalyzer
from enhanced_signal_generator import SignalStrength
import config
from typing import Dict, List
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from io import BytesIO
import matplotlib.pyplot as plt
# from signal_analytics import SignalAnalytics  # Unused
from technical_analysis import TechnicalAnalyzer
from signal_generator import SignalGenerator
from signal_recommendation import EnhancedSignalRecommender

# Import continuous learning if available
try:
    # from continuous_learning import ContinuousLearningSystem  # Unused
    from main import continuous_learner
    CONTINUOUS_LEARNING_ENABLED = continuous_learner is not None
except ImportError:
    CONTINUOUS_LEARNING_ENABLED = False
    continuous_learner = None

logger = logging.getLogger(__name__)

class TelegramHandler:
    """Handles Telegram bot interactions with enhanced UI and recommendations."""

    def __init__(self, token: str):
        self.token = token
        self.application = Application.builder().token(token).build()

        # Initialize enhanced analyzer
        self.analyzer = EnhancedTradePulseAnalyzer(
            model_dir="models",
            min_confidence=65.0,
            enable_online_learning=CONTINUOUS_LEARNING_ENABLED
        )
        # Note: SignalGenerator will be initialized with data when needed
        self.signal_generator = None
        self.signal_recommender = EnhancedSignalRecommender()
        self.user_settings: Dict[int, Dict[str, any]] = {}

        # Setup command handlers
        self.setup_handlers()

    def setup_handlers(self):
        """Setup bot command and callback handlers."""
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CallbackQueryHandler(self.handle_callback))
        self.application.add_handler(CommandHandler("menu", self.show_menu))
        self.application.add_handler(CommandHandler("recommend", self.show_recommendation))

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle the /start command."""
        if update.message is None:
            return

        user_id = update.message.from_user.id
        self.user_settings[user_id] = {
            'pair': 'EURUSD_otc',
            'timeframe': '1m'
        }

        welcome_message = (
            "🤖 *Welcome to TradePulse Signal Bot!*\n\n"
            "I analyze market data in real-time to provide optimal binary options trading signals.\n\n"
            "Features:\n"
            "• Automated signal generation\n"
            "• Multiple timeframes (5s - 15m)\n"
            "• Major currency pairs\n"
            "• Smart recommendations\n"
            "• Real-time analysis\n\n"
            "Available Timeframes: 5s, 15s, 30s, 1m, 3m, 5m, 10m, 15m\n\n"
            "Use the menu below to get started:"
        )

        await self.show_menu(update, context, welcome_message)

    async def show_menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE, custom_message: str = None):
        """Show the main menu with current settings and recommendation."""
        try:
            # Get user settings
            user_id = update.effective_user.id
            settings = self.user_settings.get(user_id, {
                'pair': 'EURUSD_otc',
                'timeframe': '1m'
            })

            # Get current recommendation
            market_data = fetch_market_data([settings['pair']], [settings['timeframe']])
            recommendation = self.signal_recommender.get_recommendation(market_data)

            # Create menu message
            message = custom_message or (
                "🤖 *Binary Options Signal Bot - Main Menu*\n\n"
                "*Current Settings:*\n"
                f"• Pair: {settings['pair'].replace('_otc', '')}\n"
                f"• Timeframe: {settings['timeframe']}\n\n"
                "*Market Overview:*\n"
            )

            # Add market overview if available
            try:
                market_data = fetch_market_data([settings['pair']], [settings['timeframe']])

                # Initialize signal generator if not already done
                if self.signal_generator is None:
                    self.signal_generator = SignalGenerator(market_data[settings['timeframe']])

                regime_info = self.signal_generator.regime_detector.detect_regime(market_data[settings['timeframe']])

                message += (
                    f"• Regime: {regime_info['regime']}\n"
                    f"• Volatility: {'High' if regime_info['metrics']['volatility'] > 1.5 else 'Medium' if regime_info['metrics']['volatility'] > 1.0 else 'Low'}\n"
                    f"• Trend Strength: {regime_info['metrics']['trend_strength']*100:.1f}%\n\n"
                )
            except Exception as e:
                message += "\n*Market data temporarily unavailable*\n\n"

            message += "*Select an option below:*"

            # Create keyboard layout with enhanced UI
            keyboard = [
                [InlineKeyboardButton("🎯 GENERATE SIGNAL 🎯", callback_data="generate_signal")],
                [
                    InlineKeyboardButton("💱 ASSET", callback_data="select_pair"),
                    InlineKeyboardButton("⏱ TIMEFRAME", callback_data="select_timeframe")
                ],
                [
                    InlineKeyboardButton("📊 PERFORMANCE", callback_data="show_performance"),
                    InlineKeyboardButton("📈 ANALYSIS", callback_data="view_analysis")
                ],
                self._create_pair_category_buttons(),
                self._create_timeframe_buttons()
            ]

            # Flatten the list of buttons
            keyboard = [button for row in keyboard for button in (row if isinstance(row, list) else [row])]

            # Create button grid with 2 buttons per row
            grid_keyboard = [keyboard[i:i + 2] for i in range(0, len(keyboard), 2)]

            reply_markup = InlineKeyboardMarkup(grid_keyboard)

            # Send or edit message
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

            # Send recommendation separately if confidence is high
            if recommendation['confidence'] >= 55:
                rec_message = (
                    "🔥 *HOT RECOMMENDATION*\n\n"
                    f"I recommend *{recommendation['asset'].replace('_otc', '')}* "
                    f"{recommendation['signal']} at {recommendation['timeframe']} expiration"
                )

                if update.callback_query:
                    await context.bot.send_message(
                        chat_id=update.effective_chat.id,
                        text=rec_message,
                        parse_mode=ParseMode.MARKDOWN
                    )
                else:
                    await update.message.reply_text(
                        text=rec_message,
                        parse_mode=ParseMode.MARKDOWN
                    )

        except Exception as e:
            logger.error(f"Error showing menu: {str(e)}")
            await self._send_error_message(update, context)

    def _create_pair_category_buttons(self) -> List[List[InlineKeyboardButton]]:
        """Create buttons for currency pair categories."""
        buttons = []
        for category, pairs in config.PAIR_CATEGORIES.items():
            if pairs:  # Only show categories with pairs
                emoji = self._get_category_emoji(category)
                buttons.append([InlineKeyboardButton(
                    f"{emoji} {category}",
                    callback_data=f"category_{category.lower().replace(' ', '_')}"
                )])
        return buttons

    def _create_timeframe_buttons(self) -> List[InlineKeyboardButton]:
        """Create timeframe selection buttons."""
        buttons = []
        for tf in config.TIMEFRAMES.keys():  # Use timeframes from config
            buttons.append(InlineKeyboardButton(
                tf.upper(),
                callback_data=f"tf_{tf}"
            ))
        return buttons

    def _get_category_emoji(self, category: str) -> str:
        """Get emoji for currency pair category."""
        emoji_map = {
            'USD Pairs': '💵',
            'EUR Pairs': '💶',
            'GBP Pairs': '💷',
            'JPY Pairs': '💴',
            'CHF Pairs': '🏦',
            'CAD Pairs': '🍁',
            'AUD Pairs': '🦘',
            'NZD Pairs': '🥝'
        }
        return emoji_map.get(category, '💱')

    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks."""
        try:
            query = update.callback_query
            await query.answer()

            if query.data == "generate_signal":
                await self._generate_signal(update, context)
            elif query.data == "select_pair":
                await self._show_pair_selection(update, context)
            elif query.data == "select_timeframe":
                await self._show_timeframe_selection(update, context)
            elif query.data == "view_analysis":
                await self._show_detailed_analysis(update, context)
            elif query.data == "view_chart":
                await self._show_chart(update, context)
            elif query.data == "show_performance":
                await self._show_performance_stats(update, context)
            elif query.data == "view_performance_chart":
                await self._show_performance_chart(update, context)
            elif query.data == "refresh_stats":
                await self._show_performance_stats(update, context)
            elif query.data.startswith("category_"):
                category = query.data.replace("category_", "").replace("_", " ").title()
                await self._show_category_pairs(update, context, category)
            elif query.data.startswith("pair_"):
                pair = query.data.replace("pair_", "")
                await self._set_pair(update, context, pair)
            elif query.data.startswith("tf_"):
                timeframe = query.data.replace("tf_", "")
                await self._set_timeframe(update, context, timeframe)
            elif query.data == "back_to_menu":
                await self.show_menu(update, context)
            elif query.data == "view_analysis":
                await self._show_detailed_analysis(update, context)
            elif query.data == "view_chart":
                await self._show_chart(update, context)
            elif query.data == "view_performance_stats":
                await self._show_performance_stats(update, context)
            elif query.data == "view_performance_chart":
                await self._show_performance_chart(update, context)

        except Exception as e:
            logger.error(f"Error handling callback: {str(e)}")
            await self._send_error_message(update, context)

    async def _generate_signal(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send a trading signal."""
        try:
            user_id = update.effective_user.id
            settings = self.user_settings.get(user_id, {
                'pair': 'EURUSD_otc',
                'timeframe': '1m'
            })

            # Get market data for multiple timeframes
            timeframes = [settings['timeframe']]
            if settings['timeframe'] != '1m':
                timeframes.append('1m')  # Always include 1m for better analysis
            if settings['timeframe'] != '5m':
                timeframes.append('5m')  # Include 5m for trend confirmation

            market_data = fetch_market_data([settings['pair']], timeframes)

            # Generate enhanced signal
            signal = self.analyzer.analyze_market(market_data)

            if not signal:
                await update.callback_query.edit_message_text(
                    text="No valid signals found at this time. Market conditions may be unfavorable.",
                    reply_markup=InlineKeyboardMarkup([[
                        InlineKeyboardButton("🔄 REFRESH", callback_data="generate_signal"),
                        InlineKeyboardButton("⬅️ BACK", callback_data="back_to_menu")
                    ]]),
                    parse_mode=ParseMode.MARKDOWN
                )
                return

            # Format enhanced signal message
            message = (
                f"🎯 *{settings['pair'].replace('_otc', '')} ({settings['timeframe']})*\n\n"
                f"*Signal: {signal.direction}*\n"
                f"*Signal Details:*\n"
                f"• Confidence: {signal.confidence}%\n"
                f"• Signal Strength: {signal.strength.value}\n"
                f"• Market Regime: {signal.metadata.regime}\n"
                f"• Entry Price: {signal.entry_price:.5f}\n"
                f"• Stop Loss: {signal.stop_loss:.5f}\n"
                f"• Take Profit: {signal.take_profit:.5f}\n\n"
                f"*Analysis Metrics:*\n"
                f"• Regime Confidence: {signal.metadata.regime_confidence:.1f}%\n"
                f"• Model Agreement: {signal.metadata.ensemble_agreement:.1f}%\n"
                f"• Uncertainty: {signal.metadata.ml_uncertainty:.1f}%\n"
                f"Time: {signal.timestamp.strftime('%H:%M:%S')}\n\n"
                "⚠️ *Risk Warning*: Trading involves risk. Never trade more than you can afford to lose."
            )

            # Add buttons based on signal strength
            keyboard = []

            # Signal action buttons
            if signal.strength in [SignalStrength.STRONG, SignalStrength.MODERATE]:
                keyboard.append([
                    InlineKeyboardButton("📊 VIEW ANALYSIS", callback_data="view_analysis"),
                    InlineKeyboardButton("📈 CHART", callback_data="view_chart")
                ])

            # Standard navigation buttons
            keyboard.append([
                InlineKeyboardButton("🔄 REFRESH", callback_data="generate_signal"),
                InlineKeyboardButton("⬅️ BACK", callback_data="back_to_menu")
            ])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error generating signal: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_pair_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show pair selection menu."""
        try:
            message = (
                "💱 *Select Trading Pair*\n\n"
                "Choose a currency pair to trade:"
            )

            # Create buttons for pair categories
            keyboard = []
            for category, pairs in config.PAIR_CATEGORIES.items():
                if pairs:  # Only show categories with pairs
                    emoji = self._get_category_emoji(category)
                    keyboard.append([InlineKeyboardButton(
                        f"{emoji} {category}",
                        callback_data=f"category_{category.lower().replace(' ', '_')}"
                    )])

            # Add back button
            keyboard.append([InlineKeyboardButton("⬅️ BACK", callback_data="back_to_menu")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error showing pair selection: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_timeframe_selection(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show timeframe selection menu."""
        try:
            message = (
                "⏱ *Select Timeframe*\n\n"
                "Choose your preferred trading timeframe:"
            )

            # Create buttons for timeframes
            keyboard = []
            timeframes = list(config.TIMEFRAMES.keys())

            # Create rows of 3 buttons each
            for i in range(0, len(timeframes), 3):
                row = []
                for j in range(i, min(i + 3, len(timeframes))):
                    tf = timeframes[j]
                    row.append(InlineKeyboardButton(
                        tf.upper(),
                        callback_data=f"tf_{tf}"
                    ))
                keyboard.append(row)

            # Add back button
            keyboard.append([InlineKeyboardButton("⬅️ BACK", callback_data="back_to_menu")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error showing timeframe selection: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_category_pairs(self, update: Update, context: ContextTypes.DEFAULT_TYPE, category: str):
        """Show pairs for selected category."""
        try:
            pairs = config.PAIR_CATEGORIES.get(category, [])
            if not pairs:
                await self._send_error_message(update, context)
                return

            # Create buttons for each pair
            keyboard = []
            for pair in pairs:
                display_pair = pair.replace('_otc', '')
                keyboard.append([InlineKeyboardButton(
                    display_pair,
                    callback_data=f"pair_{pair}"
                )])

            # Add back button
            keyboard.append([InlineKeyboardButton("⬅️ BACK", callback_data="back_to_menu")])

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.callback_query.edit_message_text(
                text=f"*Select {category}:*",
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error showing category pairs: {str(e)}")
            await self._send_error_message(update, context)

    async def _set_pair(self, update: Update, context: ContextTypes.DEFAULT_TYPE, pair: str):
        """Set the selected trading pair."""
        try:
            user_id = update.effective_user.id
            if user_id not in self.user_settings:
                self.user_settings[user_id] = {}
            self.user_settings[user_id]['pair'] = pair

            # Show updated menu
            await self.show_menu(update, context)

        except Exception as e:
            logger.error(f"Error setting pair: {str(e)}")
            await self._send_error_message(update, context)

    async def _set_timeframe(self, update: Update, context: ContextTypes.DEFAULT_TYPE, timeframe: str):
        """Set the selected timeframe."""
        try:
            user_id = update.effective_user.id
            if user_id not in self.user_settings:
                self.user_settings[user_id] = {}
            self.user_settings[user_id]['timeframe'] = timeframe

            # Show updated menu
            await self.show_menu(update, context)

        except Exception as e:
            logger.error(f"Error setting timeframe: {str(e)}")
            await self._send_error_message(update, context)

    async def _send_error_message(self, update: Update, _context: ContextTypes.DEFAULT_TYPE):
        """Send error message to user."""
        message = (
            "❌ *Error*\n\n"
            "Sorry, an error occurred. Please try again later or contact support."
        )

        if update.callback_query:
            await update.callback_query.edit_message_text(
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )
        else:
            await update.message.reply_text(
                text=message,
                parse_mode=ParseMode.MARKDOWN
            )

    async def _show_detailed_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed market analysis and signal reasoning."""
        try:
            user_id = update.effective_user.id
            settings = self.user_settings.get(user_id, {
                'pair': 'EURUSD_otc',
                'timeframe': '1m'
            })

            # Get market data
            timeframes = [settings['timeframe'], '1m', '5m']
            market_data = fetch_market_data([settings['pair']], timeframes)

            # Get latest signal and analysis
            signal = self.analyzer.analyze_market(market_data)
            if not signal:
                await update.callback_query.answer("No recent signal data available")
                return

            # Create detailed analysis message
            message = (
                f"📊 *Detailed Analysis for {settings['pair'].replace('_otc', '')}*\n\n"
                f"*Market Regime Analysis:*\n"
                f"• Current Regime: {signal.metadata.regime}\n"
                f"• Regime Confidence: {signal.metadata.regime_confidence:.1f}%\n\n"

                f"*Technical Analysis:*\n"
                f"• Analysis Status: Complete\n"
                f"• Signal Quality: Enhanced\n"
                f"• Processing: Real-time\n\n"

                f"*ML Model Analysis:*\n"
                f"• Signal Confidence: {signal.confidence:.1f}%\n"
                f"• Model Agreement: {signal.metadata.ensemble_agreement:.1f}%\n"
                f"• Prediction Uncertainty: {signal.metadata.ml_uncertainty:.1f}%\n\n"

                f"*Risk Management:*\n"
                f"• Recommended Position Size: {'Large' if signal.strength == SignalStrength.STRONG else 'Medium' if signal.strength == SignalStrength.MODERATE else 'Small'}\n"
                f"• Stop Loss: {abs((signal.stop_loss - signal.entry_price) / signal.entry_price * 100):.2f}%\n"
                f"• Take Profit: {abs((signal.take_profit - signal.entry_price) / signal.entry_price * 100):.2f}%\n\n"

                f"*Validation Results:*\n"
                f"• {'✅' if signal.metadata.validation_score > 0 else '⚠️'} Signal Quality Score: {signal.metadata.validation_score}\n"
            )

            # Add validation reasons
            for reason in signal.metadata.validation_reasons:
                message += f"• {reason}\n"

            message += "\n⚠️ *Risk Warning*: This analysis is for informational purposes only. Always manage your risk appropriately."

            # Create keyboard with chart options
            keyboard = [
                [
                    InlineKeyboardButton("📈 Price Chart", callback_data="view_chart"),
                    InlineKeyboardButton("📊 Indicators", callback_data="view_indicators")
                ],
                [
                    InlineKeyboardButton("🔄 Refresh", callback_data="generate_signal"),
                    InlineKeyboardButton("⬅️ Back", callback_data="back_to_menu")
                ]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.callback_query.edit_message_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error showing detailed analysis: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send a technical analysis chart."""
        try:
            user_id = update.effective_user.id
            settings = self.user_settings.get(user_id, {
                'pair': 'EURUSD_otc',
                'timeframe': '1m'
            })

            # Get market data
            market_data = fetch_market_data([settings['pair']], [settings['timeframe']])
            df = market_data[settings['timeframe']]

            # Create figure with secondary y-axis
            fig = go.Figure()

            # Add candlestick chart
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ))

            # Add EMA lines
            ema20 = TechnicalAnalyzer.calculate_ema(df['close'], 20)
            ema50 = TechnicalAnalyzer.calculate_ema(df['close'], 50)

            fig.add_trace(go.Scatter(
                x=df.index,
                y=ema20,
                line=dict(color='blue'),
                name='EMA 20'
            ))

            fig.add_trace(go.Scatter(
                x=df.index,
                y=ema50,
                line=dict(color='red'),
                name='EMA 50'
            ))

            # Update layout
            fig.update_layout(
                title=f'{settings["pair"].replace("_otc", "")} - {settings["timeframe"]} Timeframe',
                yaxis_title='Price',
                template='plotly_dark',
                xaxis_rangeslider_visible=False
            )

            # Save to BytesIO
            img_stream = BytesIO()
            fig.write_image(img_stream, format='png')
            img_stream.seek(0)

            # Send chart
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_stream,
                caption=f"Technical Analysis Chart for {settings['pair'].replace('_otc', '')}"
            )

        except Exception as e:
            logger.error(f"Error generating chart: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_performance_stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show detailed performance statistics."""
        try:
            # Get performance metrics
            metrics = self.analyzer.get_performance_metrics()

            # Format message
            message = (
                "📊 *Performance Statistics*\n\n"

                "*Signal Summary:*\n"
                f"• Total Signals: {metrics['total_signals']}\n"
                f"• Overall Accuracy: {metrics['accuracy']:.1f}%\n\n"

                "*Market Regimes:*\n"
            )

            # Add regime distribution
            if 'signals_by_regime' in metrics:
                total_signals = sum(metrics['signals_by_regime'].values())
                for regime, count in metrics['signals_by_regime'].items():
                    percentage = (count / total_signals * 100) if total_signals > 0 else 0
                    message += f"• {regime}: {count} ({percentage:.1f}%)\n"

            message += (
                f"\n*Signal Quality:*\n"
                f"• Average Confidence: {metrics.get('average_confidence', 0):.1f}%\n"
                f"• Average Uncertainty: {metrics.get('average_uncertainty', 0):.1f}%\n\n"

                f"*System Updates:*\n"
                f"• Regime Changes: {metrics['regime_changes']}\n"
                f"• Model Updates: {metrics['model_updates']}\n"
            )

            # Add keyboard
            keyboard = [
                [
                    InlineKeyboardButton("📈 Performance Chart", callback_data="view_performance_chart"),
                    InlineKeyboardButton("🔄 Refresh", callback_data="refresh_stats")
                ],
                [InlineKeyboardButton("⬅️ Back to Menu", callback_data="back_to_menu")]
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
            logger.error(f"Error showing performance stats: {str(e)}")
            await self._send_error_message(update, context)

    async def _show_performance_chart(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Generate and send comprehensive performance visualization charts."""
        try:
            # Get signal history
            with open('signal_history.json', 'r') as f:
                history = json.load(f)

            signals_df = pd.DataFrame(history['signals'])
            signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'])

            # Create figure with multiple subplots
            fig = plt.figure(figsize=(12, 15))
            gs = fig.add_gridspec(4, 2)

            # 1. Signal Confidence Distribution (Top Left)
            ax1 = fig.add_subplot(gs[0, 0])
            if len(signals_df) > 0:
                signals_df['confidence'].hist(ax=ax1, bins=20, color='skyblue', alpha=0.7)
                ax1.set_title('Signal Confidence Distribution')
                ax1.set_xlabel('Confidence Level (%)')
                ax1.set_ylabel('Number of Signals')
                ax1.grid(True, alpha=0.3)

            # 2. Regime Distribution (Top Right)
            ax2 = fig.add_subplot(gs[0, 1])
            if 'regime' in signals_df.columns:
                regime_counts = signals_df['regime'].value_counts()
                ax2.pie(regime_counts.values,
                       labels=regime_counts.index,
                       autopct='%1.1f%%',
                       startangle=90,
                       colors=plt.cm.Pastel1(np.linspace(0, 1, len(regime_counts))))
                ax2.set_title('Market Regime Distribution')

            # 3. Signal Performance Over Time (Middle Left)
            ax3 = fig.add_subplot(gs[1, 0])
            if 'accuracy' in signals_df.columns:
                signals_df['accuracy'].rolling(window=20).mean().plot(ax=ax3, color='green', alpha=0.7)
                ax3.set_title('Signal Accuracy (20-period MA)')
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Accuracy (%)')
                ax3.grid(True, alpha=0.3)

            # 4. Win Rate by Market Regime (Middle Right)
            ax4 = fig.add_subplot(gs[1, 1])
            if 'regime' in signals_df.columns and 'success' in signals_df.columns:
                win_rates = signals_df.groupby('regime')['success'].mean() * 100
                win_rates.plot(kind='bar', ax=ax4, color='orange', alpha=0.7)
                ax4.set_title('Win Rate by Market Regime')
                ax4.set_xlabel('Market Regime')
                ax4.set_ylabel('Win Rate (%)')
                ax4.grid(True, alpha=0.3)

            # 5. Signal Distribution by Hour (Bottom Left)
            ax5 = fig.add_subplot(gs[2, 0])
            signals_df['hour'] = signals_df['timestamp'].dt.hour
            signals_df['hour'].value_counts().sort_index().plot(kind='bar', ax=ax5, color='purple', alpha=0.7)
            ax5.set_title('Signal Distribution by Hour')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Number of Signals')
            ax5.grid(True, alpha=0.3)

            # 6. Model Performance Metrics (Bottom Right)
            ax6 = fig.add_subplot(gs[2, 1])
            if 'model_confidence' in signals_df.columns and 'model_uncertainty' in signals_df.columns:
                signals_df[['model_confidence', 'model_uncertainty']].boxplot(ax=ax6)
                ax6.set_title('Model Performance Metrics')
                ax6.set_ylabel('Value (%)')
                ax6.grid(True, alpha=0.3)

            # 7. Strategy Performance Comparison (Bottom)
            ax7 = fig.add_subplot(gs[3, :])
            if 'strategy' in signals_df.columns and 'success' in signals_df.columns:
                strategy_performance = signals_df.groupby('strategy').agg({
                    'success': ['mean', 'count']
                })
                strategy_performance['success']['mean'].plot(kind='bar', ax=ax7, color='teal', alpha=0.7)
                ax7.set_title('Strategy Performance Comparison')
                ax7.set_xlabel('Strategy')
                ax7.set_ylabel('Success Rate (%)')
                ax7.grid(True, alpha=0.3)

            plt.tight_layout()

            # Save to BytesIO
            img_stream = BytesIO()
            plt.savefig(img_stream, format='png', dpi=300, bbox_inches='tight')
            img_stream.seek(0)
            plt.close()

            # Create caption with summary statistics
            total_signals = len(signals_df)
            avg_confidence = signals_df['confidence'].mean() if 'confidence' in signals_df.columns else 0
            overall_accuracy = (signals_df['success'].mean() * 100) if 'success' in signals_df.columns else 0

            caption = (
                "📊 Performance Analysis Summary\n"
                f"Total Signals: {total_signals}\n"
                f"Average Confidence: {avg_confidence:.1f}%\n"
                f"Overall Accuracy: {overall_accuracy:.1f}%"
            )

            # Send chart
            await context.bot.send_photo(
                chat_id=update.effective_chat.id,
                photo=img_stream,
                caption=caption
            )

        except Exception as e:
            logger.error(f"Error generating performance chart: {str(e)}")
            await self._send_error_message(update, context)

    async def show_recommendation(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show trading recommendation command handler."""
        try:
            user_id = update.effective_user.id
            settings = self.user_settings.get(user_id, {
                'pair': 'EURUSD_otc',
                'timeframe': '1m'
            })

            # Get market data
            market_data = fetch_market_data([settings['pair']], [settings['timeframe']])

            # Get recommendation
            recommendation = self.signal_recommender.get_recommendation(market_data)

            # Format recommendation message
            message = (
                f"🎯 *Trading Recommendation*\n\n"
                f"*Asset:* {recommendation['asset'].replace('_otc', '')}\n"
                f"*Signal:* {recommendation['signal']}\n"
                f"*Timeframe:* {recommendation['timeframe']}\n"
                f"*Confidence:* {recommendation['confidence']:.1f}%\n\n"
                f"*Analysis:* {recommendation.get('explanation', 'Based on current market conditions')}\n\n"
                "⚠️ *Risk Warning*: This is a recommendation only. Always do your own analysis."
            )

            # Create keyboard
            keyboard = [
                [InlineKeyboardButton("🎯 GENERATE SIGNAL", callback_data="generate_signal")],
                [InlineKeyboardButton("⬅️ BACK TO MENU", callback_data="back_to_menu")]
            ]

            reply_markup = InlineKeyboardMarkup(keyboard)

            await update.message.reply_text(
                text=message,
                reply_markup=reply_markup,
                parse_mode=ParseMode.MARKDOWN
            )

        except Exception as e:
            logger.error(f"Error showing recommendation: {str(e)}")
            await self._send_error_message(update, context)



    def run(self):
        """Start the bot."""
        self.application.run_polling()