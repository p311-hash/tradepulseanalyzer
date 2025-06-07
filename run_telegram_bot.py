#!/usr/bin/env python3
"""
MasterTrade Bot - Professional Binary Options Trading Signal Bot
Enhanced with candlestick patterns, trading modes, and Pocket Option integration.
"""

import os
import logging
from datetime import datetime
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ContextTypes

# Enhanced feature imports
from signal_analytics import SignalAnalytics
from enhanced_ml_model import EnhancedMLPredictor
from deep_market_structure import DeepMarketStructureAnalyzer
from market_regime import MarketRegimeDetector
from signal_correlation import SignalCorrelationAnalyzer
from enhanced_signal_recommendation import EnhancedSignalRecommender  # Add enhanced signal recommender
from risk_manager import RiskManager
from continuous_learning import ContinuousLearningSystem
from pattern_continuous_learning import PatternContinuousLearning
from pattern_recognition import EnhancedPatternRecognizer, EnhancedPatternValidation
from market_microstructure import MarketMicrostructureAnalyzer
from feature_engineering import FeatureEngineer

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
signal_recommender = None  # Enhanced signal recommender

# Initialize feature availability tracking
ENHANCED_FEATURES = {
    'market_regime': True,
    'deep_market': True,
    'pattern_recognition': True,
    'correlation': True,
    'continuous_learning': True,
    'risk_management': True,
    'microstructure': True
}

# Initialize enhanced components with fallback handling
try:
    # Create default empty DataFrame for initialization
    import pandas as pd
    default_data = pd.DataFrame({'open': [], 'high': [], 'low': [], 'close': [], 'volume': []})
    
    market_regime_detector = MarketRegimeDetector(data=default_data)
except Exception as e:
    logger.error(f"Failed to initialize market regime detector: {str(e)}")
    ENHANCED_FEATURES['market_regime'] = False
    market_regime_detector = None

try:
    # Use same default data for pattern recognizer
    pattern_recognizer = EnhancedPatternRecognizer(data=default_data)
except Exception as e:
    logger.error(f"Failed to initialize pattern recognizer: {str(e)}")
    ENHANCED_FEATURES['pattern_recognition'] = False
    pattern_recognizer = None

try:
    signal_correlation_analyzer = SignalCorrelationAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize correlation analyzer: {str(e)}")
    ENHANCED_FEATURES['correlation'] = False
    signal_correlation_analyzer = None

try:
    deep_market_analyzer = DeepMarketStructureAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize deep market analyzer: {str(e)}")
    ENHANCED_FEATURES['deep_market'] = False
    deep_market_analyzer = None

try:
    market_microstructure = MarketMicrostructureAnalyzer()
except Exception as e:
    logger.error(f"Failed to initialize market microstructure analyzer: {str(e)}")
    ENHANCED_FEATURES['microstructure'] = False
    market_microstructure = None

try:
    risk_manager = RiskManager(initial_capital=100000)
except Exception as e:
    logger.error(f"Failed to initialize risk manager: {str(e)}")
    ENHANCED_FEATURES['risk_management'] = False
    risk_manager = None

# Initialize continuous learning system
try:
    # Create required directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    continuous_learner = ContinuousLearningSystem(
        model_path="models/latest_model",
        history_path="data/signal_history.json",
        feedback_path="data/feedback.json",
        performance_path="data/model_performance.json"
    )
    
    continuous_learner.load_data()  # Load existing data
    continuous_learner.start_learning_thread()  # Start background learning thread
    logger.info("Continuous learning system initialized and started")
    
except Exception as e:
    logger.error(f"Failed to initialize continuous learning system: {str(e)}")
    ENHANCED_FEATURES['continuous_learning'] = False
    continuous_learner = None

def get_available_features() -> str:
    """Get formatted string of available enhanced features."""
    available = []
    disabled = []
    
    feature_names = {
        'market_regime': 'Market Regime Detection',
        'deep_market': 'Deep Market Analysis',
        'pattern_recognition': 'Pattern Recognition',
        'correlation': 'Correlation Analysis',
        'continuous_learning': 'Continuous Learning',
        'risk_management': 'Risk Management',
        'microstructure': 'Market Microstructure'
    }
    
    for key, value in ENHANCED_FEATURES.items():
        if value:
            available.append(feature_names[key])
        else:
            disabled.append(feature_names[key])
            
    message = "✅ *Available Enhanced Features:*\n"
    message += "\n".join(f"• {feature}" for feature in available)
    
    if disabled:
        message += "\n\n❌ *Temporarily Disabled Features:*\n"
        message += "\n".join(f"• {feature}" for feature in disabled)
        
    return message

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send welcome message with TradeMind Bot branding."""
    user_id = update.effective_user.id
    user_name = update.effective_user.first_name or "Trader"

    # Initialize user settings with enhanced features
    telegram_handler.user_settings[user_id] = {
        'pair': 'EURUSD_otc',
        'timeframe': '2m',
        'category': 'Currencies',
        'trading_mode': 'demo',  # Default to demo mode for safety
        'risk_level': 'moderate',  # Default risk level
        'auto_ml': True,  # Enable automated ML predictions
        'pattern_alerts': True,  # Enable pattern recognition alerts
        'correlation_analysis': True,  # Enable correlation analysis
        'regime_detection': True  # Enable market regime detection
    }

    welcome_message = (
        "🤖 *Welcome to MasterTrade Bot | Professional Trading*\n\n"
        f"Hello {user_name}! I'm your advanced binary options trading assistant.\n\n"
        "🎯 *Enhanced Features:*\n"
        "• AI-Powered Signal Generation\n"
        "• Market Regime Detection\n"
        "• Deep Market Structure Analysis\n"
        "• Pattern Recognition (15+ patterns)\n"
        "• Correlation Analysis\n"
        "• Risk Management System\n"
        "• Continuous Learning\n"
        "• Multiple Timeframes\n"
        "• Demo/Real Trading Modes\n\n"
        "🧠 *AI & Analysis:*\n"
        "• Real-time market regime detection\n"
        "• Deep learning price prediction\n"
        "• Advanced pattern recognition\n"
        "• Market microstructure analysis\n"
        "• Signal correlation validation\n\n"
        "⚙️ *Risk Management:*\n"
        "• Adaptive position sizing\n"
        "• Market regime based risk control\n"
        "• Multi-level validation system\n"
        "• Performance tracking\n\n"
        "🎮 *Trading Modes:*\n"
        "• Demo Mode - Safe practice environment\n"
        "• Real Mode - Live trading (with risk management)\n\n"
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

async def settings_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle settings command to customize enhanced features."""
    try:
        user_id = update.effective_user.id
        settings = telegram_handler.user_settings.get(user_id, {})

        settings_message = (
            "⚙️ *TradeMind Bot Settings*\n\n"
            "*Current Configuration:*\n"
            f"• Trading Mode: {'Demo' if settings.get('trading_mode') == 'demo' else 'Real'}\n"
            f"• Risk Level: {settings.get('risk_level', 'moderate').title()}\n"
            f"• Auto ML: {'Enabled' if settings.get('auto_ml', True) else 'Disabled'}\n"
            f"• Pattern Alerts: {'Enabled' if settings.get('pattern_alerts', True) else 'Disabled'}\n"
            f"• Correlation Analysis: {'Enabled' if settings.get('correlation_analysis', True) else 'Disabled'}\n"
            f"• Regime Detection: {'Enabled' if settings.get('regime_detection', True) else 'Disabled'}\n\n"
            "*Risk Management:*\n"
            f"• Max Position Size: {settings.get('max_position_size', '5%')} of balance\n"
            f"• Stop Loss: {settings.get('stop_loss_pct', '2%')} per trade\n"
            f"• Take Profit: {settings.get('take_profit_pct', '4%')} per trade\n\n"
            "Use the buttons below to modify settings:"
        )
        
        from telegram import InlineKeyboardButton, InlineKeyboardMarkup

        keyboard = [
            [
                InlineKeyboardButton("🔄 Trading Mode", callback_data="settings_mode"),
                InlineKeyboardButton("⚠️ Risk Level", callback_data="settings_risk")
            ],
            [
                InlineKeyboardButton("🧠 AI Features", callback_data="settings_auto_ml"),
                InlineKeyboardButton("📊 Analysis", callback_data="settings_analysis")
            ],
            [
                InlineKeyboardButton("💰 Risk Management", callback_data="settings_risk_mgmt"),
                InlineKeyboardButton("📈 Performance", callback_data="settings_performance")
            ],
            [InlineKeyboardButton("🔙 Back to Menu", callback_data="back_to_menu")]
        ]
        
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        await update.message.reply_text(
            settings_message,
            reply_markup=reply_markup,
            parse_mode='Markdown'
        )

    except Exception as e:
        logger.error(f"Error showing settings: {str(e)}")
        await update.message.reply_text(
            "❌ Error accessing settings. Please try again later."
        )

async def signal_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Generate enhanced trading signal with comprehensive analysis."""
    try:
        user_id = update.effective_user.id
        settings = telegram_handler.user_settings.get(user_id, {})

        # Get market data
        market_data = telegram_handler.get_market_data(settings['pair'], settings['timeframe'])

        # Initialize signal components
        signal_components = {
            'regime_info': None,
            'market_structure': None,
            'micro_analysis': None,
            'patterns': None,
            'correlation_data': None,
            'risk_assessment': None,
            'signal_recommendation': None  # Add signal recommendation component
        }

        # 1. Market Regime Detection
        if ENHANCED_FEATURES['market_regime'] and market_regime_detector:
            try:
                market_regime = market_regime_detector.detect_regime(market_data)
                signal_components['regime_info'] = market_regime
                
                # Validate market conditions based on regime
                if market_regime['regime'] == 'VOLATILE':
                    logger.warning("Market conditions are highly volatile, adjusting risk parameters")
                    risk_manager.adjust_risk_parameters(volatility_multiplier=0.7)  # Reduce position size in volatile markets
                elif market_regime['regime'] == 'TRENDING':
                    risk_manager.adjust_risk_parameters(volatility_multiplier=1.2)  # Increase position size in trending markets
                
            except Exception as e:
                logger.error(f"Market regime detection failed: {str(e)}")
        
        # 2. Deep Market Structure Analysis with real-time validation
        if ENHANCED_FEATURES['deep_market'] and deep_market_analyzer:
            try:
                market_structure = deep_market_analyzer.analyze_structure(market_data)
                signal_components['market_structure'] = market_structure
                
                # Validate current market structure against historical patterns
                structure_match_score = deep_market_analyzer.validate_structure(market_structure)
                if structure_match_score < 0.6:  # Less than 60% match with known patterns
                    logger.warning("Current market structure shows unusual patterns")
                    signal_components['warnings'] = signal_components.get('warnings', []) + ["Unusual market structure detected"]
            except Exception as e:
                logger.error(f"Deep market analysis failed: {str(e)}")
        
        # 3. Microstructure Analysis
        if ENHANCED_FEATURES['microstructure'] and market_microstructure:
            try:
                signal_components['micro_analysis'] = market_microstructure.analyze(market_data)
            except Exception as e:
                logger.error(f"Microstructure analysis failed: {str(e)}")
        
        # 4. Pattern Recognition with enhanced validation and continuous learning
        if ENHANCED_FEATURES['pattern_recognition'] and pattern_recognizer:
            try:
                # Initialize enhanced pattern validation
                pattern_validator = EnhancedPatternValidation()
                pattern_learner = PatternContinuousLearning()
                
                # Detect patterns
                patterns = pattern_recognizer.find_patterns(market_data)
                signal_components['patterns'] = patterns

                # Enhanced pattern validation
                pattern_reliability = pattern_validator.validate_pattern_reliability(
                    patterns, 
                    market_regime=signal_components.get('regime_info')
                )
                signal_components['pattern_reliability'] = pattern_reliability
                
                # Market context analysis for patterns
                market_context = pattern_validator.analyze_pattern_context(patterns, market_data)
                signal_components['pattern_context'] = market_context
                
                # Check for pattern conflicts
                conflicting_patterns = pattern_validator.check_pattern_conflicts(patterns)
                if conflicting_patterns:
                    logger.warning("Detected conflicting patterns")
                    signal_components['warnings'] = signal_components.get('warnings', []) + [
                        f"Conflicting patterns detected: {', '.join(conflicting_patterns)}"
                    ]
                
                # Enhanced pattern completion analysis
                incomplete_patterns = pattern_validator.check_pattern_completion(patterns)
                if incomplete_patterns:
                    signal_components['pattern_status'] = 'INCOMPLETE'
                    signal_components['warnings'] = signal_components.get('warnings', []) + [
                        f"Incomplete patterns detected: {', '.join(incomplete_patterns)}"
                    ]
                
                # Get historical pattern performance
                if patterns:
                    for pattern_name in patterns:
                        performance = pattern_learner.get_pattern_performance(pattern_name, '7d')
                        optimal_conditions = pattern_learner.get_optimal_conditions(pattern_name)
                        
                        # Add performance context to signal components
                        if 'pattern_performance' not in signal_components:
                            signal_components['pattern_performance'] = {}
                            
                        signal_components['pattern_performance'][pattern_name] = {
                            'historical_performance': performance,
                            'optimal_conditions': optimal_conditions
                        }
                        
                        # Adjust reliability based on historical performance
                        if performance['reliability_score'] < 0.5:
                            signal_components['warnings'] = signal_components.get('warnings', []) + [
                                f"Pattern {pattern_name} has low historical reliability ({performance['reliability_score']:.2f})"
                            ]

            except Exception as e:
                logger.error(f"Enhanced pattern recognition failed: {str(e)}")
        
        # 5. Signal Correlation Analysis with real-time validation
        if ENHANCED_FEATURES['correlation'] and signal_correlation_analyzer:
            try:
                correlations = signal_correlation_analyzer.analyze_correlations(market_data)
                signal_components['correlation_data'] = correlations

                # Validate correlation strength
                strong_correlations = signal_correlation_analyzer.get_strong_correlations(correlations)
                if strong_correlations:
                    signal_components['correlation_strength'] = 'STRONG'
                    signal_components['correlated_pairs'] = strong_correlations

                # Check for correlation breakdowns
                correlation_breakdowns = signal_correlation_analyzer.detect_correlation_breakdowns(correlations)
                if correlation_breakdowns:
                    logger.warning("Detected correlation breakdowns")
                    signal_components['warnings'] = signal_components.get('warnings', []) + [
                        f"Correlation breakdowns detected in pairs: {', '.join(correlation_breakdowns)}"
                    ]

                # Analyze correlation stability
                correlation_stability = signal_correlation_analyzer.analyze_correlation_stability(correlations)
                signal_components['correlation_stability'] = correlation_stability

            except Exception as e:
                logger.error(f"Correlation analysis failed: {str(e)}")
        
        # 6. Risk Assessment with dynamic adjustment
        if ENHANCED_FEATURES['risk_management'] and risk_manager:
            try:
                # Get base risk assessment
                risk_assessment = risk_manager.assess_trade(
                    pair=settings['pair'],
                    signal=ml_signal,
                    regime=signal_components['regime_info']['regime'] if signal_components['regime_info'] else 'unknown',
                    mode=settings['trading_mode']
                )
                
                # Apply dynamic risk adjustments based on market conditions
                risk_modifiers = []
                
                if signal_components.get('regime_info', {}).get('regime') == 'VOLATILE':
                    risk_modifiers.append(('volatility', 0.7))  # Reduce risk in volatile markets
                    
                if signal_components.get('pattern_reliability', 0) < 0.7:
                    risk_modifiers.append(('pattern_uncertainty', 0.8))  # Reduce risk when patterns are uncertain
                    
                if signal_components.get('correlation_stability', 1.0) < 0.8:
                    risk_modifiers.append(('correlation_instability', 0.9))  # Reduce risk when correlations are unstable
                
                # Calculate final risk assessment with modifiers
                final_risk_assessment = risk_manager.apply_risk_modifiers(risk_assessment, risk_modifiers)
                signal_components['risk_assessment'] = final_risk_assessment

            except Exception as e:
                logger.error(f"Risk assessment failed: {str(e)}")

        # 7. Generate ML-based prediction
        ml_signal = signal_analytics.generate_ml_signal(
            market_data, 
            signal_components['regime_info'],
            signal_components['market_structure'],
            signal_components['micro_analysis']
        )

        # 8. Final Signal Validation and Continuous Learning Integration
        validation_failures = []
        signal_recommendation = None

        try:
            # Get base signal recommendation
            signal_recommendation = signal_recommender.get_recommendation(
                market_data=market_data,
                ml_signal=ml_signal,
                regime_info=signal_components['regime_info'],
                patterns=signal_components['patterns'],
                correlations=signal_components['correlation_data'],
                risk_assessment=signal_components['risk_assessment']
            )
            
            # Enhanced validation with pattern performance
            if signal_components.get('pattern_performance'):
                pattern_confidence = 0.0
                pattern_count = 0
                
                for pattern_name, perf_data in signal_components['pattern_performance'].items():
                    historical_perf = perf_data['historical_performance']
                    optimal_conditions = perf_data['optimal_conditions']
                    
                    # Check if current conditions match optimal conditions
                    current_regime = signal_components['regime_info']['regime'] if signal_components.get('regime_info') else 'UNKNOWN'
                    conditions_match = (
                        current_regime == optimal_conditions.get('best_regime', current_regime) and
                        (not optimal_conditions.get('volume_confirmation_important', False) or 
                         signal_components['pattern_context']['volume_confirmed']) and
                        (not optimal_conditions.get('trend_alignment_important', False) or 
                         signal_components['pattern_context']['trend_aligned'])
                    )
                    
                    if conditions_match:
                        pattern_confidence += historical_perf['reliability_score']
                        pattern_count += 1
                    else:
                        validation_failures.append(f"Suboptimal conditions for {pattern_name}")
                
                if pattern_count > 0:
                    avg_pattern_confidence = pattern_confidence / pattern_count
                    # Adjust final signal confidence based on pattern confidence
                    signal_recommendation['confidence'] *= (0.7 + 0.3 * avg_pattern_confidence)

            # Record patterns for continuous learning
            if ENHANCED_FEATURES['continuous_learning'] and pattern_learner:
                pattern_learner.record_pattern(
                    patterns=signal_components['patterns'],
                    market_context={
                        'market_regime': signal_components.get('regime_info', {}),
                        'volume_confirmed': signal_components['pattern_context']['volume_confirmed'],
                        'trend_aligned': signal_components['pattern_context']['trend_aligned'],
                        'support_resistance_validated': signal_components['pattern_context']['support_resistance_validated']
                    },
                    signal_result='PENDING'  # Will be updated after trade completion
                )

        except Exception as e:
            logger.error(f"Signal recommendation generation failed: {str(e)}")
            signal_recommendation = {
                'direction': 'NEUTRAL',
                'confidence': 0,
                'explanation': f"Error generating recommendation: {str(e)}",
                'validation_failures': [f"Error: {str(e)}"]
            }

        # Store the recommendation
        signal_components['signal_recommendation'] = signal_recommendation
        signal_components['validation_failures'] = validation_failures

        # Generate final signal with metadata
        final_signal = signal_analytics.validate_signal(
            base_signal=ml_signal,
            patterns=signal_components['patterns'],
            correlations=signal_components['correlation_data'],
            risk_assessment=signal_components['risk_assessment'],
            regime_info=signal_components['regime_info'],
            recommendation=signal_components['signal_recommendation']
        )

        # Add comprehensive performance tracking metadata
        final_signal['metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'market_regime': signal_components['regime_info']['regime'] if signal_components['regime_info'] else 'unknown',
            'pattern_reliability': signal_components.get('pattern_reliability', 0),
            'pattern_context': signal_components.get('pattern_context', {}),
            'correlation_stability': signal_components.get('correlation_stability', 0),
            'validation_failures': validation_failures,
            'pattern_performance': signal_components.get('pattern_performance', {}),
            'market_conditions': {
                'volume_confirmed': signal_components['pattern_context']['volume_confirmed'],
                'trend_aligned': signal_components['pattern_context']['trend_aligned'],
                'support_resistance_validated': signal_components['pattern_context']['support_resistance_validated']
            }
        }

        # Generate and send the enhanced signal message
        await telegram_handler._generate_professional_signal(
            update,
            context,
            final_signal,
            signal_components['regime_info'],
            signal_components['patterns'],
            signal_components['risk_assessment'],
            signal_components['signal_recommendation']
        )

    except Exception as e:
        logger.error(f"Error generating signal: {str(e)}")
        await update.message.reply_text(
            "❌ Error generating signal. Some features may be temporarily unavailable.\n"
            "Basic signal generation will continue to function."
        )

async def menu_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Show the main menu."""
    await telegram_handler.show_main_menu(update, context)

async def handle_callback_query(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle callback queries from inline keyboards."""
    await telegram_handler.handle_callback(update, context)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors encountered by the telegram bot."""
    logger.error('Exception while handling an update:', exc_info=context.error)
    
    try:
        # Log the stack trace
        import traceback
        traceback.print_exception(None, context.error, context.error.__traceback__)

        if update and update.effective_message:
            error_message = "❌ Sorry, a network error occurred. The bot will automatically reconnect. Please try again in a moment."
            await update.effective_message.reply_text(error_message)
            
    except Exception as e:
        logger.error(f"Exception in error handler: {str(e)}")

def main() -> None:
    """Start the enhanced TradeMind Bot."""
    global telegram_handler, signal_recommender
    try:
        # Get bot token from environment
        token = os.getenv('TELEGRAM_TOKEN')
        if not token:
            logger.error("Telegram token not found in environment variables")
            logger.info("Please set TELEGRAM_TOKEN environment variable")
            return

        # Initialize enhanced components
        logger.info("Initializing enhanced features...")

        # Ensure required directories exist
        os.makedirs("models", exist_ok=True)
        os.makedirs("data", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        # Initialize feature engineering and ML model
        feature_engineer = FeatureEngineer()
        feature_names = feature_engineer.get_feature_names()
        
        # Initialize or load ML model
        predictor = EnhancedMLPredictor(
            input_size=len(feature_names),
            feature_names=feature_names
        )        # Initialize signal recommender
        signal_recommender = EnhancedSignalRecommender()
        
        # Initialize risk manager with default settings
        risk_manager.initialize_settings({
            'max_position_size_pct': 5.0,
            'stop_loss_pct': 2.0,
            'take_profit_pct': 4.0,
            'max_daily_trades': 10,
            'max_daily_loss_pct': 15.0,
            'risk_per_trade_pct': 2.0
        })

        # Initialize MasterTrade Bot handler with enhanced components
        from enhanced_telegram_handler import MasterTradeBotHandler
        telegram_handler = MasterTradeBotHandler(
            token=token,
            predictor=predictor,
            market_regime_detector=market_regime_detector,
            signal_correlation_analyzer=signal_correlation_analyzer,
            pattern_recognizer=pattern_recognizer,
            deep_market_analyzer=deep_market_analyzer,
            market_microstructure=market_microstructure,
            risk_manager=risk_manager,
            continuous_learner=continuous_learner,
            signal_recommender=signal_recommender  # Add signal recommender
        )

        # Create application with custom settings for better reliability
        from telegram.ext import Application, CommandHandler, CallbackQueryHandler
        application = (
            Application.builder()
            .token(token)
            .read_timeout(30)  # Increase read timeout
            .write_timeout(30)  # Increase write timeout
            .connect_timeout(30)  # Increase connection timeout
            .pool_timeout(30)  # Increase pool timeout
            .job_queue(None)  # Disable job queue
            .build()
        )

        # Add error handler
        application.add_error_handler(error_handler)        # Add command handlers with proper error handling
        handlers = [
            CommandHandler("start", start),
            CommandHandler("help", help_command),
            CommandHandler("signal", signal_command),
            CommandHandler("menu", menu_command),
            CommandHandler("settings", settings_command),
            # Admin handlers
            CommandHandler("admin_login", admin_login_command),
            CommandHandler("admin_logout", admin_logout_command),
            CommandHandler("admin_panel", admin_panel_command),
            CommandHandler("admin", admin_panel_command),  # Shortcut
            # Add callback query handler for button interactions
            CallbackQueryHandler(handle_callback_query)  # Use our async handler function
        ]
        
        # Add all handlers with error catching
        for handler in handlers:
            try:
                application.add_handler(handler)
            except Exception as e:
                logger.error(f"Error adding handler {handler.__class__.__name__}: {str(e)}")
                continue

        # Start continuous learning system if enabled
        if ENHANCED_FEATURES['continuous_learning'] and continuous_learner:
            continuous_learner.start_learning_thread()

        # Log startup information
        logger.info("\n" + "="*50)
        logger.info("🤖 Enhanced TradeMind Bot Starting")
        logger.info("="*50)
        logger.info("✅ Enhanced features initialized:")
        logger.info("   • AI-Powered Signal Generation")
        logger.info("   • Market Regime Detection")
        logger.info("   • Deep Market Structure Analysis")
        logger.info("   • Pattern Recognition")
        logger.info("   • Correlation Analysis")
        logger.info("   • Risk Management")
        logger.info("   • Continuous Learning")
        logger.info("   • Advanced Settings Management")
        logger.info("="*50)
        logger.info("Bot is ready to receive commands!")
        logger.info("="*50 + "\n")

        # Start polling with automatic reconnection
        application.run_polling(
            drop_pending_updates=True,  # Don't process old updates on startup
            allowed_updates=Update.ALL_TYPES,  # Allow all update types
            close_loop=False  # Keep the event loop running for automatic reconnection
        )

    except Exception as e:
        logger.error(f"Error starting bot: {str(e)}")
        import traceback
        traceback.print_exc()
        # Wait a bit before restarting
        import time
        time.sleep(5)
        # Try to restart the bot
        main()

if __name__ == '__main__':
    main()