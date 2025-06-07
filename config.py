import os
from datetime import timezone

# Telegram Bot Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', 'your-token-here')

# Admin Authentication Configuration
BOT_OWNER_ID = int(os.environ.get('BOT_OWNER_ID', '0'))  # Primary owner Telegram user ID
BOT_ADMINS = [
    # Add admin user IDs here (can also be set via environment variable)
    # Example: 123456789, 987654321
]

# Load additional admins from environment variable (comma-separated)
ADMIN_IDS_ENV = os.environ.get('BOT_ADMIN_IDS', '')
if ADMIN_IDS_ENV:
    try:
        additional_admins = [int(uid.strip()) for uid in ADMIN_IDS_ENV.split(',') if uid.strip()]
        BOT_ADMINS.extend(additional_admins)
    except ValueError:
        pass

# Admin authentication settings
ADMIN_PASSWORD = os.environ.get('ADMIN_PASSWORD', 'MasterTrade2024!')  # Fallback password
REQUIRE_PASSWORD_AUTH = os.environ.get('REQUIRE_PASSWORD_AUTH', 'true').lower() == 'true'
SESSION_TIMEOUT_HOURS = int(os.environ.get('SESSION_TIMEOUT_HOURS', '24'))  # Admin session timeout

# Permission levels
PERMISSION_LEVELS = {
    'OWNER': 100,      # Full access, can manage admins
    'ADMIN': 50,       # Can access admin panel, manage users
    'MODERATOR': 25,   # Can view stats, moderate content
    'USER': 1          # Regular user access
}

# Trading Configuration - Enhanced with additional timeframes
TIMEFRAMES = {
    '5s': 5,      # 5 seconds
    '15s': 15,    # 15 seconds (NEW)
    '30s': 30,    # 30 seconds (NEW)
    '1m': 60,     # 1 minute in seconds
    '2m': 120,    # 2 minutes in seconds
    '3m': 180,    # 3 minutes in seconds
    '5m': 300,    # 5 minutes in seconds
    '10m': 600,   # 10 minutes in seconds (NEW)
    '15m': 900    # 15 minutes in seconds (NEW)
}

CURRENCY_PAIRS = [
    'EURUSD_otc',    # EUR/USD OTC
    'GBPUSD_otc',    # GBP/USD OTC
    'EURGBP_otc',    # EUR/GBP OTC
    'EURJPY_otc',    # EUR/JPY OTC
    'EURAUD_otc',    # EUR/AUD OTC
    'EURCHF_otc',    # EUR/CHF OTC - New
    'GBPCHF_otc',    # GBP/CHF OTC - New
    'GBPCAD_otc',    # GBP/CAD OTC
    'AUDUSD_otc',    # AUD/USD OTC
    'USDCHF_otc',    # USD/CHF OTC
    'NZDUSD_otc',    # NZD/USD OTC
    'EURNZD_otc',    # EUR/NZD OTC
    'AUDJPY_otc',    # AUD/JPY OTC
    'GBPAUD_otc'     # GBP/AUD OTC
    # Tech Stocks
    'AAPL_otc',      # Apple OTC
    'MSFT_otc',      # Microsoft OTC
    'AMZN_otc',      # Amazon OTC
    'TSLA_otc',      # Tesla OTC
    'INTC_otc',      # Intel OTC
    'BA_otc',        # Boeing OTC
    'JNJ_otc'        # Johnson & Johnson OTC
]

# Organized pair categories for UI
PAIR_CATEGORIES = {
    'USD Pairs': ['EURUSD_otc', 'GBPUSD_otc', 'USDJPY_otc', 'USDCAD_otc', 'USDCHF_otc', 'AUDUSD_otc', 'NZDUSD_otc'],
    'EUR Pairs': ['EURUSD_otc', 'EURGBP_otc', 'EURJPY_otc', 'EURAUD_otc', 'EURNZD_otc', 'EURCHF_otc'],
    'GBP Pairs': ['GBPUSD_otc', 'EURGBP_otc', 'GBPCHF_otc', 'GBPCAD_otc', 'GBPAUD_otc'],
    'JPY Pairs': ['USDJPY_otc', 'EURJPY_otc', 'AUDJPY_otc'],
    'AUD Pairs': ['EURAUD_otc', 'AUDUSD_otc', 'AUDJPY_otc', 'GBPAUD_otc'],
    'NZD Pairs': ['NZDUSD_otc', 'EURNZD_otc'],
    'CHF Pairs': ['USDCHF_otc', 'GBPCHF_otc', 'EURCHF_otc'],
    'CAD Pairs': ['USDCAD_otc', 'GBPCAD_otc'],
    'Tech Stocks': ['AAPL_otc', 'MSFT_otc', 'AMZN_otc', 'TSLA_otc', 'INTC_otc', 'BA_otc', 'JNJ_otc']
}

UTC_TIMEZONE = timezone.utc

# Technical Analysis Parameters
INDICATOR_CONFIG = {
    'VORTEX': {
        'period': 14,
        'signal_threshold': 0.1  # Minimum difference between VI+ and VI-
    },
    'SAR': {
        'acceleration': 0.02,
        'maximum': 0.2,
        'reversal_threshold': 0.0001  # Minimum price movement for reversal
    },
    'RSI': {
        'period': 14,
        'overbought': 70,
        'oversold': 30,
        'neutral_zone': (45, 55)  # Range for neutral market
    },
    'MACD': {
        'fast_period': 12,
        'slow_period': 26,
        'signal_period': 9,
        'signal_threshold': 0.0002  # Minimum difference for signal
    },
    'CCI': {
        'period': 45,
        'threshold': 100,
        'extreme_threshold': 200  # For stronger signals
    },
    'BOLLINGER': {
        'period': 20,
        'std_dev': 2,
        'squeeze_threshold': 0.1  # For volatility breakout detection
    },
    'STOCHASTIC': {
        'k_period': 14,
        'd_period': 3,
        'smooth_k': 3,
        'overbought': 80,
        'oversold': 20
    }
}
STOCHASTIC_K_PERIOD = 14
STOCHASTIC_D_PERIOD = 3
EMA50_PERIOD = 50  # Strategy-specific parameter
EMA21_PERIOD = 21  # Strategy-specific parameter

# Individual indicator parameters for backward compatibility
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2

# Strategy Parameters
PSAR_CCI_STRATEGY = {
    # Major pairs - typically have larger movements
    'EURUSD_otc': {'target_points': (7, 12)},
    'GBPUSD_otc': {'target_points': (7, 15)},
    'USDJPY_otc': {'target_points': (6, 10)},
    'USDCHF_otc': {'target_points': (6, 11)},
    'USDCAD_otc': {'target_points': (6, 10)},
    'AUDUSD_otc': {'target_points': (5, 8)},
    'NZDUSD_otc': {'target_points': (5, 8)},

    # Cross rates - EUR pairs
    'EURGBP_otc': {'target_points': (5, 9)},
    'EURJPY_otc': {'target_points': (8, 14)},
    'EURAUD_otc': {'target_points': (8, 13)},
    'EURNZD_otc': {'target_points': (9, 15)},

    # Cross rates - GBP pairs
    'GBPCHF_otc': {'target_points': (8, 14)},
    'GBPCAD_otc': {'target_points': (8, 13)},
    'GBPAUD_otc': {'target_points': (9, 15)},

    # Asian pairs
    'AUDJPY_otc': {'target_points': (7, 12)}
}

PAIR_SETTINGS = {
    # Major pairs
    'EURUSD_otc': {'target_points': (7, 12)},
    'GBPUSD_otc': {'target_points': (7, 15)},
    'USDJPY_otc': {'target_points': (6, 10)},
    'USDCHF_otc': {'target_points': (6, 11)},
    'USDCAD_otc': {'target_points': (6, 10)},
    'AUDUSD_otc': {'target_points': (5, 8)},
    'NZDUSD_otc': {'target_points': (5, 8)},

    # Cross rates - EUR pairs
    'EURGBP_otc': {'target_points': (5, 9)},
    'EURJPY_otc': {'target_points': (8, 14)},
    'EURAUD_otc': {'target_points': (8, 13)},
    'EURNZD_otc': {'target_points': (9, 15)},
    'EURCHF_otc': {'target_points': (5, 9)},

    # Cross rates - GBP pairs
    'GBPCHF_otc': {'target_points': (8, 14)},
    'GBPCAD_otc': {'target_points': (8, 13)},
    'GBPAUD_otc': {'target_points': (9, 15)},

    # Tech Stocks OTC - percentage based targets
    'AAPL_otc': {'target_points': (0.15, 0.3)},
    'MSFT_otc': {'target_points': (0.15, 0.3)},
    'AMZN_otc': {'target_points': (0.2, 0.4)},
    'TSLA_otc': {'target_points': (0.3, 0.6)},
    'INTC_otc': {'target_points': (0.2, 0.4)},
    'BA_otc': {'target_points': (0.25, 0.5)},
    'JNJ_otc': {'target_points': (0.15, 0.3)}
}

# Trading Parameters
USE_PSAR_CCI_STRATEGY = True  # Enable the strategy
PRIMARY_TIMEFRAME = 'M1'  # Primary timeframe for the strategy
SECONDARY_TIMEFRAME = 'M5'  # Secondary timeframe for trend confirmation

# Pattern Recognition Parameters
PATTERN_CONFIDENCE = 80  # minimum confidence level for pattern recognition
PRICE_TOLERANCE = 0.0001  # tolerance for price comparison

# Support/Resistance Parameters
SR_PERIODS = 20  # periods to look back for S/R levels
SR_TOLERANCE = 0.0002  # tolerance for S/R level comparison

# Logging Configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_LEVEL = 'INFO'

# Default settings for signal generation
DEFAULT_PAIR = 'EURUSD_otc'
DEFAULT_TIMEFRAME = '15s'

# Machine Learning Settings
ML_MODEL_PATH = 'binary_options_model.pt'  # Path to the ML model
USE_ML_PREDICTIONS = True  # Whether to use ML predictions
ML_CONFIDENCE_THRESHOLD = 0.45  # Reduced threshold to produce more definitive signals

# UI Settings
BUTTON_ROW_SIZE = 2  # Reduced number of buttons per row for larger UI elements
MAX_BUTTONS_PER_PAGE = 8  # Maximum number of buttons per page to avoid crowding

# API Settings for data sources
MARKET_DATA_API = "https://api.coingecko.com/api/v3"
FOREX_API = "https://open.er-api.com/v6/latest/USD"
STOCK_API = "https://www.alphavantage.co/query"

# User data files
USERS_DATA_FILE = "data/users.json"
SUPPORTED_ASSETS = CURRENCY_PAIRS

# Sentiment Analysis Settings
SENTIMENT = {
    'weights': {
        'technicals': 0.6,    # Technical analysis weight
        'news': 0.25,         # News sentiment weight
        'social': 0.15        # Social media sentiment weight
    },
    'thresholds': {
        'strong_bullish': 25,    # Minimum score for strong bullish sentiment
        'strong_bearish': -20,   # Maximum score for strong bearish sentiment
        'min_confidence': 65     # Minimum confidence required for sentiment signals
    },
    'cache_duration': 300,    # Cache sentiment results for 5 minutes
    'news_sources': ['marketaux', 'newsapi'],
    'social_sources': ['reddit', 'stocktwits', 'twitter']
}

# API Keys for Sentiment Analysis
NEWS_API_KEY = os.environ.get('NEWS_API_KEY', '')
MARKETAUX_API_KEY = os.environ.get('MARKETAUX_API_KEY', '')
REDDIT_API_TOKEN = os.environ.get('REDDIT_API_TOKEN', '')
TWITTER_API_TOKEN = os.environ.get('TWITTER_API_TOKEN', '')
STOCKTWITS_API_TOKEN = os.environ.get('STOCKTWITS_API_TOKEN', '')

# Default settings for signal generation
DEFAULT_PAIR = 'EURUSD_otc'
DEFAULT_TIMEFRAME = '15s'
