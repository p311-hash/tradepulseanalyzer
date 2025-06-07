# Social sentiment analysis configuration

# API Configuration
SENTIMENT_APIS = {
    'twitter': {
        'api_key': '',  # Add your Twitter API key
        'api_secret': '',
        'bearer_token': ''
    },
    'stocktwits': {
        'api_key': ''  # Add your Stocktwits API key
    },
    'news_api': {
        'api_key': ''  # Add your NewsAPI key
    }
}

# Sentiment Analysis Parameters
SENTIMENT_CONFIG = {
    # Weight configuration for different data sources
    'source_weights': {
        'twitter': 0.3,
        'stocktwits': 0.3,
        'news': 0.4
    },
    
    # Scoring thresholds
    'thresholds': {
        'very_bullish': 0.8,
        'bullish': 0.6,
        'neutral': 0.4,
        'bearish': 0.3,
        'very_bearish': 0.2
    },
    
    # Time windows for analysis (in minutes)
    'time_windows': {
        'short_term': 60,    # 1 hour
        'medium_term': 360,  # 6 hours
        'long_term': 1440    # 24 hours
    },
    
    # Cache configuration
    'cache_duration': 15,  # minutes
    
    # Volume thresholds for significance
    'min_mentions': 10,      # Minimum mentions to consider
    'volume_thresholds': {
        'low': 20,
        'medium': 50,
        'high': 100
    },
    
    # Keyword weights for specific terms
    'keyword_weights': {
        'highly_bullish': ['moon', 'rocket', 'breakout', 'surge'],
        'bullish': ['buy', 'long', 'support', 'uptrend'],
        'bearish': ['sell', 'short', 'resistance', 'downtrend'],
        'highly_bearish': ['crash', 'dump', 'collapse', 'plunge']
    }
}

# Trading signal adjustments
SIGNAL_ADJUSTMENTS = {
    'very_bullish': {
        'confidence_boost': 15,
        'take_profit_multiplier': 1.5,
        'stop_loss_multiplier': 0.8
    },
    'bullish': {
        'confidence_boost': 10,
        'take_profit_multiplier': 1.3,
        'stop_loss_multiplier': 0.9
    },
    'bearish': {
        'confidence_boost': -10,
        'take_profit_multiplier': 0.9,
        'stop_loss_multiplier': 1.2
    },
    'very_bearish': {
        'confidence_boost': -15,
        'take_profit_multiplier': 0.8,
        'stop_loss_multiplier': 1.5
    }
}
