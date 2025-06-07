"""
Production Configuration Management System
Handles all configuration settings, environment variables, and deployment settings.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from pathlib import Path
import secrets

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    type: str = "sqlite"
    host: str = "localhost"
    port: int = 5432
    name: str = "tradepulse.db"
    username: str = ""
    password: str = ""
    ssl_mode: str = "prefer"

@dataclass
class TradingConfig:
    """Trading engine configuration."""
    initial_balance: float = 10000.0
    max_position_size: float = 0.02  # 2% of account
    max_daily_loss: float = 0.05     # 5% of account
    max_drawdown: float = 0.10       # 10% of account
    max_positions: int = 5
    commission_rate: float = 0.001   # 0.1%
    use_real_data: bool = False
    paper_trading: bool = True

@dataclass
class APIConfig:
    """API keys and external service configuration."""
    # Trading APIs
    pocketoption_ssid: str = ""
    
    # Data APIs
    alpha_vantage_api_key: str = ""
    crypto_compare_api_key: str = ""
    yahoo_finance_enabled: bool = True
    
    # Social Media APIs
    twitter_api_key: str = ""
    twitter_api_secret: str = ""
    twitter_bearer_token: str = ""
    
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "TradePulse/1.0"
    
    stocktwits_api_key: str = ""
    
    # News APIs
    news_api_key: str = ""
    
    # Telegram Bot
    telegram_bot_token: str = ""

@dataclass
class SecurityConfig:
    """Security and authentication settings."""
    secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    jwt_secret_key: str = field(default_factory=lambda: secrets.token_hex(32))
    password_salt: str = field(default_factory=lambda: secrets.token_hex(16))
    session_timeout: int = 3600  # 1 hour
    max_login_attempts: int = 5
    lockout_duration: int = 900  # 15 minutes

@dataclass
class NotificationConfig:
    """Notification settings."""
    email_enabled: bool = False
    email_recipients: list = field(default_factory=list)
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    
    telegram_enabled: bool = False
    telegram_chat_id: str = ""
    
    error_email_enabled: bool = False
    error_email_recipients: list = field(default_factory=list)
    error_telegram_enabled: bool = False
    error_telegram_chat_id: str = ""

@dataclass
class LoggingConfig:
    """Logging configuration."""
    level: str = "INFO"
    file_enabled: bool = True
    console_enabled: bool = True
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5
    log_directory: str = "logs"

@dataclass
class WebConfig:
    """Web dashboard configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    ssl_enabled: bool = False
    ssl_cert_path: str = ""
    ssl_key_path: str = ""

class ProductionConfigManager:
    """
    Comprehensive configuration management for production deployment.
    """
    
    def __init__(self, config_file: str = "config.json", env_file: str = ".env"):
        self.config_file = Path(config_file)
        self.env_file = Path(env_file)
        
        # Configuration sections
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.api = APIConfig()
        self.security = SecurityConfig()
        self.notifications = NotificationConfig()
        self.logging = LoggingConfig()
        self.web = WebConfig()
        
        # Load configuration
        self._load_configuration()
        
        # Setup logging
        self._setup_logging()
    
    def _load_configuration(self):
        """Load configuration from environment variables and config file."""
        # Load from environment variables first
        self._load_from_environment()
        
        # Load from config file (overrides environment)
        if self.config_file.exists():
            self._load_from_file()
        else:
            # Create default config file
            self._create_default_config()
    
    def _load_from_environment(self):
        """Load configuration from environment variables."""
        # Database configuration
        self.database.type = os.getenv('DB_TYPE', self.database.type)
        self.database.host = os.getenv('DB_HOST', self.database.host)
        self.database.port = int(os.getenv('DB_PORT', self.database.port))
        self.database.name = os.getenv('DB_NAME', self.database.name)
        self.database.username = os.getenv('DB_USERNAME', self.database.username)
        self.database.password = os.getenv('DB_PASSWORD', self.database.password)
        
        # Trading configuration
        self.trading.initial_balance = float(os.getenv('INITIAL_BALANCE', self.trading.initial_balance))
        self.trading.use_real_data = os.getenv('USE_REAL_DATA', 'false').lower() == 'true'
        self.trading.paper_trading = os.getenv('PAPER_TRADING', 'true').lower() == 'true'
        
        # API configuration
        self.api.pocketoption_ssid = os.getenv('POCKETOPTION_SSID', self.api.pocketoption_ssid)
        self.api.alpha_vantage_api_key = os.getenv('ALPHA_VANTAGE_API_KEY', self.api.alpha_vantage_api_key)
        self.api.crypto_compare_api_key = os.getenv('CRYPTO_COMPARE_API_KEY', self.api.crypto_compare_api_key)
        
        # Social media APIs
        self.api.twitter_api_key = os.getenv('TWITTER_API_KEY', self.api.twitter_api_key)
        self.api.twitter_api_secret = os.getenv('TWITTER_API_SECRET', self.api.twitter_api_secret)
        self.api.twitter_bearer_token = os.getenv('TWITTER_BEARER_TOKEN', self.api.twitter_bearer_token)
        
        self.api.reddit_client_id = os.getenv('REDDIT_CLIENT_ID', self.api.reddit_client_id)
        self.api.reddit_client_secret = os.getenv('REDDIT_CLIENT_SECRET', self.api.reddit_client_secret)
        
        self.api.stocktwits_api_key = os.getenv('STOCKTWITS_API_KEY', self.api.stocktwits_api_key)
        self.api.news_api_key = os.getenv('NEWS_API_KEY', self.api.news_api_key)
        
        # Telegram
        self.api.telegram_bot_token = os.getenv('TELEGRAM_BOT_TOKEN', self.api.telegram_bot_token)
        
        # Security
        self.security.secret_key = os.getenv('SECRET_KEY', self.security.secret_key)
        
        # Notifications
        self.notifications.email_enabled = os.getenv('EMAIL_ENABLED', 'false').lower() == 'true'
        self.notifications.smtp_username = os.getenv('SMTP_USERNAME', self.notifications.smtp_username)
        self.notifications.smtp_password = os.getenv('SMTP_PASSWORD', self.notifications.smtp_password)
        
        # Web configuration
        self.web.host = os.getenv('WEB_HOST', self.web.host)
        self.web.port = int(os.getenv('WEB_PORT', self.web.port))
        self.web.debug = os.getenv('WEB_DEBUG', 'false').lower() == 'true'
        
        # Logging
        self.logging.level = os.getenv('LOG_LEVEL', self.logging.level)
    
    def _load_from_file(self):
        """Load configuration from JSON file."""
        try:
            with open(self.config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configuration sections
            if 'database' in config_data:
                self._update_dataclass(self.database, config_data['database'])
            
            if 'trading' in config_data:
                self._update_dataclass(self.trading, config_data['trading'])
            
            if 'api' in config_data:
                self._update_dataclass(self.api, config_data['api'])
            
            if 'security' in config_data:
                self._update_dataclass(self.security, config_data['security'])
            
            if 'notifications' in config_data:
                self._update_dataclass(self.notifications, config_data['notifications'])
            
            if 'logging' in config_data:
                self._update_dataclass(self.logging, config_data['logging'])
            
            if 'web' in config_data:
                self._update_dataclass(self.web, config_data['web'])
                
            logger.info(f"Configuration loaded from {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error loading configuration file: {e}")
    
    def _update_dataclass(self, dataclass_instance, data: Dict[str, Any]):
        """Update dataclass instance with dictionary data."""
        for key, value in data.items():
            if hasattr(dataclass_instance, key):
                setattr(dataclass_instance, key, value)
    
    def _create_default_config(self):
        """Create default configuration file."""
        try:
            config_data = {
                'database': self._dataclass_to_dict(self.database),
                'trading': self._dataclass_to_dict(self.trading),
                'api': self._dataclass_to_dict(self.api),
                'security': self._dataclass_to_dict(self.security),
                'notifications': self._dataclass_to_dict(self.notifications),
                'logging': self._dataclass_to_dict(self.logging),
                'web': self._dataclass_to_dict(self.web)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Default configuration created at {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error creating default configuration: {e}")
    
    def _dataclass_to_dict(self, dataclass_instance) -> Dict[str, Any]:
        """Convert dataclass instance to dictionary."""
        return {
            field.name: getattr(dataclass_instance, field.name)
            for field in dataclass_instance.__dataclass_fields__.values()
        }
    
    def _setup_logging(self):
        """Setup logging configuration."""
        try:
            # Create logs directory
            log_dir = Path(self.logging.log_directory)
            log_dir.mkdir(exist_ok=True)
            
            # Configure logging
            log_level = getattr(logging, self.logging.level.upper(), logging.INFO)
            
            # Create formatter
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            
            # Root logger
            root_logger = logging.getLogger()
            root_logger.setLevel(log_level)
            
            # Clear existing handlers
            root_logger.handlers.clear()
            
            # Console handler
            if self.logging.console_enabled:
                console_handler = logging.StreamHandler()
                console_handler.setLevel(log_level)
                console_handler.setFormatter(formatter)
                root_logger.addHandler(console_handler)
            
            # File handler
            if self.logging.file_enabled:
                from logging.handlers import RotatingFileHandler
                
                file_handler = RotatingFileHandler(
                    log_dir / 'tradepulse.log',
                    maxBytes=self.logging.max_file_size,
                    backupCount=self.logging.backup_count
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
            
            logger.info("Logging configuration completed")
            
        except Exception as e:
            print(f"Error setting up logging: {e}")
    
    def save_configuration(self):
        """Save current configuration to file."""
        try:
            config_data = {
                'database': self._dataclass_to_dict(self.database),
                'trading': self._dataclass_to_dict(self.trading),
                'api': self._dataclass_to_dict(self.api),
                'security': self._dataclass_to_dict(self.security),
                'notifications': self._dataclass_to_dict(self.notifications),
                'logging': self._dataclass_to_dict(self.logging),
                'web': self._dataclass_to_dict(self.web)
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration saved to {self.config_file}")
            
        except Exception as e:
            logger.error(f"Error saving configuration: {e}")
    
    def create_env_template(self):
        """Create .env template file."""
        try:
            env_template = """# TradePulse Environment Configuration

# Database Configuration
DB_TYPE=sqlite
DB_HOST=localhost
DB_PORT=5432
DB_NAME=tradepulse.db
DB_USERNAME=
DB_PASSWORD=

# Trading Configuration
INITIAL_BALANCE=10000.0
USE_REAL_DATA=false
PAPER_TRADING=true

# API Keys
POCKETOPTION_SSID=
ALPHA_VANTAGE_API_KEY=
CRYPTO_COMPARE_API_KEY=

# Social Media APIs
TWITTER_API_KEY=
TWITTER_API_SECRET=
TWITTER_BEARER_TOKEN=
REDDIT_CLIENT_ID=
REDDIT_CLIENT_SECRET=
STOCKTWITS_API_KEY=
NEWS_API_KEY=

# Telegram Bot
TELEGRAM_BOT_TOKEN=

# Security
SECRET_KEY=

# Email Notifications
EMAIL_ENABLED=false
SMTP_USERNAME=
SMTP_PASSWORD=

# Web Configuration
WEB_HOST=0.0.0.0
WEB_PORT=5000
WEB_DEBUG=false

# Logging
LOG_LEVEL=INFO
"""
            
            with open('.env.template', 'w') as f:
                f.write(env_template)
            
            logger.info("Environment template created at .env.template")
            
        except Exception as e:
            logger.error(f"Error creating environment template: {e}")
    
    def validate_configuration(self) -> Dict[str, List[str]]:
        """Validate configuration and return any issues."""
        issues = {
            'errors': [],
            'warnings': [],
            'info': []
        }
        
        # Check critical settings
        if self.trading.use_real_data and not self.api.pocketoption_ssid:
            issues['errors'].append("Real data enabled but no PocketOption SSID provided")
        
        if not self.security.secret_key or len(self.security.secret_key) < 32:
            issues['errors'].append("Secret key is missing or too short")
        
        # Check API keys
        if not self.api.alpha_vantage_api_key:
            issues['warnings'].append("Alpha Vantage API key not provided - limited data sources")
        
        if not self.api.telegram_bot_token:
            issues['warnings'].append("Telegram bot token not provided - bot features disabled")
        
        # Check notification settings
        if self.notifications.email_enabled and not self.notifications.smtp_username:
            issues['warnings'].append("Email notifications enabled but SMTP credentials missing")
        
        # Info messages
        if self.trading.paper_trading:
            issues['info'].append("Paper trading mode enabled - no real money at risk")
        
        if self.web.debug:
            issues['info'].append("Web debug mode enabled - disable for production")
        
        return issues
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get configuration summary for display."""
        return {
            'trading_mode': 'Real' if self.trading.use_real_data else 'Demo',
            'paper_trading': self.trading.paper_trading,
            'initial_balance': self.trading.initial_balance,
            'web_port': self.web.port,
            'log_level': self.logging.level,
            'api_keys_configured': {
                'pocketoption': bool(self.api.pocketoption_ssid),
                'alpha_vantage': bool(self.api.alpha_vantage_api_key),
                'twitter': bool(self.api.twitter_api_key),
                'telegram': bool(self.api.telegram_bot_token)
            }
        }

# Global configuration instance
config_manager = ProductionConfigManager()

# Export configuration for backward compatibility
TELEGRAM_BOT_TOKEN = config_manager.api.telegram_bot_token
POCKETOPTION_SSID = config_manager.api.pocketoption_ssid
USE_REAL_DATA = config_manager.trading.use_real_data
SECRET_KEY = config_manager.security.secret_key
ALPHA_VANTAGE_API_KEY = config_manager.api.alpha_vantage_api_key
CRYPTO_COMPARE_API_KEY = config_manager.api.crypto_compare_api_key
TWITTER_API_KEY = config_manager.api.twitter_api_key
TWITTER_API_SECRET = config_manager.api.twitter_api_secret
TWITTER_BEARER_TOKEN = config_manager.api.twitter_bearer_token
REDDIT_CLIENT_ID = config_manager.api.reddit_client_id
REDDIT_CLIENT_SECRET = config_manager.api.reddit_client_secret
REDDIT_USER_AGENT = config_manager.api.reddit_user_agent
STOCKTWITS_API_KEY = config_manager.api.stocktwits_api_key
NEWS_API_KEY = config_manager.api.news_api_key
