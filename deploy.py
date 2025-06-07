#!/usr/bin/env python3
"""
TradePulseAnalyzer Production Deployment Script
Handles complete deployment setup, configuration, and validation.
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Any
import json
import shutil
from production_config import ProductionConfigManager

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TradePulseDeployer:
    """
    Comprehensive deployment manager for TradePulse.
    """
    
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.config_manager = ProductionConfigManager()
        self.required_packages = [
            'pandas>=1.5.0',
            'numpy>=1.21.0',
            'aiohttp>=3.8.0',
            'asyncio',
            'flask>=2.0.0',
            'flask-socketio>=5.0.0',
            'flask-login>=0.6.0',
            'plotly>=5.0.0',
            'scikit-learn>=1.0.0',
            'textblob>=0.17.0',
            'vaderSentiment>=3.3.0',
            'beautifulsoup4>=4.10.0',
            'requests>=2.28.0',
            'websockets>=10.0',
            'python-telegram-bot>=20.0',
            'psutil>=5.8.0',
            'yfinance>=0.1.87',
            'ccxt>=2.0.0'
        ]
    
    def deploy(self, mode: str = 'production', skip_tests: bool = False):
        """
        Complete deployment process.
        
        Args:
            mode: Deployment mode ('production', 'staging', 'development')
            skip_tests: Skip integration tests
        """
        logger.info(f"🚀 Starting TradePulse deployment in {mode} mode...")
        
        try:
            # Step 1: Environment setup
            self._setup_environment()
            
            # Step 2: Install dependencies
            self._install_dependencies()
            
            # Step 3: Configuration setup
            self._setup_configuration(mode)
            
            # Step 4: Create directories
            self._create_directories()
            
            # Step 5: Database setup
            self._setup_database()
            
            # Step 6: Run tests (unless skipped)
            if not skip_tests:
                self._run_integration_tests()
            
            # Step 7: Security setup
            self._setup_security()
            
            # Step 8: Service configuration
            self._setup_services(mode)
            
            # Step 9: Final validation
            self._validate_deployment()
            
            logger.info("✅ TradePulse deployment completed successfully!")
            self._print_deployment_summary(mode)
            
        except Exception as e:
            logger.error(f"❌ Deployment failed: {e}")
            raise
    
    def _setup_environment(self):
        """Setup Python environment and virtual environment."""
        logger.info("Setting up environment...")
        
        # Check Python version
        python_version = sys.version_info
        if python_version < (3, 8):
            raise RuntimeError("Python 3.8 or higher is required")
        
        logger.info(f"✓ Python {python_version.major}.{python_version.minor} detected")
        
        # Check if we're in a virtual environment
        if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
            logger.warning("⚠️  Not running in a virtual environment. Consider using venv or conda.")
        
        # Create .env template if it doesn't exist
        if not Path('.env').exists():
            self.config_manager.create_env_template()
            logger.info("✓ Created .env template file")
    
    def _install_dependencies(self):
        """Install required Python packages."""
        logger.info("Installing dependencies...")
        
        try:
            # Upgrade pip first
            subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'], 
                         check=True, capture_output=True)
            
            # Install packages
            for package in self.required_packages:
                logger.info(f"Installing {package}...")
                result = subprocess.run(
                    [sys.executable, '-m', 'pip', 'install', package],
                    capture_output=True, text=True
                )
                
                if result.returncode != 0:
                    logger.warning(f"⚠️  Failed to install {package}: {result.stderr}")
                else:
                    logger.info(f"✓ Installed {package}")
            
            logger.info("✓ Dependencies installation completed")
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to install dependencies: {e}")
            raise
    
    def _setup_configuration(self, mode: str):
        """Setup configuration for deployment mode."""
        logger.info(f"Setting up configuration for {mode} mode...")
        
        # Adjust configuration based on mode
        if mode == 'production':
            self.config_manager.web.debug = False
            self.config_manager.logging.level = 'WARNING'
            self.config_manager.trading.paper_trading = False
        elif mode == 'staging':
            self.config_manager.web.debug = False
            self.config_manager.logging.level = 'INFO'
            self.config_manager.trading.paper_trading = True
        else:  # development
            self.config_manager.web.debug = True
            self.config_manager.logging.level = 'DEBUG'
            self.config_manager.trading.paper_trading = True
        
        # Save configuration
        self.config_manager.save_configuration()
        
        # Validate configuration
        issues = self.config_manager.validate_configuration()
        
        if issues['errors']:
            logger.error("Configuration errors found:")
            for error in issues['errors']:
                logger.error(f"  ❌ {error}")
            raise RuntimeError("Configuration validation failed")
        
        if issues['warnings']:
            logger.warning("Configuration warnings:")
            for warning in issues['warnings']:
                logger.warning(f"  ⚠️  {warning}")
        
        logger.info("✓ Configuration setup completed")
    
    def _create_directories(self):
        """Create necessary directories."""
        logger.info("Creating directories...")
        
        directories = [
            'logs',
            'data',
            'backups',
            'static',
            'templates',
            'uploads'
        ]
        
        for directory in directories:
            dir_path = self.project_root / directory
            dir_path.mkdir(exist_ok=True)
            logger.info(f"✓ Created directory: {directory}")
        
        logger.info("✓ Directory creation completed")
    
    def _setup_database(self):
        """Setup database and initialize tables."""
        logger.info("Setting up database...")
        
        # For SQLite, just ensure the database file exists
        if self.config_manager.database.type == 'sqlite':
            db_path = Path(self.config_manager.database.name)
            if not db_path.exists():
                db_path.touch()
                logger.info(f"✓ Created SQLite database: {db_path}")
        
        # TODO: Add database schema initialization here
        logger.info("✓ Database setup completed")
    
    def _run_integration_tests(self):
        """Run comprehensive integration tests."""
        logger.info("Running integration tests...")
        
        try:
            # Import and run tests
            from comprehensive_integration_tests import run_comprehensive_tests
            
            success = run_comprehensive_tests()
            
            if success:
                logger.info("✅ All integration tests passed")
            else:
                logger.error("❌ Some integration tests failed")
                raise RuntimeError("Integration tests failed")
                
        except ImportError:
            logger.warning("⚠️  Integration tests not available")
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            raise
    
    def _setup_security(self):
        """Setup security configurations."""
        logger.info("Setting up security...")
        
        # Ensure secret keys are set
        if not self.config_manager.security.secret_key:
            import secrets
            self.config_manager.security.secret_key = secrets.token_hex(32)
            logger.info("✓ Generated new secret key")
        
        # Set file permissions
        try:
            # Make config files readable only by owner
            os.chmod('config.json', 0o600)
            if Path('.env').exists():
                os.chmod('.env', 0o600)
            
            logger.info("✓ Set secure file permissions")
            
        except OSError as e:
            logger.warning(f"⚠️  Could not set file permissions: {e}")
        
        logger.info("✓ Security setup completed")
    
    def _setup_services(self, mode: str):
        """Setup system services and process management."""
        logger.info("Setting up services...")
        
        # Create systemd service file for Linux
        if sys.platform.startswith('linux'):
            self._create_systemd_service()
        
        # Create startup scripts
        self._create_startup_scripts(mode)
        
        logger.info("✓ Services setup completed")
    
    def _create_systemd_service(self):
        """Create systemd service file for Linux."""
        service_content = f"""[Unit]
Description=TradePulse Trading Bot
After=network.target

[Service]
Type=simple
User={os.getenv('USER', 'tradepulse')}
WorkingDirectory={self.project_root}
Environment=PATH={sys.executable}
ExecStart={sys.executable} main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""
        
        service_file = Path('/tmp/tradepulse.service')
        with open(service_file, 'w') as f:
            f.write(service_content)
        
        logger.info(f"✓ Created systemd service file: {service_file}")
        logger.info("To install: sudo cp /tmp/tradepulse.service /etc/systemd/system/")
    
    def _create_startup_scripts(self, mode: str):
        """Create startup scripts for different platforms."""
        
        # Create start script
        start_script_content = f"""#!/bin/bash
# TradePulse Startup Script

cd "{self.project_root}"

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set environment
export DEPLOYMENT_MODE={mode}

# Start the application
python main.py
"""
        
        start_script = self.project_root / 'start.sh'
        with open(start_script, 'w') as f:
            f.write(start_script_content)
        
        # Make executable
        os.chmod(start_script, 0o755)
        
        # Create Windows batch file
        batch_content = f"""@echo off
cd /d "{self.project_root}"

REM Activate virtual environment if it exists
if exist venv\\Scripts\\activate.bat (
    call venv\\Scripts\\activate.bat
)

REM Set environment
set DEPLOYMENT_MODE={mode}

REM Start the application
python main.py
pause
"""
        
        batch_file = self.project_root / 'start.bat'
        with open(batch_file, 'w') as f:
            f.write(batch_content)
        
        logger.info("✓ Created startup scripts")
    
    def _validate_deployment(self):
        """Validate the deployment."""
        logger.info("Validating deployment...")
        
        # Check critical files
        critical_files = [
            'main.py',
            'signal_generator.py',
            'enhanced_data_reliability.py',
            'live_trading_engine.py',
            'config.json'
        ]
        
        for file_name in critical_files:
            file_path = self.project_root / file_name
            if not file_path.exists():
                raise RuntimeError(f"Critical file missing: {file_name}")
        
        # Check configuration
        summary = self.config_manager.get_config_summary()
        logger.info(f"Configuration summary: {summary}")
        
        # Test imports
        try:
            import signal_generator
            import enhanced_data_reliability
            import live_trading_engine
            logger.info("✓ All modules import successfully")
        except ImportError as e:
            raise RuntimeError(f"Module import failed: {e}")
        
        logger.info("✓ Deployment validation completed")
    
    def _print_deployment_summary(self, mode: str):
        """Print deployment summary."""
        summary = self.config_manager.get_config_summary()
        
        print("\n" + "="*60)
        print("🎉 TRADEPULSE DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Deployment Mode: {mode.upper()}")
        print(f"Trading Mode: {summary['trading_mode']}")
        print(f"Paper Trading: {'Enabled' if summary['paper_trading'] else 'Disabled'}")
        print(f"Initial Balance: ${summary['initial_balance']:,.2f}")
        print(f"Web Port: {summary['web_port']}")
        print(f"Log Level: {summary['log_level']}")
        print("\nAPI Keys Configured:")
        for api, configured in summary['api_keys_configured'].items():
            status = "✅" if configured else "❌"
            print(f"  {api}: {status}")
        
        print("\n📋 NEXT STEPS:")
        print("1. Review and update .env file with your API keys")
        print("2. Start the application: python main.py")
        print("3. Access web dashboard: http://localhost:5000")
        print("4. Monitor logs in the logs/ directory")
        
        if mode == 'production':
            print("\n⚠️  PRODUCTION NOTES:")
            print("- Ensure all API keys are configured")
            print("- Set up SSL certificates for HTTPS")
            print("- Configure firewall rules")
            print("- Set up monitoring and alerting")
            print("- Regular backups of configuration and data")
        
        print("="*60)

def main():
    """Main deployment function."""
    parser = argparse.ArgumentParser(description='TradePulse Deployment Script')
    parser.add_argument('--mode', choices=['production', 'staging', 'development'], 
                       default='development', help='Deployment mode')
    parser.add_argument('--skip-tests', action='store_true', 
                       help='Skip integration tests')
    
    args = parser.parse_args()
    
    deployer = TradePulseDeployer()
    deployer.deploy(mode=args.mode, skip_tests=args.skip_tests)

if __name__ == '__main__':
    main()
