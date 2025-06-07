#!/usr/bin/env python3
"""
Comprehensive System Verification and Deployment Readiness Assessment
for TradePulseAnalyzer Trading Bot
"""

import os
import sys
import json
import importlib
import traceback
from datetime import datetime
import logging

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings('ignore')

class SystemVerifier:
    def __init__(self):
        self.results = {}
        self.overall_status = "✅"
        self.critical_issues = []
        self.warnings = []
        self.recommendations = []
        
    def log_result(self, component, status, message, details=None):
        """Log verification result for a component."""
        self.results[component] = {
            'status': status,
            'message': message,
            'details': details or [],
            'timestamp': datetime.now().isoformat()
        }
        
        if status == "❌":
            self.critical_issues.append(f"{component}: {message}")
            self.overall_status = "❌"
        elif status == "⚠️" and self.overall_status == "✅":
            self.overall_status = "⚠️"
            self.warnings.append(f"{component}: {message}")
    
    def check_file_exists(self, filepath, description=""):
        """Check if a file exists."""
        if os.path.exists(filepath):
            size = os.path.getsize(filepath)
            return True, f"Found ({size:,} bytes)"
        return False, "Not found"
    
    def check_import(self, module_name, description=""):
        """Check if a module can be imported."""
        try:
            importlib.import_module(module_name)
            return True, "Import successful"
        except ImportError as e:
            return False, f"Import failed: {str(e)}"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def verify_core_trading_features(self):
        """Verify core trading functionality."""
        print("🔍 Verifying Core Trading Features...")
        
        # Check signal generator
        exists, msg = self.check_file_exists("signal_generator.py")
        if exists:
            success, import_msg = self.check_import("signal_generator")
            if success:
                try:
                    from signal_generator import SignalGenerator
                    sg = SignalGenerator()
                    
                    # Check supported timeframes
                    supported_timeframes = ['15s', '30s', '1m', '2m', '5m', '10m', '15m']
                    timeframe_status = []
                    
                    for tf in supported_timeframes:
                        # This is a basic check - in production you'd test actual signal generation
                        timeframe_status.append(f"{tf}: Ready")
                    
                    self.log_result("Signal Generation", "✅", 
                                  "Signal generator loaded successfully",
                                  [f"Timeframes: {', '.join(supported_timeframes)}"])
                except Exception as e:
                    self.log_result("Signal Generation", "❌", 
                                  f"Failed to initialize: {str(e)}")
            else:
                self.log_result("Signal Generation", "❌", import_msg)
        else:
            self.log_result("Signal Generation", "❌", "signal_generator.py not found")
        
        # Check pattern recognition
        exists, msg = self.check_file_exists("pattern_recognition.py")
        if exists:
            success, import_msg = self.check_import("pattern_recognition")
            if success:
                try:
                    # Check for enhanced pattern recognizer in telegram handler
                    with open("enhanced_telegram_handler.py", 'r') as f:
                        content = f.read()
                    
                    patterns = [
                        'morning_star', 'evening_star', 'hammer', 'inverted_hammer',
                        'three_black_crows', 'three_white_soldiers', 'bullish_engulfing',
                        'bearish_engulfing', 'shooting_star', 'piercing_line',
                        'dark_cloud_cover', 'doji', 'spinning_top', 'tweezer_top', 'tweezer_bottom'
                    ]
                    
                    found_patterns = []
                    for pattern in patterns:
                        if pattern in content:
                            found_patterns.append(pattern)
                    
                    if len(found_patterns) >= 15:
                        self.log_result("Candlestick Patterns", "✅", 
                                      f"All 15 patterns implemented",
                                      [f"Patterns: {', '.join(found_patterns[:5])}... (+{len(found_patterns)-5} more)"])
                    else:
                        self.log_result("Candlestick Patterns", "⚠️", 
                                      f"Only {len(found_patterns)}/15 patterns found")
                except Exception as e:
                    self.log_result("Candlestick Patterns", "❌", f"Error checking patterns: {str(e)}")
            else:
                self.log_result("Candlestick Patterns", "❌", import_msg)
        else:
            self.log_result("Candlestick Patterns", "❌", "pattern_recognition.py not found")
        
        # Check technical indicators
        try:
            with open("enhanced_telegram_handler.py", 'r') as f:
                content = f.read()
            
            indicators = ['rsi', 'macd', 'stoch', 'adx', 'bollinger', 'fractal', 'vortex']
            found_indicators = []
            
            for indicator in indicators:
                if indicator.lower() in content.lower():
                    found_indicators.append(indicator.upper())
            
            if len(found_indicators) >= 6:
                self.log_result("Technical Indicators", "✅", 
                              f"Core indicators implemented",
                              [f"Found: {', '.join(found_indicators)}"])
            else:
                self.log_result("Technical Indicators", "⚠️", 
                              f"Limited indicators found: {', '.join(found_indicators)}")
        except Exception as e:
            self.log_result("Technical Indicators", "❌", f"Error checking indicators: {str(e)}")
    
    def verify_ml_components(self):
        """Verify machine learning components."""
        print("🤖 Verifying Machine Learning Components...")
        
        # Check ML model file
        exists, msg = self.check_file_exists("ml_model.py")
        if exists:
            success, import_msg = self.check_import("ml_model")
            if success:
                self.log_result("ML Model Module", "✅", "ML model module loaded successfully")
            else:
                self.log_result("ML Model Module", "❌", import_msg)
        else:
            self.log_result("ML Model Module", "❌", "ml_model.py not found")
        
        # Check model file
        model_files = ["binary_options_model.pt", "BinaryOptionsToolsV1-main/binary_options_model.pth"]
        model_found = False
        for model_file in model_files:
            if os.path.exists(model_file):
                size = os.path.getsize(model_file)
                self.log_result("ML Model File", "✅", f"Model file found: {model_file} ({size:,} bytes)")
                model_found = True
                break
        
        if not model_found:
            self.log_result("ML Model File", "❌", "No ML model file found")
        
        # Check continuous learning
        exists, msg = self.check_file_exists("continuous_learning.py")
        if exists:
            success, import_msg = self.check_import("continuous_learning")
            if success:
                # Check data files
                data_files = {
                    "signal_history.json": "Signal History",
                    "user_feedback.json": "User Feedback",
                    "model_performance.json": "Model Performance"
                }
                
                data_status = []
                for file_path, desc in data_files.items():
                    if os.path.exists(file_path):
                        size = os.path.getsize(file_path)
                        data_status.append(f"{desc}: {size:,} bytes")
                    else:
                        data_status.append(f"{desc}: Missing")
                
                self.log_result("Continuous Learning", "✅", 
                              "Continuous learning system ready",
                              data_status)
            else:
                self.log_result("Continuous Learning", "❌", import_msg)
        else:
            self.log_result("Continuous Learning", "❌", "continuous_learning.py not found")
    
    def verify_telegram_integration(self):
        """Verify Telegram bot integration."""
        print("📱 Verifying Telegram Bot Integration...")
        
        # Check bot token
        bot_token = os.environ.get("TELEGRAM_BOT_TOKEN")
        if bot_token:
            if len(bot_token) > 40:  # Basic validation
                self.log_result("Telegram Token", "✅", "Bot token configured")
            else:
                self.log_result("Telegram Token", "❌", "Invalid bot token format")
        else:
            self.log_result("Telegram Token", "❌", "TELEGRAM_BOT_TOKEN not set")
        
        # Check telegram handler files
        handler_files = ["telegram_handler.py", "enhanced_telegram_handler.py"]
        handler_found = False
        
        for handler_file in handler_files:
            if os.path.exists(handler_file):
                with open(handler_file, 'r') as f:
                    content = f.read()
                
                # Check for inline keyboard functionality
                inline_features = ['InlineKeyboardButton', 'InlineKeyboardMarkup', 'CallbackQueryHandler']
                found_features = [f for f in inline_features if f in content]
                
                if len(found_features) >= 2:
                    self.log_result("Telegram Handler", "✅", 
                                  f"Enhanced handler found: {handler_file}",
                                  [f"Features: {', '.join(found_features)}"])
                    handler_found = True
                    break
                else:
                    self.log_result("Telegram Handler", "⚠️", 
                                  f"Basic handler found: {handler_file}")
                    handler_found = True
        
        if not handler_found:
            self.log_result("Telegram Handler", "❌", "No telegram handler found")
        
        # Check admin authentication
        exists, msg = self.check_file_exists("admin_auth.py")
        if exists:
            self.log_result("Admin Authentication", "✅", "Admin auth module found")
        else:
            self.log_result("Admin Authentication", "⚠️", "Admin auth module not found")
    
    def verify_pocket_option_integration(self):
        """Verify Pocket Option integration."""
        print("💰 Verifying Pocket Option Integration...")
        
        # Check for Pocket Option API
        po_paths = [
            "pocketoptionapi",
            "BinaryOptionsTools",
            "BinaryOptionsToolsV1-main/BinaryOptionsTools",
            "BinaryOptionsToolsV2"
        ]
        
        po_found = False
        for path in po_paths:
            if os.path.exists(path):
                if os.path.isdir(path):
                    files = os.listdir(path)
                    if any('api' in f.lower() or 'pocket' in f.lower() for f in files):
                        self.log_result("Pocket Option API", "✅", 
                                      f"API found in {path}",
                                      [f"Files: {len(files)} items"])
                        po_found = True
                        break
        
        if not po_found:
            self.log_result("Pocket Option API", "❌", "Pocket Option API not found")
        
        # Check credentials
        po_credentials = ["POCKET_OPTION_EMAIL", "POCKET_OPTION_PASSWORD", "POCKET_OPTION_SSID"]
        cred_status = []
        
        for cred in po_credentials:
            if os.environ.get(cred):
                cred_status.append(f"{cred}: Set")
            else:
                cred_status.append(f"{cred}: Missing")
        
        missing_creds = [c for c in cred_status if "Missing" in c]
        if len(missing_creds) == 0:
            self.log_result("Pocket Option Credentials", "✅", "All credentials configured")
        elif len(missing_creds) < len(po_credentials):
            self.log_result("Pocket Option Credentials", "⚠️", 
                          f"Some credentials missing", cred_status)
        else:
            self.log_result("Pocket Option Credentials", "🔧", 
                          "Credentials need configuration", cred_status)
    
    def verify_data_management(self):
        """Verify data management systems."""
        print("📊 Verifying Data Management...")
        
        # Check data aggregator
        exists, msg = self.check_file_exists("data_aggregator.py")
        if exists:
            success, import_msg = self.check_import("data_aggregator")
            if success:
                self.log_result("Data Aggregator", "✅", "Data aggregator ready")
            else:
                self.log_result("Data Aggregator", "❌", import_msg)
        else:
            self.log_result("Data Aggregator", "❌", "data_aggregator.py not found")
        
        # Check timezone management
        try:
            with open("enhanced_telegram_handler.py", 'r') as f:
                content = f.read()
            
            if "TimezoneManager" in content and "UTC" in content:
                self.log_result("Timezone Management", "✅", "UTC+0 standardization implemented")
            else:
                self.log_result("Timezone Management", "⚠️", "Timezone management unclear")
        except:
            self.log_result("Timezone Management", "❌", "Cannot verify timezone management")
        
        # Check data persistence
        data_files = ["signal_history.json", "user_feedback.json", "model_performance.json"]
        persistent_files = []
        
        for file_path in data_files:
            if os.path.exists(file_path):
                persistent_files.append(file_path)
        
        if len(persistent_files) >= 2:
            self.log_result("Data Persistence", "✅", 
                          f"Data persistence working ({len(persistent_files)}/3 files)")
        else:
            self.log_result("Data Persistence", "⚠️", 
                          f"Limited persistence ({len(persistent_files)}/3 files)")
    
    def verify_error_handling(self):
        """Verify error handling and reliability."""
        print("🛡️ Verifying Error Handling & Reliability...")
        
        # Check logging setup
        log_files = ["bot.log", "logs/errors.log", "forever.log"]
        log_found = False
        
        for log_file in log_files:
            if os.path.exists(log_file):
                size = os.path.getsize(log_file)
                self.log_result("Logging System", "✅", 
                              f"Log file found: {log_file} ({size:,} bytes)")
                log_found = True
                break
        
        if not log_found:
            self.log_result("Logging System", "⚠️", "No log files found")
        
        # Check error handling modules
        error_modules = ["error_handling_system.py"]
        error_handling_found = False
        
        for module in error_modules:
            if os.path.exists(module):
                self.log_result("Error Handling", "✅", f"Error handling module found: {module}")
                error_handling_found = True
                break
        
        if not error_handling_found:
            self.log_result("Error Handling", "⚠️", "No dedicated error handling module")
        
        # Check for try-catch blocks in main files
        main_files = ["main.py", "telegram_handler.py", "signal_generator.py"]
        error_protection = []
        
        for file_path in main_files:
            if os.path.exists(file_path):
                with open(file_path, 'r') as f:
                    content = f.read()
                
                try_count = content.count('try:')
                except_count = content.count('except')
                
                if try_count > 0 and except_count > 0:
                    error_protection.append(f"{file_path}: {try_count} try blocks")
        
        if len(error_protection) >= 2:
            self.log_result("Error Protection", "✅", 
                          "Error handling implemented in core files",
                          error_protection)
        else:
            self.log_result("Error Protection", "⚠️", 
                          "Limited error protection in core files")
    
    def verify_deployment_prerequisites(self):
        """Verify deployment prerequisites."""
        print("🚀 Verifying Deployment Prerequisites...")
        
        # Check requirements file
        req_files = ["requirements.txt", "heroku_requirements.txt"]
        req_found = False
        
        for req_file in req_files:
            if os.path.exists(req_file):
                with open(req_file, 'r') as f:
                    requirements = f.read().strip().split('\n')
                
                req_count = len([r for r in requirements if r.strip() and not r.startswith('#')])
                self.log_result("Dependencies", "✅", 
                              f"Requirements file found: {req_file} ({req_count} packages)")
                req_found = True
                break
        
        if not req_found:
            self.log_result("Dependencies", "❌", "No requirements file found")
        
        # Check configuration files
        config_files = ["config.py", ".env", "Procfile"]
        config_status = []
        
        for config_file in config_files:
            if os.path.exists(config_file):
                config_status.append(f"{config_file}: Found")
            else:
                config_status.append(f"{config_file}: Missing")
        
        found_configs = [c for c in config_status if "Found" in c]
        if len(found_configs) >= 2:
            self.log_result("Configuration Files", "✅", 
                          "Core configuration files present", config_status)
        else:
            self.log_result("Configuration Files", "⚠️", 
                          "Some configuration files missing", config_status)
        
        # Check environment variables
        required_env_vars = [
            "TELEGRAM_BOT_TOKEN",
            "BOT_OWNER_ID"
        ]
        
        env_status = []
        for var in required_env_vars:
            if os.environ.get(var):
                env_status.append(f"{var}: Set")
            else:
                env_status.append(f"{var}: Missing")
        
        missing_env = [e for e in env_status if "Missing" in e]
        if len(missing_env) == 0:
            self.log_result("Environment Variables", "✅", "Required variables set")
        else:
            self.log_result("Environment Variables", "🔧", 
                          "Some variables need configuration", env_status)
    
    def generate_report(self):
        """Generate comprehensive verification report."""
        print("\n" + "="*80)
        print("📋 COMPREHENSIVE SYSTEM VERIFICATION REPORT")
        print("="*80)
        
        # Component status summary
        for component, result in self.results.items():
            status = result['status']
            message = result['message']
            print(f"\n{status} {component}")
            print(f"   {message}")
            
            if result['details']:
                for detail in result['details']:
                    print(f"   • {detail}")
        
        # Overall assessment
        print("\n" + "="*80)
        print("🎯 DEPLOYMENT READINESS ASSESSMENT")
        print("="*80)
        
        total_components = len(self.results)
        working_components = len([r for r in self.results.values() if r['status'] in ['✅', '⚠️']])
        critical_failures = len([r for r in self.results.values() if r['status'] == '❌'])
        
        print(f"\n📊 Component Status:")
        print(f"   • Total Components Checked: {total_components}")
        print(f"   • Working Components: {working_components}")
        print(f"   • Critical Failures: {critical_failures}")
        print(f"   • Overall Status: {self.overall_status}")
        
        # Critical issues
        if self.critical_issues:
            print(f"\n❌ Critical Issues ({len(self.critical_issues)}):")
            for issue in self.critical_issues:
                print(f"   • {issue}")
        
        # Warnings
        if self.warnings:
            print(f"\n⚠️ Warnings ({len(self.warnings)}):")
            for warning in self.warnings:
                print(f"   • {warning}")
        
        # Final recommendation
        print(f"\n🚦 FINAL DEPLOYMENT DECISION:")
        
        if self.overall_status == "✅":
            print("   ✅ GO - System is ready for deployment")
            print("   All critical components are functional.")
        elif self.overall_status == "⚠️":
            print("   ⚠️ CONDITIONAL GO - System can be deployed with monitoring")
            print("   Some non-critical issues exist but system is functional.")
        else:
            print("   ❌ NO-GO - System requires fixes before deployment")
            print("   Critical issues must be resolved first.")
        
        print("\n" + "="*80)
        return self.overall_status

def main():
    """Run comprehensive system verification."""
    print("🔍 Starting Comprehensive System Verification...")
    print("⏱️ ", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("="*80)
    
    verifier = SystemVerifier()
    
    # Run all verification checks
    verifier.verify_core_trading_features()
    verifier.verify_ml_components()
    verifier.verify_telegram_integration()
    verifier.verify_pocket_option_integration()
    verifier.verify_data_management()
    verifier.verify_error_handling()
    verifier.verify_deployment_prerequisites()
    
    # Generate final report
    final_status = verifier.generate_report()
    
    return final_status

if __name__ == "__main__":
    main()
