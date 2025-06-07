"""
Enhanced logging system for MasterTrade Bot
"""

import json
import logging
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

@dataclass
class ErrorContext:
    timestamp: str
    error_type: str
    error_message: str
    stack_trace: str
    component: str
    severity: str
    additional_data: Dict[str, Any]

class EnhancedLogger:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Configure logging
        self.logger = logging.getLogger("MasterTrade")
        self.logger.setLevel(logging.DEBUG)
        
        # File handler for all logs
        fh = logging.FileHandler(self.log_dir / "bot.log")
        fh.setLevel(logging.DEBUG)
        
        # File handler for errors only
        error_fh = logging.FileHandler(self.log_dir / "errors.log")
        error_fh.setLevel(logging.ERROR)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        error_fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add handlers
        self.logger.addHandler(fh)
        self.logger.addHandler(error_fh)
        self.logger.addHandler(ch)
        
        # Trade logging
        self.trade_log_path = self.log_dir / "trades.json"
        self.trades = self._load_trades()
        
        # Error tracking
        self.error_history = []
        self.error_stats = {
            'total_errors': 0,
            'errors_by_type': {},
            'errors_by_component': {}
        }
        
    def _load_trades(self) -> Dict:
        """Load existing trade history"""
        if self.trade_log_path.exists():
            try:
                with open(self.trade_log_path, 'r') as f:
                    return json.load(f)
            except json.JSONDecodeError:
                return {"trades": []}
        return {"trades": []}
        
    def log_trade(self, trade_data: Dict):
        """Log trade details with enhanced tracking"""
        trade_entry = {
            "timestamp": datetime.now().isoformat(),
            "trade_id": len(self.trades["trades"]) + 1,
            **trade_data
        }
        
        self.trades["trades"].append(trade_entry)
        
        # Save to file
        with open(self.trade_log_path, 'w') as f:
            json.dump(self.trades, f, indent=2)
            
        # Log to main log
        self.logger.info(f"Trade executed: {trade_entry}")
        
    def log_signal(self, signal_data: Dict):
        """Log signal generation details"""
        self.logger.info(f"Signal generated: {signal_data}")
        
    def log_model_update(self, update_info: Dict):
        """Log model update information"""
        self.logger.info(
            f"Model updated: {json.dumps(update_info, default=str)}"
        )
        
    def log_error(self, error: Exception, component: str, 
                  additional_data: Optional[Dict] = None):
        """Log error with enhanced context and tracking"""
        error_context = ErrorContext(
            timestamp=datetime.now().isoformat(),
            error_type=type(error).__name__,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            component=component,
            severity=self._determine_severity(error),
            additional_data=additional_data or {}
        )
        
        # Update error statistics
        self._update_error_stats(error_context)
        
        # Log error
        self.logger.error(
            f"Error in {component}: {str(error)}\n"
            f"Context: {json.dumps(asdict(error_context), indent=2)}"
        )
        
        # Save error history
        self.error_history.append(error_context)
        self._save_error_history()
        
    def get_error_summary(self) -> Dict:
        """Get summary of error statistics"""
        return {
            'total_errors': self.error_stats['total_errors'],
            'errors_by_type': self.error_stats['errors_by_type'],
            'errors_by_component': self.error_stats['errors_by_component'],
            'recent_errors': self.error_history[-10:]  # Last 10 errors
        }
        
    def _determine_severity(self, error: Exception) -> str:
        """Determine error severity based on type"""
        critical_errors = (
            SystemError, MemoryError, RuntimeError, 
            ConnectionError, TimeoutError
        )
        
        if isinstance(error, critical_errors):
            return "CRITICAL"
        elif isinstance(error, (ValueError, KeyError, AttributeError)):
            return "ERROR"
        else:
            return "WARNING"
            
    def _update_error_stats(self, error_context: ErrorContext):
        """Update error statistics"""
        self.error_stats['total_errors'] += 1
        
        # Update by type
        error_type = error_context.error_type
        self.error_stats['errors_by_type'][error_type] = \
            self.error_stats['errors_by_type'].get(error_type, 0) + 1
            
        # Update by component
        component = error_context.component
        self.error_stats['errors_by_component'][component] = \
            self.error_stats['errors_by_component'].get(component, 0) + 1
            
    def _save_error_history(self):
        """Save error history to file"""
        try:
            history_file = self.log_dir / "error_history.json"
            with open(history_file, 'w') as f:
                json.dump(
                    [asdict(err) for err in self.error_history],
                    f,
                    indent=2,
                    default=str
                )
        except Exception as e:
            self.logger.error(f"Failed to save error history: {str(e)}")

# Global logger instance
logger = EnhancedLogger()
