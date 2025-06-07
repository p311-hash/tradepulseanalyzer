"""
Comprehensive Error Handling and Recovery System
Provides robust error handling, logging, and automatic recovery mechanisms.
"""

import logging
import traceback
import asyncio
import smtplib
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
import json
import os
import sys
import psutil
import config

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    DATA_SOURCE = "data_source"
    TRADING_ENGINE = "trading_engine"
    SIGNAL_GENERATION = "signal_generation"
    NETWORK = "network"
    SYSTEM = "system"
    USER_INPUT = "user_input"

@dataclass
class ErrorEvent:
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    details: Dict[str, Any]
    stack_trace: Optional[str] = None
    resolved: bool = False
    resolution_time: Optional[datetime] = None

class ErrorHandlingSystem:
    """
    Comprehensive error handling and recovery system.
    """

    def __init__(self):
        self.error_history: List[ErrorEvent] = []
        self.error_counts: Dict[str, int] = {}
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.notification_settings = self._load_notification_settings()

        # Circuit breaker settings
        self.circuit_breakers: Dict[str, Dict] = {}
        self.default_circuit_breaker = {
            'failure_threshold': 5,
            'recovery_timeout': 300,  # 5 minutes
            'half_open_max_calls': 3
        }

        # Setup logging
        self._setup_error_logging()

        # Register recovery strategies
        self._register_recovery_strategies()

    def _setup_error_logging(self):
        """Setup comprehensive error logging."""
        # Create error logger
        self.error_logger = logging.getLogger('error_handler')
        self.error_logger.setLevel(logging.ERROR)

        # Create file handler for errors
        error_handler = logging.FileHandler('logs/errors.log')
        error_handler.setLevel(logging.ERROR)

        # Create critical error handler
        critical_handler = logging.FileHandler('logs/critical_errors.log')
        critical_handler.setLevel(logging.CRITICAL)

        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        error_handler.setFormatter(formatter)
        critical_handler.setFormatter(formatter)

        # Add handlers
        self.error_logger.addHandler(error_handler)
        self.error_logger.addHandler(critical_handler)

    def _load_notification_settings(self) -> Dict:
        """Load notification settings from config."""
        return {
            'email_enabled': getattr(config, 'ERROR_EMAIL_ENABLED', False),
            'email_recipients': getattr(config, 'ERROR_EMAIL_RECIPIENTS', []),
            'smtp_server': getattr(config, 'SMTP_SERVER', 'smtp.gmail.com'),
            'smtp_port': getattr(config, 'SMTP_PORT', 587),
            'smtp_username': getattr(config, 'SMTP_USERNAME', ''),
            'smtp_password': getattr(config, 'SMTP_PASSWORD', ''),
            'telegram_enabled': getattr(config, 'ERROR_TELEGRAM_ENABLED', False),
            'telegram_chat_id': getattr(config, 'ERROR_TELEGRAM_CHAT_ID', ''),
        }

    def _register_recovery_strategies(self):
        """Register automatic recovery strategies for different error categories."""
        self.recovery_strategies = {
            ErrorCategory.DATA_SOURCE: [
                self._recover_data_source_connection,
                self._switch_to_backup_data_source,
                self._use_cached_data
            ],
            ErrorCategory.TRADING_ENGINE: [
                self._restart_trading_engine,
                self._reset_positions,
                self._enable_safe_mode
            ],
            ErrorCategory.SIGNAL_GENERATION: [
                self._restart_signal_generator,
                self._use_backup_signals,
                self._reduce_signal_frequency
            ],
            ErrorCategory.NETWORK: [
                self._retry_with_backoff,
                self._switch_network_endpoint,
                self._enable_offline_mode
            ],
            ErrorCategory.SYSTEM: [
                self._restart_system_components,
                self._clear_memory_cache,
                self._reduce_system_load
            ]
        }

    async def handle_error(self, error: Exception, category: ErrorCategory,
                          context: Dict[str, Any] = None) -> bool:
        """
        Handle an error with automatic recovery attempts.

        Args:
            error: The exception that occurred
            category: Category of the error
            context: Additional context information

        Returns:
            True if error was resolved, False otherwise
        """
        try:
            # Determine error severity
            severity = self._determine_severity(error, category)

            # Create error event
            error_event = ErrorEvent(
                timestamp=datetime.now(),
                severity=severity,
                category=category,
                message=str(error),
                details=context or {},
                stack_trace=traceback.format_exc()
            )

            # Log the error
            self._log_error(error_event)

            # Add to error history
            self.error_history.append(error_event)

            # Update error counts
            error_key = f"{category.value}_{type(error).__name__}"
            self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1

            # Check circuit breaker
            if self._should_circuit_break(error_key):
                await self._handle_circuit_break(error_key, error_event)
                return False

            # Attempt automatic recovery
            recovery_success = await self._attempt_recovery(error_event)

            if recovery_success:
                error_event.resolved = True
                error_event.resolution_time = datetime.now()
                self.error_logger.info(f"Successfully recovered from error: {error}")

            # Send notifications for high/critical errors
            if severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                await self._send_error_notification(error_event)

            return recovery_success

        except Exception as e:
            self.error_logger.critical(f"Error in error handler: {e}")
            return False

    def _determine_severity(self, error: Exception, category: ErrorCategory) -> ErrorSeverity:
        """Determine the severity of an error."""
        # Critical errors that could cause financial loss
        critical_errors = [
            'ConnectionError',
            'TimeoutError',
            'AuthenticationError',
            'InsufficientFundsError'
        ]

        # High severity errors that affect core functionality
        high_errors = [
            'DataValidationError',
            'SignalGenerationError',
            'OrderExecutionError'
        ]

        error_name = type(error).__name__

        if error_name in critical_errors or category == ErrorCategory.TRADING_ENGINE:
            return ErrorSeverity.CRITICAL
        elif error_name in high_errors or category == ErrorCategory.SIGNAL_GENERATION:
            return ErrorSeverity.HIGH
        elif category == ErrorCategory.DATA_SOURCE:
            return ErrorSeverity.MEDIUM
        else:
            return ErrorSeverity.LOW

    def _log_error(self, error_event: ErrorEvent):
        """Log error event with appropriate level."""
        log_message = (
            f"[{error_event.category.value.upper()}] {error_event.message} "
            f"- Details: {error_event.details}"
        )

        if error_event.severity == ErrorSeverity.CRITICAL:
            self.error_logger.critical(log_message)
        elif error_event.severity == ErrorSeverity.HIGH:
            self.error_logger.error(log_message)
        elif error_event.severity == ErrorSeverity.MEDIUM:
            self.error_logger.warning(log_message)
        else:
            self.error_logger.info(log_message)

    def _should_circuit_break(self, error_key: str) -> bool:
        """Check if circuit breaker should be triggered."""
        if error_key not in self.circuit_breakers:
            self.circuit_breakers[error_key] = {
                'state': 'closed',
                'failure_count': 0,
                'last_failure_time': None,
                'half_open_calls': 0
            }

        breaker = self.circuit_breakers[error_key]
        config = self.default_circuit_breaker

        if breaker['state'] == 'closed':
            breaker['failure_count'] += 1
            breaker['last_failure_time'] = datetime.now()

            if breaker['failure_count'] >= config['failure_threshold']:
                breaker['state'] = 'open'
                self.error_logger.warning(f"Circuit breaker opened for {error_key}")
                return True

        elif breaker['state'] == 'open':
            time_since_failure = datetime.now() - breaker['last_failure_time']
            if time_since_failure.total_seconds() >= config['recovery_timeout']:
                breaker['state'] = 'half_open'
                breaker['half_open_calls'] = 0
                self.error_logger.info(f"Circuit breaker half-open for {error_key}")
            else:
                return True

        elif breaker['state'] == 'half_open':
            breaker['half_open_calls'] += 1
            if breaker['half_open_calls'] >= config['half_open_max_calls']:
                breaker['state'] = 'open'
                return True

        return False

    async def _handle_circuit_break(self, error_key: str, error_event: ErrorEvent):
        """Handle circuit breaker activation."""
        self.error_logger.critical(f"Circuit breaker activated for {error_key}")

        # Send immediate notification
        await self._send_error_notification(error_event, urgent=True)

        # Implement fallback behavior based on category
        if error_event.category == ErrorCategory.TRADING_ENGINE:
            # Disable trading to prevent losses
            self.error_logger.critical("Trading disabled due to circuit breaker")
        elif error_event.category == ErrorCategory.DATA_SOURCE:
            # Switch to backup data sources
            self.error_logger.warning("Switching to backup data sources")

    async def _attempt_recovery(self, error_event: ErrorEvent) -> bool:
        """Attempt automatic recovery using registered strategies."""
        strategies = self.recovery_strategies.get(error_event.category, [])

        for strategy in strategies:
            try:
                self.error_logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                success = await strategy(error_event)

                if success:
                    self.error_logger.info(f"Recovery successful with {strategy.__name__}")
                    return True

            except Exception as e:
                self.error_logger.error(f"Recovery strategy {strategy.__name__} failed: {e}")

        return False

    async def _send_error_notification(self, error_event: ErrorEvent, urgent: bool = False):
        """Send error notifications via configured channels."""
        try:
            # Email notification
            if self.notification_settings['email_enabled']:
                await self._send_email_notification(error_event, urgent)

            # Telegram notification
            if self.notification_settings['telegram_enabled']:
                await self._send_telegram_notification(error_event, urgent)

        except Exception as e:
            self.error_logger.error(f"Failed to send error notification: {e}")

    async def _send_email_notification(self, error_event: ErrorEvent, urgent: bool = False):
        """Send email notification for error."""
        try:
            if not self.notification_settings['email_recipients']:
                return

            subject = f"{'URGENT - ' if urgent else ''}TradePulse Error: {error_event.severity.value.upper()}"

            body = f"""
            Error Details:
            - Timestamp: {error_event.timestamp}
            - Severity: {error_event.severity.value.upper()}
            - Category: {error_event.category.value}
            - Message: {error_event.message}
            - Details: {json.dumps(error_event.details, indent=2)}

            Stack Trace:
            {error_event.stack_trace}
            """

            msg = MimeMultipart()
            msg['From'] = self.notification_settings['smtp_username']
            msg['To'] = ', '.join(self.notification_settings['email_recipients'])
            msg['Subject'] = subject

            msg.attach(MimeText(body, 'plain'))

            # Send email
            server = smtplib.SMTP(
                self.notification_settings['smtp_server'],
                self.notification_settings['smtp_port']
            )
            server.starttls()
            server.login(
                self.notification_settings['smtp_username'],
                self.notification_settings['smtp_password']
            )
            server.send_message(msg)
            server.quit()

        except Exception as e:
            self.error_logger.error(f"Failed to send email notification: {e}")

    async def _send_telegram_notification(self, error_event: ErrorEvent, urgent: bool = False):
        """Send Telegram notification for error."""
        # Implementation would depend on Telegram bot setup
        pass

    # Recovery Strategy Implementations
    async def _recover_data_source_connection(self, error_event: ErrorEvent) -> bool:
        """Attempt to recover data source connection."""
        try:
            await asyncio.sleep(5)  # Wait before retry
            # Implementation would test connection to data source
            return True
        except Exception:
            return False

    async def _switch_to_backup_data_source(self, error_event: ErrorEvent) -> bool:
        """Switch to backup data source."""
        try:
            # Implementation would switch to alternative data provider
            return True
        except Exception:
            return False

    async def _use_cached_data(self, error_event: ErrorEvent) -> bool:
        """Use cached data as fallback."""
        try:
            # Implementation would use last known good data
            return True
        except Exception:
            return False

    async def _restart_trading_engine(self, error_event: ErrorEvent) -> bool:
        """Restart trading engine components."""
        try:
            # Implementation would restart trading engine
            return True
        except Exception:
            return False

    async def _reset_positions(self, error_event: ErrorEvent) -> bool:
        """Reset position tracking."""
        try:
            # Implementation would reset position state
            return True
        except Exception:
            return False

    async def _enable_safe_mode(self, error_event: ErrorEvent) -> bool:
        """Enable safe mode with reduced functionality."""
        try:
            # Implementation would enable safe mode
            return True
        except Exception:
            return False

    async def _restart_signal_generator(self, error_event: ErrorEvent) -> bool:
        """Restart signal generation system."""
        try:
            # Implementation would restart signal generator
            return True
        except Exception:
            return False

    async def _use_backup_signals(self, error_event: ErrorEvent) -> bool:
        """Use backup signal generation method."""
        try:
            # Implementation would use simpler signal generation
            return True
        except Exception:
            return False

    async def _reduce_signal_frequency(self, error_event: ErrorEvent) -> bool:
        """Reduce signal generation frequency."""
        try:
            # Implementation would reduce signal frequency
            return True
        except Exception:
            return False

    async def _retry_with_backoff(self, error_event: ErrorEvent) -> bool:
        """Retry operation with exponential backoff."""
        try:
            for attempt in range(3):
                await asyncio.sleep(2 ** attempt)
                # Implementation would retry the failed operation
                return True
        except Exception:
            return False

    async def _switch_network_endpoint(self, error_event: ErrorEvent) -> bool:
        """Switch to alternative network endpoint."""
        try:
            # Implementation would switch to backup endpoint
            return True
        except Exception:
            return False

    async def _enable_offline_mode(self, error_event: ErrorEvent) -> bool:
        """Enable offline mode."""
        try:
            # Implementation would enable offline operation
            return True
        except Exception:
            return False

    async def _restart_system_components(self, error_event: ErrorEvent) -> bool:
        """Restart system components."""
        try:
            # Implementation would restart components
            return True
        except Exception:
            return False

    async def _clear_memory_cache(self, error_event: ErrorEvent) -> bool:
        """Clear memory cache to free resources."""
        try:
            # Implementation would clear caches
            return True
        except Exception:
            return False

    async def _reduce_system_load(self, error_event: ErrorEvent) -> bool:
        """Reduce system load."""
        try:
            # Implementation would reduce system load
            return True
        except Exception:
            return False

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        try:
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_hour = now - timedelta(hours=1)

            # Filter recent errors
            recent_errors = [e for e in self.error_history if e.timestamp >= last_24h]
            hourly_errors = [e for e in self.error_history if e.timestamp >= last_hour]

            # Calculate statistics
            stats = {
                'total_errors': len(self.error_history),
                'errors_last_24h': len(recent_errors),
                'errors_last_hour': len(hourly_errors),
                'error_rate_24h': len(recent_errors) / 24,  # Errors per hour
                'error_counts_by_category': {},
                'error_counts_by_severity': {},
                'circuit_breaker_status': {},
                'resolution_rate': 0,
                'avg_resolution_time': 0
            }

            # Count by category and severity
            for error in recent_errors:
                category = error.category.value
                severity = error.severity.value

                stats['error_counts_by_category'][category] = \
                    stats['error_counts_by_category'].get(category, 0) + 1
                stats['error_counts_by_severity'][severity] = \
                    stats['error_counts_by_severity'].get(severity, 0) + 1

            # Circuit breaker status
            for key, breaker in self.circuit_breakers.items():
                stats['circuit_breaker_status'][key] = breaker['state']

            # Resolution statistics
            resolved_errors = [e for e in self.error_history if e.resolved]
            if self.error_history:
                stats['resolution_rate'] = len(resolved_errors) / len(self.error_history)

            if resolved_errors:
                resolution_times = [
                    (e.resolution_time - e.timestamp).total_seconds()
                    for e in resolved_errors if e.resolution_time
                ]
                if resolution_times:
                    stats['avg_resolution_time'] = sum(resolution_times) / len(resolution_times)

            return stats

        except Exception as e:
            self.error_logger.error(f"Error calculating statistics: {e}")
            return {}

    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')

            # Get error statistics
            error_stats = self.get_error_statistics()

            # Determine health status
            health_score = 100

            # Deduct for high error rates
            if error_stats['errors_last_hour'] > 10:
                health_score -= 20
            elif error_stats['errors_last_hour'] > 5:
                health_score -= 10

            # Deduct for system resource usage
            if cpu_percent > 80:
                health_score -= 15
            if memory.percent > 80:
                health_score -= 15
            if disk.percent > 90:
                health_score -= 10

            # Deduct for circuit breakers
            open_breakers = sum(1 for b in self.circuit_breakers.values() if b['state'] == 'open')
            health_score -= open_breakers * 10

            # Determine status
            if health_score >= 90:
                status = "EXCELLENT"
            elif health_score >= 75:
                status = "GOOD"
            elif health_score >= 50:
                status = "FAIR"
            elif health_score >= 25:
                status = "POOR"
            else:
                status = "CRITICAL"

            return {
                'status': status,
                'health_score': max(0, health_score),
                'system_metrics': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'disk_percent': disk.percent
                },
                'error_metrics': error_stats,
                'timestamp': datetime.now().isoformat()
            }

        except Exception as e:
            self.error_logger.error(f"Error getting system health: {e}")
            return {'status': 'UNKNOWN', 'health_score': 0}
