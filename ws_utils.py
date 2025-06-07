"""Utility functions and classes for WebSocket operations."""
import json
import logging
from typing import Any, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum, auto

class WebSocketError(Exception):
    """Base exception class for WebSocket errors."""
    pass

class ConnectionError(WebSocketError):
    """Exception for connection-related errors."""
    pass

class MessageError(WebSocketError):
    """Exception for message-related errors."""
    pass

class TimeoutError(WebSocketError):
    """Exception for timeout-related errors."""
    pass

class MessageType(Enum):
    """Enumeration of WebSocket message types."""
    PING = auto()
    PONG = auto()
    AUTH = auto()
    TIME_SYNC = auto()
    TRADE = auto()
    MARKET_DATA = auto()
    ERROR = auto()
    UNKNOWN = auto()

@dataclass
class WSMessage:
    """Structured representation of a WebSocket message."""
    type: MessageType
    content: Any
    message_id: Optional[str] = None
    timestamp: Optional[float] = None
    is_response: bool = False

class MessageValidator:
    """Validates and processes WebSocket messages."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def validate_message(self, message: Union[str, Dict, bytes]) -> bool:
        """
        Validate message format and content.
        
        Args:
            message: The message to validate
            
        Returns:
            bool: True if message is valid, False otherwise
        """
        try:
            if isinstance(message, bytes):
                return True  # Binary messages are passed through
            
            if isinstance(message, str):
                # Handle ping/pong messages
                if message in ("2", "3"):
                    return True
                
                # Try to parse as JSON
                try:
                    parsed = json.loads(message)
                    return self._validate_json_structure(parsed)
                except json.JSONDecodeError:
                    # Accept plain string messages
                    return True
            
            elif isinstance(message, dict):
                return self._validate_json_structure(message)
            
            return False
            
        except Exception as e:
            self.logger.error(f"Message validation error: {str(e)}")
            return False

    def _validate_json_structure(self, data: Dict) -> bool:
        """
        Validate the structure of a JSON message.
        
        Args:
            data: The parsed JSON data
            
        Returns:
            bool: True if structure is valid, False otherwise
        """
        # Check for required fields based on message type
        if isinstance(data, list) and len(data) >= 2:
            # Handle socket.io protocol messages
            return True
        
        if isinstance(data, dict):
            # Check for common required fields
            if "type" in data or "action" in data or "message" in data:
                return True
        
        return True  # Accept other well-formed JSON

    def parse_message(self, message: Union[str, Dict, bytes]) -> Optional[WSMessage]:
        """
        Parse a raw message into a structured WSMessage object.
        
        Args:
            message: The raw message to parse
            
        Returns:
            Optional[WSMessage]: Structured message object if parsing succeeds, None otherwise
        """
        try:
            if isinstance(message, bytes):
                return WSMessage(type=MessageType.UNKNOWN, content=message)
            
            if isinstance(message, str):
                # Handle ping/pong messages
                if message == "2":
                    return WSMessage(type=MessageType.PING, content=message)
                if message == "3":
                    return WSMessage(type=MessageType.PONG, content=message)
                
                # Try to parse JSON messages
                if message.startswith('42["'):
                    try:
                        data = json.loads(message[2:])
                        if isinstance(data, list) and len(data) >= 2:
                            msg_type = self._determine_message_type(data[0])
                            return WSMessage(
                                type=msg_type,
                                content=data[1] if len(data) > 1 else None,
                                message_id=data[0]
                            )
                    except json.JSONDecodeError:
                        pass
            
            if isinstance(message, dict):
                msg_type = self._determine_message_type(message.get("type") or message.get("action"))
                return WSMessage(type=msg_type, content=message)
            
            return WSMessage(type=MessageType.UNKNOWN, content=message)
            
        except Exception as e:
            self.logger.error(f"Message parsing error: {str(e)}")
            return None

    def _determine_message_type(self, type_str: Optional[str]) -> MessageType:
        """Map string message types to MessageType enum."""
        if not type_str:
            return MessageType.UNKNOWN
            
        type_map = {
            "auth": MessageType.AUTH,
            "time-sync": MessageType.TIME_SYNC,
            "trade": MessageType.TRADE,
            "market": MessageType.MARKET_DATA,
            "error": MessageType.ERROR
        }
        
        return type_map.get(type_str.lower(), MessageType.UNKNOWN)

class WSErrorHandler:
    """Handles WebSocket errors and implements error recovery strategies."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self._error_counts: Dict[str, int] = {}
        self._max_errors = 5
        self._error_window = 300  # 5 minutes

    def handle_error(self, error: Exception, context: str = "") -> bool:
        """
        Handle an error and determine if operation should continue.
        
        Args:
            error: The exception that occurred
            context: Additional context about where the error occurred
            
        Returns:
            bool: True if operation should continue, False if it should abort
        """
        error_key = f"{type(error).__name__}:{context}"
        
        # Increment error count
        self._error_counts[error_key] = self._error_counts.get(error_key, 0) + 1
        
        # Log the error
        self.logger.error(f"WebSocket error in {context}: {str(error)}")
        
        # Check if we've hit the error threshold
        if self._error_counts[error_key] >= self._max_errors:
            self.logger.critical(
                f"Error threshold reached for {error_key}. "
                f"Had {self._error_counts[error_key]} errors in context: {context}"
            )
            return False
        
        return True

    def clear_error_counts(self, context: Optional[str] = None):
        """Clear error counts for the given context or all contexts."""
        if context:
            self._error_counts = {k: v for k, v in self._error_counts.items() if not k.endswith(context)}
        else:
            self._error_counts.clear()
