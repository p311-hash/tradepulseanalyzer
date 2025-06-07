import asyncio
import json
import logging
import websockets
from typing import Optional, Any
import uuid
from ws_message_queue import MessageQueue
from ws_utils import (
    MessageValidator, 
    WSErrorHandler, 
    WSMessage, 
    MessageType, 
    WebSocketError
)

class WebsocketClient:
    """WebSocket client implementation for binary options trading."""
    
    def __init__(self, api):
        self.api = api
        self.websocket: Optional[websockets.WebSocketClientProtocol] = None
        self.logger = logging.getLogger(__name__)
        self.wss_url = "wss://api-us-north.po.market/socket.io/?EIO=4&transport=websocket"
        self.connected = False
        self.loop = asyncio.get_event_loop()
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.base_delay = 1.0  # Base delay in seconds
        self.max_delay = 60.0  # Maximum delay in seconds
        self.last_message_time = 0
        self.last_server_timestamp = None
        self._connection_lock = asyncio.Lock()
        self.message_queue = MessageQueue()
        self.message_validator = MessageValidator()
        self.error_handler = WSErrorHandler()
        self._receiver_task: Optional[asyncio.Task] = None

    async def connect(self) -> tuple[bool, Optional[str]]:
        """Establish WebSocket connection with exponential backoff."""
        async with self._connection_lock:
            while self.reconnect_attempts < self.max_reconnect_attempts:
                try:
                    delay = min(self.base_delay * (2 ** self.reconnect_attempts), self.max_delay)
                    
                    if self.reconnect_attempts > 0:
                        self.logger.info(f"Reconnection attempt {self.reconnect_attempts + 1}/{self.max_reconnect_attempts} after {delay:.1f}s delay")
                        await asyncio.sleep(delay)
                    
                    self.websocket = await websockets.connect(
                        self.wss_url,
                        ssl=True,
                        ping_interval=None,
                        max_size=10_485_760,  # 10MB max message size
                        close_timeout=10,  # 10 second timeout for clean closure
                        extra_headers={
                            "Origin": "https://pocketoption.com",
                            "Cache-Control": "no-cache"
                        }
                    )
                    
                    self.connected = True
                    self.reconnect_attempts = 0  # Reset counter on successful connection
                    self.last_message_time = asyncio.get_event_loop().time()
                    
                    # Start the background tasks
                    self._start_background_tasks()
                    
                    self.logger.info("WebSocket connection established")
                    return True, None
                    
                except Exception as e:
                    self.reconnect_attempts += 1
                    self.logger.error(f"WebSocket connection attempt {self.reconnect_attempts} failed: {str(e)}")
                    if self.reconnect_attempts >= self.max_reconnect_attempts:
                        self.logger.error("Maximum reconnection attempts reached")
                        return False, str(e)
        
        return False, "Connection failed after maximum retries"

    def _start_background_tasks(self):
        """Start all background tasks for connection monitoring and message handling."""
        # Start message queue processor
        asyncio.create_task(self.message_queue.start_processing(self._send_raw_message))
        
        # Start the connection monitoring task
        asyncio.create_task(self._monitor_connection())
        
        # Start the message receiver task
        self._receiver_task = asyncio.create_task(self._message_receiver())
        
        # Start the keep-alive task
        asyncio.create_task(self.keep_alive())

    async def send_message(self, message: Any, timeout: float = 10.0) -> Any:
        """Send message through WebSocket connection with queuing and response handling."""
        if not self.websocket or not self.connected:
            self.logger.error("Cannot send message - WebSocket not connected")
            await self.connect()  # Try to reconnect
            if not self.connected:
                raise ConnectionError("Failed to establish connection")
        
        # Validate message before sending
        if not self.message_validator.validate_message(message):
            raise ValueError(f"Invalid message format: {message}")
        
        message_id = str(uuid.uuid4())
        try:
            response_future = await self.message_queue.put_message(message_id, message, timeout)
            return await response_future
        except Exception as e:
            if not self.error_handler.handle_error(e, "send_message"):
                raise
            self.logger.error(f"Failed to send message: {str(e)}")
            raise

    async def _send_raw_message(self, message: Any) -> None:
        """Send raw message directly through WebSocket."""
        try:
            if isinstance(message, (dict, list)):
                await self.websocket.send(json.dumps(message))
            else:
                await self.websocket.send(message)
            self.last_message_time = asyncio.get_event_loop().time()
        except Exception as e:
            self.logger.error(f"Failed to send raw message: {str(e)}")
            await self._handle_connection_failure()
            raise

    async def _validate_and_parse_message(self, message: Any) -> Optional[WSMessage]:
        """Validate and parse an incoming message."""
        if not self.message_validator.validate_message(message):
            self.logger.warning(f"Invalid message received: {message}")
            return None
            
        parsed = self.message_validator.parse_message(message)
        if parsed is None:
            self.logger.warning(f"Failed to parse message: {message}")
            return None
            
        return parsed

    async def _handle_parsed_message(self, parsed: WSMessage) -> None:
        """Handle a parsed message based on its type."""
        try:
            if parsed.type == MessageType.PING:
                await self._send_raw_message("3")  # Send pong
            elif parsed.type == MessageType.TIME_SYNC:
                self.last_server_timestamp = parsed.content.get("timestamp")
            elif parsed.type == MessageType.ERROR:
                self.logger.error(f"Received error message: {parsed.content}")
                if not self.error_handler.handle_error(
                    WebSocketError(str(parsed.content)),
                    "message_handler"
                ):
                    await self._handle_connection_failure()
            elif parsed.message_id:
                # Handle response messages
                await self.message_queue.process_response(parsed.message_id, parsed.content)
        except Exception as e:
            self.logger.error(f"Error handling message: {str(e)}")
            self.error_handler.handle_error(e, "message_handler")

    async def _message_receiver(self):
        """Background task for receiving messages."""
        while self.connected:
            try:
                if self.websocket:
                    message = await self.websocket.recv()
                    self.last_message_time = asyncio.get_event_loop().time()
                    
                    # Validate and parse message
                    parsed = await self._validate_and_parse_message(message)
                    if parsed:
                        await self._handle_parsed_message(parsed)
                    
                    self.logger.debug(f"Received message: {message}")
                    
            except websockets.ConnectionClosed as e:
                if not self.error_handler.handle_error(e, "connection"):
                    self.logger.error("Connection permanently lost")
                    break
                self.logger.warning("WebSocket connection closed")
                await self._handle_connection_failure()
                break
            except Exception as e:
                if not self.error_handler.handle_error(e, "receiver"):
                    self.logger.error("Too many receiver errors, stopping")
                    break
                await asyncio.sleep(1)  # Prevent tight loop on errors

    async def close(self):
        """Close the WebSocket connection and cleanup resources."""
        self.connected = False
        
        # Stop background tasks
        if self._receiver_task:
            self._receiver_task.cancel()
            try:
                await self._receiver_task
            except asyncio.CancelledError:
                pass
        
        await self.message_queue.stop_processing()
        
        if self.websocket:
            try:
                await self.websocket.close()
                self.logger.info("WebSocket connection closed")
            except Exception as e:
                self.logger.error(f"Error closing WebSocket: {str(e)}")

    async def _monitor_connection(self):
        """Monitor connection health and reconnect if necessary."""
        while self.connected:
            current_time = asyncio.get_event_loop().time()
            if current_time - self.last_message_time > 30:  # No messages for 30 seconds
                try:
                    self.logger.warning("No messages received for 30 seconds, checking connection...")
                    await self.send_message("2")  # Send ping
                except Exception as e:
                    if not self.error_handler.handle_error(e, "monitor"):
                        self.logger.error("Connection monitoring failed permanently")
                        break
                    self.logger.error("Connection check failed, initiating reconnect")
                    await self._handle_connection_failure()
            await asyncio.sleep(5)  # Check every 5 seconds
