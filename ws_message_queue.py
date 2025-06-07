"""Message queue implementation for WebSocket communication."""
import asyncio
import logging
from typing import Any, Optional, Dict, List
from dataclasses import dataclass
from datetime import datetime
import json

@dataclass
class QueuedMessage:
    """Represents a message in the queue with metadata."""
    message_id: str
    content: Any
    retry_count: int = 0
    timestamp: datetime = datetime.now()
    response_future: Optional[asyncio.Future] = None
    timeout: float = 10.0

class MessageQueue:
    """Implements a message queue with retry and timeout capabilities."""
    
    def __init__(self, max_retries: int = 3, max_queue_size: int = 1000):
        self.logger = logging.getLogger(__name__)
        self.queue: asyncio.Queue[QueuedMessage] = asyncio.Queue(maxsize=max_queue_size)
        self.pending_messages: Dict[str, QueuedMessage] = {}
        self.max_retries = max_retries
        self._processor_task: Optional[asyncio.Task] = None
        self._response_handlers: Dict[str, List[asyncio.Future]] = {}

    async def put_message(self, message_id: str, content: Any, timeout: float = 10.0) -> asyncio.Future:
        """Queue a message for sending and return a future for the response."""
        response_future = asyncio.get_event_loop().create_future()
        queued_message = QueuedMessage(
            message_id=message_id,
            content=content,
            response_future=response_future,
            timeout=timeout
        )
        
        await self.queue.put(queued_message)
        self.pending_messages[message_id] = queued_message
        
        # Set up timeout
        asyncio.create_task(self._handle_timeout(message_id, timeout))
        return response_future

    async def process_response(self, message_id: str, response: Any) -> None:
        """Process a response message for a pending request."""
        if message_id in self.pending_messages:
            queued_message = self.pending_messages[message_id]
            if not queued_message.response_future.done():
                queued_message.response_future.set_result(response)
            del self.pending_messages[message_id]

    async def _handle_timeout(self, message_id: str, timeout: float):
        """Handle message timeout and trigger retry if needed."""
        await asyncio.sleep(timeout)
        if message_id in self.pending_messages:
            queued_message = self.pending_messages[message_id]
            if queued_message.retry_count < self.max_retries:
                # Requeue the message
                queued_message.retry_count += 1
                self.logger.warning(f"Message {message_id} timed out, retrying (attempt {queued_message.retry_count}/{self.max_retries})")
                await self.queue.put(queued_message)
            else:
                # Failed after max retries
                self.logger.error(f"Message {message_id} failed after {self.max_retries} retries")
                if not queued_message.response_future.done():
                    queued_message.response_future.set_exception(
                        TimeoutError(f"Message {message_id} timed out after {self.max_retries} retries")
                    )
                del self.pending_messages[message_id]

    def validate_message(self, message: Any) -> bool:
        """Validate message format before sending."""
        try:
            if isinstance(message, str):
                # Try to parse as JSON if it's a string
                json.loads(message)
                return True
            elif isinstance(message, (dict, list)):
                # Validate can be serialized to JSON
                json.dumps(message)
                return True
            return False
        except json.JSONDecodeError:
            return isinstance(message, str)  # Accept plain strings that aren't JSON
        except Exception:
            return False

    async def start_processing(self, send_callback):
        """Start processing messages in the queue."""
        async def processor():
            while True:
                try:
                    message = await self.queue.get()
                    if self.validate_message(message.content):
                        await send_callback(message.content)
                    self.queue.task_done()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Error processing message: {str(e)}")
                    await asyncio.sleep(1)  # Prevent tight loop on errors
        
        self._processor_task = asyncio.create_task(processor())

    async def stop_processing(self):
        """Stop the message processor task."""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass
            self._processor_task = None
