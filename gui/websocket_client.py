import websocket
import threading
import queue
import time
import json
import logging
import random
from typing import Callable, Dict, Any, Optional, List, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("websocket_client")

class RobustWebSocket:
    """
    A robust WebSocket client with automatic reconnection,
    message queuing, and clean shutdown.
    """
    
    def __init__(self, url: str, on_message: Callable, max_queue_size: int = 500):
        """
        Initialize a robust WebSocket connection.
        
        Args:
            url: WebSocket URL to connect to
            on_message: Callback function that receives parsed messages
            max_queue_size: Maximum size of message queue (drops oldest if full)
        """
        self.url = url
        self.user_callback = on_message
        self.message_queue = queue.Queue(maxsize=max_queue_size)
        self.ws = None
        self.connected = False
        self.should_reconnect = True
        self.reconnect_delay = 1  # Start with 1 second delay
        self.max_reconnect_delay = 300  # Maximum 5 minutes
        self.stop_event = threading.Event()
        self.connect_lock = threading.Lock()
        
        # Start message processing thread
        self.process_thread = threading.Thread(target=self._process_messages)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        # Start connection in a separate thread to avoid blocking GUI
        threading.Thread(target=self._connect_in_thread, daemon=True).start()
        
        logger.info(f"WebSocket initialized for {url}")
    
    def _connect_in_thread(self):
        """Run _connect in a separate thread to avoid blocking the GUI thread."""
        with self.connect_lock:
            self._connect()
    
    def _connect(self):
        """Establish WebSocket connection."""
        try:
            # Close existing connection if any
            if self.ws:
                try:
                    self.ws.close()
                except Exception as e:
                    logger.warning(f"Error closing existing WebSocket: {e}")
            
            self.ws = websocket.WebSocketApp(
                self.url,
                on_message=self._on_message,
                on_error=self._on_error,
                on_close=self._on_close,
                on_open=self._on_open
            )
            
            # Start the WebSocket connection in a thread
            self.ws_thread = threading.Thread(target=self.ws.run_forever, kwargs={
                'ping_interval': 15,  # Send ping every 15 seconds
                'ping_timeout': 10    # Wait 10 seconds for pong before considering connection dead
            })
            self.ws_thread.daemon = True
            self.ws_thread.start()
            
            logger.info(f"WebSocket connecting to {self.url}")
            
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
            self._schedule_reconnect()
    
    def _on_message(self, ws, message):
        """Handle incoming WebSocket message."""
        try:
            # Try to add to queue, discard if full
            try:
                self.message_queue.put_nowait(message)
                logger.debug(f"WebSocket message queued: {message[:100]}...")
            except queue.Full:
                logger.warning("WebSocket message queue full, dropping oldest message")
                # Discard oldest and add new
                try:
                    self.message_queue.get_nowait()
                    self.message_queue.put_nowait(message)
                except queue.Empty:
                    logger.warning("Failed to get item from queue that was reported as full")
                except queue.Full:
                    logger.warning("Failed to add new message after removing old one")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
    
    def _on_error(self, ws, error):
        """Handle WebSocket error."""
        logger.error(f"WebSocket error: {error}")
    
    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closure."""
        logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.connected = False
        if self.should_reconnect and not self.stop_event.is_set():
            self._schedule_reconnect()
    
    def _on_open(self, ws):
        """Handle WebSocket connection establishment."""
        logger.info(f"WebSocket connected to {self.url}")
        self.connected = True
        self.reconnect_delay = 1  # Reset reconnect delay
    
    def _schedule_reconnect(self):
        """Schedule reconnection with exponential backoff and jitter."""
        if not self.should_reconnect or self.stop_event.is_set():
            return
            
        # Add jitter to avoid thundering herd problem
        jitter_factor = random.uniform(0.8, 1.2)  # ±20% jitter
        delay_with_jitter = self.reconnect_delay * jitter_factor
        
        logger.info(f"Reconnecting in {delay_with_jitter:.2f} seconds...")
        time.sleep(delay_with_jitter)
        
        # Exponential backoff
        self.reconnect_delay = min(self.reconnect_delay * 2, self.max_reconnect_delay)
        
        # Reconnect in a separate thread to avoid blocking
        threading.Thread(target=self._connect_in_thread, daemon=True).start()
    
    def _process_messages(self):
        """Process messages from the queue and deliver to user callback."""
        last_log_time = datetime.now()
        log_summary_interval = 5.0  # Summarize logs every 5 seconds

        while not self.stop_event.is_set():
            try:
                # Block for 0.5 seconds, then check if we should stop
                try:
                    message = self.message_queue.get(timeout=0.5)
                    # Parse JSON and deliver to user callback
                    try:
                        data = json.loads(message)
                        # Log summary of messages periodically instead of every message
                        now = datetime.now()
                        time_diff = (now - last_log_time).total_seconds()
                        if time_diff >= log_summary_interval:
                            last_log_time = now
                            logger.debug(f"Processed {self.message_queue.qsize() + 1} WebSocket messages in last {time_diff:.1f} seconds")
                        
                        # Deliver message to user callback
                        self.user_callback(data)
                    except json.JSONDecodeError:
                        logger.error(f"Invalid JSON in WebSocket message: {message[:100]}...")
                except queue.Empty:
                    continue
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
            
        logger.info("WebSocket message processor stopped")
    
    def send(self, message: Union[Dict[str, Any], str]) -> bool:
        """
        Send a message through the WebSocket.
        
        Args:
            message: Either a dictionary to be JSON-serialized and sent,
                    or a string to be sent directly as a text message
        
        Returns:
            Success or failure
        """
        if not self.connected:
            logger.warning("Cannot send message, WebSocket not connected")
            return False
        
        try:
            # Handle both dictionary and string messages
            if isinstance(message, dict):
                self.ws.send(json.dumps(message))
            else:
                self.ws.send(str(message))
            return True
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            return False
    
    def stop(self):
        """Stop the WebSocket connection and message processor."""
        logger.info("Stopping WebSocket...")
        self.should_reconnect = False
        self.stop_event.set()
        
        if self.ws:
            try:
                self.ws.close()
            except Exception as e:
                logger.warning(f"Error closing WebSocket: {e}")
        
        # Wait for processing thread to finish
        try:
            self.process_thread.join(timeout=1.0)
        except Exception as e:
            logger.warning(f"Error joining processing thread: {e}")
            
        logger.info("WebSocket stopped")


class WebSocketManager:
    """
    Manager for WebSocket connections to handle multiple connections
    and provide a facade for the GUI components.
    """
    
    def __init__(self, base_url: str = "ws://localhost:8000"):
        """
        Initialize WebSocket manager.
        
        Args:
            base_url: Base URL for WebSocket connections (will be converted from http to ws if needed)
        """
        # Convert http:// to ws:// or https:// to wss:// if necessary
        if base_url.startswith("http://"):
            base_url = base_url.replace("http://", "ws://")
        elif base_url.startswith("https://"):
            base_url = base_url.replace("https://", "wss://")
            
        self.base_url = base_url
        self.active_connections: Dict[str, RobustWebSocket] = {}
        self.logger = logger
        
        self.logger.info(f"WebSocketManager initialized with base URL: {base_url}")
    
    def create_workflow_connection(self, workflow_run_id: str, callback: Callable[[Dict[str, Any]], None]) -> RobustWebSocket:
        """
        Create a WebSocket connection for a workflow run.
        
        Args:
            workflow_run_id: ID of the workflow run
            callback: Function to call when messages are received
            
        Returns:
            WebSocket connection
        """
        url = f"{self.base_url}/ws/workflow/{workflow_run_id}"
        
        # Close existing connection if any
        if workflow_run_id in self.active_connections:
            self.close_connection(workflow_run_id)
        
        # Create new connection
        ws = RobustWebSocket(url, callback)
        self.active_connections[workflow_run_id] = ws
        return ws
    
    def create_dpo_training_connection(self, job_id: str, callback: Callable[[Dict[str, Any]], None]) -> RobustWebSocket:
        """
        Create a WebSocket connection for a DPO training job.
        
        Args:
            job_id: ID of the DPO training job
            callback: Function to call when messages are received
            
        Returns:
            WebSocket connection
        """
        url = f"{self.base_url}/dpo/ws/{job_id}"
        
        # Close existing connection if any
        connection_key = f"dpo_{job_id}"
        if connection_key in self.active_connections:
            self.close_connection(connection_key)
        
        # Create new connection
        ws = RobustWebSocket(url, callback)
        self.active_connections[connection_key] = ws
        return ws
    
    def close_connection(self, key: str) -> bool:
        """
        Close a specific WebSocket connection.
        
        Args:
            key: Connection key (workflow_run_id or "dpo_" + job_id)
            
        Returns:
            Success or failure
        """
        if key in self.active_connections:
            try:
                self.active_connections[key].stop()
                del self.active_connections[key]
                return True
            except Exception as e:
                self.logger.error(f"Error closing WebSocket connection {key}: {e}")
        return False
    
    def close_all(self):
        """Close all active WebSocket connections."""
        for key in list(self.active_connections.keys()):
            self.close_connection(key)