import asyncio
import json
from typing import Dict, Any, Optional, Callable, Awaitable, List
import redis.asyncio as redis
from redis.asyncio.client import PubSub
from loguru import logger
import os
from enum import Enum, auto

class EventChannel(Enum):
    """Enum for Redis PubSub event channels"""
    WORKFLOW_STATUS = auto()  # Channel for workflow status updates
    WORKFLOW_LOG = auto()  # Channel for workflow log entries
    DPO_STATUS = auto()  # Channel for DPO training job status updates
    DPO_OUTPUT = auto()  # Channel for DPO training output updates

class RedisService:
    """Service for Redis operations including PubSub for event-driven architecture"""
    
    def __init__(self, redis_url: Optional[str] = None):
        """
        Initialize Redis service.
        
        Args:
            redis_url: Redis connection URL (defaults to env variable REDIS_URL)
        """
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://localhost:6379/0")
        self.redis_client: Optional[redis.Redis] = None
        self.pubsub: Optional[PubSub] = None
        self.subscriptions: Dict[str, List[Callable[[Dict[str, Any]], Awaitable[None]]]] = {}
        self._stopping = False
        self._connected = False
        self._subscriber_task = None
        logger.info(f"Redis service initialized with URL: {self.redis_url}")
    
    async def connect(self):
        """Connect to Redis server"""
        if self.redis_client:
            await self.disconnect()
            
        try:
            self.redis_client = redis.from_url(
                self.redis_url, 
                decode_responses=True,
                health_check_interval=30
            )
            self.pubsub = self.redis_client.pubsub()
            self._connected = True
            self._stopping = False
            logger.info("Connected to Redis server")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Redis: {str(e)}")
            self._connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Redis server"""
        self._stopping = True
        
        if self._subscriber_task:
            try:
                self._subscriber_task.cancel()
                await asyncio.wait_for(asyncio.shield(self._subscriber_task), timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._subscriber_task = None
            
        if self.pubsub:
            try:
                await self.pubsub.close()
            except Exception as e:
                logger.error(f"Error closing Redis PubSub: {str(e)}")
        
        if self.redis_client:
            try:
                await self.redis_client.close()
            except Exception as e:
                logger.error(f"Error closing Redis connection: {str(e)}")
        
        self._connected = False
        logger.info("Disconnected from Redis server")
    
    def _get_channel_name(self, channel: EventChannel, id: str) -> str:
        """
        Get the full channel name for a given event type and ID
        
        Args:
            channel: The event channel type
            id: The resource ID (workflow run ID, job ID, etc.)
            
        Returns:
            Full channel name
        """
        return f"{channel.name.lower()}:{id}"

    async def publish_event(self, channel: EventChannel, id: str, data: Dict[str, Any]) -> bool:
        """
        Publish an event to a channel
        
        Args:
            channel: The event channel type
            id: The resource ID (workflow run ID, job ID, etc.)
            data: The event data to publish
            
        Returns:
            Success status
        """
        if not self.redis_client or not self._connected:
            if not await self.connect():
                return False
        
        try:
            channel_name = self._get_channel_name(channel, id)
            serialized = json.dumps(data)
            await self.redis_client.publish(channel_name, serialized)
            logger.debug(f"Published event to {channel_name}")
            return True
        except Exception as e:
            logger.error(f"Error publishing event to Redis: {str(e)}")
            return False
    
    async def subscribe(
        self,
        channel: EventChannel,
        id: str,
        callback: Callable[[Dict[str, Any]], Awaitable[None]]
    ) -> bool:
        """
        Subscribe to events on a channel
        
        Args:
            channel: The event channel type
            id: The resource ID (workflow run ID, job ID, etc.)
            callback: Async function to call when an event is received
            
        Returns:
            Success status
        """
        if not self.redis_client or not self._connected:
            if not await self.connect():
                return False
        
        try:
            channel_name = self._get_channel_name(channel, id)
            
            # Register callback
            if channel_name not in self.subscriptions:
                self.subscriptions[channel_name] = []
            self.subscriptions[channel_name].append(callback)
            
            # Subscribe to channel if not already subscribed
            await self.pubsub.subscribe(channel_name)
            
            # Start subscriber task if not already running
            if not self._subscriber_task or self._subscriber_task.done():
                self._subscriber_task = asyncio.create_task(self._message_listener())
            
            logger.info(f"Subscribed to channel {channel_name}")
            return True
        except Exception as e:
            logger.error(f"Error subscribing to Redis channel: {str(e)}")
            return False
    
    async def unsubscribe(self, channel: EventChannel, id: str, callback: Optional[Callable] = None) -> bool:
        """
        Unsubscribe from events on a channel
        
        Args:
            channel: The event channel type
            id: The resource ID (workflow run ID, job ID, etc.)
            callback: Specific callback to unsubscribe (None for all)
            
        Returns:
            Success status
        """
        if not self.pubsub:
            return False
        
        try:
            channel_name = self._get_channel_name(channel, id)
            
            if callback and channel_name in self.subscriptions:
                self.subscriptions[channel_name] = [
                    cb for cb in self.subscriptions[channel_name] if cb != callback
                ]
                
                # If no more callbacks, unsubscribe from channel
                if not self.subscriptions[channel_name]:
                    await self.pubsub.unsubscribe(channel_name)
                    del self.subscriptions[channel_name]
            else:
                # Unsubscribe all callbacks
                if channel_name in self.subscriptions:
                    await self.pubsub.unsubscribe(channel_name)
                    del self.subscriptions[channel_name]
            
            logger.info(f"Unsubscribed from channel {channel_name}")
            return True
        except Exception as e:
            logger.error(f"Error unsubscribing from Redis channel: {str(e)}")
            return False
    
    async def _message_listener(self):
        """Listen for messages from subscribed channels and dispatch to callbacks"""
        try:
            logger.debug("Starting Redis message listener")
            while not self._stopping and self._connected:
                try:
                    message = await self.pubsub.get_message(ignore_subscribe_messages=True, timeout=1.0)
                    if message and message["type"] == "message":
                        channel = message["channel"]
                        try:
                            data = json.loads(message["data"])
                            
                            # Call registered callbacks
                            if channel in self.subscriptions:
                                for callback in self.subscriptions[channel]:
                                    try:
                                        await callback(data)
                                    except Exception as callback_err:
                                        logger.error(f"Error in Redis event callback: {str(callback_err)}")
                        except json.JSONDecodeError:
                            logger.error(f"Invalid JSON in Redis message: {message['data']}")
                    
                    # Small sleep to prevent CPU spinning
                    await asyncio.sleep(0.01)
                    
                except redis.ConnectionError as ce:
                    logger.warning(f"Redis connection error: {str(ce)}. Reconnecting...")
                    self._connected = False
                    await asyncio.sleep(1)
                    
                    # Try to reconnect
                    try:
                        await self.connect()
                        # Resubscribe to all channels
                        for channel_name in self.subscriptions:
                            await self.pubsub.subscribe(channel_name)
                    except Exception as re:
                        logger.error(f"Error reconnecting to Redis: {str(re)}")
                        await asyncio.sleep(5)  # Wait longer before next retry
                
        except asyncio.CancelledError:
            logger.debug("Redis message listener cancelled")
        except Exception as e:
            logger.error(f"Error in Redis message listener: {str(e)}")