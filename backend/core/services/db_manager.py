from typing import Dict, List, Optional, Any, Union, Tuple, Set
from datetime import datetime, timedelta
import json
import asyncio
from sqlalchemy import select, and_, or_, desc, func, text, String
from sqlalchemy.orm import Session, joinedload
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.sql import expression
from sqlalchemy.dialects.postgresql import JSONB
from loguru import logger
import os
from functools import lru_cache
import time

from ...db.models.logs import InteractionLog, DPOAnnotation
from ...db.base import Base

# Check if we're using PostgreSQL for proper JSON search
IS_POSTGRES = os.getenv("DATABASE_URL", "").startswith("postgresql")


class DBManager:
    """
    Enhanced manager for database operations with async support, caching,
    batch operations, and advanced queries.
    """
    
    def __init__(self, db_session_factory):
        """
        Initialize with db session factory
        
        Args:
            db_session_factory: Function that returns a new database session
        """
        self.db_session_factory = db_session_factory
        self._cache = {}
        self._cache_ttl = {}
        self._cache_enabled = True
        self._cache_default_ttl = 300  # 5 minutes
        
        # Create async engine and session factory if using PostgreSQL
        self.async_session_factory = None
        if IS_POSTGRES:
            try:
                # Convert sync DB URL to async
                db_url = os.getenv("DATABASE_URL", "")
                async_db_url = db_url.replace("postgresql://", "postgresql+asyncpg://")
                
                # Create async engine and session factory
                async_engine = create_async_engine(
                    async_db_url,
                    pool_size=20,
                    max_overflow=40,
                    pool_pre_ping=True,
                    pool_recycle=3600,
                    echo=False
                )
                self.async_session_factory = async_sessionmaker(
                    async_engine, expire_on_commit=False, class_=AsyncSession
                )
                logger.info("Async database session factory initialized")
            except ImportError:
                logger.warning("asyncpg not installed, async database operations not available")
            except Exception as e:
                logger.error(f"Error initializing async database: {str(e)}")
    
    def _get_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate a cache key from prefix and kwargs"""
        # Sort kwargs by key to ensure consistent cache keys
        sorted_items = sorted(kwargs.items())
        key_parts = [prefix]
        for k, v in sorted_items:
            if v is not None:
                key_parts.append(f"{k}={v}")
        return ":".join(key_parts)
    
    def _set_cache(self, key: str, value: Any, ttl: int = None) -> None:
        """Set a value in the cache with TTL"""
        if not self._cache_enabled:
            return
        
        ttl = ttl or self._cache_default_ttl
        self._cache[key] = value
        self._cache_ttl[key] = time.time() + ttl
    
    def _get_cache(self, key: str) -> Tuple[bool, Any]:
        """Get a value from the cache, returns (hit, value)"""
        if not self._cache_enabled:
            return False, None
        
        if key not in self._cache:
            return False, None
        
        # Check if expired
        if time.time() > self._cache_ttl.get(key, 0):
            del self._cache[key]
            if key in self._cache_ttl:
                del self._cache_ttl[key]
            return False, None
        
        return True, self._cache[key]
    
    def clear_cache(self, prefix: Optional[str] = None) -> None:
        """
        Clear the cache, optionally only for keys with a specific prefix
        
        Args:
            prefix: Optional prefix to clear only matching keys
        """
        if prefix:
            keys_to_delete = [k for k in self._cache.keys() if k.startswith(prefix)]
            for k in keys_to_delete:
                if k in self._cache:
                    del self._cache[k]
                if k in self._cache_ttl:
                    del self._cache_ttl[k]
            logger.debug(f"Cleared {len(keys_to_delete)} cache entries with prefix '{prefix}'")
        else:
            self._cache.clear()
            self._cache_ttl.clear()
            logger.debug("Cleared entire cache")
    
    def get_logs(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_annotation: Optional[bool] = None,
        keyword: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        sort_by: str = "timestamp",
        sort_desc: bool = True,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Query logs with various filters
        
        Args:
            workflow_id: Filter by workflow run ID
            agent_name: Filter by agent name
            agent_type: Filter by agent type
            start_date: Filter by timestamp >= start_date
            end_date: Filter by timestamp <= end_date
            has_annotation: Filter for logs with/without annotations
            keyword: Search for keyword in input/output data
            offset: Query offset for pagination
            limit: Query limit for pagination
            sort_by: Field to sort by
            sort_desc: Whether to sort descending
            use_cache: Whether to use cache
            
        Returns:
            List of log entries as dictionaries
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(
                "logs",
                workflow_id=workflow_id,
                agent_name=agent_name,
                agent_type=agent_type,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                has_annotation=has_annotation,
                keyword=keyword,
                offset=offset,
                limit=limit,
                sort_by=sort_by,
                sort_desc=sort_desc
            )
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for logs query: {cache_key}")
                return cached_value
        
        db = next(self.db_session_factory())
        try:
            # Start with base query
            query = select(InteractionLog)
            
            # Apply filters
            if workflow_id:
                query = query.where(InteractionLog.workflow_run_id == workflow_id)
            
            if agent_name:
                query = query.where(InteractionLog.agent_name == agent_name)
            
            if agent_type:
                query = query.where(InteractionLog.agent_type == agent_type)
            
            if start_date:
                query = query.where(InteractionLog.timestamp >= start_date)
            
            if end_date:
                query = query.where(InteractionLog.timestamp <= end_date)
            
            if has_annotation is not None:
                # This requires a join with DPOAnnotation to check existence
                if has_annotation:
                    query = query.join(DPOAnnotation)
                else:
                    # This is a left join + filter for logs without annotations
                    query = query.outerjoin(DPOAnnotation).where(DPOAnnotation.id == None)
            
            if keyword:
                # Improved keyword search based on database type
                if IS_POSTGRES:
                    # Use PostgreSQL's JSON operators for better search
                    keyword_lower = f"%{keyword.lower()}%"
                    query = query.where(
                        or_(
                            func.lower(func.cast(InteractionLog.input_data, type_=String)).like(keyword_lower),
                            func.lower(func.cast(InteractionLog.output_data, type_=String)).like(keyword_lower),
                            # Search in specific JSON fields if known
                            func.lower(InteractionLog.input_data['content'].astext).like(keyword_lower),
                            func.lower(InteractionLog.output_data['content'].astext).like(keyword_lower)
                        )
                    )
                else:
                    # Fallback for SQLite - less efficient but works
                    keyword_lower = f"%{keyword.lower()}%"
                    query = query.where(
                        or_(
                            func.lower(func.json_extract(InteractionLog.input_data, '$')).like(keyword_lower),
                            func.lower(func.json_extract(InteractionLog.output_data, '$')).like(keyword_lower)
                        )
                    )
            
            # Apply sorting
            if sort_by == "timestamp":
                order_by = InteractionLog.timestamp.desc() if sort_desc else InteractionLog.timestamp
            elif sort_by == "agent_name":
                order_by = InteractionLog.agent_name.desc() if sort_desc else InteractionLog.agent_name
            elif sort_by == "agent_type":
                order_by = InteractionLog.agent_type.desc() if sort_desc else InteractionLog.agent_type
            else:
                # Default to timestamp
                order_by = InteractionLog.timestamp.desc()
            
            query = query.order_by(order_by)
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Eager load annotations to avoid N+1 query problem
            query = query.options(joinedload(InteractionLog.annotations))
            
            # Execute query
            results = db.execute(query).scalars().all()
            
            # Convert results to dictionaries
            logs = []
            for log in results:
                annotation_count = len(log.annotations)
                
                log_dict = {
                    "id": log.id,
                    "workflow_run_id": log.workflow_run_id,
                    "timestamp": log.timestamp.isoformat(),
                    "agent_name": log.agent_name,
                    "agent_type": log.agent_type,
                    "input_data": log.input_data,
                    "output_data": log.output_data,
                    "has_annotation": annotation_count > 0,
                    "annotation_count": annotation_count,
                    "metadata": log.metadata_json
                }
                logs.append(log_dict)
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, logs)
            
            return logs
        
        finally:
            db.close()
    
    async def get_logs_async(
        self,
        workflow_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        agent_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        has_annotation: Optional[bool] = None,
        keyword: Optional[str] = None,
        offset: int = 0,
        limit: int = 100,
        sort_by: str = "timestamp",
        sort_desc: bool = True,
        use_cache: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Async version of get_logs
        
        This method uses asyncio and SQLAlchemy's async API for better performance
        with FastAPI's async endpoints.
        """
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.get_logs(
                workflow_id, agent_name, agent_type, start_date, end_date,
                has_annotation, keyword, offset, limit, sort_by, sort_desc, use_cache
            )
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key(
                "logs_async",
                workflow_id=workflow_id,
                agent_name=agent_name,
                agent_type=agent_type,
                start_date=start_date.isoformat() if start_date else None,
                end_date=end_date.isoformat() if end_date else None,
                has_annotation=has_annotation,
                keyword=keyword,
                offset=offset,
                limit=limit,
                sort_by=sort_by,
                sort_desc=sort_desc
            )
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for async logs query: {cache_key}")
                return cached_value
        
        async with self.async_session_factory() as session:
            # Start with base query
            query = select(InteractionLog)
            
            # Apply filters (same as sync version)
            if workflow_id:
                query = query.where(InteractionLog.workflow_run_id == workflow_id)
            
            if agent_name:
                query = query.where(InteractionLog.agent_name == agent_name)
            
            if agent_type:
                query = query.where(InteractionLog.agent_type == agent_type)
            
            if start_date:
                query = query.where(InteractionLog.timestamp >= start_date)
            
            if end_date:
                query = query.where(InteractionLog.timestamp <= end_date)
            
            if has_annotation is not None:
                if has_annotation:
                    query = query.join(DPOAnnotation)
                else:
                    query = query.outerjoin(DPOAnnotation).where(DPOAnnotation.id == None)
            
            if keyword:
                # Same improved keyword search as sync version
                if IS_POSTGRES:
                    keyword_lower = f"%{keyword.lower()}%"
                    query = query.where(
                        or_(
                            func.lower(func.cast(InteractionLog.input_data, type_=String)).like(keyword_lower),
                            func.lower(func.cast(InteractionLog.output_data, type_=String)).like(keyword_lower),
                            func.lower(InteractionLog.input_data['content'].astext).like(keyword_lower),
                            func.lower(InteractionLog.output_data['content'].astext).like(keyword_lower)
                        )
                    )
                else:
                    keyword_lower = f"%{keyword.lower()}%"
                    query = query.where(
                        or_(
                            func.lower(func.json_extract(InteractionLog.input_data, '$')).like(keyword_lower),
                            func.lower(func.json_extract(InteractionLog.output_data, '$')).like(keyword_lower)
                        )
                    )
            
            # Apply sorting
            if sort_by == "timestamp":
                order_by = InteractionLog.timestamp.desc() if sort_desc else InteractionLog.timestamp
            elif sort_by == "agent_name":
                order_by = InteractionLog.agent_name.desc() if sort_desc else InteractionLog.agent_name
            elif sort_by == "agent_type":
                order_by = InteractionLog.agent_type.desc() if sort_desc else InteractionLog.agent_type
            else:
                order_by = InteractionLog.timestamp.desc()
            
            query = query.order_by(order_by)
            
            # Apply pagination
            query = query.offset(offset).limit(limit)
            
            # Eager load annotations
            query = query.options(joinedload(InteractionLog.annotations))
            
            # Execute query asynchronously
            result = await session.execute(query)
            logs_data = result.scalars().all()
            
            # Convert results to dictionaries
            logs = []
            for log in logs_data:
                annotation_count = len(log.annotations)
                
                log_dict = {
                    "id": log.id,
                    "workflow_run_id": log.workflow_run_id,
                    "timestamp": log.timestamp.isoformat(),
                    "agent_name": log.agent_name,
                    "agent_type": log.agent_type,
                    "input_data": log.input_data,
                    "output_data": log.output_data,
                    "has_annotation": annotation_count > 0,
                    "annotation_count": annotation_count,
                    "metadata": log.metadata_json
                }
                logs.append(log_dict)
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, logs)
            
            return logs
    
    def get_log_by_id(self, log_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get a log entry by ID
        
        Args:
            log_id: ID of the log entry
            use_cache: Whether to use cache
            
        Returns:
            Log entry as dictionary or None if not found
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("log_by_id", log_id=log_id)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for log_by_id: {log_id}")
                return cached_value
        
        db = next(self.db_session_factory())
        try:
            # Use joinedload to avoid N+1 query problem
            query = select(InteractionLog).options(
                joinedload(InteractionLog.annotations)
            ).where(InteractionLog.id == log_id)
            
            result = db.execute(query).scalars().first()
            
            if not result:
                return None
            
            log_dict = {
                "id": result.id,
                "workflow_run_id": result.workflow_run_id,
                "timestamp": result.timestamp.isoformat(),
                "agent_name": result.agent_name,
                "agent_type": result.agent_type,
                "input_data": result.input_data,
                "output_data": result.output_data,
                "has_annotation": len(result.annotations) > 0,
                "annotations": [
                    {
                        "id": ann.id,
                        "rating": ann.rating,
                        "rationale": ann.rationale,
                        "timestamp": ann.timestamp.isoformat()
                    } for ann in result.annotations
                ],
                "metadata": result.metadata_json
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, log_dict)
            
            return log_dict
        
        finally:
            db.close()
    
    async def get_log_by_id_async(self, log_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Async version of get_log_by_id"""
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.get_log_by_id(log_id, use_cache)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("log_by_id_async", log_id=log_id)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for log_by_id_async: {log_id}")
                return cached_value
        
        async with self.async_session_factory() as session:
            query = select(InteractionLog).options(
                joinedload(InteractionLog.annotations)
            ).where(InteractionLog.id == log_id)
            
            result = await session.execute(query)
            log = result.scalars().first()
            
            if not log:
                return None
            
            log_dict = {
                "id": log.id,
                "workflow_run_id": log.workflow_run_id,
                "timestamp": log.timestamp.isoformat(),
                "agent_name": log.agent_name,
                "agent_type": log.agent_type,
                "input_data": log.input_data,
                "output_data": log.output_data,
                "has_annotation": len(log.annotations) > 0,
                "annotations": [
                    {
                        "id": ann.id,
                        "rating": ann.rating,
                        "rationale": ann.rationale,
                        "timestamp": ann.timestamp.isoformat()
                    } for ann in log.annotations
                ],
                "metadata": log.metadata_json
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, log_dict)
            
            return log_dict
    
    def get_logs_summary(self, use_cache: bool = True) -> Dict[str, Any]:
        """
        Get summary statistics about logs
        
        Args:
            use_cache: Whether to use cache
            
        Returns:
            Dictionary with summary statistics
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("logs_summary")
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug("Cache hit for logs_summary")
                return cached_value
        
        db = next(self.db_session_factory())
        try:
            # Total logs
            total_logs = db.query(func.count(InteractionLog.id)).scalar()
            
            # Logs with annotations
            logs_with_annotations = db.query(
                func.count(InteractionLog.id)
            ).join(DPOAnnotation).scalar()
            
            # Logs by agent type
            logs_by_agent = db.query(
                InteractionLog.agent_type,
                func.count(InteractionLog.id)
            ).group_by(InteractionLog.agent_type).all()
            
            # Logs in last 24 hours
            recent_logs = db.query(
                func.count(InteractionLog.id)
            ).filter(
                InteractionLog.timestamp >= datetime.utcnow() - timedelta(days=1)
            ).scalar()
            
            # Logs by workflow
            logs_by_workflow = db.query(
                InteractionLog.workflow_run_id,
                func.count(InteractionLog.id)
            ).group_by(InteractionLog.workflow_run_id).all()
            
            # Get top workflows by log count
            top_workflows = sorted(
                logs_by_workflow,
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Get latest logs timestamp
            latest_log_time = db.query(
                func.max(InteractionLog.timestamp)
            ).scalar()
            
            # Get earliest logs timestamp
            earliest_log_time = db.query(
                func.min(InteractionLog.timestamp)
            ).scalar()
            
            # Build summary
            summary = {
                "total_logs": total_logs,
                "logs_with_annotations": logs_with_annotations,
                "logs_by_agent": {agent: count for agent, count in logs_by_agent},
                "recent_logs": recent_logs,
                "top_workflows": {workflow_id: count for workflow_id, count in top_workflows},
                "latest_log_time": latest_log_time.isoformat() if latest_log_time else None,
                "earliest_log_time": earliest_log_time.isoformat() if earliest_log_time else None,
                "annotation_percentage": (logs_with_annotations / total_logs * 100) if total_logs > 0 else 0
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, summary, ttl=60)  # Shorter TTL for summary
            
            return summary
        
        finally:
            db.close()
    
    async def get_logs_summary_async(self, use_cache: bool = True) -> Dict[str, Any]:
        """Async version of get_logs_summary"""
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.get_logs_summary(use_cache)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("logs_summary_async")
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug("Cache hit for logs_summary_async")
                return cached_value
        
        async with self.async_session_factory() as session:
            # Total logs
            result = await session.execute(select(func.count(InteractionLog.id)))
            total_logs = result.scalar()
            
            # Logs with annotations
            result = await session.execute(
                select(func.count(InteractionLog.id))
                .join(DPOAnnotation)
            )
            logs_with_annotations = result.scalar()
            
            # Logs by agent type
            result = await session.execute(
                select(InteractionLog.agent_type, func.count(InteractionLog.id))
                .group_by(InteractionLog.agent_type)
            )
            logs_by_agent = result.all()
            
            # Logs in last 24 hours
            result = await session.execute(
                select(func.count(InteractionLog.id))
                .where(InteractionLog.timestamp >= datetime.utcnow() - timedelta(days=1))
            )
            recent_logs = result.scalar()
            
            # Logs by workflow
            result = await session.execute(
                select(InteractionLog.workflow_run_id, func.count(InteractionLog.id))
                .group_by(InteractionLog.workflow_run_id)
            )
            logs_by_workflow = result.all()
            
            # Get top workflows by log count
            top_workflows = sorted(
                logs_by_workflow,
                key=lambda x: x[1],
                reverse=True
            )[:10]
            
            # Get latest logs timestamp
            result = await session.execute(select(func.max(InteractionLog.timestamp)))
            latest_log_time = result.scalar()
            
            # Get earliest logs timestamp
            result = await session.execute(select(func.min(InteractionLog.timestamp)))
            earliest_log_time = result.scalar()
            
            # Build summary
            summary = {
                "total_logs": total_logs,
                "logs_with_annotations": logs_with_annotations,
                "logs_by_agent": {agent: count for agent, count in logs_by_agent},
                "recent_logs": recent_logs,
                "top_workflows": {workflow_id: count for workflow_id, count in top_workflows},
                "latest_log_time": latest_log_time.isoformat() if latest_log_time else None,
                "earliest_log_time": earliest_log_time.isoformat() if earliest_log_time else None,
                "annotation_percentage": (logs_with_annotations / total_logs * 100) if total_logs > 0 else 0
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, summary, ttl=60)  # Shorter TTL for summary
            
            return summary
    
    def get_annotation(self, log_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """
        Get annotation for a specific log entry
        
        Args:
            log_id: ID of the log entry
            use_cache: Whether to use cache
            
        Returns:
            Annotation as dictionary or None if not found
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("annotation", log_id=log_id)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for annotation: {log_id}")
                return cached_value
        
        db = next(self.db_session_factory())
        try:
            annotation = db.query(DPOAnnotation).filter(
                DPOAnnotation.log_entry_id == log_id
            ).first()
            
            if not annotation:
                return None
            
            annotation_dict = {
                "id": annotation.id,
                "log_entry_id": annotation.log_entry_id,
                "rating": annotation.rating,
                "rationale": annotation.rationale,
                "chosen_prompt": annotation.chosen_prompt,
                "rejected_prompt": annotation.rejected_prompt,
                "dpo_context": annotation.dpo_context,
                "user_id": annotation.user_id,
                "timestamp": annotation.timestamp.isoformat()
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, annotation_dict)
            
            return annotation_dict
        
        finally:
            db.close()
    
    async def get_annotation_async(self, log_id: int, use_cache: bool = True) -> Optional[Dict[str, Any]]:
        """Async version of get_annotation"""
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.get_annotation(log_id, use_cache)
        
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("annotation_async", log_id=log_id)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for annotation_async: {log_id}")
                return cached_value
        
        async with self.async_session_factory() as session:
            query = select(DPOAnnotation).where(DPOAnnotation.log_entry_id == log_id)
            result = await session.execute(query)
            annotation = result.scalars().first()
            
            if not annotation:
                return None
            
            annotation_dict = {
                "id": annotation.id,
                "log_entry_id": annotation.log_entry_id,
                "rating": annotation.rating,
                "rationale": annotation.rationale,
                "chosen_prompt": annotation.chosen_prompt,
                "rejected_prompt": annotation.rejected_prompt,
                "dpo_context": annotation.dpo_context,
                "user_id": annotation.user_id,
                "timestamp": annotation.timestamp.isoformat()
            }
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, annotation_dict)
            
            return annotation_dict
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> Optional[int]:
        """
        Save or update an annotation
        
        Args:
            annotation_data: Dictionary with annotation data
            
        Returns:
            ID of the saved annotation or None if failed
        """
        db = next(self.db_session_factory())
        try:
            log_id = annotation_data.get("log_entry_id")
            
            # Check if log exists
            log = db.query(InteractionLog).filter(InteractionLog.id == log_id).first()
            if not log:
                logger.error(f"Log entry {log_id} not found")
                return None
            
            # Check if annotation already exists
            existing = db.query(DPOAnnotation).filter(
                DPOAnnotation.log_entry_id == log_id
            ).first()
            
            result_id = None
            if existing:
                # Update existing annotation
                for key, value in annotation_data.items():
                    if key != "id" and hasattr(existing, key):
                        setattr(existing, key, value)
                
                db.commit()
                result_id = existing.id
            else:
                # Create new annotation
                annotation = DPOAnnotation(**annotation_data)
                db.add(annotation)
                db.commit()
                result_id = annotation.id
            
            # Clear related caches
            self.clear_cache(f"annotation")
            self.clear_cache("logs_summary")
            self.clear_cache("logs")  # Clear logs cache as has_annotation may have changed
            
            return result_id
        
        except Exception as e:
            logger.error(f"Error saving annotation: {str(e)}")
            db.rollback()
            return None
        
        finally:
            db.close()
    
    def delete_annotation(self, annotation_id: int) -> bool:
        """
        Delete an annotation
        
        Args:
            annotation_id: ID of the annotation to delete
            
        Returns:
            True if deleted successfully, False otherwise
        """
        db = next(self.db_session_factory())
        try:
            annotation = db.query(DPOAnnotation).filter(
                DPOAnnotation.id == annotation_id
            ).first()
            
            if not annotation:
                return False
            
            # Get the log_id before deleting
            log_id = annotation.log_entry_id
            
            db.delete(annotation)
            db.commit()
            
            # Clear related caches
            self.clear_cache(f"annotation")
            self.clear_cache("logs_summary")
            self.clear_cache("logs")  # Clear logs cache as has_annotation may have changed
            
            return True
        
        except Exception as e:
            logger.error(f"Error deleting annotation: {str(e)}")
            db.rollback()
            return False
        
        finally:
            db.close()
    
    def get_dpo_ready_annotations(self, agent_type: str, use_cache: bool = True) -> List[Dict[str, Any]]:
        """
        Get all annotations ready for DPO training for a specific agent type
        
        Args:
            agent_type: Type of agent to get annotations for
            use_cache: Whether to use cache
            
        Returns:
            List of annotations ready for DPO training
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("dpo_ready_annotations", agent_type=agent_type)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for dpo_ready_annotations: {agent_type}")
                return cached_value
        
        db = next(self.db_session_factory())
        try:
            # Join with logs to filter by agent type
            annotations = db.query(DPOAnnotation).join(
                InteractionLog,
                DPOAnnotation.log_entry_id == InteractionLog.id
            ).filter(
                InteractionLog.agent_type == agent_type,
                DPOAnnotation.chosen_prompt != None,
                DPOAnnotation.rejected_prompt != None
            ).all()
            
            result = [{
                "id": ann.id,
                "log_entry_id": ann.log_entry_id,
                "dpo_context": ann.dpo_context,
                "chosen": ann.chosen_prompt,
                "rejected": ann.rejected_prompt
            } for ann in annotations]
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, result)
            
            return result
        
        finally:
            db.close()
    
    def export_dpo_data(self, agent_type: str, output_format: str = "jsonl", use_cache: bool = True) -> str:
        """
        Export DPO training data for a specific agent type
        
        Args:
            agent_type: Type of agent to export data for
            output_format: Format to export data in (currently only "jsonl" is supported)
            use_cache: Whether to use cache
            
        Returns:
            String containing the exported data
        """
        # Check cache if enabled
        if use_cache:
            cache_key = self._get_cache_key("export_dpo_data", agent_type=agent_type, output_format=output_format)
            cache_hit, cached_value = self._get_cache(cache_key)
            if cache_hit:
                logger.debug(f"Cache hit for export_dpo_data: {agent_type}, {output_format}")
                return cached_value
        
        annotations = self.get_dpo_ready_annotations(agent_type, use_cache=use_cache)
        
        if output_format == "jsonl":
            import json
            lines = []
            for ann in annotations:
                lines.append(json.dumps({
                    "prompt": ann["dpo_context"],
                    "chosen": ann["chosen"],
                    "rejected": ann["rejected"]
                }))
            result = "\n".join(lines)
            
            # Cache the result if caching is enabled
            if use_cache:
                self._set_cache(cache_key, result)
            
            return result
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def batch_save_annotations(self, annotations_data: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Save multiple annotations in a single transaction
        
        Args:
            annotations_data: List of annotation data dictionaries
            
        Returns:
            List of saved annotation IDs or None for failed entries
        """
        if not annotations_data:
            return []
            
        db = next(self.db_session_factory())
        try:
            result_ids = []
            log_ids = set()
            
            for annotation_data in annotations_data:
                log_id = annotation_data.get("log_entry_id")
                if not log_id:
                    result_ids.append(None)
                    continue
                    
                log_ids.add(log_id)
                
                # Check if log exists
                log = db.query(InteractionLog).filter(InteractionLog.id == log_id).first()
                if not log:
                    logger.error(f"Log entry {log_id} not found")
                    result_ids.append(None)
                    continue
                
                # Check if annotation already exists
                existing = db.query(DPOAnnotation).filter(
                    DPOAnnotation.log_entry_id == log_id
                ).first()
                
                if existing:
                    # Update existing annotation
                    for key, value in annotation_data.items():
                        if key != "id" and hasattr(existing, key):
                            setattr(existing, key, value)
                    
                    result_ids.append(existing.id)
                else:
                    # Create new annotation
                    annotation = DPOAnnotation(**annotation_data)
                    db.add(annotation)
                    # We'll get the ID after commit
                    result_ids.append(None)
            
            # Commit all changes in a single transaction
            db.commit()
            
            # Clear related caches
            self.clear_cache("annotation")
            self.clear_cache("logs_summary")
            self.clear_cache("logs")
            self.clear_cache("dpo_ready_annotations")
            
            return result_ids
            
        except Exception as e:
            logger.error(f"Error in batch save annotations: {str(e)}")
            db.rollback()
            return [None] * len(annotations_data)
            
        finally:
            db.close()
    
    async def batch_save_annotations_async(self, annotations_data: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Async version of batch_save_annotations"""
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.batch_save_annotations(annotations_data)
            
        if not annotations_data:
            return []
            
        async with self.async_session_factory() as session:
            try:
                result_ids = []
                log_ids = set()
                
                for annotation_data in annotations_data:
                    log_id = annotation_data.get("log_entry_id")
                    if not log_id:
                        result_ids.append(None)
                        continue
                        
                    log_ids.add(log_id)
                    
                    # Check if log exists
                    query = select(InteractionLog).where(InteractionLog.id == log_id)
                    result = await session.execute(query)
                    log = result.scalars().first()
                    
                    if not log:
                        logger.error(f"Log entry {log_id} not found")
                        result_ids.append(None)
                        continue
                    
                    # Check if annotation already exists
                    query = select(DPOAnnotation).where(DPOAnnotation.log_entry_id == log_id)
                    result = await session.execute(query)
                    existing = result.scalars().first()
                    
                    if existing:
                        # Update existing annotation
                        for key, value in annotation_data.items():
                            if key != "id" and hasattr(existing, key):
                                setattr(existing, key, value)
                        
                        result_ids.append(existing.id)
                    else:
                        # Create new annotation
                        annotation = DPOAnnotation(**annotation_data)
                        session.add(annotation)
                        # We'll get the ID after commit
                        result_ids.append(None)
                
                # Commit all changes in a single transaction
                await session.commit()
                
                # Clear related caches
                self.clear_cache("annotation")
                self.clear_cache("logs_summary")
                self.clear_cache("logs")
                self.clear_cache("dpo_ready_annotations")
                
                return result_ids
                
            except Exception as e:
                logger.error(f"Error in batch save annotations async: {str(e)}")
                await session.rollback()
                return [None] * len(annotations_data)
    
    def batch_save_logs(self, logs_data: List[Dict[str, Any]]) -> List[Optional[int]]:
        """
        Save multiple interaction logs in a single transaction
        
        Args:
            logs_data: List of log data dictionaries
            
        Returns:
            List of saved log IDs or None for failed entries
        """
        if not logs_data:
            return []
            
        db = next(self.db_session_factory())
        try:
            result_ids = []
            logs = []
            
            for log_data in logs_data:
                # Create new log
                log = InteractionLog(**log_data)
                db.add(log)
                logs.append(log)
                result_ids.append(None)  # Placeholder, will be updated after commit
            
            # Commit all changes in a single transaction
            db.commit()
            
            # Update result_ids with actual IDs
            for i, log in enumerate(logs):
                result_ids[i] = log.id
            
            # Clear related caches
            self.clear_cache("logs")
            self.clear_cache("logs_summary")
            
            return result_ids
            
        except Exception as e:
            logger.error(f"Error in batch save logs: {str(e)}")
            db.rollback()
            return [None] * len(logs_data)
            
        finally:
            db.close()
    
    async def batch_save_logs_async(self, logs_data: List[Dict[str, Any]]) -> List[Optional[int]]:
        """Async version of batch_save_logs"""
        if not self.async_session_factory:
            # Fallback to sync version if async not available
            return self.batch_save_logs(logs_data)
            
        if not logs_data:
            return []
            
        async with self.async_session_factory() as session:
            try:
                result_ids = []
                logs = []
                
                for log_data in logs_data:
                    # Create new log
                    log = InteractionLog(**log_data)
                    session.add(log)
                    logs.append(log)
                    result_ids.append(None)  # Placeholder, will be updated after commit
                
                # Commit all changes in a single transaction
                await session.commit()
                
                # Update result_ids with actual IDs
                for i, log in enumerate(logs):
                    result_ids[i] = log.id
                
                # Clear related caches
                self.clear_cache("logs")
                self.clear_cache("logs_summary")
                
                return result_ids
                
            except Exception as e:
                logger.error(f"Error in batch save logs async: {str(e)}")
                await session.rollback()
                return [None] * len(logs_data)