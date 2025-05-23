from typing import Tuple
from functools import lru_cache
from sqlalchemy.orm import Session
from loguru import logger

from ..core.services.config_manager import ConfigManager
from ..core.services.workflow_manager import WorkflowManager
from ..core.services.db_manager import DBManager
from ..core.services.dpo_trainer import DPOTrainer
from ..core.services.redis_service import RedisService
from ..core.services.db_monitoring import apply_db_monitoring
from ..core.services.monitoring_service import get_monitoring_service
from ..db.base import get_db

# Service singletons
_config_manager = None
_workflow_manager = None
_db_manager = None
_dpo_trainer = None
_redis_service = None


@lru_cache()
def get_config_manager() -> ConfigManager:
    """Get or create ConfigManager singleton"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def get_services(db: Session) -> Tuple[ConfigManager, WorkflowManager, DBManager, DPOTrainer]:
    """
    Get or create service instances
    
    This is the main dependency function that provides access to all services
    """
    global _workflow_manager, _db_manager, _dpo_trainer
    
    # Get or create ConfigManager
    config_manager = get_config_manager()
    
    # Get or create DBManager
    if _db_manager is None:
        # Create a DB session factory function
        def db_session_factory():
            from ..db.base import SessionLocal
            return SessionLocal()
        
        # Create DBManager and apply monitoring decorators
        db_manager = DBManager(db_session_factory)
        _db_manager = apply_db_monitoring(db_manager)
        
        # Initialize monitoring service
        monitoring_service = get_monitoring_service()
    
    # Get or create WorkflowManager
    if _workflow_manager is None:
        # Create a DB session factory function
        def db_session_factory():
            from ..db.base import SessionLocal
            return SessionLocal()
        
        _workflow_manager = WorkflowManager(config_manager, db_session_factory)
    
    # Get or create DPOTrainer
    if _dpo_trainer is None:
        _dpo_trainer = DPOTrainer(config_manager, _db_manager)
    
    return config_manager, _workflow_manager, _db_manager, _dpo_trainer


@lru_cache()
async def get_redis_service() -> RedisService:
    """Get or create RedisService singleton and ensure it's connected"""
    global _redis_service
    if _redis_service is None:
        _redis_service = RedisService()
        await _redis_service.connect()
    return _redis_service


def get_settings():
    """Get application settings"""
    return {
        "app_name": "CAMEL Extensions API",
        "version": "0.1.0",
        "debug": True,
        "api_prefix": "/api",
        "docs_url": "/docs",
        "openapi_url": "/openapi.json",
        "redoc_url": "/redoc",
        "db_url": "sqlite:///./camel_extensions.db",
    }