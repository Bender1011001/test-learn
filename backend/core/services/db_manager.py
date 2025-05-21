from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from sqlalchemy import select, and_, desc, func
from sqlalchemy.orm import Session
from loguru import logger

from ...db.models.logs import InteractionLog, DPOAnnotation


class DBManager:
    """Manager for database operations related to logs and annotations"""
    
    def __init__(self, db_session_factory):
        """Initialize with db session factory"""
        self.db_session_factory = db_session_factory
    
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
        sort_desc: bool = True
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
            
        Returns:
            List of log entries as dictionaries
        """
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
                # Note: This is a simplistic implementation and might not work well with JSON fields
                # In production, use a proper full-text search or JSON operators
                query = query.where(
                    func.lower(func.cast(InteractionLog.input_data, type_=db.String)).like(
                        f"%{keyword.lower()}%"
                    ) |
                    func.lower(func.cast(InteractionLog.output_data, type_=db.String)).like(
                        f"%{keyword.lower()}%"
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
                    "annotation_count": annotation_count
                }
                logs.append(log_dict)
            
            return logs
        
        finally:
            db.close()
    
    def get_log_by_id(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Get a log entry by ID"""
        db = next(self.db_session_factory())
        try:
            log = db.query(InteractionLog).filter(InteractionLog.id == log_id).first()
            
            if not log:
                return None
            
            return {
                "id": log.id,
                "workflow_run_id": log.workflow_run_id,
                "timestamp": log.timestamp.isoformat(),
                "agent_name": log.agent_name,
                "agent_type": log.agent_type,
                "input_data": log.input_data,
                "output_data": log.output_data,
                "has_annotation": len(log.annotations) > 0
            }
        
        finally:
            db.close()
    
    def get_logs_summary(self) -> Dict[str, Any]:
        """Get summary statistics about logs"""
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
            
            return {
                "total_logs": total_logs,
                "logs_with_annotations": logs_with_annotations,
                "logs_by_agent": {agent: count for agent, count in logs_by_agent},
                "recent_logs": recent_logs
            }
        
        finally:
            db.close()
    
    def get_annotation(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Get annotation for a specific log entry"""
        db = next(self.db_session_factory())
        try:
            annotation = db.query(DPOAnnotation).filter(
                DPOAnnotation.log_entry_id == log_id
            ).first()
            
            if not annotation:
                return None
            
            return {
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
        
        finally:
            db.close()
    
    def save_annotation(self, annotation_data: Dict[str, Any]) -> Optional[int]:
        """Save or update an annotation"""
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
            
            if existing:
                # Update existing annotation
                for key, value in annotation_data.items():
                    if key != "id" and hasattr(existing, key):
                        setattr(existing, key, value)
                
                db.commit()
                return existing.id
            else:
                # Create new annotation
                annotation = DPOAnnotation(**annotation_data)
                db.add(annotation)
                db.commit()
                return annotation.id
        
        except Exception as e:
            logger.error(f"Error saving annotation: {str(e)}")
            db.rollback()
            return None
        
        finally:
            db.close()
    
    def delete_annotation(self, annotation_id: int) -> bool:
        """Delete an annotation"""
        db = next(self.db_session_factory())
        try:
            annotation = db.query(DPOAnnotation).filter(
                DPOAnnotation.id == annotation_id
            ).first()
            
            if not annotation:
                return False
            
            db.delete(annotation)
            db.commit()
            return True
        
        except Exception as e:
            logger.error(f"Error deleting annotation: {str(e)}")
            db.rollback()
            return False
        
        finally:
            db.close()
    
    def get_dpo_ready_annotations(self, agent_type: str) -> List[Dict[str, Any]]:
        """Get all annotations ready for DPO training for a specific agent type"""
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
            
            return [{
                "id": ann.id,
                "log_entry_id": ann.log_entry_id,
                "dpo_context": ann.dpo_context,
                "chosen": ann.chosen_prompt,
                "rejected": ann.rejected_prompt
            } for ann in annotations]
        
        finally:
            db.close()
    
    def export_dpo_data(self, agent_type: str, output_format: str = "jsonl") -> str:
        """Export DPO training data for a specific agent type"""
        annotations = self.get_dpo_ready_annotations(agent_type)
        
        if output_format == "jsonl":
            import json
            lines = []
            for ann in annotations:
                lines.append(json.dumps({
                    "prompt": ann["dpo_context"],
                    "chosen": ann["chosen"],
                    "rejected": ann["rejected"]
                }))
            return "\n".join(lines)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")