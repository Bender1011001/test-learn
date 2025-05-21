from sqlalchemy import Column, Integer, String, Text, DateTime, Float, ForeignKey, JSON
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid

from ..base import Base

class InteractionLog(Base):
    """Model for storing workflow agent interaction logs."""
    __tablename__ = "interaction_logs"

    id = Column(Integer, primary_key=True, index=True)
    workflow_run_id = Column(String(36), index=True, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
    agent_name = Column(String(100), index=True)
    agent_type = Column(String(100), index=True)
    input_data = Column(JSON)  # Store input as JSON
    output_data = Column(JSON)  # Store output as JSON
    metadata_json = Column(JSON, nullable=True)  # Additional metadata

    # Relationship to annotations
    annotations = relationship("DPOAnnotation", back_populates="log_entry", cascade="all, delete-orphan")

    def __init__(self, **kwargs):
        # If workflow_run_id not provided, generate a UUID
        if 'workflow_run_id' not in kwargs:
            kwargs['workflow_run_id'] = str(uuid.uuid4())
        super().__init__(**kwargs)


class DPOAnnotation(Base):
    """Model for storing Direct Preference Optimization annotations."""
    __tablename__ = "dpo_annotations"

    id = Column(Integer, primary_key=True, index=True)
    log_entry_id = Column(Integer, ForeignKey("interaction_logs.id"), index=True)
    rating = Column(Float)  # Rating score (e.g., 1-5)
    rationale = Column(Text, nullable=True)  # Explanation for the rating
    chosen_prompt = Column(Text)  # Better action
    rejected_prompt = Column(Text)  # Worse action 
    dpo_context = Column(Text)  # Context for DPO training
    user_id = Column(String(100), nullable=True)  # Who created the annotation
    timestamp = Column(DateTime, default=datetime.utcnow)

    # Relationship to log entry
    log_entry = relationship("InteractionLog", back_populates="annotations")