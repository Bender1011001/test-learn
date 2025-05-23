"""Add task models

Revision ID: add_task_models
Revises: initial_migration
Create Date: 2025-05-22 17:05:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = 'add_task_models'
down_revision = 'initial_migration'
branch_labels = None
depends_on = None


def upgrade():
    # Create enum types
    task_difficulty_enum = sa.Enum('BEGINNER', 'INTERMEDIATE', 'ADVANCED', name='taskdifficultyenum')
    task_category_enum = sa.Enum('CODING', 'REASONING', 'CREATIVE', 'ANALYTICAL', 'PROBLEM_SOLVING', 'DATA_ANALYSIS', name='taskcategoryenum')
    task_priority_enum = sa.Enum('LOW', 'MEDIUM', 'HIGH', 'CRITICAL', name='taskpriorityenum')
    task_status_enum = sa.Enum('PENDING', 'IN_PROGRESS', 'COMPLETED', 'FAILED', 'CANCELLED', name='taskstatusenum')
    
    task_difficulty_enum.create(op.get_bind())
    task_category_enum.create(op.get_bind())
    task_priority_enum.create(op.get_bind())
    task_status_enum.create(op.get_bind())
    
    # Create tasks table
    op.create_table('tasks',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=False),
        sa.Column('title', sa.String(length=200), nullable=False),
        sa.Column('description', sa.Text(), nullable=False),
        sa.Column('category', task_category_enum, nullable=False),
        sa.Column('difficulty', task_difficulty_enum, nullable=False),
        sa.Column('priority', task_priority_enum, nullable=True),
        sa.Column('status', task_status_enum, nullable=True),
        sa.Column('complexity_score', sa.Float(), nullable=False),
        sa.Column('estimated_duration', sa.Integer(), nullable=True),
        sa.Column('success_criteria', sa.JSON(), nullable=True),
        sa.Column('evaluation_metrics', sa.JSON(), nullable=True),
        sa.Column('prerequisites', sa.JSON(), nullable=True),
        sa.Column('tags', sa.JSON(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=True),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('generated_by_agent', sa.String(length=100), nullable=True),
        sa.Column('generation_context', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_tasks_id'), 'tasks', ['id'], unique=False)
    op.create_index(op.f('ix_tasks_task_id'), 'tasks', ['task_id'], unique=True)
    op.create_index(op.f('ix_tasks_category'), 'tasks', ['category'], unique=False)
    op.create_index(op.f('ix_tasks_difficulty'), 'tasks', ['difficulty'], unique=False)
    op.create_index(op.f('ix_tasks_priority'), 'tasks', ['priority'], unique=False)
    op.create_index(op.f('ix_tasks_status'), 'tasks', ['status'], unique=False)
    op.create_index(op.f('ix_tasks_complexity_score'), 'tasks', ['complexity_score'], unique=False)
    op.create_index(op.f('ix_tasks_created_at'), 'tasks', ['created_at'], unique=False)

    # Create task_executions table
    op.create_table('task_executions',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('execution_id', sa.String(length=100), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=False),
        sa.Column('executor_agent', sa.String(length=100), nullable=True),
        sa.Column('status', task_status_enum, nullable=False),
        sa.Column('execution_output', sa.JSON(), nullable=True),
        sa.Column('success_criteria_met', sa.JSON(), nullable=True),
        sa.Column('execution_time', sa.Float(), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('quality_score', sa.Float(), nullable=True),
        sa.Column('efficiency_score', sa.Float(), nullable=True),
        sa.Column('completeness_score', sa.Float(), nullable=True),
        sa.Column('started_at', sa.DateTime(), nullable=False),
        sa.Column('completed_at', sa.DateTime(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_executions_id'), 'task_executions', ['id'], unique=False)
    op.create_index(op.f('ix_task_executions_execution_id'), 'task_executions', ['execution_id'], unique=True)
    op.create_index(op.f('ix_task_executions_task_id'), 'task_executions', ['task_id'], unique=False)
    op.create_index(op.f('ix_task_executions_status'), 'task_executions', ['status'], unique=False)
    op.create_index(op.f('ix_task_executions_started_at'), 'task_executions', ['started_at'], unique=False)
    op.create_index(op.f('ix_task_executions_completed_at'), 'task_executions', ['completed_at'], unique=False)

    # Create task_feedback table
    op.create_table('task_feedback',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('feedback_id', sa.String(length=100), nullable=False),
        sa.Column('task_id', sa.String(length=100), nullable=False),
        sa.Column('execution_id', sa.String(length=100), nullable=True),
        sa.Column('reviewer_agent', sa.String(length=100), nullable=True),
        sa.Column('feedback_type', sa.String(length=50), nullable=True),
        sa.Column('overall_rating', sa.Float(), nullable=False),
        sa.Column('strengths', sa.JSON(), nullable=True),
        sa.Column('areas_for_improvement', sa.JSON(), nullable=True),
        sa.Column('effectiveness_score', sa.Float(), nullable=True),
        sa.Column('correctness_score', sa.Float(), nullable=True),
        sa.Column('detailed_feedback', sa.Text(), nullable=True),
        sa.Column('task_difficulty_assessment', sa.String(length=50), nullable=True),
        sa.Column('task_quality_rating', sa.Float(), nullable=True),
        sa.Column('execution_quality_rating', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_feedback_id'), 'task_feedback', ['id'], unique=False)
    op.create_index(op.f('ix_task_feedback_feedback_id'), 'task_feedback', ['feedback_id'], unique=True)
    op.create_index(op.f('ix_task_feedback_task_id'), 'task_feedback', ['task_id'], unique=False)
    op.create_index(op.f('ix_task_feedback_execution_id'), 'task_feedback', ['execution_id'], unique=False)
    op.create_index(op.f('ix_task_feedback_created_at'), 'task_feedback', ['created_at'], unique=False)

    # Create task_queues table
    op.create_table('task_queues',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('queue_id', sa.String(length=100), nullable=False),
        sa.Column('queue_name', sa.String(length=100), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('max_size', sa.Integer(), nullable=True),
        sa.Column('auto_generation_enabled', sa.Boolean(), nullable=True),
        sa.Column('generation_rate_limit', sa.Integer(), nullable=True),
        sa.Column('total_tasks_generated', sa.Integer(), nullable=True),
        sa.Column('total_tasks_completed', sa.Integer(), nullable=True),
        sa.Column('average_completion_time', sa.Float(), nullable=True),
        sa.Column('average_quality_score', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.Column('last_generation_time', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_queues_id'), 'task_queues', ['id'], unique=False)
    op.create_index(op.f('ix_task_queues_queue_id'), 'task_queues', ['queue_id'], unique=True)

    # Create task_generation_settings table
    op.create_table('task_generation_settings',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('settings_id', sa.String(length=100), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('preferred_categories', sa.JSON(), nullable=True),
        sa.Column('preferred_difficulties', sa.JSON(), nullable=True),
        sa.Column('complexity_range', sa.JSON(), nullable=True),
        sa.Column('min_complexity_score', sa.Float(), nullable=True),
        sa.Column('max_complexity_score', sa.Float(), nullable=True),
        sa.Column('diversity_threshold', sa.Float(), nullable=True),
        sa.Column('quality_threshold', sa.Float(), nullable=True),
        sa.Column('max_tasks_per_minute', sa.Integer(), nullable=True),
        sa.Column('max_tasks_per_hour', sa.Integer(), nullable=True),
        sa.Column('max_tasks_per_day', sa.Integer(), nullable=True),
        sa.Column('enable_adaptive_difficulty', sa.Boolean(), nullable=True),
        sa.Column('enable_feedback_learning', sa.Boolean(), nullable=True),
        sa.Column('success_rate_target', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=True),
        sa.Column('updated_at', sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_task_generation_settings_id'), 'task_generation_settings', ['id'], unique=False)
    op.create_index(op.f('ix_task_generation_settings_settings_id'), 'task_generation_settings', ['settings_id'], unique=True)


def downgrade():
    # Drop tables
    op.drop_table('task_generation_settings')
    op.drop_table('task_queues')
    op.drop_table('task_feedback')
    op.drop_table('task_executions')
    op.drop_table('tasks')
    
    # Drop enum types
    sa.Enum(name='taskstatusenum').drop(op.get_bind())
    sa.Enum(name='taskpriorityenum').drop(op.get_bind())
    sa.Enum(name='taskcategoryenum').drop(op.get_bind())
    sa.Enum(name='taskdifficultyenum').drop(op.get_bind())