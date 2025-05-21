"""Initial migration

Revision ID: initial_migration
Revises: 
Create Date: 2025-05-21 06:33:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSON

# revision identifiers, used by Alembic.
revision = 'initial_migration'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create interaction_logs table
    op.create_table('interaction_logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('workflow_run_id', sa.String(length=36), nullable=False),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.Column('agent_name', sa.String(length=100), nullable=True),
        sa.Column('agent_type', sa.String(length=100), nullable=True),
        sa.Column('input_data', sa.JSON(), nullable=True),
        sa.Column('output_data', sa.JSON(), nullable=True),
        sa.Column('metadata_json', sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_interaction_logs_agent_name'), 'interaction_logs', ['agent_name'], unique=False)
    op.create_index(op.f('ix_interaction_logs_agent_type'), 'interaction_logs', ['agent_type'], unique=False)
    op.create_index(op.f('ix_interaction_logs_id'), 'interaction_logs', ['id'], unique=False)
    op.create_index(op.f('ix_interaction_logs_timestamp'), 'interaction_logs', ['timestamp'], unique=False)
    op.create_index(op.f('ix_interaction_logs_workflow_run_id'), 'interaction_logs', ['workflow_run_id'], unique=False)

    # Create dpo_annotations table
    op.create_table('dpo_annotations',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('log_entry_id', sa.Integer(), nullable=True),
        sa.Column('rating', sa.Float(), nullable=True),
        sa.Column('rationale', sa.Text(), nullable=True),
        sa.Column('chosen_prompt', sa.Text(), nullable=True),
        sa.Column('rejected_prompt', sa.Text(), nullable=True),
        sa.Column('dpo_context', sa.Text(), nullable=True),
        sa.Column('user_id', sa.String(length=100), nullable=True),
        sa.Column('timestamp', sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(['log_entry_id'], ['interaction_logs.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_dpo_annotations_id'), 'dpo_annotations', ['id'], unique=False)
    op.create_index(op.f('ix_dpo_annotations_log_entry_id'), 'dpo_annotations', ['log_entry_id'], unique=False)


def downgrade() -> None:
    # Drop tables in reverse order
    op.drop_index(op.f('ix_dpo_annotations_log_entry_id'), table_name='dpo_annotations')
    op.drop_index(op.f('ix_dpo_annotations_id'), table_name='dpo_annotations')
    op.drop_table('dpo_annotations')
    
    op.drop_index(op.f('ix_interaction_logs_workflow_run_id'), table_name='interaction_logs')
    op.drop_index(op.f('ix_interaction_logs_timestamp'), table_name='interaction_logs')
    op.drop_index(op.f('ix_interaction_logs_id'), table_name='interaction_logs')
    op.drop_index(op.f('ix_interaction_logs_agent_type'), table_name='interaction_logs')
    op.drop_index(op.f('ix_interaction_logs_agent_name'), table_name='interaction_logs')
    op.drop_table('interaction_logs')