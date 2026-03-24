"""add missing FK indexes for query performance

Revision ID: f2b3c4d5e6a7
Revises: e1a2b3c4d5f6
Create Date: 2026-03-23 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'f2b3c4d5e6a7'
down_revision: Union[str, None] = 'e1a2b3c4d5f6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_index('ix_performances_dancer_id', 'performances', ['dancer_id'])
    op.create_index('ix_detected_persons_performance_id', 'detected_persons', ['performance_id'])
    op.create_index('ix_performance_dancers_performance_id', 'performance_dancers', ['performance_id'])
    op.create_index('ix_frames_performance_dancer_id', 'frames', ['performance_dancer_id'])
    op.create_index('ix_analyses_performance_id', 'analyses', ['performance_id'])
    op.create_index('ix_analyses_performance_dancer_id', 'analyses', ['performance_dancer_id'])


def downgrade() -> None:
    op.drop_index('ix_analyses_performance_dancer_id', 'analyses')
    op.drop_index('ix_analyses_performance_id', 'analyses')
    op.drop_index('ix_frames_performance_dancer_id', 'frames')
    op.drop_index('ix_performance_dancers_performance_id', 'performance_dancers')
    op.drop_index('ix_detected_persons_performance_id', 'detected_persons')
    op.drop_index('ix_performances_dancer_id', 'performances')
