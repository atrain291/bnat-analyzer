"""add dancer detection tables

Revision ID: b7e2d3f4a5b6
Revises: a3f1b2c4d5e6
Create Date: 2026-03-06 23:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'b7e2d3f4a5b6'
down_revision: Union[str, None] = 'a3f1b2c4d5e6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add detection_frame_url to performances
    op.add_column('performances', sa.Column('detection_frame_url', sa.String(500), nullable=True))

    # Create detected_persons table
    op.create_table('detected_persons',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('performance_id', sa.Integer(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('bbox', sa.JSON(), nullable=False),
        sa.Column('representative_pose', sa.JSON(), nullable=False),
        sa.Column('frame_count', sa.Integer(), nullable=False),
        sa.Column('area', sa.Float(), nullable=False),
        sa.ForeignKeyConstraint(['performance_id'], ['performances.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Create performance_dancers table
    op.create_table('performance_dancers',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('performance_id', sa.Integer(), nullable=False),
        sa.Column('track_id', sa.Integer(), nullable=False),
        sa.Column('label', sa.String(200), nullable=True),
        sa.ForeignKeyConstraint(['performance_id'], ['performances.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )

    # Add performance_dancer_id to frames
    op.add_column('frames', sa.Column('performance_dancer_id', sa.Integer(), nullable=True))
    op.create_foreign_key('frames_performance_dancer_id_fkey', 'frames', 'performance_dancers',
                          ['performance_dancer_id'], ['id'], ondelete='CASCADE')

    # Add performance_dancer_id to analyses and drop unique constraint on performance_id
    op.add_column('analyses', sa.Column('performance_dancer_id', sa.Integer(), nullable=True))
    op.create_foreign_key('analyses_performance_dancer_id_fkey', 'analyses', 'performance_dancers',
                          ['performance_dancer_id'], ['id'], ondelete='CASCADE')
    op.drop_constraint('analyses_performance_id_key', 'analyses', type_='unique')


def downgrade() -> None:
    op.create_unique_constraint('analyses_performance_id_key', 'analyses', ['performance_id'])
    op.drop_constraint('analyses_performance_dancer_id_fkey', 'analyses', type_='foreignkey')
    op.drop_column('analyses', 'performance_dancer_id')
    op.drop_constraint('frames_performance_dancer_id_fkey', 'frames', type_='foreignkey')
    op.drop_column('frames', 'performance_dancer_id')
    op.drop_table('performance_dancers')
    op.drop_table('detected_persons')
    op.drop_column('performances', 'detection_frame_url')
