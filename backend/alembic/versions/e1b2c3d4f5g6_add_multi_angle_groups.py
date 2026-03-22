"""add multi angle groups

Revision ID: e1b2c3d4f5g6
Revises: d9a4b5c6e7f8
Create Date: 2026-03-22 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e1b2c3d4f5g6'
down_revision: Union[str, None] = 'd9a4b5c6e7f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Multi-angle groups table
    op.create_table(
        'multi_angle_groups',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('dancer_id', sa.Integer(), sa.ForeignKey('dancers.id'), nullable=False),
        sa.Column('item_name', sa.String(300), nullable=True),
        sa.Column('item_type', sa.String(100), nullable=True),
        sa.Column('talam', sa.String(100), nullable=True),
        sa.Column('ragam', sa.String(100), nullable=True),
        sa.Column('sync_offsets', sa.JSON(), nullable=True),
        sa.Column('sync_confidence', sa.Float(), nullable=True),
        sa.Column('status', sa.String(20), nullable=False, server_default='pending'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # Multi-angle fused analysis table
    op.create_table(
        'multi_angle_analyses',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('multi_angle_group_id', sa.Integer(),
                  sa.ForeignKey('multi_angle_groups.id', ondelete='CASCADE'), nullable=False),
        sa.Column('dancer_label', sa.String(200), nullable=True),
        sa.Column('aramandi_score', sa.Float(), nullable=True),
        sa.Column('upper_body_score', sa.Float(), nullable=True),
        sa.Column('symmetry_score', sa.Float(), nullable=True),
        sa.Column('rhythm_consistency_score', sa.Float(), nullable=True),
        sa.Column('overall_score', sa.Float(), nullable=True),
        sa.Column('per_view_scores', sa.JSON(), nullable=True),
        sa.Column('score_sources', sa.JSON(), nullable=True),
        sa.Column('llm_summary', sa.String(8000), nullable=True),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )

    # Add multi-angle columns to performances
    op.add_column('performances',
                  sa.Column('multi_angle_group_id', sa.Integer(),
                            sa.ForeignKey('multi_angle_groups.id', ondelete='SET NULL'),
                            nullable=True))
    op.add_column('performances',
                  sa.Column('camera_label', sa.String(100), nullable=True))


def downgrade() -> None:
    op.drop_column('performances', 'camera_label')
    op.drop_column('performances', 'multi_angle_group_id')
    op.drop_table('multi_angle_analyses')
    op.drop_table('multi_angle_groups')
