"""add wham 3d columns

Revision ID: d9a4b5c6e7f8
Revises: c8f3e4d5a6b7
Create Date: 2026-03-10 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'd9a4b5c6e7f8'
down_revision: Union[str, None] = 'c8f3e4d5a6b7'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Frame: WHAM 3D data
    op.add_column('frames', sa.Column('joints_3d', sa.JSON(), nullable=True))
    op.add_column('frames', sa.Column('world_position', sa.JSON(), nullable=True))
    op.add_column('frames', sa.Column('foot_contact', sa.JSON(), nullable=True))

    # JointAngleState: 3D angles
    op.add_column('joint_angle_states', sa.Column('knee_angle_3d', sa.Float(), nullable=True))
    op.add_column('joint_angle_states', sa.Column('torso_angle_3d', sa.Float(), nullable=True))
    op.add_column('joint_angle_states', sa.Column('hip_abduction_left', sa.Float(), nullable=True))
    op.add_column('joint_angle_states', sa.Column('hip_abduction_right', sa.Float(), nullable=True))
    op.add_column('joint_angle_states', sa.Column('torso_twist', sa.Float(), nullable=True))

    # BalanceMetrics: 3D center of mass
    op.add_column('balance_metrics', sa.Column('center_of_mass_3d_x', sa.Float(), nullable=True))
    op.add_column('balance_metrics', sa.Column('center_of_mass_3d_y', sa.Float(), nullable=True))
    op.add_column('balance_metrics', sa.Column('center_of_mass_3d_z', sa.Float(), nullable=True))


def downgrade() -> None:
    # BalanceMetrics
    op.drop_column('balance_metrics', 'center_of_mass_3d_z')
    op.drop_column('balance_metrics', 'center_of_mass_3d_y')
    op.drop_column('balance_metrics', 'center_of_mass_3d_x')

    # JointAngleState
    op.drop_column('joint_angle_states', 'torso_twist')
    op.drop_column('joint_angle_states', 'hip_abduction_right')
    op.drop_column('joint_angle_states', 'hip_abduction_left')
    op.drop_column('joint_angle_states', 'torso_angle_3d')
    op.drop_column('joint_angle_states', 'knee_angle_3d')

    # Frame
    op.drop_column('frames', 'foot_contact')
    op.drop_column('frames', 'world_position')
    op.drop_column('frames', 'joints_3d')
