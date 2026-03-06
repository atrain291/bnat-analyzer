"""add unique dancer name and cascade deletes

Revision ID: a3f1b2c4d5e6
Revises: 127f256e26b9
Create Date: 2026-03-06 22:45:00.000000
"""
from typing import Sequence, Union

from alembic import op


revision: str = 'a3f1b2c4d5e6'
down_revision: Union[str, None] = '127f256e26b9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Unique constraint on dancer name
    op.create_unique_constraint('uq_dancers_name', 'dancers', ['name'])

    # Add ON DELETE CASCADE to foreign keys
    # sessions -> dancers
    op.drop_constraint('sessions_dancer_id_fkey', 'sessions', type_='foreignkey')
    op.create_foreign_key('sessions_dancer_id_fkey', 'sessions', 'dancers', ['dancer_id'], ['id'], ondelete='CASCADE')

    # performances -> dancers
    op.drop_constraint('performances_dancer_id_fkey', 'performances', type_='foreignkey')
    op.create_foreign_key('performances_dancer_id_fkey', 'performances', 'dancers', ['dancer_id'], ['id'], ondelete='CASCADE')

    # performances -> sessions
    op.drop_constraint('performances_session_id_fkey', 'performances', type_='foreignkey')
    op.create_foreign_key('performances_session_id_fkey', 'performances', 'sessions', ['session_id'], ['id'], ondelete='CASCADE')

    # analyses -> performances
    op.drop_constraint('analyses_performance_id_fkey', 'analyses', type_='foreignkey')
    op.create_foreign_key('analyses_performance_id_fkey', 'analyses', 'performances', ['performance_id'], ['id'], ondelete='CASCADE')

    # frames -> performances
    op.drop_constraint('frames_performance_id_fkey', 'frames', type_='foreignkey')
    op.create_foreign_key('frames_performance_id_fkey', 'frames', 'performances', ['performance_id'], ['id'], ondelete='CASCADE')

    # balance_metrics -> frames
    op.drop_constraint('balance_metrics_frame_id_fkey', 'balance_metrics', type_='foreignkey')
    op.create_foreign_key('balance_metrics_frame_id_fkey', 'balance_metrics', 'frames', ['frame_id'], ['id'], ondelete='CASCADE')

    # joint_angle_states -> frames
    op.drop_constraint('joint_angle_states_frame_id_fkey', 'joint_angle_states', type_='foreignkey')
    op.create_foreign_key('joint_angle_states_frame_id_fkey', 'joint_angle_states', 'frames', ['frame_id'], ['id'], ondelete='CASCADE')

    # mudra_states -> frames
    op.drop_constraint('mudra_states_frame_id_fkey', 'mudra_states', type_='foreignkey')
    op.create_foreign_key('mudra_states_frame_id_fkey', 'mudra_states', 'frames', ['frame_id'], ['id'], ondelete='CASCADE')


def downgrade() -> None:
    # Remove cascades (revert to plain FKs)
    op.drop_constraint('mudra_states_frame_id_fkey', 'mudra_states', type_='foreignkey')
    op.create_foreign_key('mudra_states_frame_id_fkey', 'mudra_states', 'frames', ['frame_id'], ['id'])

    op.drop_constraint('joint_angle_states_frame_id_fkey', 'joint_angle_states', type_='foreignkey')
    op.create_foreign_key('joint_angle_states_frame_id_fkey', 'joint_angle_states', 'frames', ['frame_id'], ['id'])

    op.drop_constraint('balance_metrics_frame_id_fkey', 'balance_metrics', type_='foreignkey')
    op.create_foreign_key('balance_metrics_frame_id_fkey', 'balance_metrics', 'frames', ['frame_id'], ['id'])

    op.drop_constraint('frames_performance_id_fkey', 'frames', type_='foreignkey')
    op.create_foreign_key('frames_performance_id_fkey', 'frames', 'performances', ['performance_id'], ['id'])

    op.drop_constraint('analyses_performance_id_fkey', 'analyses', type_='foreignkey')
    op.create_foreign_key('analyses_performance_id_fkey', 'analyses', 'performances', ['performance_id'], ['id'])

    op.drop_constraint('performances_session_id_fkey', 'performances', type_='foreignkey')
    op.create_foreign_key('performances_session_id_fkey', 'performances', 'sessions', ['session_id'], ['id'])

    op.drop_constraint('performances_dancer_id_fkey', 'performances', type_='foreignkey')
    op.create_foreign_key('performances_dancer_id_fkey', 'performances', 'dancers', ['dancer_id'], ['id'])

    op.drop_constraint('sessions_dancer_id_fkey', 'sessions', type_='foreignkey')
    op.create_foreign_key('sessions_dancer_id_fkey', 'sessions', 'dancers', ['dancer_id'], ['id'])

    op.drop_unique_constraint('uq_dancers_name', 'dancers')
