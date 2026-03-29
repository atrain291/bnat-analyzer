"""add pose_summary to performance_dancers

Revision ID: g3c4d5e6f7a8
Revises: f2b3c4d5e6a7
Create Date: 2026-03-29 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'g3c4d5e6f7a8'
down_revision: Union[str, None] = '9ac63380e2ba'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('performance_dancers', sa.Column('pose_summary', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('performance_dancers', 'pose_summary')
