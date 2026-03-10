"""add beat detection columns

Revision ID: c8f3e4d5a6b7
Revises: b7e2d3f4a5b6
Create Date: 2026-03-09 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c8f3e4d5a6b7'
down_revision: Union[str, None] = 'b7e2d3f4a5b6'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('performances', sa.Column('beat_timestamps', sa.JSON(), nullable=True))
    op.add_column('performances', sa.Column('tempo_bpm', sa.Float(), nullable=True))


def downgrade() -> None:
    op.drop_column('performances', 'tempo_bpm')
    op.drop_column('performances', 'beat_timestamps')
