"""add appearance columns to detected_persons

Revision ID: e1a2b3c4d5f6
Revises: d9a4b5c6e7f8
Create Date: 2026-03-22 12:00:00.000000
"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'e1a2b3c4d5f6'
down_revision: Union[str, None] = 'd9a4b5c6e7f8'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('detected_persons', sa.Column('appearance', sa.JSON(), nullable=True))
    op.add_column('detected_persons', sa.Column('color_histogram', sa.JSON(), nullable=True))


def downgrade() -> None:
    op.drop_column('detected_persons', 'color_histogram')
    op.drop_column('detected_persons', 'appearance')
