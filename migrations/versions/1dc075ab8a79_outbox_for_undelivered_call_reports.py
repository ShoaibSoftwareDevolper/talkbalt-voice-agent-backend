"""outbox for undelivered call reports

Revision ID: 1dc075ab8a79
Revises:
Create Date: 2026-11-11 13:28:02.772500
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision = "1dc075ab8a79"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "pending_reports",
        sa.Column("call_id", sa.Uuid(), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("call_id"),
    )


def downgrade() -> None:
    op.drop_table("pending_reports")
