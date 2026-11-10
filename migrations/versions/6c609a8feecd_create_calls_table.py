"""create calls table

Revision ID: 6c609a8feecd
Revises: 
Create Date: 2026-10-22 19:19:04.319957

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = '6c609a8feecd'
down_revision: Union[str, Sequence[str], None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "calls",
        sa.Column("id", sa.Uuid(), primary_key=True),
        sa.Column("room_name", sa.String(255), nullable=False),
        sa.Column("call_type", sa.String(16), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float(), nullable=True),
        sa.Column("end_reason", sa.String(32), nullable=True),
        sa.Column("llm_model", sa.String(128), nullable=True),
        sa.Column("recording_key", sa.String(512), nullable=True),
        sa.Column("recording_status", sa.String(16), nullable=False, server_default="disabled"),
        sa.Column("egress_id", sa.String(64), nullable=True),
        sa.Column("avg_stt_ms", sa.Integer(), nullable=True),
        sa.Column("avg_llm_ttft_ms", sa.Integer(), nullable=True),
        sa.Column("avg_tts_ttfb_ms", sa.Integer(), nullable=True),
        sa.Column("avg_e2e_ms", sa.Integer(), nullable=True),
        sa.Column("turn_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column(
            "transcript",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("caller_number_masked", sa.String(16), nullable=True),
    )
    op.create_index("ix_calls_room_name", "calls", ["room_name"])
    op.create_index("ix_calls_started_at", "calls", ["started_at"])


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_calls_started_at", table_name="calls")
    op.drop_index("ix_calls_room_name", table_name="calls")
    op.drop_table("calls")
