"""create prediction_outcomes

Revision ID: 22b28d89f68c
Revises: 2d9550dc429f
Create Date: 2026-08-22 00:27:45.642305

The label side of the eventual prediction_log JOIN prediction_outcomes
retraining feed. Schema per PROJECT_PLAN.md's Phase 10a-i outcomes.py
deliverable. The unique triple (customerid, observed_at, source) is what
lets serving/outcomes.py's write path use INSERT ... ON CONFLICT DO NOTHING
to make a re-run of its CLI safe rather than a silent duplicate — that CLI
has no upstream caller in a position to notice and dedupe on its own.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "22b28d89f68c"
down_revision: str | Sequence[str] | None = "2d9550dc429f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "prediction_outcomes",
        sa.Column("outcome_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("customerid", sa.String(length=20), nullable=False),
        sa.Column("churned", sa.Boolean(), nullable=False),
        sa.Column("observed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column(
            "recorded_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("source", sa.String(length=20), nullable=False),
        sa.PrimaryKeyConstraint("outcome_id", name="pk_prediction_outcomes"),
        sa.UniqueConstraint(
            "customerid",
            "observed_at",
            "source",
            name="uq_prediction_outcomes_customer_observed_source",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("prediction_outcomes")
