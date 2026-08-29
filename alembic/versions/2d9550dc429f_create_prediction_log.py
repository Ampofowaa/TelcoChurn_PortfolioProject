"""create prediction_log

Revision ID: 2d9550dc429f
Revises: 8189f18ddc0c
Create Date: 2026-08-22 00:27:43.002494

Durable capture of every prediction served (or backtested) through
serving/app.py's /predict and /predict/batch, plus pipelines/batch_predict.py's
nightly sweep (route='nightly_batch') — the single foundation drift_check.py,
performance_check.py, and retraining read from. Schema per
PROJECT_PLAN.md's Phase 10a-i deliverable.

prediction_id is a surrogate BIGINT identity, not customerid: the same
customer is expected to recur across rows (repeat scoring, later cycles),
which is the normal shape of an event log, not a data-quality problem.
request_id deliberately carries no uniqueness constraint — a client retry
producing a second row with the same request_id is expected and harmless;
performance_check.py's join always takes the most recent prediction, so a
retried row is simply never selected, never double-counted.

Indexes: predicted_at (drift_check.py's daily window scan) and customerid
(performance_check.py's / the reserve mechanism's per-customer join) — named
explicitly here rather than left implicit, since without them both queries
degrade to a full sequential scan once the table has any real row count.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2d9550dc429f"
down_revision: str | Sequence[str] | None = "8189f18ddc0c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "prediction_log",
        sa.Column("prediction_id", sa.BigInteger(), sa.Identity(), nullable=False),
        sa.Column("request_id", sa.String(length=36), nullable=False),
        sa.Column("customerid", sa.String(length=20), nullable=True),
        sa.Column(
            "predicted_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("route", sa.String(length=20), nullable=False),
        sa.Column("feature_snapshot", postgresql.JSONB(), nullable=False),
        sa.Column("model_version", sa.String(length=20), nullable=False),
        sa.Column("run_id", sa.String(length=64), nullable=False),
        sa.Column("probability", sa.Float(), nullable=False),
        sa.Column("threshold", sa.Float(), nullable=False),
        sa.Column("decision", sa.Boolean(), nullable=False),
        sa.Column("contact", sa.Boolean(), nullable=True),
        sa.Column("dual_score_mode", sa.String(length=10), nullable=True),
        sa.Column("challenger_version", sa.String(length=20), nullable=True),
        sa.Column("challenger_probability", sa.Float(), nullable=True),
        sa.Column("champion_probability", sa.Float(), nullable=True),
        sa.Column("resolution_kind", sa.String(length=20), nullable=True),
        sa.PrimaryKeyConstraint("prediction_id", name="pk_prediction_log"),
    )
    op.create_index(
        "ix_prediction_log_predicted_at",
        "prediction_log",
        ["predicted_at"],
    )
    op.create_index(
        "ix_prediction_log_customerid",
        "prediction_log",
        ["customerid"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_prediction_log_customerid", table_name="prediction_log")
    op.drop_index("ix_prediction_log_predicted_at", table_name="prediction_log")
    op.drop_table("prediction_log")
