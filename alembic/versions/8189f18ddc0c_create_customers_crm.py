"""create customers_crm

Revision ID: 8189f18ddc0c
Revises: 2a3418a6f529
Create Date: 2026-08-22 00:27:38.327782

Reproduces sql/schema/003_create_customers_crm.sql — same LOOKUP_COLUMNS
shape as customers_raw, minus churn, plus crm_snapshot_at.
sql/schema/003_create_customers_crm.sql is deleted once this migration
lands (PROJECT_PLAN.md's Phase 10a-i).
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "8189f18ddc0c"
down_revision: str | Sequence[str] | None = "2a3418a6f529"  # pragma: allowlist secret
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "customers_crm",
        sa.Column("customerid", sa.String(length=20), nullable=False),
        sa.Column("gender", sa.String(length=10), nullable=False),
        sa.Column(
            "seniorcitizen",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.Column("has_partner", sa.String(length=3), nullable=False),
        sa.Column("dependents", sa.String(length=3), nullable=False),
        sa.Column(
            "tenure",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.Column("phoneservice", sa.String(length=3), nullable=False),
        sa.Column("multiplelines", sa.String(length=25), nullable=False),
        sa.Column("internetservice", sa.String(length=25), nullable=False),
        sa.Column("onlinesecurity", sa.String(length=25), nullable=False),
        sa.Column("onlinebackup", sa.String(length=25), nullable=False),
        sa.Column("deviceprotection", sa.String(length=25), nullable=False),
        sa.Column("techsupport", sa.String(length=25), nullable=False),
        sa.Column("streamingtv", sa.String(length=25), nullable=False),
        sa.Column("streamingmovies", sa.String(length=25), nullable=False),
        sa.Column("contract_type", sa.String(length=20), nullable=False),
        sa.Column("paperlessbilling", sa.String(length=3), nullable=False),
        sa.Column("paymentmethod", sa.String(length=45), nullable=False),
        sa.Column(
            "monthlycharges",
            sa.Numeric(precision=8, scale=2),
            nullable=False,
        ),
        sa.Column("totalcharges", sa.Numeric(precision=10, scale=2), nullable=True),
        sa.Column("crm_snapshot_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("seniorcitizen IN (0, 1)", name="ck_seniorcitizen_binary"),
        sa.CheckConstraint("tenure >= 0", name="ck_tenure_nonnegative"),
        sa.CheckConstraint("monthlycharges >= 0", name="ck_monthlycharges_nonnegative"),
        sa.PrimaryKeyConstraint("customerid", name="pk_customers_crm"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("customers_crm")
