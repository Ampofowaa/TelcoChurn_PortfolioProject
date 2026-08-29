"""create customers_raw

Revision ID: 2a3418a6f529
Revises:
Create Date: 2026-08-22 00:27:30.490181

Baseline migration — reproduces sql/schema/001_create_raw.sql's DDL exactly
(columns, types, PRIMARY KEY, CHECK constraints) so `alembic upgrade head`
against an empty database produces the table data/ingest.py already expects.
sql/schema/001_create_raw.sql is deleted once this migration lands (see
PROJECT_PLAN.md's Phase 10a-i) — this file is now the sole source of this
table's DDL, hand-written rather than imported from
telco_churn.data.tables so this migration stays reproducible independent of
any future change to that module.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "2a3418a6f529"  # pragma: allowlist secret
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "customers_raw",
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
        sa.Column(
            "churn",
            sa.SmallInteger(),
            nullable=False,
        ),
        sa.CheckConstraint("seniorcitizen IN (0, 1)", name="ck_seniorcitizen_binary"),
        sa.CheckConstraint("tenure >= 0", name="ck_tenure_nonnegative"),
        sa.CheckConstraint("monthlycharges >= 0", name="ck_monthlycharges_nonnegative"),
        sa.CheckConstraint("churn IN (0, 1)", name="ck_customers_raw_churn_binary"),
        sa.PrimaryKeyConstraint("customerid", name="pk_customers_raw"),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("customers_raw")
