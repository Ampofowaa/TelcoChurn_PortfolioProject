"""create training_pool

Revision ID: 6658a9eeadb6
Revises: 22b28d89f68c
Create Date: 2026-08-25 15:40:50.596461

The unified retraining feed for Phase 10a-ii's reserve mechanism: every
customers_raw column + churn, plus a nullable reserve_month SMALLINT (NULL =
the original one-time CSV seed, 1-6 = which reserve cohort a row's feature
values came from). sql/features/*.sql / features/build.py repoint from
customers_raw to this table so every future engineered-feature view covers
both the seeded population and every past/future reserve cohort without a
per-source SQL duplicate.

Surrogate BIGINT identity PK, not customerid — same reasoning as
prediction_log (see 2d9550dc429f_create_prediction_log.py): a customerid
legitimately recurs here (once from the one-time seed with reserve_month
NULL, once more per matured reserve cohort the cyclical reshape appends), so
customerid alone can't be unique. reserve_month is indexed since it's the
fold-forward training query's primary filter predicate.
"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "6658a9eeadb6"
down_revision: str | Sequence[str] | None = "22b28d89f68c"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "training_pool",
        sa.Column("training_pool_id", sa.BigInteger(), sa.Identity(), nullable=False),
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
        sa.Column("reserve_month", sa.SmallInteger(), nullable=True),
        sa.CheckConstraint("seniorcitizen IN (0, 1)", name="ck_seniorcitizen_binary"),
        sa.CheckConstraint("tenure >= 0", name="ck_tenure_nonnegative"),
        sa.CheckConstraint("monthlycharges >= 0", name="ck_monthlycharges_nonnegative"),
        sa.CheckConstraint("churn IN (0, 1)", name="ck_training_pool_churn_binary"),
        sa.PrimaryKeyConstraint("training_pool_id", name="pk_training_pool"),
    )
    op.create_index(
        "ix_training_pool_reserve_month",
        "training_pool",
        ["reserve_month"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_index("ix_training_pool_reserve_month", table_name="training_pool")
    op.drop_table("training_pool")
