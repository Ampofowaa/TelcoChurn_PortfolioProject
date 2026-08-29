"""Regenerate examples/sample_batch_predictions.csv from customers_crm.

Deterministic, not a one-off hand-edit: the 47 ID-only rows are
`dev_ids() ∩ customers_crm`'s customerids, sorted, first 47 — both the
dev/test split (seed 42, data/split.py) and customers_crm's customerid set
(identical to customers_raw's, see serving/crm_data.py) are fixed, so
re-running this always reproduces the same selection. Dev-partition only,
never test — the same discipline the original hand-picked demo IDs followed,
kept explicit here rather than silently dropped on regeneration: a demo file
should never surface a sealed test-set customerid, even though scoring a
real customer here isn't the kind of "evaluation" the test-set-touched-once
invariant actually governs.

The remaining 3 rows (one full-inline prospect, two partial-override
what-ifs) are fixed, hand-picked examples demonstrating /predict/batch's
other two item shapes. They are not derived from customers_crm and don't
change between runs.

Requires customers_crm to already be populated (`make crm-data`).
Usage: uv run python scripts/generate_sample_batch_predictions.py
"""

from __future__ import annotations

import csv

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.engine import Engine

from telco_churn.data.split import dev_ids
from telco_churn.serving.customer_lookup import LOOKUP_COLUMNS
from telco_churn.utils.db import get_engine
from telco_churn.utils.logging import configure_logging, get_logger
from telco_churn.utils.paths import get_project_root

logger = get_logger(__name__)

_N_ID_ONLY_ROWS = 47
_N_FETCH_DEMO_IDS = 5

_OUTPUT_PATH = get_project_root() / "examples" / "sample_batch_predictions.csv"
_CSV_HEADER = ["note", "customerid", *LOOKUP_COLUMNS]

_FULL_INLINE_PROSPECT: dict[str, str] = {
    "note": "Full inline - new prospect (no customerid; never touches Postgres)",
    "customerid": "",
    "gender": "Female",
    "seniorcitizen": "0",
    "has_partner": "No",
    "dependents": "No",
    "tenure": "0",
    "phoneservice": "Yes",
    "multiplelines": "No",
    "internetservice": "Fiber optic",
    "onlinesecurity": "No",
    "onlinebackup": "No",
    "deviceprotection": "No",
    "techsupport": "No",
    "streamingtv": "No",
    "streamingmovies": "No",
    "contract_type": "Month-to-month",
    "paperlessbilling": "Yes",
    "paymentmethod": "Electronic check",
    "monthlycharges": "75.0",
    "totalcharges": "",
}

_PARTIAL_OVERRIDE_ROWS: list[dict[str, str]] = [
    {
        "note": (
            "Partial override - what if this customer switched to "
            "month-to-month at a higher rate"
        ),
        "customerid": "9592-ERDKV",
        "contract_type": "Month-to-month",
        "monthlycharges": "95.0",
    },
    {
        "note": "Partial override - what if this customer dropped tech support",
        "customerid": "0440-EKDCF",
        "techsupport": "No",
        "monthlycharges": "52.0",
    },
]


def select_dev_partition_customer_ids(engine: Engine) -> list[str]:
    """Every customers_crm customerid that's also in the dev partition, sorted."""
    with engine.connect() as conn:
        crm_ids = {
            row[0] for row in conn.execute(text("SELECT customerid FROM customers_crm"))
        }
    dev = set(dev_ids().tolist())
    return sorted(crm_ids & dev)


def _blank_row(note: str, customerid: str) -> dict[str, str]:
    row = dict.fromkeys(_CSV_HEADER, "")
    row["note"] = note
    row["customerid"] = customerid
    return row


def build_rows(batch_ids: list[str]) -> list[dict[str, str]]:
    """ID-only rows for batch_ids, followed by the fixed prospect/override demo rows."""
    rows = [
        _blank_row("ID-only - resolved from customers_crm", cid) for cid in batch_ids
    ]
    rows.append(_FULL_INLINE_PROSPECT)
    for override in _PARTIAL_OVERRIDE_ROWS:
        row = _blank_row(override["note"], override["customerid"])
        row.update(
            {k: v for k, v in override.items() if k not in ("note", "customerid")}
        )
        rows.append(row)
    return rows


def main() -> None:
    load_dotenv()
    configure_logging()

    engine = get_engine()
    candidate_ids = select_dev_partition_customer_ids(engine)
    needed = _N_ID_ONLY_ROWS + _N_FETCH_DEMO_IDS
    if len(candidate_ids) < needed:
        raise RuntimeError(
            f"customers_crm has only {len(candidate_ids)} dev-partition rows, "
            f"need at least {needed}. Run `make crm-data` first."
        )

    batch_ids = candidate_ids[:_N_ID_ONLY_ROWS]
    fetch_demo_ids = candidate_ids[
        _N_ID_ONLY_ROWS : _N_ID_ONLY_ROWS + _N_FETCH_DEMO_IDS
    ]
    rows = build_rows(batch_ids)

    with _OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    logger.info(
        "sample_batch_predictions_written", path=str(_OUTPUT_PATH), rows=len(rows)
    )
    print(f"Wrote {len(rows)} rows to {_OUTPUT_PATH}")
    print("Fetch-button demo IDs (Score a Customer tab):")
    for cid in fetch_demo_ids:
        print(f"  {cid}")


if __name__ == "__main__":
    main()
