from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
GBD_STATS_PATH = PROJECT_ROOT / "data" / "reference" / "gbd_stats.csv"
DATA_CLINPRED_PATH = PROJECT_ROOT / "data" / "data_clinpred.csv"
OUTPUT_PATH = PROJECT_ROOT / "frontend" / "data" / "gbd_l3_indication_lookup.csv"


def _ta_code(value) -> str:
    text = str(value or "UNCLASSIFIED").strip()
    if not text or text.lower() == "nan":
        return "UNCLASSIFIED"
    if text.upper() in {"OTHER/UNCLASSIFIED", "OTHER / UNCLASSIFIED"}:
        return "UNCLASSIFIED"
    return text.upper()


def build_lookup() -> pd.DataFrame:
    stats = pd.read_csv(GBD_STATS_PATH)
    l3 = stats[stats["Level"].eq(3)].copy()
    l3["gbd_cause_id_3_ml"] = pd.to_numeric(l3["Cause ID"], errors="coerce").astype(int)
    l3["gbd_indication_name_3"] = l3["Cause Name"].astype(str)
    l3["canonical_model_ta"] = l3["model_ta"].fillna("Other/Unclassified").astype(str)
    l3["canonical_model_ta_code"] = l3["canonical_model_ta"].map(_ta_code)
    l3["sort_order"] = pd.to_numeric(l3["Sort Order"], errors="coerce").fillna(999999).astype(int)

    observed = pd.DataFrame(columns=["gbd_cause_id_3_ml", "observed_rows_total", "observed_tas", "observed_rows_by_ta"])
    if DATA_CLINPRED_PATH.exists():
        data = pd.read_csv(DATA_CLINPRED_PATH, usecols=["therapeutic_area", "gbd_cause_id_3_ml"])
        data["gbd_cause_id_3_ml"] = pd.to_numeric(
            data["gbd_cause_id_3_ml"],
            errors="coerce",
        ).fillna(0).astype(int)
        data["therapeutic_area"] = data["therapeutic_area"].map(_ta_code)

        counts = (
            data.groupby(["gbd_cause_id_3_ml", "therapeutic_area"])
            .size()
            .reset_index(name="rows")
        )

        observed_rows = []
        for cause_id, group in counts.groupby("gbd_cause_id_3_ml"):
            observed_rows.append({
                "gbd_cause_id_3_ml": int(cause_id),
                "observed_rows_total": int(group["rows"].sum()),
                "observed_tas": "|".join(sorted(group["therapeutic_area"].astype(str).unique())),
                "observed_rows_by_ta": json.dumps(
                    {
                        str(item.therapeutic_area): int(item.rows)
                        for item in group.itertuples(index=False)
                    },
                    sort_keys=True,
                ),
            })
        observed = pd.DataFrame(observed_rows)

    lookup = l3[
        [
            "gbd_cause_id_3_ml",
            "gbd_indication_name_3",
            "canonical_model_ta",
            "canonical_model_ta_code",
            "sort_order",
        ]
    ].merge(observed, on="gbd_cause_id_3_ml", how="left")

    lookup["observed_rows_total"] = lookup["observed_rows_total"].fillna(0).astype(int)
    lookup["observed_tas"] = lookup["observed_tas"].fillna("")
    lookup["observed_rows_by_ta"] = lookup["observed_rows_by_ta"].fillna("{}")

    fallback = pd.DataFrame([{
        "gbd_cause_id_3_ml": 0,
        "gbd_indication_name_3": "Other / Unclassified",
        "canonical_model_ta": "Other/Unclassified",
        "canonical_model_ta_code": "UNCLASSIFIED",
        "sort_order": 999999,
        "observed_rows_total": 0,
        "observed_tas": "UNCLASSIFIED",
        "observed_rows_by_ta": json.dumps({"UNCLASSIFIED": 0}),
    }])

    lookup = pd.concat([lookup, fallback], ignore_index=True)
    return lookup.sort_values(["sort_order", "gbd_cause_id_3_ml"]).reset_index(drop=True)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    build_lookup().to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
