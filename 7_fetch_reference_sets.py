"""
Fetch small, ChEMBL-sourced reference sets that power two features in
App.py's "Selectivity & Reference Compounds" tab:

  cathepsin_d           -> data/reference/cathepsin_d_actives.csv
      Known active compounds (IC50 <= 1 uM) against cathepsin D (ChEMBL
      target search "Cathepsin D"), the aspartic protease most associated
      with historical off-target liability in BACE1 inhibitor programs.
      Used for the similarity-based selectivity screen.

  bace1_known_inhibitors -> data/reference/bace1_known_inhibitors.csv
      BACE1-tested compounds (target CHEMBL4822, same as the main training
      pipeline) that ChEMBL records a max_phase for (i.e. reached some
      stage of clinical development) — used as the "known/clinical-stage
      BACE1 inhibitors" reference panel.

Both outputs are small (typically tens to low hundreds of rows) and are
NOT training data — they're just a similarity reference set, so no
featurization/training step is needed for them.

Usage:
    python src/7_fetch_reference_sets.py --target cathepsin_d
    python src/7_fetch_reference_sets.py --target bace1_known_inhibitors
    python src/7_fetch_reference_sets.py --target all      # fetch both

Requires internet access to the public ChEMBL API (same requirement as
src/1_fetch_chembl_data.py). If you're in a sandboxed environment without
outbound internet, run this step wherever you ran the main pipeline's
fetch step, then copy data/reference/*.csv alongside the app.
"""
import argparse
import pandas as pd
from pathlib import Path
from chembl_webresource_client.new_client import new_client
OUT_DIR = Path(__file__).resolve().parent / "data" / "reference"

CATHEPSIN_D_OUT = OUT_DIR / "cathepsin_d_actives.csv"
BACE1_KNOWN_OUT = OUT_DIR / "bace1_known_inhibitors.csv"

ACTIVE_IC50_NM_CUTOFF = 1000  # <= 1 uM => "active" for the selectivity reference set
MAX_ROWS = 300                 # cap for app responsiveness; keeps the most potent rows


def get_target_id(target_name: str) -> str:
    target_client = new_client.target
    results = target_client.search(target_name)
    if not results:
        raise ValueError(f"No ChEMBL target found for '{target_name}'")
    for r in results:
        if r.get("target_type") == "SINGLE PROTEIN" and r.get("organism") == "Homo sapiens":
            print(f"Using target: {r['pref_name']} ({r['target_chembl_id']})")
            return r["target_chembl_id"]
    print(f"Using target: {results[0]['pref_name']} ({results[0]['target_chembl_id']})")
    return results[0]["target_chembl_id"]


def fetch_ic50_activities(target_chembl_id: str) -> pd.DataFrame:
    activity_client = new_client.activity
    activities = activity_client.filter(
        target_chembl_id=target_chembl_id,
        standard_type="IC50",
    ).only(
        "molecule_chembl_id", "canonical_smiles", "standard_relation",
        "standard_value", "standard_units",
    )
    df = pd.DataFrame(activities)
    print(f"Retrieved {len(df)} raw IC50 records")
    return df


def fetch_pref_names(molecule_ids):
    """Best-effort pref_name + max_phase lookup for a list of ChEMBL
    molecule IDs. Returns a dict keyed by molecule_chembl_id."""
    molecule_client = new_client.molecule
    info = {}
    ids = list(dict.fromkeys(molecule_ids))  # dedupe, preserve order
    batch_size = 50
    for i in range(0, len(ids), batch_size):
        batch = ids[i:i + batch_size]
        try:
            records = molecule_client.filter(molecule_chembl_id__in=batch).only(
                "molecule_chembl_id", "pref_name", "max_phase"
            )
            for r in records:
                info[r["molecule_chembl_id"]] = {
                    "pref_name": r.get("pref_name"),
                    "max_phase": r.get("max_phase"),
                }
        except Exception as e:
            print(f"  (molecule lookup batch failed, continuing without names: {e})")
    return info


def build_cathepsin_d_reference():
    target_id = get_target_id("Cathepsin D")
    df = fetch_ic50_activities(target_id)
    if df.empty:
        print("No activities retrieved for cathepsin D.")
        return

    df = df[df["standard_units"] == "nM"]
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value", "canonical_smiles"])
    df = df[df["standard_relation"].isin(["=", "<", "<="])]
    df = df[df["standard_value"] <= ACTIVE_IC50_NM_CUTOFF]

    # keep the most potent record per unique compound
    df = df.sort_values("standard_value").drop_duplicates("molecule_chembl_id", keep="first")
    df = df.head(MAX_ROWS)

    names = fetch_pref_names(df["molecule_chembl_id"].tolist())
    df["name"] = df["molecule_chembl_id"].map(lambda mid: (names.get(mid) or {}).get("pref_name") or mid)

    out = df[["name", "molecule_chembl_id", "canonical_smiles", "standard_value"]].rename(
        columns={"canonical_smiles": "smiles", "standard_value": "ic50_nM"}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(CATHEPSIN_D_OUT, index=False)
    print(f"Saved {len(out)} cathepsin D actives to {CATHEPSIN_D_OUT}")


def build_bace1_known_inhibitors_reference():
    target_id = get_target_id("BACE1")
    df = fetch_ic50_activities(target_id)
    if df.empty:
        print("No activities retrieved for BACE1.")
        return

    df = df.dropna(subset=["canonical_smiles"]).drop_duplicates("molecule_chembl_id")
    names = fetch_pref_names(df["molecule_chembl_id"].tolist())
    df["max_phase"] = df["molecule_chembl_id"].map(lambda mid: (names.get(mid) or {}).get("max_phase"))
    df["name"] = df["molecule_chembl_id"].map(lambda mid: (names.get(mid) or {}).get("pref_name") or mid)

    # keep only compounds ChEMBL records a clinical phase for
    df = df.dropna(subset=["max_phase"])
    df["max_phase"] = pd.to_numeric(df["max_phase"], errors="coerce")
    df = df[df["max_phase"].fillna(0) > 0]
    df = df.sort_values("max_phase", ascending=False)

    out = df[["name", "molecule_chembl_id", "canonical_smiles", "max_phase"]].rename(
        columns={"canonical_smiles": "smiles"}
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(BACE1_KNOWN_OUT, index=False)
    print(f"Saved {len(out)} known/clinical-stage BACE1 inhibitors to {BACE1_KNOWN_OUT}")
    if out.empty:
        print(
            "No BACE1-tested compounds with a recorded max_phase were found. "
            "This can happen if ChEMBL's annotations for this target are sparse — "
            "consider seeding data/reference/bace1_known_inhibitors.csv manually "
            "with a few named compounds (e.g. from a ChEMBL/PubChem web search) "
            "instead."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", choices=["cathepsin_d", "bace1_known_inhibitors", "all"], required=True)
    args = parser.parse_args()

    if args.target in ("cathepsin_d", "all"):
        build_cathepsin_d_reference()
    if args.target in ("bace1_known_inhibitors", "all"):
        build_bace1_known_inhibitors_reference()