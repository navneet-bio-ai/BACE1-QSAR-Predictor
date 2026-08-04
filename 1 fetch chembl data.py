"""
Fetch raw bioactivity (IC50) data from ChEMBL for a given target.

Defaults to BACE1 (Beta-secretase 1), ChEMBL target CHEMBL4822, so the
existing pipeline commands keep working unchanged:

    python src/1_fetch_chembl_data.py

To pull data for a different target (e.g. to train a second AD-target model
alongside BACE1, such as AChE for a multi-target profiling tool), pass
--target and, optionally, --out:

    python src/1_fetch_chembl_data.py --target AChE --out data/raw/ache_raw.csv
    python src/1_fetch_chembl_data.py --target "Acetylcholinesterase" --out data/raw/ache_raw.csv

Output: data/raw/bace1_raw.csv (or --out path)
"""
import argparse
import pandas as pd
from pathlib import Path
from chembl_webresource_client.new_client import new_client

DEFAULT_TARGET_NAME = "BACE1"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "raw" / "bace1_raw.csv"


def get_target_id(target_name: str) -> str:
    target_client = new_client.target
    results = target_client.search(target_name)
    if not results:
        raise ValueError(f"No ChEMBL target found for '{target_name}'")

    # Prefer an exact / single-protein human target if available
    for r in results:
        if r.get("target_type") == "SINGLE PROTEIN" and r.get("organism") == "Homo sapiens":
            print(f"Using target: {r['pref_name']} ({r['target_chembl_id']})")
            return r["target_chembl_id"]

    # fall back to first hit
    print(f"Using target: {results[0]['pref_name']} ({results[0]['target_chembl_id']})")
    return results[0]["target_chembl_id"]


def fetch_activities(target_chembl_id: str) -> pd.DataFrame:
    activity_client = new_client.activity
    activities = activity_client.filter(
        target_chembl_id=target_chembl_id,
        standard_type="IC50",
    ).only(
        "molecule_chembl_id",
        "canonical_smiles",
        "standard_type",
        "standard_relation",
        "standard_value",
        "standard_units",
        "assay_chembl_id",
        "assay_description",
    )
    df = pd.DataFrame(activities)
    print(f"Retrieved {len(df)} raw IC50 records")
    return df


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", default=DEFAULT_TARGET_NAME,
                         help=f"ChEMBL target name to search for (default: {DEFAULT_TARGET_NAME})")
    parser.add_argument("--out", type=Path, default=None,
                         help=f"Output CSV path (default: {DEFAULT_OUTPUT_PATH} when --target is unset, "
                              "otherwise data/raw/<target>_raw.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if args.out is not None:
        output_path = args.out
    elif args.target == DEFAULT_TARGET_NAME:
        output_path = DEFAULT_OUTPUT_PATH
    else:
        safe_name = "".join(c if c.isalnum() else "_" for c in args.target.lower())
        output_path = DEFAULT_OUTPUT_PATH.parent / f"{safe_name}_raw.csv"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    target_id = get_target_id(args.target)
    df = fetch_activities(target_id)
    df.to_csv(output_path, index=False)
    print(f"Saved raw data to {output_path}")