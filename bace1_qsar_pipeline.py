"""
BACE1 QSAR Pipeline — single-file version
==========================================

Predicts BACE1 (Beta-secretase 1) inhibitory activity (pIC50) from
molecular structure, using RDKit descriptors + Morgan fingerprints and a
Random Forest regressor, with scaffold-aware train/test splitting,
cross-validation, and hyperparameter tuning.

Stages (run in order, or run all with --all):
  fetch      pull raw BACE1 IC50 bioactivity data from ChEMBL
  curate     clean, dedupe, convert IC50 -> pIC50
  featurize  compute RDKit descriptors + Morgan/ECFP fingerprints
  split      Bemis-Murcko scaffold split + random split (for comparison)
  train      GridSearchCV + 5-fold CV over 4 setups (desc/fp x random/scaffold)
  evaluate   predicted-vs-actual plot + SHAP summary plot for the best model

Usage:
    pip install chembl_webresource_client rdkit scikit-learn pandas numpy shap matplotlib joblib

    python bace1_qsar_pipeline.py fetch
    python bace1_qsar_pipeline.py curate
    python bace1_qsar_pipeline.py featurize
    python bace1_qsar_pipeline.py split
    python bace1_qsar_pipeline.py train
    python bace1_qsar_pipeline.py evaluate

    # or run everything in sequence:
    python bace1_qsar_pipeline.py --all

Note: `fetch` calls the public ChEMBL web API and needs outbound internet
access. Without it, download the CSV manually from
https://www.ebi.ac.uk/chembl/ (search target CHEMBL4822 = BACE1, export
activities as CSV) and place it at data/raw/bace1_raw.csv, then run from
`curate` onward.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# ── paths ────────────────────────────────────────────────────────────────
BASE = Path(__file__).resolve().parent
RAW_PATH = BASE / "data" / "raw" / "bace1_raw.csv"
CURATED_PATH = BASE / "data" / "processed" / "bace1_curated.csv"
DESC_PATH = BASE / "data" / "processed" / "descriptors.csv"
FP_PATH = BASE / "data" / "processed" / "fingerprints.npy"
FP_IDS_PATH = BASE / "data" / "processed" / "fingerprint_ids.csv"
SCAFFOLD_SPLIT_PATH = BASE / "data" / "processed" / "split_scaffold.csv"
RANDOM_SPLIT_PATH = BASE / "data" / "processed" / "split_random.csv"
MODEL_PATH = BASE / "models" / "best_model_fingerprints_scaffold.joblib"
RESULTS_DIR = BASE / "results"

TARGET_NAME = "BACE1"
FP_RADIUS = 2
FP_NBITS = 2048
TEST_FRACTION = 0.2
RANDOM_STATE = 42
PARAM_GRID = {
    "n_estimators": [200, 400],
    "max_depth": [None, 12, 24],
    "min_samples_leaf": [1, 2, 4],
}


def _ensure_dirs():
    for p in [RAW_PATH, CURATED_PATH, DESC_PATH, MODEL_PATH]:
        p.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ── stage 1: fetch ──────────────────────────────────────────────────────
def stage_fetch():
    """
    Fetch is chunked, retried, and saved incrementally to disk after every
    chunk. ChEMBL's API sometimes drops long-lived connections partway
    through large targets (BACE1 has ~27k IC50 records) — a plain single
    request for everything loses all progress on one dropped packet. This
    version survives that: if it dies partway, re-running `fetch` resumes
    from the last completed chunk instead of starting over.
    """
    import time
    from chembl_webresource_client.new_client import new_client

    CHUNK_SIZE = 500
    MAX_RETRIES = 6
    FIELDS = [
        "molecule_chembl_id", "canonical_smiles", "standard_type",
        "standard_relation", "standard_value", "standard_units",
        "assay_chembl_id", "assay_description",
    ]

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

    def fetch_chunk_with_retry(query, offset, end):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return list(query[offset:end])
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise
                wait = min(2 ** attempt, 30)
                print(f"    chunk {offset}-{end} failed (attempt {attempt}/{MAX_RETRIES}): "
                      f"{type(e).__name__}: {e}. Retrying in {wait}s...")
                time.sleep(wait)

    def fetch_activities_resumable(target_chembl_id: str):
        activity_client = new_client.activity
        query = activity_client.filter(
            target_chembl_id=target_chembl_id, standard_type="IC50"
        ).only(*FIELDS)

        total = len(query)  # one lightweight request to get the count
        print(f"Target has {total} IC50 records total; fetching in chunks of {CHUNK_SIZE}")

        # resume support: if a partial file exists from a previous failed
        # run, pick up where it left off instead of re-fetching everything
        start_offset = 0
        write_header = True
        if RAW_PATH.exists():
            existing = pd.read_csv(RAW_PATH)
            start_offset = len(existing)
            write_header = False
            print(f"Found existing partial file with {start_offset} records — resuming from there")

        offset = start_offset
        while offset < total:
            end = min(offset + CHUNK_SIZE, total)
            chunk = fetch_chunk_with_retry(query, offset, end)
            chunk_df = pd.DataFrame(chunk)
            chunk_df.to_csv(RAW_PATH, mode="a", index=False, header=write_header)
            write_header = False
            print(f"  fetched {end}/{total} records (saved to disk)")
            offset = end

        print(f"Done. Total records saved: {offset}")

    target_id = get_target_id(TARGET_NAME)
    fetch_activities_resumable(target_id)
    print(f"Saved raw data to {RAW_PATH}")


# ── stage 2: curate ─────────────────────────────────────────────────────
def stage_curate():
    from rdkit import Chem

    def is_valid_smiles(smiles: str) -> bool:
        if not isinstance(smiles, str) or not smiles:
            return False
        return Chem.MolFromSmiles(smiles) is not None

    df = pd.read_csv(RAW_PATH)
    n0 = len(df)

    df = df[df["standard_relation"] == "="]
    df = df[df["standard_units"] == "nM"]
    df = df.dropna(subset=["canonical_smiles", "standard_value"])
    df["standard_value"] = pd.to_numeric(df["standard_value"], errors="coerce")
    df = df.dropna(subset=["standard_value"])
    df = df[df["standard_value"] > 0]

    df = df[df["canonical_smiles"].apply(is_valid_smiles)]

    # IC50 (nM) -> pIC50 = -log10(IC50 in M)
    df["pIC50"] = -np.log10(df["standard_value"] * 1e-9)

    df_grouped = (
        df.groupby(["molecule_chembl_id", "canonical_smiles"])["pIC50"]
        .median()
        .reset_index()
    )

    print(f"Raw records: {n0}")
    print(f"After filtering to '=' / nM / valid SMILES: {len(df)}")
    print(f"Unique compounds after dedup (median pIC50): {len(df_grouped)}")

    df_grouped.to_csv(CURATED_PATH, index=False)
    print(f"Saved curated dataset to {CURATED_PATH}")


# ── stage 3: featurize ──────────────────────────────────────────────────
def stage_featurize():
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Lipinski
    from rdkit.Chem import rdFingerprintGenerator

    def compute_descriptors(mol) -> dict:
        return {
            "MolWt": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBD": Lipinski.NumHDonors(mol),
            "HBA": Lipinski.NumHAcceptors(mol),
            "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        }

    def compute_morgan_fp(mol, generator) -> np.ndarray:
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((FP_NBITS,), dtype=np.int8)
        for bit in fp.GetOnBits():
            arr[bit] = 1
        return arr

    df = pd.read_csv(CURATED_PATH)
    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)

    desc_rows, fp_rows, valid_ids = [], [], []
    for _, row in df.iterrows():
        mol = Chem.MolFromSmiles(row["canonical_smiles"])
        if mol is None:
            continue
        desc = compute_descriptors(mol)
        desc["molecule_chembl_id"] = row["molecule_chembl_id"]
        desc["pIC50"] = row["pIC50"]
        desc_rows.append(desc)
        fp_rows.append(compute_morgan_fp(mol, fp_generator))
        valid_ids.append(row["molecule_chembl_id"])

    desc_df = pd.DataFrame(desc_rows)
    desc_df.to_csv(DESC_PATH, index=False)
    print(f"Saved {len(desc_df)} rows of descriptors to {DESC_PATH}")

    fp_array = np.vstack(fp_rows)
    np.save(FP_PATH, fp_array)
    pd.DataFrame({"molecule_chembl_id": valid_ids}).to_csv(FP_IDS_PATH, index=False)
    print(f"Saved {fp_array.shape} fingerprint matrix to {FP_PATH}")


# ── stage 4: split ───────────────────────────────────────────────────────
def stage_split():
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from sklearn.model_selection import train_test_split

    def get_scaffold(smiles: str) -> str:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold)

    def scaffold_split(df, test_fraction, seed):
        scaffolds = defaultdict(list)
        for idx, smiles in zip(df.index, df["canonical_smiles"]):
            scaffolds[get_scaffold(smiles)].append(idx)

        rng = np.random.RandomState(seed)
        groups = list(scaffolds.values())
        rng.shuffle(groups)
        groups.sort(key=len, reverse=True)

        n_test_target = int(len(df) * test_fraction)
        test_idx = []
        for group in groups:
            if len(test_idx) < n_test_target:
                test_idx.extend(group)
            else:
                break

        split = pd.Series("train", index=df.index)
        split.loc[test_idx] = "test"
        return split

    df = pd.read_csv(CURATED_PATH)

    df["scaffold_split"] = scaffold_split(df, TEST_FRACTION, RANDOM_STATE)
    df[["molecule_chembl_id", "scaffold_split"]].rename(
        columns={"scaffold_split": "split"}
    ).to_csv(SCAFFOLD_SPLIT_PATH, index=False)

    train_ids, test_ids = train_test_split(
        df["molecule_chembl_id"], test_size=TEST_FRACTION, random_state=RANDOM_STATE
    )
    random_split = pd.DataFrame(
        {
            "molecule_chembl_id": df["molecule_chembl_id"],
            "split": np.where(df["molecule_chembl_id"].isin(train_ids), "train", "test"),
        }
    )
    random_split.to_csv(RANDOM_SPLIT_PATH, index=False)

    n_test_scaffold = (df["scaffold_split"] == "test").sum()
    print(f"Scaffold split: {len(df) - n_test_scaffold} train / {n_test_scaffold} test")
    print(f"Random split:   {len(train_ids)} train / {len(test_ids)} test")


# ── stage 5: train ──────────────────────────────────────────────────────
def stage_train():
    import joblib
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
    from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

    def evaluate(model, X_test, y_test) -> dict:
        preds = model.predict(X_test)
        return {
            "R2": r2_score(y_test, preds),
            "MAE": mean_absolute_error(y_test, preds),
            "RMSE": mean_squared_error(y_test, preds) ** 0.5,
        }

    def run_experiment(X, y, ids, split_df, label):
        split_map = dict(zip(split_df["molecule_chembl_id"], split_df["split"]))
        mask_train = ids.map(split_map).eq("train").values
        mask_test = ids.map(split_map).eq("test").values

        X_train, X_test = X[mask_train], X[mask_test]
        y_train, y_test = y[mask_train], y[mask_test]

        base_model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
        cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = GridSearchCV(base_model, PARAM_GRID, cv=cv, scoring="r2", n_jobs=-1)
        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        cv_scores = cross_val_score(best_model, X_train, y_train, cv=cv, scoring="r2")
        test_metrics = evaluate(best_model, X_test, y_test)

        result = {
            "experiment": label,
            "n_train": int(mask_train.sum()),
            "n_test": int(mask_test.sum()),
            "best_params": json.dumps(search.best_params_),
            "cv_r2_mean": float(cv_scores.mean()),
            "cv_r2_std": float(cv_scores.std()),
            **test_metrics,
        }
        print(f"\n[{label}]")
        print(f"  train/test: {result['n_train']}/{result['n_test']}")
        print(f"  best params: {result['best_params']}")
        print(f"  5-fold CV R2: {result['cv_r2_mean']:.3f} +/- {result['cv_r2_std']:.3f}")
        print(f"  test R2: {result['R2']:.3f}  MAE: {result['MAE']:.3f}  RMSE: {result['RMSE']:.3f}")
        return result, best_model

    desc_df = pd.read_csv(DESC_PATH)
    y = desc_df["pIC50"].values
    ids = desc_df["molecule_chembl_id"]
    X_desc = desc_df.drop(columns=["molecule_chembl_id", "pIC50"]).values

    fp = np.load(FP_PATH)
    fp_ids = pd.read_csv(FP_IDS_PATH)["molecule_chembl_id"]
    fp_lookup = {mid: i for i, mid in enumerate(fp_ids)}
    align_idx = [fp_lookup[mid] for mid in ids]
    X_fp = fp[align_idx]

    scaffold_split_df = pd.read_csv(SCAFFOLD_SPLIT_PATH)
    random_split_df = pd.read_csv(RANDOM_SPLIT_PATH)

    results = []
    r, _ = run_experiment(X_desc, y, ids, random_split_df, "descriptors + random_split")
    results.append(r)
    r, _ = run_experiment(X_desc, y, ids, scaffold_split_df, "descriptors + scaffold_split")
    results.append(r)
    r, _ = run_experiment(X_fp, y, ids, random_split_df, "fingerprints + random_split")
    results.append(r)
    r, best_model = run_experiment(X_fp, y, ids, scaffold_split_df, "fingerprints + scaffold_split")
    results.append(r)

    pd.DataFrame(results).to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nSaved comparison table to {RESULTS_DIR / 'model_comparison.csv'}")

    joblib.dump(best_model, MODEL_PATH)
    print(f"Saved best (fingerprints + scaffold split) model to {MODEL_PATH}")


# ── stage 6: evaluate ────────────────────────────────────────────────────
def stage_evaluate():
    import joblib
    import matplotlib.pyplot as plt

    def predicted_vs_actual_plot(y_true, y_pred, out_path):
        plt.figure(figsize=(5, 5))
        plt.scatter(y_true, y_pred, alpha=0.5, s=15)
        lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
        plt.plot(lims, lims, "r--", linewidth=1)
        plt.xlabel("Actual pIC50")
        plt.ylabel("Predicted pIC50")
        plt.title("BACE1 QSAR: Predicted vs Actual (test set)")
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    def shap_summary_plot(model, X_sample, out_path):
        import shap
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()

    model = joblib.load(MODEL_PATH)

    desc_df = pd.read_csv(DESC_PATH)
    y = desc_df["pIC50"].values
    ids = desc_df["molecule_chembl_id"]

    fp = np.load(FP_PATH)
    fp_ids = pd.read_csv(FP_IDS_PATH)["molecule_chembl_id"]
    fp_lookup = {mid: i for i, mid in enumerate(fp_ids)}
    align_idx = [fp_lookup[mid] for mid in ids]
    X = fp[align_idx]

    split_df = pd.read_csv(SCAFFOLD_SPLIT_PATH)
    split_map = dict(zip(split_df["molecule_chembl_id"], split_df["split"]))
    mask_test = ids.map(split_map).eq("test").values

    X_test, y_test = X[mask_test], y[mask_test]
    y_pred = model.predict(X_test)

    predicted_vs_actual_plot(y_test, y_pred, RESULTS_DIR / "predicted_vs_actual.png")
    print(f"Saved {RESULTS_DIR / 'predicted_vs_actual.png'}")

    sample_size = min(200, X_test.shape[0])
    rng = np.random.RandomState(42)
    sample_idx = rng.choice(X_test.shape[0], sample_size, replace=False)
    try:
        shap_summary_plot(model, X_test[sample_idx], RESULTS_DIR / "shap_summary.png")
        print(f"Saved {RESULTS_DIR / 'shap_summary.png'}")
    except Exception as e:
        print(f"SHAP plot skipped ({e}). Install/upgrade shap if this persists.")


# ── entry point ──────────────────────────────────────────────────────────
STAGES = {
    "fetch": stage_fetch,
    "curate": stage_curate,
    "featurize": stage_featurize,
    "split": stage_split,
    "train": stage_train,
    "evaluate": stage_evaluate,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BACE1 QSAR pipeline (single-file version)")
    parser.add_argument("stage", nargs="?", choices=list(STAGES.keys()), help="pipeline stage to run")
    parser.add_argument("--all", action="store_true", help="run all stages in order")
    args = parser.parse_args()

    _ensure_dirs()

    if args.all:
        for name, fn in STAGES.items():
            print(f"\n{'=' * 60}\nSTAGE: {name}\n{'=' * 60}")
            fn()
    elif args.stage:
        STAGES[args.stage]()
    else:
        parser.print_help()
