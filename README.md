# BACE1 QSAR Predictor

A QSAR (Quantitative Structure–Activity Relationship) pipeline and interactive
web app that predicts small-molecule inhibitory potency (pIC50) against
**BACE1 (Beta-secretase 1)** — a validated therapeutic target in Alzheimer's
disease — from a SMILES string alone.

Built on real ChEMBL bioactivity data (13,777 raw records → 821 curated
compounds), using RDKit molecular fingerprints and a Random Forest regressor,
with rigorous **scaffold-based validation** to give an honest estimate of how
the model performs on genuinely novel chemical structures — not just molecules
similar to what it was trained on.

🔗 **Live demo:** `streamlit run App.py` (see [Setup](#setup) below)

---

## Why this isn't just another "RDKit + Random Forest" QSAR tutorial

Most beginner QSAR projects report one R² from one random train/test split
and stop there. That number is usually inflated, because medicinal chemistry
datasets are full of analog series — near-identical molecules with small
substituent changes. A random split lets close analogs land on both sides of
the split, so the model partly memorizes chemical series instead of learning
generalizable structure-activity relationships.

This project evaluates every model on **both** a random split and a
**Bemis-Murcko scaffold split** (which keeps each core scaffold entirely on
one side), and reports both numbers side by side:

| Feature set | Split | Test R² |
|---|---|---|
| 6 physicochemical descriptors | Random | 0.480 |
| 6 physicochemical descriptors | **Scaffold** | **-0.001** |
| Morgan fingerprints (2048-bit) | Random | 0.625 |
| Morgan fingerprints (2048-bit) | **Scaffold** | **0.222** |

The gap between the random-split and scaffold-split numbers *is* the
finding: the descriptor model's apparent 0.48 R² collapses to essentially
zero on novel chemotypes, while fingerprints generalize meaningfully better
but still leave real room for improvement. That's the honest, defensible
number — and the one this project reports as its headline result.

---

## What's in this repo

```
bace1_qsar_pipeline.py   Full pipeline: fetch → curate → featurize → split → train → evaluate
App.py                   Streamlit app for interactive SMILES → pIC50 prediction
data/
  raw/                   Raw ChEMBL bioactivity export
  processed/             Curated dataset, descriptors, fingerprints, train/test splits
models/
  best_model_fingerprints_scaffold.joblib   Trained model (fingerprints + scaffold split)
results/
  model_comparison.csv        All 4 experiments' metrics
  predicted_vs_actual.png     Test-set prediction scatter plot
  shap_summary.png            Feature importance (which substructures drive potency)
```

## Pipeline

| Stage | What it does |
|---|---|
| `fetch` | Pulls all BACE1 (CHEMBL4822) IC50 bioactivity records from the ChEMBL API. Chunked, retried, and saved incrementally — resumable if the connection drops mid-fetch. |
| `curate` | Filters to exact IC50 measurements in nM, validates every SMILES with RDKit, deduplicates by compound (median pIC50 across assays). |
| `featurize` | Computes 6 classical RDKit descriptors (MolWt, LogP, TPSA, HBD, HBA, RotatableBonds) *and* Morgan/ECFP fingerprints (radius 2, 2048 bits) for direct comparison. |
| `split` | Builds both a random 80/20 split and a Bemis-Murcko scaffold-based split, so leakage effects are visible rather than hidden. |
| `train` | Random Forest regression with `GridSearchCV` hyperparameter tuning and 5-fold cross-validation, across all 4 descriptor/fingerprint × random/scaffold combinations. |
| `evaluate` | Predicted-vs-actual scatter plot and SHAP summary plot for the best (fingerprints + scaffold split) model. |

## Setup

```bash
pip install chembl_webresource_client rdkit scikit-learn pandas numpy shap matplotlib joblib streamlit
```

Run the pipeline stages in order:

```bash
python bace1_qsar_pipeline.py fetch      # pulls ~14k records from ChEMBL (needs internet)
python bace1_qsar_pipeline.py curate
python bace1_qsar_pipeline.py featurize
python bace1_qsar_pipeline.py split
python bace1_qsar_pipeline.py train      # GridSearchCV across 4 experiments, a few minutes
python bace1_qsar_pipeline.py evaluate
```

Or run everything at once with `python bace1_qsar_pipeline.py --all`.

Then launch the interactive app:

```bash
streamlit run App.py
```

Paste any SMILES string to get a predicted pIC50/IC50, structure rendering,
and the six molecular descriptors — with the model's real scaffold-split
uncertainty shown in the sidebar, not hidden.

## Interactive app

<!-- Add a screenshot here, e.g.: ![App screenshot](results/app_screenshot.png) -->

The Streamlit app (`App.py`) loads the trained model and lets you:
- Paste a SMILES string and get an instant predicted pIC50 / IC50 (nM)
- See the parsed 2D structure
- View the six classical descriptors for the compound
- See the model's honest scaffold-split R²/MAE/RMSE in the sidebar, so
  predictions are never presented without their real uncertainty

## Key takeaway

Scaffold-based validation revealed that a model which looks strong under
random-split evaluation (R² up to 0.625) drops substantially — to R² 0.222 —
when tested on structurally novel compounds. This is a known but frequently
overlooked pitfall in QSAR modeling, and surfacing it honestly (rather than
reporting only the flattering number) is the main methodological contribution
of this project over a standard tutorial-level QSAR model.

## Possible extensions

- External validation on a second BACE1 dataset (e.g. BindingDB) to check
  for ChEMBL-specific overfitting
- Compare Random Forest against gradient boosting (XGBoost/LightGBM) and a
  graph neural network baseline
- Expand the descriptor set or try learned embeddings (e.g. Mol2Vec, ChemBERTa)
- Deploy the Streamlit app publicly (Streamlit Community Cloud) for a live,
  linkable demo

## Data source

Bioactivity data: [ChEMBL](https://www.ebi.ac.uk/chembl/) database, target
CHEMBL4822 (Beta-secretase 1 / BACE1).

## Author

Built by Navneet Vishwakarma as part of an ongoing portfolio in computational
drug design, cheminformatics, and machine learning.
