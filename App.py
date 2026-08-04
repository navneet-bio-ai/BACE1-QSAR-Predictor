"""
BACE1 QSAR Predictor — Streamlit app (v3)

Loads the model trained by the src/ pipeline and provides:
  - Single-compound prediction (SMILES or drawn structure) with:
      * CNS-relevant scoring (CNS MPO, Ro5, BBB heuristic)
      * Applicability domain check (Tanimoto similarity to training set)
      * SHAP-based fragment explainability (which substructures drove the
        prediction, visualized on the molecule)
      * Full model diagnostics (test + cross-validated metrics, and the
        descriptor-vs-fingerprint / random-vs-scaffold-split comparison)
      * Downloadable per-compound PDF/CSV report
  - Batch screening from an uploaded CSV of SMILES, now with an
    applicability-domain flag per row
  - Benchmark comparison against known, approved Alzheimer's drugs
  - Selectivity screen against cathepsin D (the aspartic protease most
    associated with historical BACE1-inhibitor off-target liability) and a
    reference panel of known/clinical-stage BACE1 inhibitors — both driven
    by small ChEMBL-sourced reference sets (see src/7_fetch_reference_sets.py)

Run from the SAME folder as the src/ pipeline, after `5_train_model.py` has
produced models/best_model_fingerprints_scaffold.joblib:

    pip install -r requirements.txt
    streamlit run App.py
"""
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski
from rdkit.Chem import rdFingerprintGenerator

try:
    from rdkit.Chem import Draw
    DRAW_AVAILABLE = True
except ImportError:
    DRAW_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

try:
    from streamlit_ketcher import st_ketcher
    KETCHER_AVAILABLE = True
except ImportError:
    KETCHER_AVAILABLE = False

try:
    from fpdf import FPDF, XPos, YPos
    FPDF_AVAILABLE = True
except ImportError:
    FPDF_AVAILABLE = False

BASE = Path(__file__).resolve().parent
MODEL_PATH = BASE / "models" / "best_model_fingerprints_scaffold.joblib"
RESULTS_DIR = BASE / "results"
DATA_DIR = BASE / "data" / "processed"
FP_PATH = DATA_DIR / "fingerprints.npy"
FP_IDS_PATH = DATA_DIR / "fingerprint_ids.csv"
DESC_PATH = DATA_DIR / "descriptors.csv"
REF_DIR = BASE / "data" / "reference"
CATHEPSIN_D_REF_PATH = REF_DIR / "cathepsin_d_actives.csv"
BACE1_KNOWN_INHIBITORS_PATH = REF_DIR / "bace1_known_inhibitors.csv"

FP_RADIUS = 2
FP_NBITS = 2048

# Applicability-domain confidence bands, based on max Tanimoto similarity
# (binary Morgan fingerprints) to the nearest training-set compound. These
# thresholds are a commonly used rule of thumb for fingerprint-based QSAR
# AD screens, not a formally fitted cutoff — treat the label as directional.
AD_HIGH = 0.5
AD_MODERATE = 0.3

# Well-characterized, approved Alzheimer's drugs, for benchmark comparison.
# Note: donepezil, rivastigmine, galantamine (AChE/BuChE inhibitors) and
# memantine (NMDA antagonist) act on different targets than BACE1 — they are
# NOT BACE1 inhibitors. They're included as familiar reference points for
# drug-likeness/CNS-property comparison, not as BACE1 activity benchmarks.
KNOWN_AD_DRUGS = {
    "Donepezil (AChE inhibitor)": "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2",
    "Rivastigmine (AChE/BuChE inhibitor)": "CCN(C)C(=O)Oc1cccc(C(C)N(C)C)c1",
    "Galantamine (AChE inhibitor)": "COc1c(O)ccc2c1C[C@@H](O)C=C[C@@H]1[C@@H]3C[C@@]21CCN3C",
    "Memantine (NMDA antagonist)": "CC12CC3CC(C)(C1)CC(N)(C3)C2",
}

st.set_page_config(page_title="BACE1 QSAR Predictor", page_icon="🧬", layout="wide")

st.markdown(
    """
    <style>
    .metric-card {
        background-color: rgba(151,166,195,0.1);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.5rem;
    }
    .flag-pass { color: #2ecc71; font-weight: 600; }
    .flag-fail { color: #e74c3c; font-weight: 600; }
    .flag-warn { color: #f39c12; font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ── model / reference-data loading ─────────────────────────────────────
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_resource
def load_training_fingerprints():
    """Training-set fingerprints + IDs, used for the applicability-domain
    check. Returns (None, None) if the pipeline's processed data isn't
    present alongside the app (e.g. only the model file was deployed)."""
    if not FP_PATH.exists() or not FP_IDS_PATH.exists():
        return None, None
    fps = np.load(FP_PATH)
    ids = pd.read_csv(FP_IDS_PATH)["molecule_chembl_id"]
    return fps, ids


@st.cache_resource
def load_training_pIC50_lookup():
    if not DESC_PATH.exists():
        return {}
    d = pd.read_csv(DESC_PATH)
    return dict(zip(d["molecule_chembl_id"], d["pIC50"]))


@st.cache_data
def load_reference_set(path: Path):
    """Loads a small reference set of known actives (name, smiles[, chembl_id,
    max_phase]) used for the selectivity check / known-inhibitor panel. See
    src/7_fetch_reference_sets.py for how these CSVs are generated from
    ChEMBL. Returns None if the file hasn't been fetched yet."""
    if not path.exists():
        return None
    df = pd.read_csv(path)
    if "smiles" not in df.columns:
        return None
    return df


@st.cache_resource
def get_shap_explainer(_model):
    if not SHAP_AVAILABLE:
        return None
    return shap.TreeExplainer(_model)


@st.cache_resource
def load_model_comparison():
    path = RESULTS_DIR / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


# ── featurization ───────────────────────────────────────────────────────
def get_fp_generator():
    return rdFingerprintGenerator.GetMorganGenerator(radius=FP_RADIUS, fpSize=FP_NBITS)


def compute_morgan_fp(mol, radius=FP_RADIUS, n_bits=FP_NBITS):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = generator.GetFingerprint(mol)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr.reshape(1, -1)


def compute_morgan_fp_with_info(mol, radius=FP_RADIUS, n_bits=FP_NBITS):
    """Same fingerprint as compute_morgan_fp, but also returns RDKit's
    bit -> substructure-environment map, needed to draw *why* a given bit
    is set (used by the SHAP explainability panel)."""
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    ao = rdFingerprintGenerator.AdditionalOutput()
    ao.AllocateBitInfoMap()
    fp = generator.GetFingerprint(mol, additionalOutput=ao)
    arr = np.zeros((n_bits,), dtype=np.int8)
    for bit in fp.GetOnBits():
        arr[bit] = 1
    return arr.reshape(1, -1), ao.GetBitInfoMap()


def compute_descriptors(mol):
    return {
        "MolWt": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "HBD": Lipinski.NumHDonors(mol),
        "HBA": Lipinski.NumHAcceptors(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
    }


def predict_pIC50(model, mol_or_fp):
    fp = mol_or_fp if isinstance(mol_or_fp, np.ndarray) else compute_morgan_fp(mol_or_fp)
    pred = model.predict(fp)[0]
    return pred


# ── drug-likeness / CNS scoring (unchanged from v2) ─────────────────────
def rule_of_five(desc):
    """Lipinski Rule of 5 — standard oral drug-likeness screen."""
    violations = []
    if desc["MolWt"] > 500:
        violations.append("MW > 500")
    if desc["LogP"] > 5:
        violations.append("LogP > 5")
    if desc["HBD"] > 5:
        violations.append("HBD > 5")
    if desc["HBA"] > 10:
        violations.append("HBA > 10")
    return violations


def _desirability_decreasing(x, low, high):
    """1.0 below `low`, 0.0 above `high`, linear in between."""
    if x <= low:
        return 1.0
    if x >= high:
        return 0.0
    return (high - x) / (high - low)


def _desirability_hump(x, low1, low2, high2, high1):
    """0 below low1, ramps to 1 at low2, holds to high2, ramps to 0 at high1."""
    if x <= low1 or x >= high1:
        return 0.0
    if low2 <= x <= high2:
        return 1.0
    if x < low2:
        return (x - low1) / (low2 - low1)
    return (high1 - x) / (high1 - high2)


def cns_mpo_score(desc):
    """
    Simplified CNS MPO (Multiparameter Optimization) score, based on Wager
    et al. 2010 — the standard composite desirability score used in CNS
    drug discovery to balance potency-relevant properties against
    brain-penetration and safety liabilities.

    This is a SIMPLIFIED version using the 4 of 6 standard parameters we can
    reliably compute from SMILES alone (MW, LogP as a proxy for both cLogP
    and cLogD, TPSA, HBD) — it omits cLogD and the most basic pKa, which
    need additional prediction tools. Max score here is 4 (full CNS MPO is
    out of 6). Treat as directional, not a substitute for the full score.
    """
    s_mw = _desirability_decreasing(desc["MolWt"], 360, 500)
    s_logp = _desirability_decreasing(desc["LogP"], 3, 5)
    s_tpsa = _desirability_hump(desc["TPSA"], 20, 40, 90, 120)
    s_hbd = _desirability_decreasing(desc["HBD"], 0, 3)
    total = s_mw + s_logp + s_tpsa + s_hbd
    return total, {"MW": s_mw, "LogP": s_logp, "TPSA": s_tpsa, "HBD": s_hbd}


def bbb_heuristic(desc):
    """
    Simple rule-based CNS/BBB permeability screen (TPSA < 90, MW < 450,
    -1 < LogP < 4 — commonly cited rule-of-thumb ranges for CNS penetrants).

    This is a coarse heuristic, not a trained classifier. For a rigorous
    estimate, a dedicated trained BBB permeability model (e.g. a
    RandomForestClassifier on the MoleculeNet BBBP dataset) is a natural
    companion tool — pairing this BACE1 potency model with a separately
    trained BBB permeability classifier would give a fuller CNS drug-likeness
    picture, since a potent BACE1 inhibitor that can't cross the BBB has
    limited therapeutic value.
    """
    checks = {
        "TPSA < 90 Å²": desc["TPSA"] < 90,
        "MolWt < 450": desc["MolWt"] < 450,
        "-1 < LogP < 4": -1 < desc["LogP"] < 4,
    }
    passed = sum(checks.values())
    return passed, checks


# ── applicability domain ────────────────────────────────────────────────
def tanimoto_bulk(query_fp, ref_fps):
    """Tanimoto similarity between one binary fingerprint and a matrix of
    binary fingerprints. query_fp: (n_bits,); ref_fps: (N, n_bits)."""
    q = query_fp.astype(bool)
    r = ref_fps.astype(bool)
    inter = np.logical_and(r, q).sum(axis=1)
    union = np.logical_or(r, q).sum(axis=1)
    union = np.where(union == 0, 1, union)
    return inter / union


def applicability_domain(query_fp):
    """Returns similarity of the query to the training set, or None if the
    training fingerprints weren't shipped alongside the model."""
    fps, ids = load_training_fingerprints()
    if fps is None:
        return None
    sims = tanimoto_bulk(query_fp.flatten(), fps)
    nn_idx = int(np.argmax(sims))
    max_sim = float(sims[nn_idx])
    nn_id = ids.iloc[nn_idx]
    pic50_lookup = load_training_pIC50_lookup()
    top5 = np.sort(sims)[-5:]
    return {
        "max_similarity": max_sim,
        "mean_top5_similarity": float(top5.mean()),
        "nearest_id": nn_id,
        "nearest_pIC50": pic50_lookup.get(nn_id),
    }


def ad_confidence_label(max_sim):
    if max_sim >= AD_HIGH:
        return "Within applicability domain", "flag-pass", "success"
    elif max_sim >= AD_MODERATE:
        return "Moderate confidence — structurally somewhat novel", "flag-warn", "warning"
    else:
        return "Outside applicability domain — treat prediction as low-confidence", "flag-fail", "error"


# ── SHAP explainability ─────────────────────────────────────────────────
def explain_prediction(model, mol, fp):
    """Per-molecule SHAP attribution over the 2048 fingerprint bits,
    restricted to bits actually present in this molecule (the only ones
    that can be drawn as a concrete substructure)."""
    explainer = get_shap_explainer(model)
    if explainer is None:
        return None

    fp_with_info, bit_info = compute_morgan_fp_with_info(mol)
    shap_values = explainer.shap_values(fp_with_info)
    sv = np.array(shap_values).flatten()

    base_value = explainer.expected_value
    if isinstance(base_value, (list, np.ndarray)):
        base_value = np.array(base_value).flatten()[0]

    set_bits = np.where(fp_with_info.flatten() == 1)[0]
    drawable = [(int(b), float(sv[b])) for b in set_bits if b in bit_info]

    top_positive = sorted([x for x in drawable if x[1] > 0], key=lambda x: -x[1])[:5]
    top_negative = sorted([x for x in drawable if x[1] < 0], key=lambda x: x[1])[:5]

    return {
        "base_value": float(base_value),
        "contribution_sum": float(sv.sum()),
        "top_positive": top_positive,
        "top_negative": top_negative,
        "bit_info": bit_info,
    }


def render_bit_image(mol, bit_id, bit_info, size=(200, 200)):
    if not DRAW_AVAILABLE:
        return None
    try:
        return Draw.DrawMorganBit(mol, int(bit_id), bit_info, useSVG=False, molSize=size)
    except Exception:
        return None


# ── selectivity check ───────────────────────────────────────────────────
@st.cache_data
def _reference_fingerprints(df: pd.DataFrame):
    """Fingerprints for a reference set dataframe with a `smiles` column.
    Rows with unparseable SMILES are dropped."""
    generator = get_fp_generator()
    rows, valid_idx = [], []
    for i, smi in enumerate(df["smiles"]):
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            continue
        fp = generator.GetFingerprint(mol)
        arr = np.zeros((FP_NBITS,), dtype=np.int8)
        for b in fp.GetOnBits():
            arr[b] = 1
        rows.append(arr)
        valid_idx.append(i)
    if not rows:
        return None, None
    return np.vstack(rows), df.iloc[valid_idx].reset_index(drop=True)


def selectivity_check(query_fp):
    """Compares the query molecule's similarity to known cathepsin D actives
    against a rough BACE1-relevance baseline. Flags molecules that look
    structurally close to known cathepsin D binders, since off-target
    aspartic-protease activity (especially cathepsin D) has historically
    been a liability for BACE1 inhibitor programs."""
    ref_df = load_reference_set(CATHEPSIN_D_REF_PATH)
    if ref_df is None:
        return None
    fps, valid_df = _reference_fingerprints(ref_df)
    if fps is None:
        return None
    sims = tanimoto_bulk(query_fp.flatten(), fps)
    nn_idx = int(np.argmax(sims))
    return {
        "max_similarity_to_cathepsin_d_actives": float(sims[nn_idx]),
        "nearest_name": valid_df.iloc[nn_idx].get("name", valid_df.iloc[nn_idx].get("molecule_chembl_id", "unknown")),
        "n_reference_compounds": len(valid_df),
    }


# ── PDF report ───────────────────────────────────────────────────────────
_PDF_UNICODE_REPLACEMENTS = {
    "\u2014": "-", "\u2013": "-",   # em dash, en dash
    "\u2265": ">=", "\u2264": "<=",
    "\u00b1": "+/-",
    "\u00c5": "A",                   # Å
    "\u2192": "->", "\u2019": "'", "\u2018": "'",
    "\u201c": '"', "\u201d": '"',
}


def _pdf_safe(text: str) -> str:
    """The core PDF fonts (Helvetica/Courier) are Latin-1 only, so common
    Unicode punctuation used elsewhere in the app (em dashes, ≥, ±, Å, ...)
    has to be swapped for an ASCII equivalent before it reaches fpdf, or
    report generation raises."""
    for u, ascii_ in _PDF_UNICODE_REPLACEMENTS.items():
        text = text.replace(u, ascii_)
    return text.encode("latin-1", "replace").decode("latin-1")


def build_pdf_report(smiles, desc, pred_pIC50, pred_ic50_nM, mpo, bbb_passed,
                      ro5_violations, ad_result, sel_result):
    if not FPDF_AVAILABLE:
        return None
    pdf = FPDF()
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _pdf_safe("BACE1 QSAR Prediction Report"), new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 6, _pdf_safe("Research/portfolio tool - not a validated assay. Treat as directional."),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Compound", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Courier", "", 9)
    pdf.multi_cell(0, 5, _pdf_safe(smiles))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Prediction", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 6, _pdf_safe(f"Predicted pIC50: {pred_pIC50:.2f}   (IC50 ~ {pred_ic50_nM:,.1f} nM)"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if ad_result is not None:
        label, _, _ = ad_confidence_label(ad_result["max_similarity"])
        pdf.multi_cell(0, 6, _pdf_safe(
            f"Applicability domain: {label} (max Tanimoto sim = "
            f"{ad_result['max_similarity']:.2f} to {ad_result['nearest_id']})"
        ))
    pdf.ln(2)

    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Drug-likeness / CNS properties", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("Helvetica", "", 10)
    for k, v in desc.items():
        pdf.cell(0, 6, _pdf_safe(f"{k}: {v:.2f}" if isinstance(v, float) else f"{k}: {v}"),
                 new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"Simplified CNS MPO: {mpo:.2f} / 4", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, f"BBB heuristic checks passed: {bbb_passed} / 3", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.cell(0, 6, _pdf_safe(f"Rule-of-5 violations: {len(ro5_violations)} "
                             f"({', '.join(ro5_violations) if ro5_violations else 'none'})"),
             new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    if sel_result is not None:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Selectivity screen (vs. cathepsin D)", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
        pdf.set_font("Helvetica", "", 10)
        pdf.multi_cell(0, 6, _pdf_safe(
            f"Max similarity to known cathepsin D actives: "
            f"{sel_result['max_similarity_to_cathepsin_d_actives']:.2f} "
            f"(nearest: {sel_result['nearest_name']}, n={sel_result['n_reference_compounds']} reference compounds)"
        ))

    return bytes(pdf.output())


# ── single-compound rendering ────────────────────────────────────────────
def render_single_result(mol, smiles_str, model, key_prefix=""):
    desc = compute_descriptors(mol)
    fp, bit_info = compute_morgan_fp_with_info(mol)
    pred_pIC50 = predict_pIC50(model, fp)
    pred_ic50_nM = 10 ** (9 - pred_pIC50)

    col1, col2, col3 = st.columns([1, 1, 1.2])

    with col1:
        if DRAW_AVAILABLE:
            img = Draw.MolToImage(mol, size=(260, 260))
            st.image(img, caption="Parsed structure")
        else:
            st.caption("Structure image unavailable in this environment — prediction unaffected.")

    with col2:
        st.metric("Predicted pIC50", f"{pred_pIC50:.2f}")
        st.metric("Predicted IC50", f"{pred_ic50_nM:,.1f} nM")
        if pred_pIC50 >= 8:
            st.success("High predicted potency (pIC50 ≥ 8)")
        elif pred_pIC50 >= 6:
            st.info("Moderate predicted potency (6 ≤ pIC50 < 8)")
        else:
            st.warning("Low predicted potency (pIC50 < 6)")

    with col3:
        mpo, mpo_parts = cns_mpo_score(desc)
        st.metric("Simplified CNS MPO", f"{mpo:.2f} / 4")
        bbb_passed, bbb_checks = bbb_heuristic(desc)
        st.metric("BBB heuristic checks passed", f"{bbb_passed} / 3")
        ro5_violations = rule_of_five(desc)
        if not ro5_violations:
            st.markdown('<span class="flag-pass">✓ Passes Rule of 5</span>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<span class="flag-warn">⚠ Ro5 violations: {", ".join(ro5_violations)}</span>',
                unsafe_allow_html=True,
            )

    # ── applicability domain ──
    st.divider()
    ad_result = applicability_domain(fp)
    st.subheader("🎯 Applicability domain")
    if ad_result is None:
        st.caption(
            "Training-set fingerprints not found alongside this app "
            f"(expected at `{FP_PATH.relative_to(BASE)}`) — applicability-domain "
            "check skipped. Ship `data/processed/fingerprints.npy` and "
            "`fingerprint_ids.csv` with the model to enable it."
        )
    else:
        label, css_class, alert_kind = ad_confidence_label(ad_result["max_similarity"])
        getattr(st, alert_kind)(
            f"{label} — nearest training compound is **{ad_result['nearest_id']}** "
            f"(Tanimoto similarity = {ad_result['max_similarity']:.2f}"
            + (f", known pIC50 = {ad_result['nearest_pIC50']:.2f}" if ad_result['nearest_pIC50'] is not None else "")
            + ")"
        )
        c1, c2 = st.columns(2)
        c1.metric("Max similarity to any training compound", f"{ad_result['max_similarity']:.2f}")
        c2.metric("Mean similarity to 5 nearest neighbors", f"{ad_result['mean_top5_similarity']:.2f}")
        st.caption(
            "Rule of thumb: similarity ≥ 0.5 → prediction is on reasonably "
            "familiar chemical space; 0.3–0.5 → moderate extrapolation; "
            "< 0.3 → the model is extrapolating well outside what it was "
            "trained on and the pIC50 above should not be trusted at face value."
        )

    # ── selectivity vs cathepsin D ──
    sel_result = selectivity_check(fp)
    if sel_result is not None:
        st.subheader("⚖️ Selectivity screen (vs. cathepsin D)")
        sim = sel_result["max_similarity_to_cathepsin_d_actives"]
        if sim >= 0.5:
            st.warning(
                f"Structurally close to a known cathepsin D active "
                f"(similarity = {sim:.2f} to **{sel_result['nearest_name']}**) — "
                "worth a closer look at aspartic-protease selectivity before "
                "reading the BACE1 potency in isolation."
            )
        else:
            st.success(
                f"No strong structural similarity to known cathepsin D actives "
                f"(max similarity = {sim:.2f}, n={sel_result['n_reference_compounds']} reference compounds)."
            )
        st.caption(
            "Similarity-based heuristic against a small ChEMBL-sourced "
            "reference set, not a trained selectivity classifier — a low "
            "similarity score doesn't rule out off-target activity by "
            "another mechanism."
        )

    # ── SHAP explainability ──
    with st.expander("🔍 Why this prediction? (fragment-level explainability)"):
        if not SHAP_AVAILABLE:
            st.caption("Install `shap` (already in requirements.txt) to enable this panel.")
        else:
            explanation = explain_prediction(model, mol, fp)
            if explanation is None:
                st.caption("SHAP explainer unavailable.")
            else:
                st.caption(
                    f"Base (average) prediction: {explanation['base_value']:.2f} pIC50. "
                    f"This molecule's fingerprint bits shift that by "
                    f"{explanation['contribution_sum']:+.2f} to reach the "
                    f"{pred_pIC50:.2f} shown above. Only substructures actually "
                    "present in this molecule are shown below (fingerprint bits "
                    "that are 'off' can't be drawn as a fragment)."
                )
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.markdown("**Pushing potency up**")
                    if not explanation["top_positive"]:
                        st.caption("No positively-contributing fragments identified.")
                    for bit_id, contrib in explanation["top_positive"]:
                        img = render_bit_image(mol, bit_id, explanation["bit_info"])
                        if img is not None:
                            st.image(img, caption=f"Bit {bit_id}  (+{contrib:.3f} pIC50)")
                        else:
                            st.write(f"Bit {bit_id}: +{contrib:.3f} pIC50")
                with ec2:
                    st.markdown("**Pulling potency down**")
                    if not explanation["top_negative"]:
                        st.caption("No negatively-contributing fragments identified.")
                    for bit_id, contrib in explanation["top_negative"]:
                        img = render_bit_image(mol, bit_id, explanation["bit_info"])
                        if img is not None:
                            st.image(img, caption=f"Bit {bit_id}  ({contrib:.3f} pIC50)")
                        else:
                            st.write(f"Bit {bit_id}: {contrib:.3f} pIC50")

    with st.expander("Molecular descriptors & CNS property breakdown"):
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Descriptors**")
            st.table(pd.DataFrame(desc.items(), columns=["Descriptor", "Value"]).set_index("Descriptor"))
        with c2:
            st.write("**BBB heuristic checks**")
            for label_, passed in bbb_checks.items():
                icon = "✅" if passed else "❌"
                st.write(f"{icon} {label_}")
            st.write("**CNS MPO component scores** (each 0–1)")
            st.table(pd.DataFrame(mpo_parts.items(), columns=["Component", "Score"]).set_index("Component"))

    with st.expander("📊 Full model diagnostics"):
        comparison = load_model_comparison()
        if comparison is None:
            st.caption(f"Run the `evaluate`/`train` stage to populate `{RESULTS_DIR / 'model_comparison.csv'}`.")
        else:
            st.caption(
                "How the deployed model (fingerprints + scaffold split) compares "
                "to the basic-tutorial baseline (descriptors + random split) and "
                "the two intermediate setups — the gap between random-split and "
                "scaffold-split R² is the honest measure of how well this "
                "generalizes to genuinely new chemotypes."
            )
            show_cols = ["experiment", "n_train", "n_test", "cv_r2_mean", "cv_r2_std", "R2", "MAE", "RMSE"]
            show_cols = [c for c in show_cols if c in comparison.columns]
            st.dataframe(comparison[show_cols], use_container_width=True, hide_index=True)

    # ── report export ──
    st.divider()
    dl1, dl2 = st.columns(2)
    row = {
        "smiles": smiles_str,
        "pIC50": round(pred_pIC50, 3),
        "IC50_nM": round(pred_ic50_nM, 2),
        "CNS_MPO": round(mpo, 2),
        "BBB_heuristic_passed": bbb_passed,
        "Ro5_violations": len(ro5_violations),
        "AD_max_similarity": round(ad_result["max_similarity"], 3) if ad_result else None,
        "AD_nearest_training_id": ad_result["nearest_id"] if ad_result else None,
        "cathepsin_d_max_similarity": round(sel_result["max_similarity_to_cathepsin_d_actives"], 3) if sel_result else None,
        **{k: round(v, 3) if isinstance(v, float) else v for k, v in desc.items()},
    }
    with dl1:
        st.download_button(
            "Download this result as CSV",
            data=pd.DataFrame([row]).to_csv(index=False).encode("utf-8"),
            file_name="bace1_prediction.csv",
            mime="text/csv",
            key=f"{key_prefix}csv_dl",
        )
    with dl2:
        if FPDF_AVAILABLE:
            pdf_bytes = build_pdf_report(
                smiles_str, desc, pred_pIC50, pred_ic50_nM, mpo, bbb_passed,
                ro5_violations, ad_result, sel_result,
            )
            st.download_button(
                "Download PDF report",
                data=pdf_bytes,
                file_name="bace1_prediction_report.pdf",
                mime="application/pdf",
                key=f"{key_prefix}pdf_dl",
            )
        else:
            st.caption("Install `fpdf2` to enable PDF report export.")

    return row


# ── header ──────────────────────────────────────────────────────────────
st.title("🧬 BACE1 QSAR Predictor")
st.caption(
    "Predicts BACE1 (Beta-secretase 1) inhibitory potency for Alzheimer's drug discovery, "
    "with CNS-relevant drug-likeness scoring — Random Forest model on Morgan fingerprints, "
    "validated with scaffold-based train/test splitting."
)

model = load_model()

if model is None:
    st.error(
        f"No trained model found at `{MODEL_PATH}`.\n\n"
        "Run the pipeline's `train` stage first:\n\n"
        "```\npython src/5_train_model.py\n```"
    )
    st.stop()

with st.sidebar:
    st.subheader("Model performance")
    st.caption(
        "Reported on a scaffold-based test split (compounds with novel core "
        "structures the model never saw in training) — a stricter, more "
        "realistic estimate than a random split."
    )
    comparison_df = load_model_comparison()
    if comparison_df is not None:
        row = comparison_df[comparison_df["experiment"] == "fingerprints + scaffold_split"]
        if not row.empty:
            r = row.iloc[0]
            st.metric("Test R²", f"{r['R2']:.3f}")
            st.metric("Test MAE (pIC50)", f"{r['MAE']:.3f}")
            st.metric("Test RMSE (pIC50)", f"{r['RMSE']:.3f}")
            if "cv_r2_mean" in r and "cv_r2_std" in r:
                st.metric("5-fold CV R² (train set)", f"{r['cv_r2_mean']:.3f} ± {r['cv_r2_std']:.3f}")
    else:
        st.caption("Run `evaluate` stage to populate metrics here.")

    st.divider()
    ad_status = "available" if FP_PATH.exists() else "not found"
    sel_status = "available" if CATHEPSIN_D_REF_PATH.exists() else "not fetched yet"
    st.caption(f"Applicability domain data: **{ad_status}**")
    st.caption(f"Cathepsin D reference set: **{sel_status}**")
    st.divider()
    st.caption(
        "⚠️ Research/portfolio tool, not a validated assay. Scaffold-split "
        "R² reflects real uncertainty on novel chemotypes — treat "
        "predictions as directional, not definitive."
    )

# ── tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🔬 Single Prediction",
    "📋 Batch Screening",
    "💊 Compare to Known AD Drugs",
    "⚖️ Selectivity & Reference Compounds",
])

with tab1:
    example_smiles = "COc1cc2c(cc1OC)C(=O)C(CC1CCN(C)CC1)C2"

    if KETCHER_AVAILABLE:
        input_mode = st.radio("Input method", ["Type / paste SMILES", "Draw structure"], horizontal=True)
    else:
        input_mode = "Type / paste SMILES"

    smiles_input = None
    if input_mode == "Draw structure":
        st.caption("Draw a structure, then click the ✔ / export button in the editor to send it to the predictor.")
        smiles_input = st_ketcher("", key="ketcher_input")
    else:
        smiles_input = st.text_input(
            "SMILES string",
            value="",
            placeholder=f"e.g. {example_smiles}",
            help="Paste a SMILES string for the compound you want to screen against BACE1.",
        )

    if smiles_input:
        mol = Chem.MolFromSmiles(smiles_input)
        if mol is None:
            st.error("That SMILES string couldn't be parsed. Check it and try again.")
        else:
            render_single_result(mol, smiles_input, model)
    else:
        st.info("Enter or draw a structure above to get a prediction.")
        if input_mode == "Type / paste SMILES":
            st.caption(f"Try the example: `{example_smiles}`")

with tab2:
    st.write(
        "Upload a CSV with a `smiles` column to screen multiple compounds at once — "
        "useful for triaging a virtual library or a set of analogs."
    )
    uploaded = st.file_uploader("CSV file", type=["csv"])
    if uploaded is not None:
        try:
            batch_df = pd.read_csv(uploaded)
        except Exception as e:
            st.error(f"Couldn't read that CSV: {e}")
            batch_df = None

        if batch_df is not None:
            smiles_col = None
            for candidate in ["smiles", "SMILES", "Smiles", "canonical_smiles"]:
                if candidate in batch_df.columns:
                    smiles_col = candidate
                    break

            if smiles_col is None:
                st.error("No `smiles` column found. Expected a column named `smiles` (case-insensitive).")
            else:
                results = []
                progress = st.progress(0, text="Screening compounds...")
                n = len(batch_df)
                for i, smi in enumerate(batch_df[smiles_col]):
                    mol = Chem.MolFromSmiles(str(smi))
                    if mol is None:
                        results.append({"smiles": smi, "valid": False})
                        continue
                    desc = compute_descriptors(mol)
                    fp = compute_morgan_fp(mol)
                    pred = predict_pIC50(model, fp)
                    mpo, _ = cns_mpo_score(desc)
                    bbb_passed, _ = bbb_heuristic(desc)
                    ro5_violations = rule_of_five(desc)
                    ad_result = applicability_domain(fp)
                    results.append({
                        "smiles": smi, "valid": True,
                        "pIC50": round(pred, 2),
                        "IC50_nM": round(10 ** (9 - pred), 1),
                        "CNS_MPO": round(mpo, 2),
                        "BBB_heuristic_passed": bbb_passed,
                        "Ro5_violations": len(ro5_violations),
                        "AD_max_similarity": round(ad_result["max_similarity"], 3) if ad_result else None,
                        "AD_confidence": ad_confidence_label(ad_result["max_similarity"])[0] if ad_result else "n/a",
                        **{k: round(v, 2) if isinstance(v, float) else v for k, v in desc.items()},
                    })
                    progress.progress((i + 1) / n, text=f"Screening compounds... {i+1}/{n}")

                results_df = pd.DataFrame(results)
                st.success(f"Screened {n} compounds ({results_df['valid'].sum()} valid).")
                st.dataframe(results_df, use_container_width=True)

                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results as CSV",
                    data=csv_bytes,
                    file_name="bace1_batch_predictions.csv",
                    mime="text/csv",
                )
    else:
        st.caption("Example CSV format: a single column named `smiles`, one SMILES string per row.")

with tab3:
    st.write(
        "For reference, here's how the model scores four well-known, approved "
        "Alzheimer's drugs on the same properties. **Note:** none of these are "
        "BACE1 inhibitors — donepezil, rivastigmine, and galantamine are "
        "acetylcholinesterase (AChE) inhibitors, and memantine is an NMDA "
        "receptor antagonist. They're shown here as familiar reference points "
        "for CNS drug-likeness comparison, not as a BACE1 activity benchmark."
    )
    ref_results = []
    for name, smi in KNOWN_AD_DRUGS.items():
        mol = Chem.MolFromSmiles(smi)
        desc = compute_descriptors(mol)
        fp = compute_morgan_fp(mol)
        pred = predict_pIC50(model, fp)
        mpo, _ = cns_mpo_score(desc)
        bbb_passed, _ = bbb_heuristic(desc)
        ref_results.append({
            "Drug": name,
            "Predicted pIC50 (BACE1 model)": round(pred, 2),
            "CNS MPO (/4)": round(mpo, 2),
            "BBB heuristic (/3)": bbb_passed,
            "MolWt": round(desc["MolWt"], 1),
            "LogP": round(desc["LogP"], 2),
            "TPSA": round(desc["TPSA"], 1),
        })
    st.dataframe(pd.DataFrame(ref_results), use_container_width=True)
    st.caption(
        "The 'Predicted pIC50' column here is what the BACE1 model outputs for these "
        "structures — since these drugs don't target BACE1, this number isn't a "
        "meaningful potency estimate for them. It's included only to show the model "
        "produces sane, bounded output on real, approved drug-like structures rather "
        "than degenerate values."
    )

with tab4:
    st.write(
        "BACE1 inhibitor programs have historically had to watch selectivity "
        "against other aspartic proteases — cathepsin D in particular. This "
        "tab screens whichever compound you last predicted in **Single "
        "Prediction** against a small ChEMBL-sourced reference set of known "
        "cathepsin D actives, and lists known/clinical-stage BACE1 inhibitors "
        "for context."
    )

    cath_df = load_reference_set(CATHEPSIN_D_REF_PATH)
    st.subheader("Cathepsin D reference set")
    if cath_df is None:
        st.warning(
            "Reference set not found. Fetch it once (needs internet access "
            "to the ChEMBL API):\n\n"
            "```\npython src/7_fetch_reference_sets.py --target cathepsin_d\n```\n\n"
            f"This writes `{CATHEPSIN_D_REF_PATH.relative_to(BASE)}`, after "
            "which the selectivity screen in the Single Prediction tab "
            "activates automatically."
        )
    else:
        st.caption(f"{len(cath_df)} known cathepsin D actives loaded from ChEMBL.")
        st.dataframe(cath_df.head(20), use_container_width=True, hide_index=True)

    st.subheader("Known / clinical-stage BACE1 inhibitors")
    bace_ref_df = load_reference_set(BACE1_KNOWN_INHIBITORS_PATH)
    if bace_ref_df is None:
        st.warning(
            "Reference set not found. Fetch it once:\n\n"
            "```\npython src/7_fetch_reference_sets.py --target bace1_known_inhibitors\n```\n\n"
            f"This writes `{BACE1_KNOWN_INHIBITORS_PATH.relative_to(BASE)}` "
            "with ChEMBL-sourced BACE1-annotated compounds filtered to those "
            "with a recorded clinical phase, and this panel will score them "
            "on the same properties as the AD-drug comparison tab."
        )
    else:
        rows = []
        for _, r in bace_ref_df.iterrows():
            mol = Chem.MolFromSmiles(str(r["smiles"]))
            if mol is None:
                continue
            desc = compute_descriptors(mol)
            fp = compute_morgan_fp(mol)
            pred = predict_pIC50(model, fp)
            mpo, _ = cns_mpo_score(desc)
            rows.append({
                "Name": r.get("name", r.get("molecule_chembl_id", "?")),
                "ChEMBL ID": r.get("molecule_chembl_id", ""),
                "Max phase": r.get("max_phase", ""),
                "Predicted pIC50": round(pred, 2),
                "CNS MPO (/4)": round(mpo, 2),
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("Reference file found but no rows could be parsed.")