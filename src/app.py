
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors, Draw
from rdkit.Chem import rdFingerprintGenerator


# --------------------------------------------------
# PATHS AND SETTINGS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "drugonaut_fingerprint.npz"
SCAFFOLD_MODEL_PATH = PROJECT_ROOT / "models" / "drugonaut_scaffold.npz"
DATA_PATH = PROJECT_ROOT / "data" / "BBBP_prepared.csv"

FINGERPRINT_BITS = 1024
RADIUS = 2

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=FINGERPRINT_BITS,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Drugonaut | Molecular Research",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@500;600;700&display=swap');

    .stApp {
        background-color: #F7F6F0;
        color: #234E3D;
        font-family: 'DM Sans', sans-serif;
    }

    h1, h2, h3 {
        color: #234E3D !important;
        font-family: 'Playfair Display', serif !important;
    }

    p, label, li {
        color: #343D36;
    }

    [data-testid="stSidebar"] {
        background-color: #EEEFE6;
    }

    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #DDE3D9;
        border-radius: 12px;
        padding: 16px;
    }

    div.stButton > button,
    div.stDownloadButton > button {
        background-color: #234E3D;
        color: white;
        border: 1px solid #234E3D;
        border-radius: 8px;
        min-height: 42px;
    }

    div.stButton > button:hover,
    div.stDownloadButton > button:hover {
        background-color: #35674F;
        color: white;
        border-color: #35674F;
    }

    .eyebrow {
        text-transform: uppercase;
        letter-spacing: 3px;
        color: #64806A;
        font-size: 12px;
        font-weight: 700;
    }

    .research-note {
        background: #E9EEE5;
        border-left: 4px solid #234E3D;
        padding: 14px 18px;
        border-radius: 4px;
        margin-top: 18px;
        color: #343D36;
    }

    .footer {
        color: #777E73;
        font-size: 12px;
        text-align: center;
        padding: 24px 0 10px 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# COMMON FUNCTIONS
# --------------------------------------------------

def sigmoid(z):
    z = np.clip(z, -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def calculate_auc(y_true, scores):
    positives = scores[y_true == 1]
    negatives = scores[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return float("nan")

    greater = (positives[:, None] > negatives[None, :]).mean()
    tied = (positives[:, None] == negatives[None, :]).mean()

    return float(greater + 0.5 * tied)


def calculate_metrics(y_true, scores):
    predictions = (scores >= 0.5).astype(int)

    tn = int(np.sum((y_true == 0) & (predictions == 0)))
    fp = int(np.sum((y_true == 0) & (predictions == 1)))
    fn = int(np.sum((y_true == 1) & (predictions == 0)))
    tp = int(np.sum((y_true == 1) & (predictions == 1)))

    recall_0 = tn / (tn + fp) if tn + fp else 0.0
    recall_1 = tp / (tp + fn) if tp + fn else 0.0

    return {
        "accuracy": float(np.mean(predictions == y_true)),
        "balanced_accuracy": (recall_0 + recall_1) / 2,
        "roc_auc": calculate_auc(y_true, scores),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall_0": recall_0,
        "recall_1": recall_1,
        "predictions": predictions,
    }


# --------------------------------------------------
# MODEL AND DATA LOADING
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, "Model file not found: models/drugonaut_fingerprint.npz"

    try:
        with np.load(MODEL_PATH) as saved:
            weights = saved["weights"].astype(float)
            bias = float(np.asarray(saved["bias"]).reshape(-1)[0])

        if weights.size != FINGERPRINT_BITS:
            return None, (
                f"Expected {FINGERPRINT_BITS} weights, "
                f"but found {weights.size}."
            )

        return {"weights": weights, "bias": bias}, None

    except Exception as exc:
        return None, f"Could not load model: {exc}"


@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(DATA_PATH)

        if not {"smiles", "p_np"}.issubset(df.columns):
            return pd.DataFrame()

        if "name" not in df.columns:
            df["name"] = "Unnamed compound"

        return df

    except Exception:
        return pd.DataFrame()


# --------------------------------------------------
# MOLECULAR FUNCTIONS
# --------------------------------------------------

def make_fingerprint(mol):
    return GENERATOR.GetFingerprint(mol)


def fingerprint_array(mol):
    fingerprint = make_fingerprint(mol)
    values = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return values


def predict_bbb(mol, model):
    if model is None:
        return None

    values = fingerprint_array(mol).astype(float)
    score = float(sigmoid(
        np.dot(values, model["weights"]) + model["bias"]
    ))

    return {
        "prediction": int(score >= 0.5),
        "score": score,
    }


def find_similar_molecules(mol, df, top_n=5):
    if df.empty:
        return pd.DataFrame()

    target_fp = make_fingerprint(mol)
    target_smiles = Chem.MolToSmiles(mol)
    matches = []

    for _, row in df.iterrows():
        smiles_value = row.get("smiles")

        if not isinstance(smiles_value, str) or not smiles_value.strip():
            continue

        candidate = Chem.MolFromSmiles(smiles_value)

        if candidate is None:
            continue

        canonical = Chem.MolToSmiles(candidate)

        if canonical == target_smiles:
            continue

        similarity = DataStructs.TanimotoSimilarity(
            target_fp,
            make_fingerprint(candidate),
        )

        try:
            label = int(row["p_np"])
        except (ValueError, TypeError):
            label = row["p_np"]

        matches.append({
            "Compound": str(row.get("name", "Unnamed compound")),
            "Similarity (%)": round(similarity * 100, 2),
            "Dataset label": label,
            "SMILES": canonical,
        })

    if not matches:
        return pd.DataFrame()

    result = pd.DataFrame(matches)
    return (
        result.sort_values("Similarity (%)", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )


# --------------------------------------------------
# RANDOM-SPLIT MODEL EVALUATION
# --------------------------------------------------

@st.cache_data(show_spinner="Evaluating random-split model...")
def evaluate_saved_model():
    if not MODEL_PATH.exists() or not DATA_PATH.exists():
        return None, "The model or prepared BBBP dataset was not found."

    model, error = load_model()

    if model is None:
        return None, error

    try:
        df = pd.read_csv(DATA_PATH)
        fingerprints, labels, names, smiles_list = [], [], [], []
        skipped = 0

        for _, row in df.iterrows():
            if pd.isna(row["smiles"]) or pd.isna(row["p_np"]):
                skipped += 1
                continue

            molecule = Chem.MolFromSmiles(str(row["smiles"]))

            if molecule is None:
                skipped += 1
                continue

            try:
                label = int(row["p_np"])
            except (ValueError, TypeError):
                skipped += 1
                continue

            if label not in (0, 1):
                skipped += 1
                continue

            fingerprints.append(fingerprint_array(molecule))
            labels.append(label)
            names.append(str(row.get("name", "Unnamed compound")))
            smiles_list.append(str(row["smiles"]))

        X = np.asarray(fingerprints, dtype=np.float32)
        y = np.asarray(labels, dtype=np.int64)

        if len(y) == 0 or len(np.unique(y)) != 2:
            return None, "Evaluation requires valid examples from both classes."

        # Reproduce the existing random model's training split.
        rng = np.random.default_rng(42)
        train_parts, test_parts = [], []

        for label in (0, 1):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            split = int(len(indices) * 0.8)
            train_parts.append(indices[:split])
            test_parts.append(indices[split:])

        train_idx = np.concatenate(train_parts)
        test_idx = np.concatenate(test_parts)
        rng.shuffle(train_idx)
        rng.shuffle(test_idx)

        scores = sigmoid(X[test_idx] @ model["weights"] + model["bias"])
        metrics = calculate_metrics(y[test_idx], scores)
        predictions = metrics.pop("predictions")

        test_rows = pd.DataFrame({
            "Compound": np.asarray(names, dtype=object)[test_idx],
            "SMILES": np.asarray(smiles_list, dtype=object)[test_idx],
            "Actual label": y[test_idx],
            "Predicted label": predictions,
            "Raw model score": scores,
            "Correct?": predictions == y[test_idx],
        })

        metrics.update({
            "test_size": len(test_idx),
            "train_size": len(train_idx),
            "total_size": len(y),
            "skipped": skipped,
            "class_0_count": int(np.sum(y[test_idx] == 0)),
            "class_1_count": int(np.sum(y[test_idx] == 1)),
            "test_rows": test_rows,
        })

        return metrics, None

    except Exception as exc:
        return None, f"Random-split evaluation failed: {exc}"


# --------------------------------------------------
# SCAFFOLD-SPLIT MODEL EVALUATION
# Read the saved held-out results from the scaffold script.
# --------------------------------------------------

@st.cache_data(show_spinner="Loading scaffold-split evaluation...")
def evaluate_scaffold_model():
    if not SCAFFOLD_MODEL_PATH.exists():
        return None, (
            "Scaffold model not found. Run "
            "src/train_scaffold_model.py first."
        )

    try:
        with np.load(SCAFFOLD_MODEL_PATH, allow_pickle=False) as saved:
            required = {
                "weights", "bias", "train_indices", "test_indices",
                "y_test", "test_scores", "test_predictions",
                "accuracy", "balanced_accuracy", "roc_auc",
                "tn", "fp", "fn", "tp",
            }

            missing = required.difference(saved.files)

            if missing:
                return None, (
                    "The scaffold model file is missing fields: "
                    + ", ".join(sorted(missing))
                )

            weights = saved["weights"].astype(float)
            y_test = saved["y_test"].astype(int)
            scores = saved["test_scores"].astype(float)
            predictions = saved["test_predictions"].astype(int)
            train_idx = saved["train_indices"].astype(int)
            test_idx = saved["test_indices"].astype(int)

            if not (
                len(y_test) == len(scores) == len(predictions) == len(test_idx)
            ):
                return None, "Saved scaffold test arrays have inconsistent sizes."

            # Verify the saved predictions and metrics against saved test scores.
            calculated = calculate_metrics(y_test, scores)

            for key in ("accuracy", "balanced_accuracy", "roc_auc"):
                stored = float(np.asarray(saved[key]).reshape(-1)[0])
                current = calculated[key]

                if not np.isclose(stored, current, atol=1e-8, equal_nan=True):
                    return None, (
                        f"Saved scaffold metric '{key}' does not match "
                        "the saved test scores."
                    )

            for key in ("tn", "fp", "fn", "tp"):
                stored = int(np.asarray(saved[key]).reshape(-1)[0])

                if stored != calculated[key]:
                    return None, (
                        f"Saved scaffold confusion count '{key}' "
                        "does not match the saved test scores."
                    )

            if not np.array_equal(predictions, calculated["predictions"]):
                return None, (
                    "Saved scaffold predictions do not match the saved scores."
                )

            if len(np.unique(y_test)) != 2:
                return None, "The scaffold test set must contain both classes."

            train_scaffolds = set()
            test_scaffolds = set()

            if "scaffold_test_groups" in saved.files:
                test_scaffolds = set(
                    saved["scaffold_test_groups"].astype(str).tolist()
                )

            # Reconstruct train scaffold keys from the dataset and saved indices.
            # This checks the scaffold split against the current prepared dataset.
            df = pd.read_csv(DATA_PATH)

            if not {"smiles", "p_np"}.issubset(df.columns):
                return None, "The prepared dataset is missing required columns."

            valid_smiles = []
            valid_labels = []
            valid_scaffolds = []

            # Use RDKit's Murcko scaffold calculation as in the training script.
            from rdkit.Chem.Scaffolds import MurckoScaffold

            for _, row in df.iterrows():
                if pd.isna(row["smiles"]) or pd.isna(row["p_np"]):
                    continue

                mol = Chem.MolFromSmiles(str(row["smiles"]))

                if mol is None:
                    continue

                try:
                    label = int(row["p_np"])
                except (ValueError, TypeError):
                    continue

                if label not in (0, 1):
                    continue

                scaffold = MurckoScaffold.GetScaffoldForMol(mol)

                if scaffold is not None and scaffold.GetNumAtoms() > 0:
                    key = "SCAFFOLD:" + Chem.MolToSmiles(scaffold)
                else:
                    key = "ACYCLIC:" + Chem.MolToSmiles(mol)

                valid_smiles.append(str(row["smiles"]))
                valid_labels.append(label)
                valid_scaffolds.append(key)

            valid_labels = np.asarray(valid_labels, dtype=int)
            valid_scaffolds = np.asarray(valid_scaffolds, dtype=object)

            if (
                len(valid_labels) != len(np.unique(np.concatenate([train_idx, test_idx])))
                or (len(train_idx) and train_idx.max() >= len(valid_labels))
                or (len(test_idx) and test_idx.max() >= len(valid_labels))
            ):
                return None, (
                    "The prepared dataset no longer matches the saved scaffold "
                    "indices. Re-run src/train_scaffold_model.py."
                )

            train_scaffolds = set(valid_scaffolds[train_idx].tolist())
            actual_test_scaffolds = set(valid_scaffolds[test_idx].tolist())
            overlap = train_scaffolds.intersection(actual_test_scaffolds)

            if overlap:
                return None, (
                    "Scaffold overlap was detected. Re-run scaffold validation "
                    "before relying on these results."
                )

            if test_scaffolds and test_scaffolds != actual_test_scaffolds:
                return None, (
                    "Saved scaffold groups do not match the current dataset. "
                    "Re-run src/train_scaffold_model.py."
                )

            test_names = []
            test_smiles = []

            for idx in test_idx:
                row = df.iloc[int(idx)]
                test_names.append(str(row.get("name", "Unnamed compound")))
                test_smiles.append(str(row["smiles"]))

            test_rows = pd.DataFrame({
                "Compound": test_names,
                "SMILES": test_smiles,
                "Actual label": y_test,
                "Predicted label": predictions,
                "Raw model score": scores,
                "Correct?": predictions == y_test,
            })

            metrics = {
                **calculated,
                "test_size": len(test_idx),
                "train_size": len(train_idx),
                "total_size": len(valid_labels),
                "skipped": len(df) - len(valid_labels),
                "class_0_count": int(np.sum(y_test == 0)),
                "class_1_count": int(np.sum(y_test == 1)),
                "train_scaffold_count": len(train_scaffolds),
                "test_scaffold_count": len(actual_test_scaffolds),
                "scaffold_overlap": len(overlap),
                "test_rows": test_rows,
            }

            return metrics, None

    except Exception as exc:
        return None, f"Scaffold evaluation failed: {exc}"


# --------------------------------------------------
# SIDEBAR AND NAVIGATION
# --------------------------------------------------

with st.sidebar:
    st.markdown("## 🧪 Drugonaut")
    st.caption("Molecular research explorer")
    st.divider()

    page = st.radio(
        "Navigate",
        ["Molecule Explorer", "Model Evaluation"],
        index=0,
    )

    st.divider()
    st.markdown("### Model information")
    st.caption("Dataset: BBBP")
    st.caption("Fingerprint: Morgan, radius 2")
    st.caption("Fingerprint size: 1024 bits")

    st.info(
        "Educational research prototype. Predictions are not clinical evidence."
    )


model, model_error = load_model()
df = load_dataset()


# --------------------------------------------------
# MODEL EVALUATION PAGE
# --------------------------------------------------

if page == "Model Evaluation":
    st.markdown(
        '<div class="eyebrow">MODEL VALIDATION • HELD-OUT DATA</div>',
        unsafe_allow_html=True,
    )
    st.title("Evaluate the model.")
    st.write(
        "Compare random-split validation with scaffold-split validation "
        "to explore how performance changes when chemical scaffolds are held out."
    )

    st.divider()

    with st.expander("What do these metrics measure?"):
        st.markdown(
            """
            - **Accuracy:** fraction of test molecules classified correctly.
            - **Balanced accuracy:** average recall across the two classes.
            - **ROC-AUC:** how well scores rank positive examples above negative ones.
            - **Confusion matrix:** counts correct and incorrect predictions.
            - **Random split:** molecules are separated by label, but related
              chemical scaffolds may appear in both sets.
            - **Scaffold split:** each Murcko scaffold group is kept on one side
              of the train/test boundary.
            """
        )

    with st.spinner("Loading both evaluations..."):
        random_eval, random_error = evaluate_saved_model()
        scaffold_eval, scaffold_error = evaluate_scaffold_model()

    st.header("01 — Validation comparison")

    if random_eval is not None and scaffold_eval is not None:
        comparison = pd.DataFrame({
            "Metric": ["Accuracy", "Balanced accuracy", "ROC-AUC"],
            "Random split": [
                random_eval["accuracy"],
                random_eval["balanced_accuracy"],
                random_eval["roc_auc"],
            ],
            "Scaffold split": [
                scaffold_eval["accuracy"],
                scaffold_eval["balanced_accuracy"],
                scaffold_eval["roc_auc"],
            ],
        })

        comparison["Difference (scaffold − random)"] = (
            comparison["Scaffold split"] - comparison["Random split"]
        )

        st.dataframe(
            comparison.style.format({
                "Random split": "{:.3f}",
                "Scaffold split": "{:.3f}",
                "Difference (scaffold − random)": "{:+.3f}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        r1, r2, r3 = st.columns(3)
        r1.metric(
            "Random split accuracy",
            f"{random_eval['accuracy']:.1%}",
        )
        r2.metric(
            "Scaffold split accuracy",
            f"{scaffold_eval['accuracy']:.1%}",
            delta=f"{scaffold_eval['accuracy'] - random_eval['accuracy']:+.1%}",
            delta_color="inverse",
        )
        r3.metric(
            "Scaffold overlap",
            scaffold_eval["scaffold_overlap"],
        )

        st.caption(
            "A negative accuracy difference means performance was lower on "
            "the scaffold split. That can indicate that predicting molecules "
            "from unfamiliar scaffold groups is more challenging."
        )

    else:
        if random_eval is None:
            st.warning(f"Random-split evaluation unavailable: {random_error}")
        if scaffold_eval is None:
            st.warning(f"Scaffold-split evaluation unavailable: {scaffold_error}")

    if scaffold_eval is not None:
        st.divider()
        st.header("02 — Scaffold split details")

        st.caption(
            "These metrics are loaded from the saved scaffold test predictions "
            "and verified against the stored scores."
        )

        s1, s2, s3 = st.columns(3)
        s1.metric("Accuracy", f"{scaffold_eval['accuracy']:.1%}")
        s2.metric(
            "Balanced accuracy",
            f"{scaffold_eval['balanced_accuracy']:.1%}",
        )
        auc = scaffold_eval["roc_auc"]
        s3.metric("ROC-AUC", f"{auc:.3f}" if np.isfinite(auc) else "N/A")

        a, b, c, d = st.columns(4)
        a.metric("Training molecules", scaffold_eval["train_size"])
        b.metric("Test molecules", scaffold_eval["test_size"])
        c.metric("Train scaffold groups", scaffold_eval["train_scaffold_count"])
        d.metric("Test scaffold groups", scaffold_eval["test_scaffold_count"])

        st.success(
            f"Scaffold overlap verified: {scaffold_eval['scaffold_overlap']}"
        )

        st.subheader("Scaffold confusion matrix")
        scaffold_matrix = pd.DataFrame(
            [
                [scaffold_eval["tn"], scaffold_eval["fp"]],
                [scaffold_eval["fn"], scaffold_eval["tp"]],
            ],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(scaffold_matrix, use_container_width=True)

        st.subheader("Scaffold class-level performance")
        st.dataframe(
            pd.DataFrame({
                "Class": ["0", "1"],
                "Recall": [
                    scaffold_eval["recall_0"],
                    scaffold_eval["recall_1"],
                ],
                "Correct predictions": [
                    scaffold_eval["tn"],
                    scaffold_eval["tp"],
                ],
                "Incorrect predictions": [
                    scaffold_eval["fp"],
                    scaffold_eval["fn"],
                ],
            }).style.format({"Recall": "{:.1%}"}),
            use_container_width=True,
            hide_index=True,
        )

        scaffold_rows = scaffold_eval["test_rows"]
        with st.expander("Inspect incorrect scaffold predictions"):
            incorrect = scaffold_rows[~scaffold_rows["Correct?"]]
            if incorrect.empty:
                st.success("No incorrect predictions in this test set.")
            else:
                st.dataframe(
                    incorrect.round({"Raw model score": 4}),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Inspect all scaffold test predictions"):
            st.dataframe(
                scaffold_rows.round({"Raw model score": 4}),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download scaffold test predictions (CSV)",
            data=scaffold_rows.to_csv(index=False).encode("utf-8"),
            file_name="drugonaut_scaffold_test_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if random_eval is not None:
        st.divider()
        st.header("03 — Random split details")

        st.caption(
            "The random model is evaluated on its held-out random test split."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{random_eval['accuracy']:.1%}")
        m2.metric(
            "Balanced accuracy",
            f"{random_eval['balanced_accuracy']:.1%}",
        )
        auc = random_eval["roc_auc"]
        m3.metric("ROC-AUC", f"{auc:.3f}" if np.isfinite(auc) else "N/A")

        st.subheader("Random-split confusion matrix")
        random_matrix = pd.DataFrame(
            [
                [random_eval["tn"], random_eval["fp"]],
                [random_eval["fn"], random_eval["tp"]],
            ],
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"],
        )
        st.dataframe(random_matrix, use_container_width=True)

        random_rows = random_eval["test_rows"]

        with st.expander("Inspect incorrect random-split predictions"):
            incorrect = random_rows[~random_rows["Correct?"]]
            if incorrect.empty:
                st.success("No incorrect predictions in this test set.")
            else:
                st.dataframe(
                    incorrect.round({"Raw model score": 4}),
                    use_container_width=True,
                    hide_index=True,
                )

        with st.expander("Inspect all random test predictions"):
            st.dataframe(
                random_rows.round({"Raw model score": 4}),
                use_container_width=True,
                hide_index=True,
            )

        st.download_button(
            "Download random test predictions (CSV)",
            data=random_rows.to_csv(index=False).encode("utf-8"),
            file_name="drugonaut_random_test_predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="research-note">
        <strong>Research interpretation:</strong> Scaffold-split validation
        tests generalization across held-out chemical scaffold groups and is
        often more challenging than a random split. Differences between these
        results are informative, but do not by themselves prove that scaffold
        separation is the only cause: the two test sets contain different
        molecules. A high score does not establish experimental BBB permeability,
        safety, or clinical usefulness. Raw sigmoid scores are not necessarily
        calibrated probabilities.
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()
    st.markdown(
        '<div class="footer">DRUGONAUT · Model Evaluation</div>',
        unsafe_allow_html=True,
    )
    st.stop()


# --------------------------------------------------
# MOLECULE EXPLORER PAGE
# --------------------------------------------------

st.markdown(
    '<div class="eyebrow">MOLECULAR RESEARCH • MADE EXPLORABLE</div>',
    unsafe_allow_html=True,
)
st.title("Understand molecules. Explore possibilities.")
st.markdown(
    """
    Investigate molecular properties, explore a model's BBBP classification,
    and find structurally similar compounds from the research dataset.
    """
)
st.divider()

DEFAULT_SMILES = "Cn1c(=O)c2c(ncn2C)n(C)c1=O"

with st.form("molecule_form"):
    smiles = st.text_area(
        "Molecule SMILES",
        value=DEFAULT_SMILES,
        height=90,
        help="Enter a molecule as a SMILES string. Caffeine is the example.",
        placeholder="Enter a valid SMILES string...",
    )
    analyze = st.form_submit_button(
        "Analyze molecule →",
        use_container_width=True,
    )


if analyze:
    if not smiles.strip():
        st.warning("Please enter a SMILES string.")
        st.stop()

    with st.spinner("Analyzing molecular structure..."):
        mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        st.error(
            "This SMILES string could not be parsed. "
            "Please check the input and try again."
        )
        st.stop()

    canonical = Chem.MolToSmiles(mol)
    st.success("Molecular structure parsed successfully.")

    st.header("01 — Molecular structure")
    left, right = st.columns([1, 1.5])

    with left:
        try:
            st.image(
                Draw.MolToImage(mol, size=(420, 320)),
                caption="2D molecular structure",
                use_container_width=True,
            )
        except Exception as exc:
            st.warning(f"Structure drawing was unavailable: {exc}")

    with right:
        st.markdown("**Input SMILES**")
        st.code(smiles.strip(), language=None)
        st.markdown("**Canonical SMILES**")
        st.code(canonical, language=None)
        st.caption(
            "Canonical SMILES is a standardized representation generated by RDKit."
        )

    st.divider()
    st.header("02 — Molecular properties")

    molecular_weight = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)
    hba = Lipinski.NumHAcceptors(mol)
    rotatable_bonds = Lipinski.NumRotatableBonds(mol)
    ring_count = rdMolDescriptors.CalcNumRings(mol)

    col1, col2, col3 = st.columns(3)
    col1.metric("Molecular weight", f"{molecular_weight:.2f} g/mol")
    col2.metric("LogP", f"{logp:.2f}")
    col3.metric("TPSA", f"{tpsa:.2f} Å²")

    col4, col5, col6, col7 = st.columns(4)
    col4.metric("H-bond donors", hbd)
    col5.metric("H-bond acceptors", hba)
    col6.metric("Rotatable bonds", rotatable_bonds)
    col7.metric("Ring count", ring_count)

    with st.expander("What do these properties mean?"):
        st.markdown(
            """
            - **Molecular weight:** approximate molecular mass.
            - **LogP:** estimates partitioning between an oily phase and water.
            - **TPSA:** topological polar surface area.
            - **H-bond donors:** groups that can donate a hydrogen bond.
            - **H-bond acceptors:** atoms or groups that can accept a hydrogen bond.
            - **Rotatable bonds:** a rough measure of structural flexibility.
            - **Ring count:** the number of rings identified by RDKit.
            """
        )

    st.divider()
    st.header("03 — BBBP model classification")
    result = predict_bbb(mol, model)

    if result is None:
        st.error(model_error or "The model could not be loaded.")
        st.caption(
            "Molecular properties and similar-compound search remain available."
        )
    else:
        if result["prediction"] == 1:
            st.subheader("Model classification: Positive")
            st.write(
                "The model assigns this molecule to class 1 in the BBBP "
                "dataset's label scheme."
            )
        else:
            st.subheader("Model classification: Negative")
            st.write(
                "The model assigns this molecule to class 0 in the BBBP "
                "dataset's label scheme."
            )

        st.metric("Raw model score", f"{result['score']:.3f}")
        st.progress(float(result["score"]))
        st.caption(
            "This is a sigmoid output, not a calibrated real-world probability "
            "or experimental confirmation of blood-brain barrier passage."
        )

    st.divider()
    st.header("04 — Structurally similar compounds")

    with st.spinner("Searching the BBBP dataset..."):
        similar_df = find_similar_molecules(mol, df, top_n=5)

    if similar_df.empty:
        st.info(
            "No similar compounds could be retrieved. "
            "Check that BBBP_prepared.csv exists and contains valid SMILES."
        )
    else:
        st.dataframe(similar_df, use_container_width=True, hide_index=True)
        st.caption(
            "Similarity uses Tanimoto similarity between Morgan fingerprints. "
            "Structural similarity does not establish identical biological activity."
        )

    st.divider()
    st.header("05 — Download research report")

    report = {
        "Analysis date": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "Input SMILES": smiles.strip(),
        "Canonical SMILES": canonical,
        "Molecular weight (g/mol)": round(molecular_weight, 3),
        "LogP": round(logp, 3),
        "TPSA (A^2)": round(tpsa, 3),
        "H-bond donors": hbd,
        "H-bond acceptors": hba,
        "Rotatable bonds": rotatable_bonds,
        "Ring count": ring_count,
    }

    if result is not None:
        report["BBBP model classification"] = (
            "Positive" if result["prediction"] == 1 else "Negative"
        )
        report["Raw model score (not calibrated)"] = round(result["score"], 5)

    report_df = pd.DataFrame(list(report.items()), columns=["Field", "Value"])
    download_col1, download_col2 = st.columns(2)

    with download_col1:
        st.download_button(
            "Download analysis report (CSV)",
            data=report_df.to_csv(index=False).encode("utf-8"),
            file_name="drugonaut_analysis.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with download_col2:
        if not similar_df.empty:
            st.download_button(
                "Download similar compounds (CSV)",
                data=similar_df.to_csv(index=False).encode("utf-8"),
                file_name="drugonaut_similar_compounds.csv",
                mime="text/csv",
                use_container_width=True,
            )
        else:
            st.button(
                "No similar compounds to download",
                disabled=True,
                use_container_width=True,
            )

    st.markdown(
        """
        <div class="research-note">
        <strong>Research note:</strong> These computational estimates do not
        establish safety, efficacy, clinical suitability, or experimentally
        verified blood-brain barrier permeability.
        </div>
        """,
        unsafe_allow_html=True,
    )

else:
    st.markdown("### Start with a molecule")
    st.write(
        "A caffeine SMILES example is already filled in. "
        "Click **Analyze molecule →** to explore the dashboard."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 01. Structure")
        st.write("Visualize the molecular structure and its SMILES.")

    with col2:
        st.markdown("#### 02. Properties")
        st.write("Explore molecular weight, LogP, TPSA, and more.")

    with col3:
        st.markdown("#### 03. Similarity")
        st.write("Find related structures in the BBBP dataset.")

    if model is None:
        st.warning(model_error or "The prediction model could not be loaded.")

    if df.empty:
        st.warning(
            "The prepared BBBP dataset could not be loaded. "
            "Check data/BBBP_prepared.csv."
        )


st.divider()
st.markdown(
    """
    <div class="footer">
        DRUGONAUT · Educational molecular research prototype<br>
        Not intended for clinical or medical decision-making.
    </div>
    """,
    unsafe_allow_html=True,
)
