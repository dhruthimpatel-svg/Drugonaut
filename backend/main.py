from pathlib import Path
from io import BytesIO
import csv
import os

import numpy as np
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel, Field
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator


# -----------------------------------------------------
# PROJECT PATHS
# -----------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "drugonaut_fingerprint.npz"
DATA_PATH = PROJECT_ROOT / "data" / "BBBP_prepared.csv"

FINGERPRINT_BITS = 1024
RADIUS = 2
SIMILARITY_LIMIT = 10


# -----------------------------------------------------
# RDKit FINGERPRINT GENERATOR
# -----------------------------------------------------

GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
    radius=RADIUS,
    fpSize=FINGERPRINT_BITS,
)


# -----------------------------------------------------
# FASTAPI APP
# -----------------------------------------------------

app = FastAPI(
    title="Drugonaut API",
    description="API for molecular exploration and research predictions.",
    version="0.3.0",
)


# -----------------------------------------------------
# CORS
# -----------------------------------------------------

# Keep local development URLs working.
allowed_origins = [
    f"http://localhost:{port}"
    for port in range(5173, 5177)
] + [
    f"http://127.0.0.1:{port}"
    for port in range(5173, 5177)
]

# Add the deployed frontend URL from the environment.
frontend_url = os.getenv("DRUGONAUT_FRONTEND_URL", "").strip().rstrip("/")

if frontend_url:
    allowed_origins.append(frontend_url)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


# -----------------------------------------------------
# REQUEST MODEL
# -----------------------------------------------------

class AnalyzeRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=2000)


# -----------------------------------------------------
# MODEL UTILITIES
# -----------------------------------------------------

def sigmoid(value):
    value = np.clip(value, -500, 500)
    return 1.0 / (1.0 + np.exp(-value))


def load_model():
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")

    with np.load(MODEL_PATH) as saved:
        weights = saved["weights"].astype(float)
        bias = float(np.asarray(saved["bias"]).reshape(-1)[0])

    if weights.size != FINGERPRINT_BITS:
        raise RuntimeError(
            f"Expected {FINGERPRINT_BITS} model weights, "
            f"but found {weights.size}."
        )

    return {"weights": weights, "bias": bias}


def molecule_fingerprint(mol):
    return GENERATOR.GetFingerprint(mol)


# -----------------------------------------------------
# SIMILAR MOLECULE DATASET
# -----------------------------------------------------

def load_compounds():
    """Read the BBBP CSV and prepare valid molecular structures."""
    if not DATA_PATH.exists():
        raise RuntimeError(f"Dataset file not found: {DATA_PATH}")

    compounds = []

    with DATA_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:
        reader = csv.DictReader(file)

        required = {"name", "p_np", "smiles"}
        available = set(reader.fieldnames or [])

        if not required.issubset(available):
            raise RuntimeError(
                "Dataset must contain name, p_np, and smiles columns."
            )

        for row in reader:
            smiles = (row.get("smiles") or "").strip()
            mol = Chem.MolFromSmiles(smiles)

            if mol is None:
                continue

            try:
                fp = molecule_fingerprint(mol)
            except Exception:
                continue

            label_raw = (row.get("p_np") or "").strip()

            try:
                label = int(float(label_raw))
            except (ValueError, TypeError):
                label = None

            compounds.append({
                "name": (
                    (row.get("name") or "").strip()
                    or "Unnamed compound"
                ),
                "smiles": Chem.MolToSmiles(mol),
                "dataset_label": label,
                "fingerprint": fp,
            })

    return compounds


# -----------------------------------------------------
# CACHE DATASET IN MEMORY
# -----------------------------------------------------

_compound_cache = None


def get_compounds():
    global _compound_cache

    if _compound_cache is None:
        _compound_cache = load_compounds()

    return _compound_cache


# -----------------------------------------------------
# ROOT
# -----------------------------------------------------

@app.get("/")
def root():
    return {
        "name": "Drugonaut API",
        "status": "running",
        "message": "Molecular research backend is online.",
    }


# -----------------------------------------------------
# HEALTH CHECK
# -----------------------------------------------------

@app.get("/api/health")
def health():
    return {"status": "healthy"}


# -----------------------------------------------------
# MODEL EVALUATION
# -----------------------------------------------------

@app.get("/api/model-evaluation")
def model_evaluation():
    """
    Return previously recorded evaluation results.

    These results are historical project measurements.
    This endpoint does not rerun model training or evaluation.
    """

    random_split = {
        "name": "Random split",
        "accuracy": 0.875,
        "balanced_accuracy": 0.839,
        "roc_auc": 0.928,
        "description": (
            "Randomly partitioned evaluation. Similar molecular "
            "structures may occur in both training and test sets."
        ),
    }

    scaffold_split = {
        "name": "Scaffold split",
        "accuracy": 0.790,
        "balanced_accuracy": 0.714,
        "roc_auc": 0.802,
        "confusion_matrix": {
            "labels": [0, 1],
            "matrix": [
                [76, 59],
                [55, 353],
            ],
            "layout": (
                "Rows are actual classes [0, 1]; "
                "columns are predicted classes [0, 1]."
            ),
        },
        "description": (
            "Scaffold-based evaluation designed to separate "
            "molecular scaffold groups between training and test data."
        ),
    }

    return {
        "model": "Drugonaut Morgan fingerprint logistic regression",
        "fingerprint": {
            "type": "Morgan",
            "radius": RADIUS,
            "bits": FINGERPRINT_BITS,
        },
        "dataset": {
            "name": "BBBP",
            "valid_molecules": 2039,
            "scaffold_groups": 1196,
            "scaffold_train_size": 1496,
            "scaffold_test_size": 543,
            "scaffold_overlap": 0,
        },
        "evaluation": {
            "status": "previously_recorded",
            "recomputed_on_request": False,
            "splits": [
                random_split,
                scaffold_split,
            ],
        },
        "metric_definitions": {
            "accuracy": "Fraction of predictions that are correct.",
            "balanced_accuracy": (
                "Average recall across the two classes."
            ),
            "roc_auc": (
                "Area under the ROC curve; measures ranking "
                "performance across classification thresholds."
            ),
        },
        "interpretation": (
            "The recorded scaffold-split metrics are lower than "
            "the random-split metrics. This is consistent with "
            "scaffold evaluation being a more demanding test of "
            "generalization to different molecular structures. "
            "These results are historical and have not been "
            "recalculated by this endpoint."
        ),
        "disclaimer": (
            "Evaluation metrics describe model performance on "
            "particular dataset splits. They do not establish "
            "clinical validity, drug safety, efficacy, or BBB "
            "permeability for an individual compound."
        ),
    }


# -----------------------------------------------------
# MOLECULE IMAGE
# -----------------------------------------------------

@app.get("/api/molecule-image")
def molecule_image(smiles: str):
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid SMILES.",
        )

    try:
        from rdkit.Chem import Draw

        image = Draw.MolToImage(mol, size=(500, 400))
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        return Response(
            content=buffer.getvalue(),
            media_type="image/png",
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Could not render molecule: {exc}",
        ) from exc


# -----------------------------------------------------
# ANALYZE MOLECULE
# -----------------------------------------------------

@app.post("/api/analyze")
def analyze_molecule(request: AnalyzeRequest):
    smiles = request.smiles.strip()
    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid SMILES. Please check the molecular string.",
        )

    try:
        model = load_model()
        fingerprint = molecule_fingerprint(mol)

        values = np.zeros(FINGERPRINT_BITS, dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fingerprint, values)

        score = float(
            sigmoid(
                np.dot(values.astype(float), model["weights"])
                + model["bias"]
            )
        )

        descriptors = {
            "molecular_weight": round(Descriptors.MolWt(mol), 3),
            "logP": round(Descriptors.MolLogP(mol), 3),
            "tpsa": round(Descriptors.TPSA(mol), 3),
            "hbd": int(Lipinski.NumHDonors(mol)),
            "hba": int(Lipinski.NumHAcceptors(mol)),
            "rotatable_bonds": int(Lipinski.NumRotatableBonds(mol)),
            "ring_count": int(rdMolDescriptors.CalcNumRings(mol)),
        }

        return {
            "smiles": Chem.MolToSmiles(mol),
            "prediction": {
                "label": (
                    "Model predicts BBB penetration"
                    if score >= 0.5
                    else "Model predicts no BBB penetration"
                ),
                "class": int(score >= 0.5),
                "score": round(score, 5),
                "threshold": 0.5,
                "score_note": (
                    "Raw model score, not a calibrated probability."
                ),
            },
            "descriptors": descriptors,
            "disclaimer": (
                "Research prototype only. This prediction is not "
                "proof of blood-brain barrier passage, safety, or efficacy."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Molecule analysis failed: {exc}",
        ) from exc


# -----------------------------------------------------
# SIMILAR MOLECULE SEARCH
# -----------------------------------------------------

@app.get("/api/similar-molecules")
def similar_molecules(
    smiles: str = Query(min_length=1, max_length=2000),
    limit: int = Query(default=SIMILARITY_LIMIT, ge=1, le=25),
):
    mol = Chem.MolFromSmiles(smiles.strip())

    if mol is None:
        raise HTTPException(
            status_code=422,
            detail="Invalid SMILES. Please check the molecular string.",
        )

    try:
        query_smiles = Chem.MolToSmiles(mol)
        query_fp = molecule_fingerprint(mol)
        compounds = get_compounds()

        matches = []

        for compound in compounds:
            if compound["smiles"] == query_smiles:
                continue

            similarity = DataStructs.TanimotoSimilarity(
                query_fp,
                compound["fingerprint"],
            )

            matches.append({
                "name": compound["name"],
                "smiles": compound["smiles"],
                "dataset_label": compound["dataset_label"],
                "similarity": round(float(similarity), 4),
            })

        matches.sort(
            key=lambda item: item["similarity"],
            reverse=True,
        )

        return {
            "query_smiles": query_smiles,
            "metric": "Morgan fingerprint Tanimoto similarity",
            "count": min(limit, len(matches)),
            "results": matches[:limit],
            "note": (
                "Similarity measures structural resemblance only. "
                "It does not establish equivalent biological activity, "
                "safety, efficacy, or BBB permeability."
            ),
        }

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Similar molecule search failed: {exc}",
        ) from exc
