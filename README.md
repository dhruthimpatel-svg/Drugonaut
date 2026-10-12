# Drugonaut

An exploratory molecular research platform for studying blood-brain barrier (BBB) permeability.

## Features
- Molecular structure visualization using RDKit.
- BBB permeability scoring using a trained fingerprint-based model.
- Molecular descriptors including molecular weight, LogP, TPSA, hydrogen-bond donors and acceptors, rotatable bonds, and ring count.
- Similar-molecule search using Morgan fingerprints and Tanimoto similarity.
- Recorded random-split and scaffold-split evaluation results.
- React frontend, FastAPI backend, and Streamlit application.

## Technology Stack
- Python
- React and Vite
- FastAPI
- RDKit
- NumPy and pandas
- Streamlit

## Run Locally

### Backend
From the project root:

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

### Frontend
Open a second terminal:

    cd frontend
    npm run dev

## Scientific Limitations
Drugonaut is an educational research prototype, not a validated clinical or drug-development tool. A model score is not necessarily a calibrated probability, and a prediction does not prove that a molecule crosses the blood-brain barrier. Predictions do not establish safety, efficacy, or toxicity.

## Dataset Attribution
Document the original BBBP dataset source, licence, and attribution requirements before redistributing dataset files.

## Disclaimer
This project is intended for learning and research exploration only.
