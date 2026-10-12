# Drugonaut

Drugonaut is an educational molecular research prototype for exploring blood-brain barrier (BBB) permeability using molecular fingerprints and machine learning.

## Features

- Visualize molecular structures using RDKit.
- Explore molecular descriptors: molecular weight, LogP, TPSA, hydrogen-bond donors and acceptors, rotatable bonds, and ring count.
- Generate a model score for BBB permeability research.
- Find similar molecules using Morgan fingerprints and Tanimoto similarity.
- Review recorded random-split and scaffold-split evaluation results.
- Use the React frontend, FastAPI backend, or Streamlit application.

## Technology Stack

- Python
- React and Vite
- FastAPI
- Streamlit
- RDKit
- NumPy and pandas

## Run Locally

### 1. Start the API

From the project root, run:

    .\.venv\Scripts\python.exe -m uvicorn backend.main:app --reload --port 8000

API health check: `http://127.0.0.1:8000/api/health`

API documentation: `http://127.0.0.1:8000/docs`

### 2. Start the React frontend

Open a second terminal:

    cd frontend
    npm run dev

Open `http://localhost:5173`.

### 3. Start the Streamlit application

From the project root, run:

    .\.venv\Scripts\python.exe -m streamlit run src\app.py

## Dataset Attribution

Drugonaut uses the Blood-Brain Barrier Penetration (BBBP) dataset from MoleculeNet, distributed through DeepChem.

- Dataset source: https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/BBBP.csv
- Dataset loader and documentation: https://github.com/deepchem/deepchem/blob/master/deepchem/molnet/load_function/bbbp_datasets.py
- Dataset documentation and licence information: https://huggingface.co/datasets/zpn/bbbp
- Research paper: Z. Wu et al., *MoleculeNet: A Benchmark for Molecular Machine Learning* (2018). https://arxiv.org/abs/1703.00564

The dataset documentation lists the dataset as MIT-licensed. Retain appropriate attribution and verify the applicable licence terms for the exact dataset files and any modifications redistributed with this project.

## Scientific Limitations

Drugonaut is an educational research prototype, not a validated clinical or drug-development tool.

- A model score is not necessarily a calibrated probability.
- A prediction does not prove that a molecule crosses the blood-brain barrier.
- Predictions do not establish a molecule's safety, efficacy, or toxicity.
- Evaluation results depend on the dataset, split strategy, and model.
- Experimental validation is necessary before drawing scientific or medical conclusions.

## Disclaimer

This project is intended for learning and research exploration only. It must not be used as the sole basis for medical, clinical, or drug-development decisions.