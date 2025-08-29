# Roundabout - Hackathon Project

## Overview
This project is a Machine Learning / NLP pipeline to evaluate the quality and relevancy of Google location reviews.  
It includes preprocessing, feature extraction, model training, policy enforcement, and a simple UI for predictions.

---

## START UP

### 1. Clone the repository
```bash
git clone https://github.com/xyanjun02/roundabout.git
cd roundabout
````

### 2. Set up the environment

```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate roundabout
```

### 3. Install additional dependencies (if needed)

```bash
pip install -r requirements.txt
```

---

## Folder Structure

```
tiktok-techjam/
├─ README.md
├─ .gitignore
├─ data/
│  ├─ raw/            # Original unprocessed datasets
│  └─ processed/      # Cleaned and preprocessed data
├─ scripts/           # Standalone scripts (e.g., download data)
├─ src/
│  ├─ preprocess/     # Data cleaning and text preprocessing
│  ├─ features/       # Feature extraction modules
│  ├─ policies/       # Rule-based or policy enforcement modules
│  ├─ llm/            # ML/NLP models (training and inference)
│  └─ utils/          # Utility functions (logging, configs, etc.)
├─ ui/                # Frontend UI (Streamlit / FastAPI)
├─ outputs/
│  ├─ models/         # Saved trained models
│  └─ predictions/    # Output predictions / results
├─ experiments/       # Jupyter notebooks or experiments
└─ requirements.txt   # Optional pip requirements file
```

---

## Running the Project

---

## Notes

* Ensure the `roundabout` conda environment is **active** before running scripts.
* Outputs (models, predictions) are saved in `outputs/`.
* Raw and processed data are **not tracked** in git (`.gitignore`) for team convenience.

---

## Team / Contributors

* Xie Yanjun
* Tong Jia Jun
* Venice Phua
* Lee Sze Ying