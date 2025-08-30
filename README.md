# Roundabout - TikTok TechJam 2025 Project

## Overview

Roundabout is an end-to-end Machine Learning/NLP pipeline for evaluating the quality and relevancy of Google location reviews.  
It covers data preprocessing, feature extraction, policy-based classification, and provides a simple UI for predictions.

**Key principle:**  
The system ensures robust and reproducible results by avoiding incremental training: each training run uses the full labeled dataset (concatenate old and new data if needed).

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/xyanjun02/roundabout.git
cd roundabout
```

### 2. Set Up the Environment

```bash
conda env create -f environment.yml
conda activate roundabout
```

### 3. Install Additional Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

- Obtain an OpenAI API key: [Get an API key here](https://platform.openai.com/settings/organization/api-keys)
- Create a `.env` file in the project root:

  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```

### 5. (Optional) Set Up Local LLM Fallback

- Install [Ollama](https://ollama.com/download) and pull the Gemma model:
  ```bash
  ollama pull gemma:2b
  ollama serve
  ```

---

## Folder Structure

```
roundabout/
├─ README.md
├─ environment.yml
├─ requirements.txt
├─ data/
│  ├─ raw/            # Original datasets
│  ├─ processed/      # Cleaned and preprocessed data
│  └─ labeled/        # Labeled data with policy/relevance labels
├─ scripts/           # Standalone scripts for pipeline stages
├─ src/
│  ├─ preprocess/     # Data cleaning
│  ├─ features/       # Feature extraction
│  ├─ policies/       # Classifiers & policy enforcement
│  ├─ llm/            # LLM-based labeling
│  └─ utils/          # Utility functions
├─ ui/                # Streamlit/FastAPI UI
├─ outputs/
│  ├─ models/         # Saved trained models
│  └─ predictions/    # Output predictions
└─ experiments/       # Jupyter notebooks or experiments
```

---

## Pipeline Flow

Follow these steps to robustly reproduce results:

### 1. Prepare Data

- Place raw reviews in `data/raw/reviews.csv`.
- Run preprocessing:
  ```bash
  python scripts/01_prepare_reviews.py
  ```
- Output: `data/processed/reviews_processed.csv`

### 2. Generate Labels

- Generate policy violation labels with LLM:
  ```bash
  python scripts/02_generate_labels.py --overwrite
  ```
- Output: `data/labeled/reviews_with_labels.csv`
- **Tip:** Always use `--overwrite` when regenerating labels to avoid incremental label inconsistencies.

### 3. Train Classifiers

- Train BERT-based classifiers for each policy:
  ```bash
  python scripts/03_train_classifiers.py
  ```
- Output: `outputs/models/`
- **Important:** Incremental training is **not allowed**. Each run uses the entire labeled dataset (concatenate old and new data if needed).

### 4. Evaluate Models

- Evaluate the trained classifiers:
  ```bash
  python scripts/04_eval.py
  ```
- Output: `outputs/predictions/evaluation_results.json`
- **Note:** Robust evaluation handles small test sets and avoids stratification errors.

### 5. Batch Inference on New Data

- Run predictions on new/unlabeled reviews:
  ```bash
  python scripts/05_batch_inference.py --input data/processed/new_reviews.csv
  ```
- Output: `outputs/predictions/batch_predictions.csv`
- **Input must have a `text_clean` column (preprocessed).**

### 6. Run the UI (Optional)

- Launch the Streamlit app for interactive predictions:
  ```bash
  streamlit run ui/streamlit_app.py
  ```
- Upload a CSV with a `text_clean` column and download predictions.

---

## Robustness & Troubleshooting

- **Encoding:** Always ensure CSVs are UTF-8.
- **Incremental Training Not Allowed:** Combine all labeled data before retraining.
- **Fallback:** Ollama/Gemma is used if OpenAI API is unavailable.
- **Small Test Sets:** Use larger test sets for more precise evaluation.
- **Duplicates & Missing Columns:** Scripts handle them to prevent
- **Environment:** Always activate the `roundabout` conda environment.
- **Data Consistency:** Ensure `text_clean` and label columns exist.
- **Encoding:** Convert to UTF-8 to avoid Unicode issues:
  ```python
  df = pd.read_csv('file.csv', encoding='latin1')
  df.to_csv('file_utf8.csv', index=False, encoding='utf-8')
  ```
- **Module Import:** Ensure `__init__.py` exists and run scripts from project root.
  Set Python path if needed:
  ```powershell
  $env:PYTHONPATH = $PWD
  ```

---

## Extending the Pipeline

- **Adding Data:** Add raw reviews → preprocess → generate labels → retrain models.
- **New Policies:** Update labeling prompts and extend `src/policies`.
- **No Incremental Training:** Always train from a fresh combined dataset to ensure consistency.

---

## Contributors

- Steve Chia
- Xie Yanjun
- Venice Phua
- Tong Jia Jun
- Lee Sze Ying

---