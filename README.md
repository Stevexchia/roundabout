# Roundabout - TikTok TechJam 2025 Project

## Overview

Roundabout is an end-to-end Machine Learning/NLP pipeline for evaluating the quality and relevancy of Google location reviews.  
It covers data preprocessing, feature extraction, model training, policy enforcement, and provides a simple UI for predictions.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/xyanjun02/roundabout.git
cd roundabout
```

### 2. Set Up the Environment

Create and activate the conda environment:

```bash
conda env create -f environment.yml
conda activate roundabout
```

### 3. Install Additional Dependencies

Some dependencies are managed via pip:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys

- Obtain an OpenAI API key: [Get an API key here](https://platform.openai.com/settings/organization/api-keys)
- Create a `.env` file in the project root with the following content:

  ```
  OPENAI_API_KEY=your_openai_api_key_here
  ```

---

## Folder Structure

```
roundabout/
├─ README.md
├─ .gitignore
├─ environment.yml
├─ requirements.txt
├─ data/
│  ├─ raw/            # Original unprocessed datasets
│  ├─ processed/      # Cleaned and preprocessed data
│  └─ labeled/        # Labeled data with policy/relevance labels
├─ scripts/           # Standalone scripts for each pipeline stage
├─ src/
│  ├─ preprocess/     # Data cleaning and text preprocessing
│  ├─ features/       # Feature extraction modules
│  ├─ policies/       # Policy enforcement and classifiers
│  ├─ llm/            # LLM-based labeling utilities
│  └─ utils/          # Utility functions
├─ ui/                # Streamlit/FastAPI UI
├─ outputs/
│  ├─ models/         # Saved trained models
│  └─ predictions/    # Output predictions/results
└─ experiments/       # Jupyter notebooks or experiments
```

---

## How to Reproduce Results

1. **Prepare Data**  
   Place your raw reviews CSV at `data/raw/reviews.csv`.

2. **Preprocess Reviews**  
   Run the preprocessing script to clean and normalize the data:
   ```bash
   python scripts/01_prepare_reviews.py
   ```
   Output: `data/processed/reviews_processed.csv`

3. **Generate Pseudo-Labels**  
   Use the LLM client to generate policy violation labels:
   ```bash
   python scripts/02_generate_labels.py
   ```
   Output: `data/labeled/reviews_with_labels.csv`

4. **Train Classifiers**  
   Train BERT-based classifiers for each policy:
   ```bash
   python scripts/03_train_classifiers.py
   ```
   Output: Trained models in `outputs/models/`

5. **Evaluate Models**  
   Evaluate the trained classifiers:
   ```bash
   python scripts/04_eval.py
   ```
   Output: Evaluation results in `outputs/predictions/evaluation_results.json`

6. **Run the UI (Optional)**  
   Launch the Streamlit app for interactive predictions:
   ```bash
   streamlit run ui/streamlit_app.py
   ```

---

## Notes & Troubleshooting

- Ensure the `roundabout` conda environment is **active** before running scripts.
- Data files and outputs are not tracked by git (see `.gitignore`).
- If you encounter "src not found" errors, check that all `__init__.py` files exist and are correct, then run in the root directory "$env:PYTHONPATH = $PWD"
- If OpenAI API fails or rate limits, pseudo-labeling will fall back to a local LLM via Ollama. See [Gemma + Ollama integration](https://ai.google.dev/gemma/docs/integrations/ollama) for setup.
- Error codes for OpenAI API are listed in the [OpenAI documentation](https://platform.openai.com/docs/guides/error-codes).

---

## Team / Contributors

- Steve Chia
- Xie Yanjun
- Venice Phua
- Tong Jia Jun
- Lee Sze Ying