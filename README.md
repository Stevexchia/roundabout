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
- Ensure that you have gotten an API Key through the API request for GPT-4
- While you can provide an api_key keyword argument, we recommend using python-dotenv to add OPENAI_API_KEY="My API Key" to your .env file so that your API key is not stored in source control. [[Get an API key here](https://platform.openai.com/settings/organization/api-keys)].

Error codes are as follows:
Status Code	Error Type
400	BadRequestError
401	AuthenticationError
403	PermissionDeniedError
404	NotFoundError
422	UnprocessableEntityError
429	RateLimitError
>=500	InternalServerError
N/A	APIConnectionError

---

## FAQ/Notes

* Ensure the `roundabout` conda environment is **active** before running scripts.
* Outputs (models, predictions) are saved in `outputs/`.
* Raw and processed data are **not tracked** in git (`.gitignore`) for team convenience.

1. If src not found, do ensure that init.py has the correct syntax and your file path is correct

---

## Team / Contributors

* Steve Chia
* Xie Yanjun
* Venice Phua
* Tong Jia Jun
* Lee Sze Ying