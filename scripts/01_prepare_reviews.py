"""
Normalize & clean raw Google reviews using modular preprocessing.
"""

from pathlib import Path
import pandas as pd
from src.preprocess import preprocess_dataframe

RAW_PATH = Path("data/raw/reviews.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "reviews_processed.csv"


def main():
    assert RAW_PATH.exists(), f"Input not found: {RAW_PATH}. Put your raw CSV there."
    
    df_raw = pd.read_csv(RAW_PATH)
    
    df_processed = preprocess_dataframe(df_raw)
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_processed.to_csv(OUT_PATH, index=False)
    
    # Report of results
    kept = len(df_processed)
    total = len(df_raw)
    print(f"Saved {kept}/{total} rows → {OUT_PATH}")
    print(df_processed.head())
    
    print(f"📊 Average review length: {df_processed['text_length'].mean():.1f} chars")
    print(f"📊 Average word count: {df_processed['word_count'].mean():.1f} words")


if __name__ == "__main__":
    main()