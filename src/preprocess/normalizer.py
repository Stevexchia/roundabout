import pandas as pd
from typing import Dict, Optional
from .text_cleaner import basic_clean
from .filters import should_keep_review

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize column names and structure."""
    # Handle different possible column names
    place_col = None
    for col in ["business_name", "place_name", "location", "venue"]:
        if col in df.columns:
            place_col = col
            break
    
    if place_col is None:
        raise ValueError("No place/business name column found")
    
    # Create normalized structure
    df_norm = pd.DataFrame({
        "review_id": range(1, len(df) + 1),
        "place_name": df[place_col].astype(str),
        "rating": df.get("rating"),
        "rating_category": df.get("rating_category"),
        "text": df["text"].astype(str),
    })
    
    return df_norm


def add_processed_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned text and metadata features."""
    df = df.copy()
    
    # Clean text
    df["text_clean"] = df["text"].map(basic_clean)
    
    # Add metadata features
    df["text_length"] = df["text"].str.len()
    df["word_count"] = df["text_clean"].str.split().str.len()
    df["has_rating"] = df["rating"].notna()
    
    return df

def preprocess_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    if "text" not in df_raw.columns:
        raise ValueError("Input DataFrame must contain a 'text' column.")
    
    df = df_raw.dropna(subset=["text"]).copy()
    df = df[df["text"].astype(str).map(should_keep_review)]
    
    df_norm = normalize_columns(df)
    
    df_processed = add_processed_features(df_norm)
    
    return df_processed