"""
Normalize & clean raw Google reviews into a single schema.

Input  (default): data/raw/reviews.csv  (columns seen in your file)
  - business_name, author_name, text, photo, rating, rating_category

Output: data/processed/reviews.csv  (columns)
  - review_id, place_name, rating, rating_category, text, text_clean

Notes:
- Keeps English-only if langdetect is installed; otherwise skips language filter.
- Fixes weird encodings if ftfy is installed; otherwise skips that step.
- Drops rows with empty text.
"""

from pathlib import Path
import re
import pandas as pd

# ---- Optional deps: used if available, otherwise skipped gracefully
try:
    from ftfy import fix_text
except Exception:
    fix_text = None

try:
    from langdetect import detect, DetectorFactory
    DetectorFactory.seed = 0
except Exception:
    detect = None

RAW_PATH = Path("data/raw/reviews.csv")
OUT_DIR = Path("data/processed")
OUT_PATH = OUT_DIR / "reviews.csv"

def basic_clean(text: str) -> str:
    """Lowercase, trim, collapse spaces, normalize repeated punctuation."""
    if not isinstance(text, str):
        return ""
    text = text.strip()
    if fix_text:
        # fix encoding if ftfy is available
        text = fix_text(text)
    text = text.lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"([!?.,]){2,}", r"\\1", text)
    return text.strip()

def is_english(text: str) -> bool:
    """Return True if text is English; if langdetect unavailable, allow all."""
    if not text or not isinstance(text, str):
        return False
    if detect is None:
        return True
    try:
        return detect(text) == "en"
    except Exception:
        # On detection failure, keep the row (be permissive)
        return True

def main():
    assert RAW_PATH.exists(), f"Input not found: {RAW_PATH}. Put your raw CSV there."

    df_raw = pd.read_csv(RAW_PATH)

    # sanity: require a 'text' column
    if "text" not in df_raw.columns:
        raise ValueError("Expected a 'text' column in the raw CSV.")

    # drop rows without text
    df = df_raw.dropna(subset=["text"]).copy()

    # optional language filter
    df = df[df["text"].astype(str).map(is_english)]

    # build normalized frame
    place_col = "business_name" if "business_name" in df.columns else "place_name"
    df_norm = pd.DataFrame({
        "review_id": range(1, len(df) + 1),
        "place_name": df[place_col].astype(str),
        "rating": df["rating"] if "rating" in df.columns else None,
        "rating_category": df["rating_category"] if "rating_category" in df.columns else None,
        "text": df["text"].astype(str),
    })

    # cleaned text
    df_norm["text_clean"] = df_norm["text"].map(basic_clean)

    # ensure output dir
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df_norm.to_csv(OUT_PATH, index=False)

    # quick report
    kept = len(df_norm)
    total = len(df_raw)
    print(f"✅ Saved {kept}/{total} rows → {OUT_PATH}")
    print(df_norm.head(5))

if __name__ == "__main__":
    main()
