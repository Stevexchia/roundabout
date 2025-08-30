"""
Utility script to add derived relevance labels to existing labeled data.
This is useful if you already have policy labels but want to add the overall relevance metric.

Input: data/labeled/reviews_with_labels.csv
Output: Updates the same file with is_relevant column
"""

import pandas as pd
from pathlib import Path


def add_relevance_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived relevance labels based on policy violations."""
    df = df.copy()
    
    # Required columns else unable to decide whether is_relevant
    required_cols = ['is_advertisement', 'is_irrelevant', 'is_rant_without_visit']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # A review is relevant if it violates NONE of the three policies
    df['is_relevant'] = ~(
        df['is_advertisement'] | 
        df['is_irrelevant'] | 
        df['is_rant_without_visit']
    )
    
    return df


def main():
    # Path to labeled data
    input_path = Path("data/labeled/reviews_with_labels.csv")
    
    if not input_path.exists():
        raise FileNotFoundError(f"Labeled data not found: {input_path}")
    
    print(" Loading labeled data...")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} reviews")
    
    # Check if relevance labels already exist
    if 'is_relevant' in df.columns:
        print(" Relevance labels already exist. Overwriting...")
    
    # Add relevance labels
    print(" Adding relevance labels...")
    df_with_relevance = add_relevance_labels(df)
    
    # Show statistics
    total = len(df_with_relevance)
    relevant = df_with_relevance['is_relevant'].sum()
    not_relevant = total - relevant
    
    print(f"\n Relevance Statistics:")
    print(f"   Total reviews: {total}")
    print(f"   Relevant: {relevant} ({relevant/total:.1%})")
    print(f"   Not relevant: {not_relevant} ({not_relevant/total:.1%})")
    
    print(f"\n Violation Breakdown:")
    ad_only = (df_with_relevance['is_advertisement'] & 
               ~df_with_relevance['is_irrelevant'] & 
               ~df_with_relevance['is_rant_without_visit']).sum()
    irrelevant_only = (~df_with_relevance['is_advertisement'] & 
                       df_with_relevance['is_irrelevant'] & 
                       ~df_with_relevance['is_rant_without_visit']).sum()
    rant_only = (~df_with_relevance['is_advertisement'] & 
                 ~df_with_relevance['is_irrelevant'] & 
                 df_with_relevance['is_rant_without_visit']).sum()
    multiple = (df_with_relevance[['is_advertisement', 'is_irrelevant', 'is_rant_without_visit']].sum(axis=1) > 1).sum()
    
    print(f"   Advertisement only: {ad_only}")
    print(f"   Irrelevant content only: {irrelevant_only}")
    print(f"   Rant without visit only: {rant_only}")
    print(f"   Multiple violations: {multiple}")
    
    df_with_relevance.to_csv(input_path, index=False)
    print(f"\n Updated labeled data saved to: {input_path}")
    
    print(f"\n Sample updated reviews:")
    sample_cols = ['review_id', 'text', 'is_advertisement', 'is_irrelevant', 
                   'is_rant_without_visit', 'is_relevant']
    available_cols = [col for col in sample_cols if col in df_with_relevance.columns]
    print(df_with_relevance[available_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()