"""
Train BERT-based classifiers for policy violation detection.

Input: data/labeled/reviews_with_labels.csv
Output: Trained models in outputs/models/
"""

import pandas as pd
from pathlib import Path
from src.policies import MultiPolicyClassifier
import argparse
import json

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train policy classifiers')
    parser.add_argument('--model-name', type=str, default='distilbert-base-uncased',
                       help='Base model to use')
    parser.add_argument('--min-samples', type=int, default=5,
                       help='Minimum samples per class for training')
    args = parser.parse_args()
    
    input_path = Path("data/labeled/reviews_validation.csv")
    output_dir = Path("outputs/models")
    
    print("Loading labeled data...")
    if not input_path.exists():
        raise FileNotFoundError(f"Labeled data not found: {input_path}")
    labeled_df = pd.read_csv(input_path)
    print(f"Loaded {len(labeled_df)} labeled reviews")
    
    required_columns = ['rating_category', 'text_clean', 'is_advertisement', 'is_irrelevant', 'is_rant_without_visit']
    missing_cols = [col for col in required_columns if col not in labeled_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    labeled_df = labeled_df.dropna(subset=['text_clean'])
    labeled_df = labeled_df.dropna(subset=['rating_category'])
    labeled_df = labeled_df.reset_index(drop=True)
    print(f"After filtering: {len(labeled_df)} reviews")
    
    print(f"\nClass Distribution:")
    for policy in ['advertisement', 'irrelevant', 'rant_without_visit']:
        col_name = f'is_{policy}'
        positive_count = labeled_df[col_name].sum()
        negative_count = len(labeled_df) - positive_count
        print(f"   {policy}: {positive_count} positive, {negative_count} negative")
        
        if positive_count < args.min_samples:
            print(f"   Warning: Only {positive_count} positive samples for {policy} (min: {args.min_samples})")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nInitializing classifiers with {args.model_name}...")
    multi_classifier = MultiPolicyClassifier()
    
    # Set model name for all classifiers
    for classifier in multi_classifier.classifiers.values():
        classifier.model_name = args.model_name
    
    # Train all classifiers
    print(f"\nTraining classifiers...")
    try:
        multi_classifier.train_all(labeled_df)
        print(f"All classifiers trained successfully!")
        
        metadata = {
            'model_name': args.model_name,
            'training_samples': len(labeled_df),
            'class_distribution': {
                'advertisement': int(labeled_df['is_advertisement'].sum()),
                'irrelevant': int(labeled_df['is_irrelevant'].sum()),
                'rant_without_visit': int(labeled_df['is_rant_without_visit'].sum())
            },
            'training_completed': True
        }
        
        metadata_path = output_dir / 'training_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Training metadata saved to: {metadata_path}")
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise
    
    print(f"\nTraining completed! Models saved in: {output_dir}")
    print(f"Next step: Run evaluation with scripts/04_eval.py")


if __name__ == "__main__":
    main()