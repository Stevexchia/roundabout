"""
Evaluate trained BERT classifiers on test data.

Input: data/labeled/reviews_with_labels.csv + trained models
Output: outputs/predictions/evaluation_results.json
"""

import pandas as pd
from pathlib import Path
from src.policies import MultiPolicyClassifier
from sklearn.model_selection import train_test_split
import json
import argparse
from datetime import datetime


def create_detailed_report(evaluation_results: dict, test_df: pd.DataFrame) -> dict:
    """Create a detailed evaluation report."""
    report = {
        'evaluation_timestamp': datetime.now().isoformat(),
        'test_set_size': len(test_df),
        'policy_evaluations': {}
    }
    
    for policy_type, metrics in evaluation_results.items():
        policy_report = {
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'confusion_matrix': metrics['confusion_matrix'],
            'classification_report': metrics['classification_report']
        }
        
        # Add interpretation
        if metrics['f1_score'] >= 0.8:
            performance = "Excellent"
        elif metrics['f1_score'] >= 0.7:
            performance = "Good"
        elif metrics['f1_score'] >= 0.6:
            performance = "Fair"
        else:
            performance = "Needs Improvement"
        
        policy_report['performance_rating'] = performance
        
        report['policy_evaluations'][policy_type] = policy_report
    
    # Overall summary
    avg_f1 = sum(m['f1_score'] for m in evaluation_results.values()) / len(evaluation_results)
    avg_precision = sum(m['precision'] for m in evaluation_results.values()) / len(evaluation_results)
    avg_recall = sum(m['recall'] for m in evaluation_results.values()) / len(evaluation_results)
    
    report['overall_metrics'] = {
        'average_f1_score': avg_f1,
        'average_precision': avg_precision,
        'average_recall': avg_recall
    }
    
    return report


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Evaluate policy classifiers')
    parser.add_argument('--test-size', type=float, default=0.2,
                       help='Test set size (0.0-1.0)')
    parser.add_argument('--random-seed', type=int, default=42,
                       help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Paths
    input_path = Path("data/labeled/reviews_with_labels.csv")
    models_dir = Path("outputs/models")
    output_dir = Path("outputs/predictions")
    output_path = output_dir / "evaluation_results.json"
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load labeled data
    print("Loading labeled data...")
    if not input_path.exists():
        raise FileNotFoundError(f"Labeled data not found: {input_path}")
    
    labeled_df = pd.read_csv(input_path)
    labeled_df = labeled_df.dropna(subset=['text_clean'])
    print(f"Loaded {len(labeled_df)} labeled reviews")
    
    # Split into train/test (matching the training split)
    train_df, test_df = train_test_split(
        labeled_df, 
        test_size=args.test_size, 
        random_state=args.random_seed,
        stratify=labeled_df[['is_advertisement', 'is_irrelevant', 'is_rant_without_visit']].sum(axis=1)
    )
    
    print(f"Test set size: {len(test_df)} reviews")
    
    # Load trained models