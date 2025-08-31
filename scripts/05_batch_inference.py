import pandas as pd
from pathlib import Path
from src.policies import MultiPolicyClassifier

def main():
    input_path = Path("data/processed/reviews_processed.csv") 
    models_dir = Path("outputs/models")
    output_path = Path("outputs/predictions/batch_predictions.csv")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    if 'text_clean' not in df.columns:
        raise ValueError("Input data must have a 'text_clean' column.")

    print(f"Loaded {len(df)} reviews for prediction")
    if 'rating_category' in df.columns:
        print("Rating categories found - will be used for enhanced predictions")
    else:
        print("No rating categories found - using text only")

    print("Loading trained classifiers...")
    multi_classifier = MultiPolicyClassifier()
    multi_classifier.load_all_models(models_dir=models_dir)

    print("Running batch inference...")
    results_df = multi_classifier.predict_all(df)

    results_df.to_csv(output_path, index=False)
    print(f"Saved batch predictions to {output_path}")
    
    # Print summary
    print(f"\nPrediction Summary:")
    print(f"Total reviews processed: {len(results_df)}")
    if 'is_relevant' in results_df.columns:
        relevant_count = results_df['is_relevant'].sum()
        print(f"Relevant reviews: {relevant_count} ({relevant_count/len(results_df):.1%})")
        
        # Show policy violation breakdown
        for policy in ['advertisement', 'irrelevant', 'rant_without_visit']:
            if f'is_{policy}' in results_df.columns:
                violation_count = results_df[f'is_{policy}'].sum()
                print(f"{policy.capitalize()} violations: {violation_count} ({violation_count/len(results_df):.1%})")

if __name__ == "__main__":
    main()