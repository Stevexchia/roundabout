import pandas as pd
from pathlib import Path
from src.policies import MultiPolicyClassifier

def main():
    input_path = Path("data/processed/reviews_processed.csv") 
    models_dir = Path("outputs/models")
    output_path = Path("outputs/predictions/batch_predictions.csv")

    df = pd.read_csv(input_path)
    if 'text_clean' not in df.columns:
        raise ValueError("Input data must have a 'text_clean' column.")

    print("Loading trained classifiers...")
    multi_classifier = MultiPolicyClassifier()
    multi_classifier.load_all_models(models_dir=models_dir)

    print("Running batch inference...")
    results_df = multi_classifier.predict_all(df['text_clean'].tolist())

    if 'review_id' in df.columns:
        results_df['review_id'] = df['review_id']

    results_df.to_csv(output_path, index=False)
    print(f"Saved batch predictions to {output_path}")

if __name__ == "__main__":
    main()