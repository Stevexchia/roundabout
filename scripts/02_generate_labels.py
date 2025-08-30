"""
Generating of pseudo-labels using LLM.
This is to minimize manual labeling effort.
Use cases as well as examples can be found here: https://github.com/openai/openai-python?tab=readme-ov-file
"""
import os
import pandas as pd
from pathlib import Path
from src.llm import LLMClient, PolicyLabel
import argparse
from dotenv import load_dotenv

load_dotenv()  # take environment variables from .env file

def main():
    # for users to parse CLI arguments
    parser = argparse.ArgumentParser(description='Generate pseudo-labels using GPT or Ollama fallback')
    parser.add_argument('--api-key', type=str, help='OpenAI API key', 
                       default=os.getenv('OPENAI_API_KEY'))
    parser.add_argument('--model', type=str, default='gpt-4o', 
                       help='GPT model to use')
    parser.add_argument('--ollama-model', type=str, default='gemma3:1b', 
                        help='Ollama local model to use for fallback')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of reviews to label (default: all)')
    parser.add_argument('--batch-size', type=int, default=5,
                       help='Batch size for API calls')
    parser.add_argument('--overwrite', action='store_true',
                       help='Whether to overwrite existing output file')
    args = parser.parse_args()
    
    if not args.api_key:
        raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable or use --api-key")
    
    # Paths
    input_path = Path("data/processed/reviews_processed.csv")
    output_dir = Path("data/labeled")
    output_path = output_dir / "reviews_with_labels.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading processed reviews...")
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    reviews_df = pd.read_csv(input_path)
    print(f"Loaded {len(reviews_df)} reviews")
    
    if args.sample_size and args.sample_size < len(reviews_df):
        reviews_df = reviews_df.sample(n=args.sample_size, random_state=42)
        print(f"Sampled {args.sample_size} reviews for labeling")
    
    print(f"Initializing LLM client (OpenAI: {args.model}, Ollama fallback: {args.ollama_model})...")
    llm_client = LLMClient(api_key=args.api_key, model=args.model, ollama_model=args.ollama_model)
    
    print("Generating pseudo-labels...")
    labels_df = llm_client.classify_batch(reviews_df, batch_size=args.batch_size, output_path=str(output_path), overwrite=args.overwrite)
    
    labeled_df = reviews_df.merge(labels_df, on='review_id', how='left')
    labeled_df.to_csv(output_path, index=False)
    
    # output summary statistics
    total_reviews = len(labeled_df)
    ad_violations = labeled_df['is_advertisement'].sum()
    irrelevant_violations = labeled_df['is_irrelevant'].sum()
    rant_violations = labeled_df['is_rant_without_visit'].sum()
    relevant_reviews = labeled_df['is_relevant'].sum()
    
    print(f"\nLabeled dataset saved to: {output_path}")
    print(f"Label Statistics:")
    print(f"   Total reviews: {total_reviews}")
    print(f"   1. Advertisement violations: {ad_violations} ({ad_violations/total_reviews:.1%})")
    print(f"   2. Irrelevant content: {irrelevant_violations} ({irrelevant_violations/total_reviews:.1%})")
    print(f"   3. Rants without visit: {rant_violations} ({rant_violations/total_reviews:.1%})")
    print(f"   Overall relevant reviews: {relevant_reviews} ({relevant_reviews/total_reviews:.1%})")
    
    # Showing of sample results
    print(f"\nSample labeled reviews:")
    sample_cols = ['review_id', 'place_name', 'text', 'is_advertisement', 
                   'is_irrelevant', 'is_rant_without_visit', 'is_relevant', 'reasoning']
    available_cols = [col for col in sample_cols if col in labeled_df.columns]
    print(labeled_df[available_cols].head(3).to_string(index=False))


if __name__ == "__main__":
    main()