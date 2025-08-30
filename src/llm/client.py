"""
LLM client for generating pseudo-labels using OpenAI GPT models.
"""

import openai
import json
import time
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

@dataclass
class PolicyLabel:
    """Container for policy violation labels."""
    is_advertisement: bool
    is_irrelevant: bool
    is_rant_without_visit: bool
    confidence_advertisement: float
    confidence_irrelevant: float
    confidence_rant: float
    reasoning: str = ""


class LLMClient:
    """Client for interacting with OpenAI GPT models."""
    
    def __init__(self, api_key: str, model: str = "gpt-5-mini"):
        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.rate_limit_delay = 1.0  # seconds between requests to ensure we dont hit the limit
    
    def classify_review(self, review_text: str, place_name: str = "") -> PolicyLabel:
        """Classify a single review for policy violations."""
        prompt = self._build_classification_prompt(review_text, place_name)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are an expert at detecting policy violations in location reviews. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # Low temperature for consistency
                max_tokens=500
            )
            
            result_text = response.choices[0].message.content.strip()
            result_json = json.loads(result_text)
            
            return PolicyLabel(
                is_advertisement=result_json.get("is_advertisement", False),
                is_irrelevant=result_json.get("is_irrelevant", False), 
                is_rant_without_visit=result_json.get("is_rant_without_visit", False),
                confidence_advertisement=result_json.get("confidence_advertisement", 0.5),
                confidence_irrelevant=result_json.get("confidence_irrelevant", 0.5),
                confidence_rant=result_json.get("confidence_rant", 0.5),
                reasoning=result_json.get("reasoning", "")
            )
            
        except Exception as e:
            print(f"Error processing review: {e}")
            return PolicyLabel(False, False, False, 0.0, 0.0, 0.0, f"Error: {str(e)}")
    
    def classify_batch(self, reviews_df: pd.DataFrame, batch_size: int = 5, output_path: Optional[str] = None) -> pd.DataFrame:
        results = []

        if output_path and Path(output_path).exists():
            existing_df = pd.read_csv(output_path)
            labeled_ids = set(existing_df['review_id'])
            reviews_df = reviews_df[~reviews_df['review_id'].isin(labeled_ids)]
            results.extend(existing_df.to_dict('records'))
            print(f"Resuming labeling, skipping {len(labeled_ids)} already labeled reviews")

        total_batches = (len(reviews_df) + batch_size - 1) // batch_size
        
        for i in range(0, len(reviews_df), batch_size):
            batch = reviews_df.iloc[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{total_batches}")
            
            for _, row in batch.iterrows():
                review_text = row.get('text', '')
                place_name = row.get('place_name', '')
                
                try:
                    label = self.classify_review(review_text, place_name)
                except Exception as e:
                    label = PolicyLabel(False, False, False, 0.0, 0.0, 0.0, f"Error: {str(e)}")
                    print(f"Error processing review {row.get('review_id')}: {e}")

                is_relevant = not (label.is_advertisement or label.is_irrelevant or label.is_rant_without_visit)
                
                #adds to the results data
                results.append({
                    'review_id': row.get('review_id', ''),
                    'is_advertisement': label.is_advertisement,
                    'is_irrelevant': label.is_irrelevant,
                    'is_rant_without_visit': label.is_rant_without_visit,
                    'is_relevant': is_relevant,
                    'confidence_advertisement': label.confidence_advertisement,
                    'confidence_irrelevant': label.confidence_irrelevant, 
                    'confidence_rant': label.confidence_rant,
                    'reasoning': label.reasoning
                })
                
                # Rate limiting
                time.sleep(self.rate_limit_delay)
            
            if output_path:
                pd.DataFrame(results).to_csv(output_path, index=False)
                print(f"Saved progress to {output_path} after batch {i//batch_size + 1}")
        
        return pd.DataFrame(results)
    
    def _build_classification_prompt(self, review_text: str, place_name: str = "") -> str:
        """Build the classification prompt for GPT."""
        return f"""
Analyze this location review and classify it for policy violations. Respond with valid JSON only.

PLACE: {place_name}
REVIEW: "{review_text}"

POLICIES TO CHECK:
1. Advertisement: Contains promotional content, links, or marketing
2. Irrelevant: Not about the location (personal stories unrelated to place)
3. Rant without visit: Complaint from someone who clearly hasn't been there

EXAMPLES:
- "Great food! Visit www.deals.com" → Advertisement: true
- "I love my new car, this place is loud" → Irrelevant: true  
- "Never been but heard it's terrible" → Rant without visit: true
- "Food was cold, service slow" → All false (legitimate complaint)

Respond with JSON format:
{{
  "is_advertisement": boolean,
  "is_irrelevant": boolean,
  "is_rant_without_visit": boolean,
  "confidence_advertisement": float (0.0-1.0),
  "confidence_irrelevant": float (0.0-1.0),
  "confidence_rant": float (0.0-1.0),
  "reasoning": "Brief explanation"
}}
"""