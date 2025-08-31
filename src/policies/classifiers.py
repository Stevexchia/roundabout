"""
BERT-based classifiers for policy violation detection.
"""

import torch
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments, pipeline
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import pickle
from pathlib import Path
from torch.utils.data import Dataset

class ReviewDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        
    def __len__(self):
        return self.encodings['input_ids'].shape[0]
    
    def __getitem__(self, idx):
        return {key: val[idx] for key, val in self.encodings.items()}

class PolicyClassifier:
    """BERT-based classifier for policy violations."""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", policy_type: str = "advertisement"):
        """Initialize the classifier."""
        self.model_name = model_name
        self.policy_type = policy_type
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.is_trained = False
        
    def prepare_model(self, num_labels: int = 2):
        """Initialize tokenizer and model."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name, 
            num_labels=num_labels
        )
    
    def _combine_text_and_rating(self, texts: List[str], rating_categories: List[str]) -> List[str]:
        """Combine text and rating category into single input."""
        combined_texts = []
        for text, rating_cat in zip(texts, rating_categories):
            # Prepend rating category as a special token
            combined_text = f"[{rating_cat.upper()}_REVIEW] {text}"
            combined_texts.append(combined_text)
        return combined_texts
        
    def prepare_data(self, texts: List[str], labels: List[int], rating_categories: List[str] = None) -> Dict:
        if self.tokenizer is None:
            raise ValueError("Model not prepared. Call prepare_model() first.")
        
        # Combine text with rating category if provided
        if rating_categories is not None:
            input_texts = self._combine_text_and_rating(texts, rating_categories)
        else:
            input_texts = texts
            
        encodings = self.tokenizer(
            input_texts,
            truncation=True,
            padding=True,
            max_length=512,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def train(self, train_texts: List[str], train_labels: List[int], 
              train_rating_categories: List[str] = None,
              val_texts: List[str] = None, val_labels: List[int] = None,
              val_rating_categories: List[str] = None,
              output_dir: str = None):
        
        if output_dir is None:
            output_dir = f"outputs/models/{self.policy_type}_classifier"
        
        self.prepare_model()
        
        train_encodings = self.prepare_data(train_texts, train_labels, train_rating_categories)
        train_dataset = ReviewDataset(train_encodings)

        if val_texts and val_labels:
            val_encodings = self.prepare_data(val_texts, val_labels, val_rating_categories)
            val_dataset = ReviewDataset(val_encodings)
        else:
            val_dataset = None
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=64,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            eval_strategy="epoch" if val_dataset else "no",
            save_strategy="epoch",
            load_best_model_at_end=True if val_dataset else False,
        )
        
        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
        )
        
        # Train
        print(f" Training {self.policy_type} classifier...")
        trainer.train()
        
        trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        self.pipeline = pipeline(
            "text-classification",
            model=output_dir,
            tokenizer=output_dir,
            return_all_scores=True
        )
        
        self.is_trained = True
        print(f"{self.policy_type} classifier trained and saved to {output_dir}")
    
    def load_model(self, model_path: str):
        """Load a pre-trained model."""
        try:
            self.pipeline = pipeline(
                "text-classification",
                model=model_path,
                tokenizer=model_path,
                return_all_scores=True
            )
            self.is_trained = True
            print(f"Loaded {self.policy_type} classifier from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
    
    def predict(self, texts: List[str], rating_categories: List[str] = None) -> List[Dict]:
        """Predict policy violations for texts."""
        if not self.is_trained or self.pipeline is None:
            raise ValueError("Model not trained or loaded. Train model or call load_model() first.")
        
        # Combine text with rating category if provided
        if rating_categories is not None:
            input_texts = self._combine_text_and_rating(texts, rating_categories)
        else:
            input_texts = texts
        
        predictions = []
        for text in input_texts:
            result = self.pipeline(text)[0]  # Get first result
            # Convert to probability of positive class (violation)
            violation_prob = next(r['score'] for r in result if r['label'] == 'LABEL_1')
            predictions.append({
                'violation_probability': violation_prob,
                'is_violation': violation_prob > 0.5,
                'confidence': max(r['score'] for r in result)
            })
        
        return predictions
    
    def evaluate(self, test_texts: List[str], test_labels: List[int], test_rating_categories: List[str] = None) -> Dict:
        """Evaluate the classifier."""
        predictions = self.predict(test_texts, test_rating_categories)
        pred_labels = [1 if p['is_violation'] else 0 for p in predictions]
        
        report = classification_report(test_labels, pred_labels, output_dict=True)
        cm = confusion_matrix(test_labels, pred_labels)
        
        return {
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'accuracy': report['accuracy'],
            'f1_score': report['1']['f1-score'] if '1' in report else 0.0,
            'precision': report['1']['precision'] if '1' in report else 0.0,
            'recall': report['1']['recall'] if '1' in report else 0.0
        }


class MultiPolicyClassifier:
    """Manages multiple policy classifiers including overall relevance."""
    
    def __init__(self):
        self.classifiers = {
            'advertisement': PolicyClassifier(policy_type='advertisement'),
            'irrelevant': PolicyClassifier(policy_type='irrelevant'),
            'rant_without_visit': PolicyClassifier(policy_type='rant_without_visit'),
            'relevance': PolicyClassifier(policy_type='relevance')  # Overall relevance classifier
        }
    
    def train_all(self, labeled_df: pd.DataFrame):
        """Train all policy classifiers including overall relevance."""
        # First, derive the relevance label if it doesn't exist
        if 'is_relevant' not in labeled_df.columns:
            labeled_df = self._add_relevance_labels(labeled_df)
        
        texts = labeled_df['text_clean'].tolist()
        rating_categories = labeled_df['rating_category'].tolist() if 'rating_category' in labeled_df.columns else None
        
        for policy_type, classifier in self.classifiers.items():
            print(f"\n Training {policy_type} classifier...")
            
            # Get labels for this policy
            if policy_type == 'rant_without_visit':
                labels = labeled_df['is_rant_without_visit'].astype(int).tolist()
            elif policy_type == 'relevance':
                labels = labeled_df['is_relevant'].astype(int).tolist()
            else:
                labels = labeled_df[f'is_{policy_type}'].astype(int).tolist()
            
            unique_labels = set(labels)
            if len(unique_labels) < 2:
                print(f"Skipping {policy_type}: only one class present ({unique_labels})")
                continue
            
            # Split data with rating categories
            if rating_categories:
                train_texts, val_texts, train_labels, val_labels, train_ratings, val_ratings = train_test_split(
                    texts, labels, rating_categories, test_size=0.2, random_state=42, stratify=labels
                )
            else:
                train_texts, val_texts, train_labels, val_labels = train_test_split(
                    texts, labels, test_size=0.2, random_state=42, stratify=labels
                )
                train_ratings, val_ratings = None, None
            
            classifier.train(train_texts, train_labels, train_ratings, val_texts, val_labels, val_ratings)
    
    def _add_relevance_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add derived relevance labels based on policy violations."""
        df = df.copy()
        
        # A review is relevant if it violates NONE of the three policies
        df['is_relevant'] = ~(
            df['is_advertisement'] | 
            df['is_irrelevant'] | 
            df['is_rant_without_visit']
        )
        
        print(f" Derived relevance labels:")
        relevant_count = df['is_relevant'].sum()
        total_count = len(df)
        print(f"   Relevant: {relevant_count} ({relevant_count/total_count:.1%})")
        print(f"   Not relevant: {total_count - relevant_count} ({(total_count - relevant_count)/total_count:.1%})")
        
        return df
    
    def load_all_models(self, models_dir: str = "outputs/models"):
        """Load all pre-trained models."""
        models_path = Path(models_dir)
        for policy_type, classifier in self.classifiers.items():
            model_path = models_path / f"{policy_type}_classifier"
            if model_path.exists():
                classifier.load_model(str(model_path))
            else:
                print(f"Model not found: {model_path}")
    
    def predict_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Predict all policy violations AND overall relevance for reviews in a DataFrame."""
        results = []
        texts = df['text_clean'].tolist()
        rating_categories = df['rating_category'].tolist() if 'rating_category' in df.columns else None
        
        for i, row in df.iterrows():
            text = row['text_clean']
            rating_category = row.get('rating_category', None)
            result = {
                'text_index': i,
                'text': text,
                'rating_category': rating_category,
            }
            
            policy_violations = {}
            for policy_type, classifier in self.classifiers.items():
                if policy_type == 'relevance':
                    continue
                if classifier.is_trained:
                    # Pass rating category to prediction
                    pred = classifier.predict([text], [rating_category] if rating_category else None)[0]
                    result[f'is_{policy_type}'] = pred['is_violation']
                    result[f'{policy_type}_confidence'] = pred['violation_probability']
                    policy_violations[policy_type] = pred['is_violation']
                else:
                    result[f'is_{policy_type}'] = False
                    result[f'{policy_type}_confidence'] = 0.0
                    policy_violations[policy_type] = False
            
            # Handle relevance prediction
            # Method 1: Derived relevance (review is relevant if it violates NONE of the policies)
            derived_relevance = not any(policy_violations.values())
            result['is_relevant_derived'] = derived_relevance
            
            # Method 2: Use relevance classifier if available
            if 'relevance' in self.classifiers and self.classifiers['relevance'].is_trained:
                relevance_pred = self.classifiers['relevance'].predict([text], [rating_category] if rating_category else None)[0]
                # For relevance classifier, treat it as predicting "irrelevance", so flip the result
                result['is_relevant'] = not relevance_pred['is_violation']
                result['relevance_confidence'] = 1 - relevance_pred['violation_probability']
            else:
                # Fall back to derived relevance
                result['is_relevant'] = derived_relevance
                result['relevance_confidence'] = 1.0 if derived_relevance else 0.0
            
            # Optionally, add review_id if present
            if 'review_id' in row:
                result['review_id'] = row['review_id']
            results.append(result)
        return pd.DataFrame(results)
    
    def evaluate_all(self, test_df: pd.DataFrame) -> Dict:
        """Evaluate all classifiers including overall relevance."""
        # Add relevance labels if missing
        if 'is_relevant' not in test_df.columns:
            test_df = self._add_relevance_labels(test_df)
            
        texts = test_df['text_clean'].tolist()
        rating_categories = test_df['rating_category'].tolist() if 'rating_category' in test_df.columns else None
        evaluation_results = {}
        
        for policy_type, classifier in self.classifiers.items():
            if classifier.is_trained:
                # Get true labels
                if policy_type == 'rant_without_visit':
                    true_labels = test_df['is_rant_without_visit'].astype(int).tolist()
                elif policy_type == 'relevance':
                    true_labels = test_df['is_relevant'].astype(int).tolist()
                else:
                    true_labels = test_df[f'is_{policy_type}'].astype(int).tolist()
                
                # Evaluate with rating categories
                evaluation_results[policy_type] = classifier.evaluate(texts, true_labels, rating_categories)
            else:
                print(f"{policy_type} classifier not trained, skipping evaluation")
        
        return evaluation_results