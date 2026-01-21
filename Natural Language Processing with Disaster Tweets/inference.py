"""
Inference module for Disaster Tweets Classification.
Handles single tweet and batch prediction with confidence scores.

This module provides utilities for:
- Single tweet prediction with confidence scores and probabilities
- Batch prediction for multiple tweets
- Model validation on sample data with debugging output
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List
from datasets import Dataset
from transformers import BertForSequenceClassification
from config import Config
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer


class DisasterTweetsInference:
    """
    End-to-end inference pipeline for disaster tweets classification.
    
    This class provides:
    - Single tweet prediction with confidence and probability breakdown
    - Batch prediction for multiple tweets
    - Sample validation with detailed debugging
    
    Attributes:
        config: Config instance
        model: Trained BertForSequenceClassification model
        preprocessor: DataPreprocessor instance
        trainer: ModelTrainer instance
        debug: Whether to print debug information
    """
    
    def __init__(
        self,
        config: Config,
        model: BertForSequenceClassification,
        preprocessor: DataPreprocessor,
        trainer: ModelTrainer
    ):
        """
        Initialize inference pipeline.
        
        Args:
            config: Config instance
            model: Trained model
            preprocessor: DataPreprocessor instance
            trainer: ModelTrainer instance
        """
        self.config = config
        self.model = model
        self.preprocessor = preprocessor
        self.trainer = trainer
        self.debug = config.DEBUG_ENABLED
    
    def predict_single_tweet(self, text: str) -> Dict[str, any]:
        """
        Predict class for a single tweet with detailed output.
        
        Processing:
        1. Clean and validate text
        2. Tokenize using BERT tokenizer
        3. Move tensors to device
        4. Forward pass through model
        5. Apply softmax for probabilities
        6. Return prediction, confidence, and probability breakdown
        
        Args:
            text: Raw tweet text
            
        Returns:
            Dictionary with keys:
            - 'text': Original tweet text
            - 'cleaned_text': Cleaned text after preprocessing
            - 'prediction': Predicted class (0 or 1)
            - 'confidence': Confidence score (0.0-1.0)
            - 'probabilities': Dict with probabilities for each class
            - 'error': Error message if prediction failed
        """
        # Clean and validate
        cleaned = self.preprocessor.clean_text(text)
        if not self.preprocessor.validate_text(cleaned):
            return {
                'text': text,
                'prediction': -1,
                'confidence': 0.0,
                'error': 'Text too short after cleaning'
            }
        
        # Tokenize
        encodings = self.preprocessor.tokenizer(
            [cleaned],
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Move to device
        for key in encodings:
            encodings[key] = encodings[key].to(self.trainer.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            probas = nn.functional.softmax(logits, dim=1).cpu().numpy()
        
        prediction = int(np.argmax(probas[0]))
        confidence = float(probas[0][prediction])
        
        return {
            'text': text,
            'cleaned_text': cleaned,
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': {
                'not_disaster': float(probas[0][0]),
                'disaster': float(probas[0][1])
            }
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """
        Predict class for multiple tweets.
        
        Args:
            texts: List of tweet text strings
            
        Returns:
            List of prediction dictionaries (one per tweet)
        """
        results = []
        for text in texts:
            results.append(self.predict_single_tweet(text))
        return results
    
    def validate_sample_predictions(
        self,
        test_dataset: Dataset,
        sample_size: int = 10
    ) -> Dict:
        """
        Validate model on random sample with detailed debugging output.
        
        This method:
        1. Randomly selects sample_size examples from dataset
        2. Gets model predictions
        3. Compares with true labels (if available)
        4. Prints detailed comparison for first 5 samples
        
        Args:
            test_dataset: HuggingFace Dataset to validate on
            sample_size: Number of samples to validate
            
        Returns:
            Dictionary with validation results:
            - 'accuracy': Sample accuracy (if labels available)
            - 'predictions': List of predicted labels
            - 'true_labels': List of true labels (if available)
        """
        sample_size = min(sample_size, len(test_dataset))
        sample_indices = np.random.choice(len(test_dataset), sample_size, replace=False)
        
        sample_data = test_dataset.select(sample_indices)
        predictions = self.trainer.predict(sample_data)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        
        if 'labels' in sample_data.column_names:
            true_labels = sample_data['labels']
            accuracy = np.mean(pred_labels == true_labels)
            
            if self.debug:
                print(f"[DEBUG] Sample validation accuracy: {accuracy:.4f}")
                for i in range(min(5, sample_size)):
                    print(f"[DEBUG] Sample {i}: True={true_labels[i]}, Pred={pred_labels[i]}")
            
            return {
                'accuracy': accuracy,
                'predictions': pred_labels.tolist(),
                'true_labels': true_labels
            }
        else:
            if self.debug:
                print(f"[DEBUG] Predictions: {pred_labels[:5]}...")
            
            return {
                'predictions': pred_labels.tolist()
            }
