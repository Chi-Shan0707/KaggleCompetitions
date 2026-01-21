"""
Main execution module for Disaster Tweets Classification.
Orchestrates the complete pipeline: data loading, preprocessing, training, and inference.

Usage:
    python main.py

This module handles:
- Data loading from Kaggle CSV files
- Preprocessing and dataset preparation
- Model training with validation
- Test set prediction
- Submission file generation
"""

import os
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from config import Config
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from inference import DisasterTweetsInference


def check_and_report_cache_status():
    """
    Report Hugging Face model cache status for user awareness.
    
    Helps users understand if models will be downloaded or loaded from cache.
    """
    cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
    
    if cache_dir.exists():
        cache_size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        print(f"[DEBUG] Hugging Face cache found at: {cache_dir}")
        print(f"[DEBUG] Cache size: {cache_size_mb:.2f} MB")
        
        # Check if BERT model is already cached
        bert_cached = any('bert-base-uncased' in str(p) for p in cache_dir.rglob('*'))
        if bert_cached:
            print(f"[DEBUG] ✓ bert-base-uncased already in cache (will use local copy, no re-download)")
        else:
            print(f"[DEBUG] ⚠ bert-base-uncased NOT in cache (will download on first use)")
    else:
        print(f"[DEBUG] Hugging Face cache not found - models will download on first use to: {cache_dir}")


def main():
    """
    Execute the complete disaster tweets classification pipeline.
    
    Pipeline stages:
    1. Configuration validation
    2. Data loading from CSV files
    3. Data preprocessing and tokenization
    4. Train/validation/test splits
    5. Class weight computation
    6. Model training with early stopping
    7. Validation on hold-out set
    8. Test set prediction
    9. Sample inference demonstration
    10. Submission file generation
    """
    
    # ---- PRE-FLIGHT CHECK: Cache Status ----
    check_and_report_cache_status()
    
    print("="*80)
    print("DISASTER TWEETS CLASSIFICATION - COMPLETE PIPELINE")
    print("="*80)
    
    # ---- STEP 1: Configuration ----
    print("\n[STEP 1] Loading configuration...")
    config = Config()
    config.validate()
    print(f"✓ Configuration loaded with MAX_LENGTH={config.MAX_LENGTH}, BATCH_SIZE={config.BATCH_SIZE}")
    
    # ---- STEP 2: Load Data ----
    print("\n[STEP 2] Loading training data...")
    train_path = Path('nlp-getting-started/train.csv')
    test_path = Path('nlp-getting-started/test.csv')
    
    if not train_path.exists():
        print(f"✗ Error: {train_path} not found. Please download from Kaggle.")
        return
    
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    print(f"✓ Loaded train: {train_df.shape[0]} samples, test: {test_df.shape[0]} samples")
    
    # Debug: Check column names
    if config.DEBUG_ENABLED:
        print(f"[DEBUG] Train columns: {train_df.columns.tolist()}")
        print(f"[DEBUG] Test columns: {test_df.columns.tolist()}")
    
    # ---- STEP 3: Preprocess Data ----
    print("\n[STEP 3] Preprocessing data...")
    preprocessor = DataPreprocessor(config)
    
    # Prepare training data with train/val split
    train_dataset, train_stats = preprocessor.prepare_dataset(
        train_df, text_col='text', label_col='target', split_type='train'
    )
    print(f"✓ Training dataset: {train_stats['n_samples']} samples")
    
    # Split into train/validation
    indices = np.arange(len(train_dataset))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=config.VALIDATION_SIZE / (config.TRAIN_SIZE),
        random_state=config.RANDOM_STATE,
        stratify=train_dataset['labels']
    )
    
    train_split = train_dataset.select(train_indices)
    val_split = train_dataset.select(val_indices)
    
    print(f"✓ Train split: {len(train_split)} samples")
    print(f"✓ Validation split: {len(val_split)} samples")
    
    # Prepare test data
    test_dataset, test_stats = preprocessor.prepare_dataset(
        test_df, text_col='text', split_type='inference'
    )
    print(f"✓ Test dataset: {test_stats['n_samples']} samples")
    
    # ---- STEP 4: Compute Class Weights ----
    print("\n[STEP 4] Computing class weights for imbalance handling...")
    trainer_instance = ModelTrainer(config)
    class_weights = trainer_instance._compute_class_weights(train_split['labels'])
    print(f"✓ Class weights computed")
    
    # ---- STEP 5: Train Model ----
    print("\n[STEP 5] Training model with Hugging Face Trainer...")
    trainer = ModelTrainer(config)
    training_result = trainer.train(train_split, val_split, class_weights=class_weights)
    print(f"✓ Training complete")
    
    # ---- STEP 6: Validation ----
    print("\n[STEP 6] Validating model predictions...")
    inference = DisasterTweetsInference(config, trainer.model, preprocessor, trainer)
    
    validation_result = inference.validate_sample_predictions(val_split, sample_size=10)
    print(f"✓ Sample validation accuracy: {validation_result['accuracy']:.4f}")
    
    # ---- STEP 7: Test Predictions ----
    print("\n[STEP 7] Generating test predictions...")
    test_predictions = trainer.predict(test_dataset)
    test_probas = trainer.predict_proba(test_dataset)
    
    print(f"✓ Generated predictions for {len(test_predictions)} test samples")
    
    if config.DEBUG_ENABLED:
        print(f"[DEBUG] Prediction distribution: {np.bincount(test_predictions)}")
        print(f"[DEBUG] Confidence stats - Min: {test_probas.max(axis=1).min():.4f}, "
              f"Max: {test_probas.max(axis=1).max():.4f}, "
              f"Mean: {test_probas.max(axis=1).mean():.4f}")
    
    # ---- STEP 8: Sample Inference Test ----
    print("\n[STEP 8] Running sample inference tests...")
    sample_tweets = [
        "Just experienced a terrible earthquake near my area.",
        "Beautiful sunset today, feeling peaceful and grateful.",
        "URGENT: Building collapse reported downtown, need rescue teams"
    ]
    
    for tweet in sample_tweets:
        result = inference.predict_single_tweet(tweet)
        pred_label = "DISASTER" if result['prediction'] == 1 else "NOT DISASTER"
        print(f"  Tweet: '{tweet[:50]}...'")
        print(f"  → Prediction: {pred_label} (confidence: {result['confidence']:.4f})")
    
    # ---- STEP 9: Prepare Submission ----
    print("\n[STEP 9] Preparing submission file...")
    submission_df = pd.DataFrame({
        'id': test_df['id'],
        'target': test_predictions
    })
    
    submission_path = 'disaster_tweets_submission.csv'
    submission_df.to_csv(submission_path, index=False)
    print(f"✓ Submission saved to {submission_path}")
    
    if config.DEBUG_ENABLED:
        print(f"[DEBUG] Submission shape: {submission_df.shape}")
        print(f"[DEBUG] First 5 rows:\n{submission_df.head()}")
    
    # ---- FINAL SUMMARY ----
    print("\n" + "="*80)
    print("PIPELINE EXECUTION COMPLETE")
    print("="*80)
    print(f"Summary:")
    print(f"  • Train samples: {len(train_split)}")
    print(f"  • Validation accuracy: {validation_result['accuracy']:.4f}")
    print(f"  • Test predictions: {len(test_predictions)}")
    print(f"  • Submission file: {submission_path}")
    print("="*80)
    
    return {
        'trainer': trainer,
        'preprocessor': preprocessor,
        'submission': submission_df,
        'inference': inference
    }


if __name__ == '__main__':
    results = main()
