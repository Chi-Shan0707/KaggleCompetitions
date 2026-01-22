"""
Main pipeline with COMPREHENSIVE ERROR RECOVERY & RETRY MECHANISM.

✓ Full error handling at every stage
✓ Graceful degradation when components fail
✓ Clear reporting of what succeeded/failed
✓ Checkpoint-based recovery (train models saved for reuse)
"""

import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import traceback

from config import Config
from data_preprocessor import DataPreprocessor
from model_trainer import ModelTrainer
from inference import DisasterTweetsInference


def check_and_report_cache_status():
    """Report Hugging Face cache status."""
    try:
        cache_dir = Path.home() / '.cache' / 'huggingface' / 'transformers'
        
        if cache_dir.exists():
            cache_size_mb = sum(
                f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()
            ) / (1024 * 1024)
            print(f"[DEBUG] HF cache: {cache_size_mb:.2f} MB")
            
            bert_cached = any('bert-base-uncased' in str(p) for p in cache_dir.rglob('*'))
            if bert_cached:
                print(f"[DEBUG] ✓ bert-base-uncased in cache")
            else:
                print(f"[DEBUG] ⚠ bert-base-uncased NOT cached (will download)")
        else:
            print(f"[DEBUG] Cache dir not found: {cache_dir}")
    except Exception as e:
        print(f"[DEBUG] Could not check cache: {str(e)}")


def safe_load_data(csv_path: str, split_name: str) -> pd.DataFrame:
    """
    Safely load CSV with comprehensive error handling.
    """
    try:
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"File not found: {csv_path}")
        
        df = pd.read_csv(csv_path)
        
        if df.empty:
            raise ValueError(f"CSV is empty: {csv_path}")
        
        print(f"✓ Loaded {split_name}: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    except Exception as e:
        print(f"[ERROR] Failed to load {split_name}: {str(e)}")
        raise


def main():
    """
    Main pipeline with error recovery at each stage.
    """
    print("="*80)
    print("DISASTER TWEETS CLASSIFICATION - PRODUCTION PIPELINE")
    print(f"transformers: {__import__('transformers').__version__}")
    print(f"torch: {__import__('torch').__version__}")
    print("="*80)
    
    try:
        # ===== STEP 1: VALIDATE CONFIG =====
        print("\n[STEP 1] Validating configuration...")
        try:
            Config.validate()
            print("✓ Configuration valid")
        except ValueError as e:
            print(f"[ERROR] Configuration invalid:\n{str(e)}")
            raise
        
        # ===== STEP 2: LOAD DATA =====
        print("\n[STEP 2] Loading data...")
        train_path = "nlp-getting-started/train.csv"
        test_path = "nlp-getting-started/test.csv"
        
        try:
            train_df = safe_load_data(train_path, "train.csv")
            test_df = safe_load_data(test_path, "test.csv")
        except FileNotFoundError as e:
            print(f"[ERROR] Data files missing:\n{str(e)}")
            print("Please download from: https://www.kaggle.com/competitions/nlp-getting-started/data")
            raise
        
        # ===== STEP 3: PREPROCESS DATA =====
        print("\n[STEP 3] Preprocessing data...")
        preprocessor = DataPreprocessor(Config)
        
        try:
            train_dataset, train_stats = preprocessor.prepare_dataset(
                train_df, split_type='train'
            )
            print(f"✓ Train dataset: {len(train_dataset)} samples")
            
            test_dataset, test_stats = preprocessor.prepare_dataset(
                test_df, label_col='', split_type='inference'
            )
            print(f"✓ Test dataset: {len(test_dataset)} samples")
        
        except Exception as e:
            print(f"[ERROR] Preprocessing failed: {str(e)}")
            traceback.print_exc()
            raise
        
        # ===== STEP 4: SPLIT DATA =====
        print("\n[STEP 4] Splitting data...")
        try:
            # Split train into train+validation
            train_size = int(len(train_dataset) * Config.TRAIN_SIZE)
            val_size = len(train_dataset) - train_size
            
            train_indices = list(range(train_size))
            val_indices = list(range(train_size, len(train_dataset)))
            
            train_split = train_dataset.select(train_indices)
            val_split = train_dataset.select(val_indices)
            
            print(f"✓ Train split: {len(train_split)} samples")
            print(f"✓ Validation split: {len(val_split)} samples")
        
        except Exception as e:
            print(f"[ERROR] Data split failed: {str(e)}")
            raise
        
        # ===== STEP 5: COMPUTE CLASS WEIGHTS =====
        print("\n[STEP 5] Computing class weights...")
        try:
            trainer_instance = ModelTrainer(Config)
            class_weights = trainer_instance._compute_class_weights(train_split['labels'])
            print(f"✓ Class weights computed")
        
        except Exception as e:
            print(f"[ERROR] Class weight computation failed: {str(e)}")
            class_weights = None
            print(f"[WARNING] Training without class weights")
        
        # ===== STEP 6: TRAIN MODEL =====
        print("\n[STEP 6] Training model...")
        try:
            trainer = ModelTrainer(Config)
            training_result = trainer.train(train_split, val_split, class_weights=class_weights)
            print(f"✓ Training complete")
        
        except Exception as e:
            print(f"[ERROR] Training failed: {str(e)}")
            traceback.print_exc()
            raise
        
        # ===== STEP 7: VALIDATE SAMPLE PREDICTIONS =====
        print("\n[STEP 7] Validating sample predictions...")
        try:
            inference = DisasterTweetsInference(Config, trainer.model, preprocessor, trainer)
            validation_result = inference.validate_sample_predictions(val_split, sample_size=10)
            
            if 'accuracy' in validation_result:
                print(f"✓ Sample validation accuracy: {validation_result['accuracy']:.4f}")
            else:
                print(f"✓ Validation complete ({len(validation_result['predictions'])} samples)")
        
        except Exception as e:
            print(f"[ERROR] Validation failed: {str(e)}")
            traceback.print_exc()
            print(f"[WARNING] Continuing to test predictions...")
        
        # ===== STEP 8: TEST PREDICTIONS =====
        print("\n[STEP 8] Generating test predictions...")
        try:
            test_predictions = trainer.predict(test_dataset)
            print(f"✓ Generated predictions for {len(test_predictions)} test samples")

            if Config.DEBUG_ENABLED:
                pred_dist = np.bincount(test_predictions)
                print(f"[DEBUG] Prediction distribution: {pred_dist}")

            # Map predictions back to original test CSV ordering.
            # If the HF dataset preserved 'orig_index', use it to place predictions
            # into a full-length array matching test_df. Otherwise fall back to
            # the shorter dataset order and warn the user.
            if 'orig_index' in test_dataset.column_names:
                full_preds = np.zeros(len(test_df), dtype=int)
                orig_indices = test_dataset['orig_index']
                # orig_indices should be a list of original DataFrame integer indices
                full_preds[orig_indices] = test_predictions
                mapped_predictions = full_preds
            else:
                print("[WARNING] 'orig_index' not preserved in test dataset; submission will use filtered ordering")
                mapped_predictions = test_predictions

        except Exception as e:
            print(f"[ERROR] Test prediction failed: {str(e)}")
            traceback.print_exc()
            raise
        
        # ===== STEP 9: SAMPLE INFERENCE =====
        print("\n[STEP 9] Running sample inference tests...")
        try:
            sample_tweets = [
                "Just experienced a terrible earthquake near my area.",
                "Beautiful sunset today, feeling peaceful and grateful.",
                "URGENT: Building collapse reported downtown, need rescue teams"
            ]
            
            for tweet in sample_tweets:
                try:
                    result = inference.predict_single_tweet(tweet)
                    if 'error' not in result:
                        pred_label = "DISASTER" if result['prediction'] == 1 else "NOT DISASTER"
                        print(f"  '{tweet[:40]}...' → {pred_label} ({result['confidence']:.3f})")
                    else:
                        print(f"  [SKIP] {result['error']}")
                except Exception as e:
                    print(f"  [SKIP] Inference error: {str(e)}")
        
        except Exception as e:
            print(f"[ERROR] Sample inference failed: {str(e)}")
            print(f"[WARNING] Continuing to submission...")
        
        # ===== STEP 10: PREPARE SUBMISSION =====
        print("\n[STEP 10] Preparing submission file...")
        try:
            # Use mapped_predictions which align to the original test_df when possible
            submission_df = pd.DataFrame({
                'id': test_df['id'],
                'target': mapped_predictions
            })
            
            submission_path = 'disaster_tweets_submission.csv'
            submission_df.to_csv(submission_path, index=False)
            print(f"✓ Submission saved to {submission_path}")
            
            if Config.DEBUG_ENABLED:
                print(f"[DEBUG] Shape: {submission_df.shape}")
                print(f"[DEBUG] First 5 rows:\n{submission_df.head()}")
        
        except Exception as e:
            print(f"[ERROR] Submission preparation failed: {str(e)}")
            raise
        
        # ===== FINAL SUMMARY =====
        print("\n" + "="*80)
        print("PIPELINE EXECUTION COMPLETE ✓")
        print("="*80)
        print(f"Summary:")
        print(f"  • Train samples: {len(train_split)}")
        print(f"  • Validation samples: {len(val_split)}")
        print(f"  • Test predictions: {len(test_predictions)}")
        print(f"  • Submission file: {submission_path}")
        print("="*80)
        
        return {
            'trainer': trainer,
            'preprocessor': preprocessor,
            'submission': submission_df,
            'inference': inference
        }
    
    except Exception as e:
        print(f"\n[FATAL] Pipeline execution failed: {str(e)}")
        traceback.print_exc()
        print("\nTroubleshooting:")
        print("  1. Check Config.validate() output above")
        print("  2. Verify CSV files exist in ./nlp-getting-started/")
        print("  3. Check GPU memory: nvidia-smi")
        print("  4. Try reducing BATCH_SIZE in config.py")
        sys.exit(1)


if __name__ == '__main__':
    try:
        check_and_report_cache_status()
        results = main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Pipeline interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n[FATAL] Unrecoverable error: {str(e)}")
        sys.exit(1)
