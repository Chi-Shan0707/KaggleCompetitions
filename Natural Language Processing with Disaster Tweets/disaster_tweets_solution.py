"""
Complete Disaster Tweets Classification Solution
Uses Hugging Face Transformers + PyTorch with built-in debugging checks
Optimized for Kaggle's free GPU and 10k dataset
"""


import re
import inspect
import numpy as np
import pandas as pd
import warnings
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import os

# os.environ['HTTP_PROXY'] = 'http://10.255.255.254:7897'
# os.environ['HTTPS_PROXY'] = 'http://10.255.255.254:7897'

# Hugging Face imports
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support

warnings.filterwarnings('ignore')


# ============================================================================
# INITIALIZATION & CACHING UTILITIES
# ============================================================================

def check_and_report_cache_status():
    """
    Report Hugging Face model cache status and usage.
    Helps users understand if models will be re-downloaded or cached.
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


# ============================================================================
# SECTION 1: CONFIGURATION CLASS - All hyperparameters in one place
# ============================================================================

class Config:
    """
    Centralized configuration for the disaster tweets pipeline.
    All magic numbers are defined here for easy adjustment.
    """
    
    # Model configuration
    MODEL_NAME: str = 'bert-base-uncased'
    MAX_LENGTH: int = 128
    NUM_LABELS: int = 2
    
    # Training hyperparameters
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01
    
    # Data configuration
    TRAIN_SIZE: float = 0.8
    VALIDATION_SIZE: float = 0.1
    TEST_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    # Text preprocessing
    MIN_TEXT_LENGTH: int = 3
    REMOVE_URLS: bool = True
    REMOVE_MENTIONS: bool = True
    REMOVE_SPECIAL_CHARS: bool = False  # BERT handles special tokens
    LOWERCASE: bool = True
    
    # Debugging and paths
    DEBUG_ENABLED: bool = True
    SAMPLE_SIZE_DEBUG: int = 100  # Rows to check for debugging
    CHECKPOINT_DIR: str = './checkpoint'
    OUTPUT_DIR: str = './outputs'
    
    @classmethod
    def validate(cls) -> None:
        """Validate configuration values are sensible."""
        assert cls.MAX_LENGTH > 0, "MAX_LENGTH must be positive"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert 0 < cls.LEARNING_RATE < 1, "LEARNING_RATE should be between 0 and 1"
        assert 0 < cls.TRAIN_SIZE < 1, "TRAIN_SIZE should be between 0 and 1"
        if cls.DEBUG_ENABLED:
            print(f"[DEBUG] Configuration validated successfully")


# ============================================================================
# SECTION 2: DATA PREPROCESSOR - Text cleaning and tokenization
# ============================================================================

class DataPreprocessor:
    """
    Handles text cleaning, tokenization, and dataset preparation.
    Includes validation checks for data quality.
    """
    
    def __init__(self, config: Config):
        """
        Initialize preprocessor with tokenizer and config.
        
        Args:
            config: Config class instance with hyperparameters
        """
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.debug = config.DEBUG_ENABLED
        
    def clean_text(self, text: str) -> str:
        """
        Remove noise from text while preserving semantic content.
        
        Steps:
        1. Handle NaN values
        2. Remove URLs (if enabled)
        3. Remove mentions (if enabled)
        4. Normalize whitespace
        5. Lowercase (if enabled)
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        # Handle missing values
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove mentions and hashtag symbols (keep hashtag content)
        if self.config.REMOVE_MENTIONS:
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Lowercase
        if self.config.LOWERCASE:
            text = text.lower()
        
        return text
    
    def validate_text(self, text: str) -> bool:
        """
        Check if text meets minimum requirements.
        
        Args:
            text: Cleaned text string
            
        Returns:
            True if text is valid, False otherwise
        """
        return len(text.split()) >= self.config.MIN_TEXT_LENGTH
    
    def tokenize_batch(self, texts: List[str]) -> Dict:
        """
        Tokenize a batch of texts using BERT tokenizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with input_ids, attention_mask, token_type_ids
        """
        return self.tokenizer(
            texts,
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'target',
        split_type: str = 'train'
    ) -> Tuple[Dataset, Optional[Dict]]:
        """
        Prepare dataset for training/inference.
        
        Args:
            df: Input DataFrame
            text_col: Name of text column
            label_col: Name of label column
            split_type: 'train', 'validation', or 'inference'
            
        Returns:
            Tuple of (HF Dataset, stats_dict)
        """
        # Debug: Check for missing values
        if self.debug:
            print(f"[DEBUG] Original shape: {df.shape}")
            print(f"[DEBUG] Missing values in {text_col}: {df[text_col].isna().sum()}")
            if label_col in df.columns:
                print(f"[DEBUG] Missing values in {label_col}: {df[label_col].isna().sum()}")
        
        # Create working copy
        df = df.copy()
        
        # Drop rows with missing text
        df = df.dropna(subset=[text_col])
        
        # Clean text
        df['cleaned_text'] = df[text_col].apply(self.clean_text)
        
        # Validate text length
        df['valid'] = df['cleaned_text'].apply(self.validate_text)
        invalid_count = (~df['valid']).sum()
        if invalid_count > 0 and self.debug:
            print(f"[DEBUG] Removing {invalid_count} rows with too-short text")
        df = df[df['valid']].drop('valid', axis=1)
        
        # Debug: Check for NaN values after cleaning
        if self.debug:
            nan_count = df['cleaned_text'].isna().sum()
            if nan_count > 0:
                print(f"[WARNING] {nan_count} NaN values after cleaning")
                df = df.dropna(subset=['cleaned_text'])
        
        # Tokenize
        encodings = self.tokenizer(
            df['cleaned_text'].tolist(),
            max_length=self.config.MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors=None
        )
        
        # Debug: Check token distribution
        if self.debug:
            token_lens = [len(ids) for ids in encodings['input_ids']]
            print(f"[DEBUG] Token length - Min: {min(token_lens)}, "
                  f"Max: {max(token_lens)}, Mean: {np.mean(token_lens):.1f}")
        
        # Prepare dataset dictionary
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings['token_type_ids'],
        }
        
        # Add labels if available
        if label_col in df.columns:
            labels = df[label_col].astype(int).tolist()
            dataset_dict['labels'] = labels
            
            # Debug: Check class distribution
            if self.debug:
                class_dist = pd.Series(labels).value_counts().to_dict()
                print(f"[DEBUG] Class distribution: {class_dist}")
        
        # Create HF Dataset
        hf_dataset = Dataset.from_dict(dataset_dict)
        
        # Prepare stats
        stats = {
            'n_samples': len(hf_dataset),
            'n_features': len(encodings['input_ids'][0]) if encodings['input_ids'] else 0,
        }
        
        return hf_dataset, stats


# ============================================================================
# SECTION 3: MODEL TRAINER - Training loop with validation
# ============================================================================

class ModelTrainer:
    """
    Handles model training, validation, and inference.
    Uses Hugging Face Trainer API for production reliability.
    """
    
    def __init__(self, config: Config):
        """
        Initialize trainer with model and configuration.
        
        Args:
            config: Config class instance
        """
        self.config = config
        self.debug = config.DEBUG_ENABLED
        self.device = self._setup_device()
        self.model = self._load_model()
        self.trainer = None
        
    def _setup_device(self) -> torch.device:
        """
        Setup GPU/CPU device with validation.
        
        Returns:
            torch.device object
        """
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if self.debug:
                print(f"[DEBUG] GPU available: {torch.cuda.get_device_name(0)}")
                print(f"[DEBUG] GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = torch.device('cpu')
            if self.debug:
                print(f"[DEBUG] GPU not available, using CPU")
        
        return device
    
    def _load_model(self) -> BertForSequenceClassification:
        """
        Load pre-trained BERT model for classification.
        
        Returns:
            BertForSequenceClassification model
        """
        model = BertForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS
        )
        model.to(self.device)
        return model
    
    def _compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """
        Compute class weights to handle imbalance.
        Uses inverse frequency weighting: weight = 1 / frequency
        
        Args:
            labels: List of label integers
            
        Returns:
            Tensor of shape (num_labels,) with class weights
        """
        labels_array = np.array(labels)
        class_counts = np.bincount(labels_array, minlength=self.config.NUM_LABELS)
        class_weights = 1.0 / (class_counts + 1e-8)  # Avoid division by zero
        class_weights = class_weights / class_weights.sum() * len(class_weights)  # Normalize
        
        if self.debug:
            for i, weight in enumerate(class_weights):
                print(f"[DEBUG] Class {i} weight: {weight:.4f}")
        
        return torch.tensor(class_weights, dtype=torch.float32).to(self.device)
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """
        Compute evaluation metrics (F1, Accuracy, Precision, Recall).
        
        Args:
            eval_preds: EvalPrediction object from Trainer
            
        Returns:
            Dictionary of metric names to values
        """
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=1)
        
        f1 = f1_score(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        precision, recall, _, _ = precision_recall_fscore_support(
            labels, predictions, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall
        }
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Train the model with early stopping.
        
        Args:
            train_dataset: HF Dataset for training
            eval_dataset: HF Dataset for validation
            class_weights: Optional tensor of class weights for imbalance handling
            
        Returns:
            Dictionary with training results
        """
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # Setup training arguments (signature-aware for compatibility across transformers versions)
        ta_kwargs = {
            'output_dir': self.config.OUTPUT_DIR,
            'num_train_epochs': self.config.NUM_EPOCHS,
            'per_device_train_batch_size': self.config.BATCH_SIZE,
            'per_device_eval_batch_size': self.config.BATCH_SIZE,
            'learning_rate': self.config.LEARNING_RATE,
            'warmup_steps': self.config.WARMUP_STEPS,
            'weight_decay': self.config.WEIGHT_DECAY,
            'logging_steps': 100,
            'save_strategy': 'epoch',
            'evaluation_strategy': 'epoch',
            'load_best_model_at_end': True,
            'metric_for_best_model': 'f1',
            'greater_is_better': True,
            'save_total_limit': 2,
            'seed': self.config.RANDOM_STATE,
            'fp16': torch.cuda.is_available()  # Use mixed precision on GPU
        }
        # Filter kwargs to those supported by this transformers version
        supported_params = inspect.signature(TrainingArguments.__init__).parameters
        filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported_params}
        ignored = set(ta_kwargs) - set(filtered_kwargs)
        if self.debug and ignored:
            print(f"[DEBUG] TrainingArguments ignored parameters (unsupported in installed transformers): {ignored}")
        training_args = TrainingArguments(**filtered_kwargs)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
        )
        
        # Apply class weights if provided
        if class_weights is not None:
            # Create custom loss function with weights
            original_model = self.model
            
            class WeightedBertModel(BertForSequenceClassification):
                def __init__(self, model, weights):
                    super().__init__(model.config)
                    self.bert = model.bert
                    self.classifier = model.classifier
                    self.class_weights = weights
                
                def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
                    outputs = super().forward(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        labels=None
                    )
                    
                    if labels is not None:
                        loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
                        loss = loss_fct(outputs.logits.view(-1, self.config.num_labels), labels.view(-1))
                        outputs.loss = loss
                    
                    return outputs
            
            self.model = WeightedBertModel(original_model, class_weights)
            self.model.to(self.device)
            self.trainer.model = self.model
            if self.debug:
                print("[DEBUG] Class weights applied to model")
        
        # Train
        if self.debug:
            print("[DEBUG] Starting training...")
        
        train_result = self.trainer.train()
        
        if self.debug:
            print(f"[DEBUG] Training complete. Loss: {train_result.training_loss:.4f}")
        
        return {
            'training_loss': train_result.training_loss,
            'model': self.model
        }
    
    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """
        Get probability predictions for a dataset.
        
        Args:
            dataset: HF Dataset to predict on
            
        Returns:
            Array of shape (n_samples, num_labels) with probabilities
        """
        if self.trainer is None:
            raise ValueError("Model must be trained first")
        
        predictions = self.trainer.predict(dataset)
        logits = predictions.predictions
        probas = nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
        
        return probas
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """
        Get class predictions for a dataset.
        
        Args:
            dataset: HF Dataset to predict on
            
        Returns:
            Array of shape (n_samples,) with class predictions
        """
        probas = self.predict_proba(dataset)
        return np.argmax(probas, axis=1)


# ============================================================================
# SECTION 4: INFERENCE MODULE - Prediction and validation
# ============================================================================

class DisasterTweetsInference:
    """
    End-to-end inference pipeline for disaster tweets classification.
    """
    
    def __init__(self, config: Config, model: BertForSequenceClassification,
                 preprocessor: DataPreprocessor, trainer: ModelTrainer):
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
        Predict class for a single tweet.
        
        Args:
            text: Tweet text
            
        Returns:
            Dictionary with prediction and confidence
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
        
        # Get model on correct device
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
            texts: List of tweet texts
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            results.append(self.predict_single_tweet(text))
        return results
    
    def validate_sample_predictions(self, test_dataset: Dataset,
                                   sample_size: int = 10) -> Dict:
        """
        Validate model on sample predictions with detailed debugging.
        
        Args:
            test_dataset: HF Dataset to test on
            sample_size: Number of samples to validate
            
        Returns:
            Dictionary with validation results
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


# ============================================================================
# SECTION 5: MAIN EXECUTION BLOCK
# ============================================================================

def main():
    """
    Main execution function with debugging checks at each stage.
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
    class_weights = trainer_instance = ModelTrainer(config)
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
