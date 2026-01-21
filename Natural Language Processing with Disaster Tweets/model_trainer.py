"""
Model training module for Disaster Tweets Classification.
Handles BERT model initialization, training, and inference.

This module provides the ModelTrainer class for:
- GPU/CPU device setup with validation
- Model loading and initialization
- Training with class weight balancing
- Evaluation metrics computation
- Batch prediction
"""

import inspect
import os
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional
import transformers
from transformers import (
    BertForSequenceClassification,
    Trainer, TrainingArguments, EarlyStoppingCallback
)
from datasets import Dataset
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
from config import Config


class ModelTrainer:
    """
    Handles model training, validation, and inference.
    
    This class encapsulates:
    - Device setup (GPU/CPU detection with memory reporting)
    - BERT model loading and initialization
    - Class weight computation for handling imbalanced datasets
    - Training loop with early stopping and validation
    - Batch prediction and probability estimation
    
    Attributes:
        config: Config instance with hyperparameters
        debug: Whether to print debug information
        device: torch.device (cuda or cpu)
        model: BertForSequenceClassification model instance
        trainer: HuggingFace Trainer instance (after training)
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
        Setup GPU/CPU device with validation and memory reporting.
        
        Returns:
            torch.device object (cuda or cpu)
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
        Load pre-trained BERT model for binary classification.
        
        The model is loaded from HuggingFace model hub with
        classification head initialized for binary classification.
        
        Returns:
            BertForSequenceClassification model on configured device
        """
        model = BertForSequenceClassification.from_pretrained(
            self.config.MODEL_NAME,
            num_labels=self.config.NUM_LABELS
        )
        model.to(self.device)
        return model
    
    def _compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """
        Compute class weights to handle imbalanced dataset.
        
        Uses inverse frequency weighting:
        weight_i = 1 / frequency_i
        
        Then normalizes so that sum(weights) = num_classes,
        keeping training dynamics stable.
        
        Args:
            labels: List of label integers (0 or 1)
            
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
        Compute evaluation metrics for validation set.
        
        Metrics computed:
        - Accuracy: (TP + TN) / (TP + TN + FP + FN)
        - F1 Score (weighted): Balances precision and recall
        - Precision (weighted)
        - Recall (weighted)
        
        Args:
            eval_preds: EvalPrediction object from Trainer
                       with 'predictions' and 'label_ids' fields
            
        Returns:
            Dictionary mapping metric names to values
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
    
    def _get_training_args(self, class_weights: Optional[torch.Tensor] = None) -> TrainingArguments:
        """
        Construct TrainingArguments with version-aware parameter handling.
        
        This method handles compatibility across different transformers versions:
        - Older versions: use 'evaluation_strategy' (deprecated in 4.30+)
        - Newer versions: may drop 'evaluation_strategy' or rename parameters
        
        The method intelligently:
        1. Inspects the TrainingArguments signature
        2. Filters kwargs to only include supported parameters
        3. Removes dependent parameters if their prerequisites are unsupported
           (e.g., remove load_best_model_at_end if evaluation_strategy unavailable)
        
        Args:
            class_weights: Optional class weights (not directly used here)
            
        Returns:
            TrainingArguments instance compatible with installed transformers
        """
        # Prepare all potential training arguments
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
        
        # Filter kwargs to those supported by installed transformers version
        supported_params = inspect.signature(TrainingArguments.__init__).parameters
        filtered_kwargs = {k: v for k, v in ta_kwargs.items() if k in supported_params}
        ignored = set(ta_kwargs) - set(filtered_kwargs)
        
        # Handle dependent parameters
        # If evaluation_strategy is not supported, remove parameters that depend on it
        if 'evaluation_strategy' not in supported_params and 'evaluation_strategy' in ignored:
            # These require evaluation_strategy to be set
            dependent_params = {'load_best_model_at_end', 'metric_for_best_model', 'greater_is_better'}
            for param in dependent_params:
                filtered_kwargs.pop(param, None)
            
            if self.debug:
                print(f"[DEBUG] Removed dependent parameters (require evaluation_strategy): {dependent_params}")
        
        if self.debug and ignored:
            print(f"[DEBUG] TrainingArguments ignored parameters (unsupported in transformers {transformers.__version__}): {ignored}")
        
        return TrainingArguments(**filtered_kwargs)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """
        Train the model with early stopping and validation.
        
        Training pipeline:
        1. Create output directory for checkpoints
        2. Setup training arguments (version-aware)
        3. Initialize HuggingFace Trainer
        4. Apply class weights via custom loss function if provided
        5. Execute training loop
        
        Args:
            train_dataset: HuggingFace Dataset for training
            eval_dataset: HuggingFace Dataset for validation
            class_weights: Optional tensor of class weights for imbalance handling
            
        Returns:
            Dictionary with training results including:
            - 'training_loss': Final training loss
            - 'model': Trained model instance
        """
        # Create output directory
        os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
        
        # Setup training arguments with version compatibility
        training_args = self._get_training_args(class_weights)
        
        # Initialize trainer with conditional callbacks
        callbacks = []
        # Only add EarlyStoppingCallback if a metric for best model is defined
        if getattr(training_args, 'metric_for_best_model', None):
            callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
        else:
            if self.debug:
                print("[DEBUG] EarlyStoppingCallback skipped: 'metric_for_best_model' undefined (likely no evaluation strategy)")

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=callbacks,
        )
        
        # Apply class weights if provided
        if class_weights is not None:
            # Create custom model with weighted loss
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
            dataset: HuggingFace Dataset to predict on
            
        Returns:
            Array of shape (n_samples, num_labels) with class probabilities
            (summed to 1.0 across labels for each sample)
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
            dataset: HuggingFace Dataset to predict on
            
        Returns:
            Array of shape (n_samples,) with predicted class labels (0 or 1)
        """
        probas = self.predict_proba(dataset)
        return np.argmax(probas, axis=1)
