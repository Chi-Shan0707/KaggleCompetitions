"""
Model training module with FULL VERSION COMPATIBILITY & ERROR RECOVERY.

✓ transformers 4.57.6 compatible
✓ Pattern P002: Type standardization (force ModelOutput)
✓ Pattern P003: Device consistency checks
✓ Pattern P007: Serialization-safe class definition (module level)
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


class WeightedBertModel(BertForSequenceClassification):
    """
    BERT with weighted loss - MODULE-LEVEL DEFINITION (Pattern P007 fix).
    
    Forces ModelOutput format for guaranteed compatibility across transformers versions.
    """
    def __init__(self, model, weights):
        super().__init__(model.config)
        self.bert = model.bert
        self.classifier = model.classifier
        self.class_weights = weights

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, labels=None, **kwargs):
        """
        Forward pass with FORCED ModelOutput (Pattern P002 standardization).
        Also includes device consistency checks (Pattern P003).
        """
        # FORCE return_dict=True for version-agnostic output
        outputs = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            labels=labels,
            return_dict=True,
            **kwargs
        )

        if labels is not None:
            # Pattern P003: Device consistency check
            if self.class_weights.device != outputs.logits.device:
                self.class_weights = self.class_weights.to(outputs.logits.device)

            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights)
            outputs.loss = loss_fct(
                outputs.logits.view(-1, self.config.num_labels),
                labels.view(-1)
            )

        return outputs


class ModelTrainer:
    """
    Production-ready trainer with comprehensive error handling.
    """
    
    def __init__(self, config: Config):
        """Initialize with validation."""
        self.config = config
        self.debug = config.DEBUG_ENABLED
        self.device = self._setup_device()
        self.model = self._load_model()
        self.trainer = None
        
    def _setup_device(self) -> torch.device:
        """Setup device with memory reporting."""
        if torch.cuda.is_available():
            device = torch.device('cuda')
            if self.debug:
                print(f"[DEBUG] GPU: {torch.cuda.get_device_name(0)}")
                gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print(f"[DEBUG] GPU memory: {gpu_mem:.2f} GB")
        else:
            device = torch.device('cpu')
            if self.debug:
                print(f"[DEBUG] GPU not available, using CPU")
        
        return device
    
    def _load_model(self) -> BertForSequenceClassification:
        """Load BERT model with error handling."""
        try:
            model = BertForSequenceClassification.from_pretrained(
                self.config.MODEL_NAME,
                num_labels=self.config.NUM_LABELS
            )
            model.to(self.device)
            return model
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to load model {self.config.MODEL_NAME}: {str(e)}")
    
    def _compute_class_weights(self, labels: List[int]) -> torch.Tensor:
        """Compute class weights with validation."""
        try:
            labels_array = np.array(labels)
            class_counts = np.bincount(labels_array, minlength=self.config.NUM_LABELS)
            
            # Inverse frequency weighting
            class_weights = 1.0 / (class_counts + 1e-8)
            class_weights = class_weights / class_weights.sum() * len(class_weights)
            
            if self.debug:
                for i, weight in enumerate(class_weights):
                    print(f"[DEBUG] Class {i} weight: {weight:.4f}")
            
            return torch.tensor(class_weights, dtype=torch.float32).to(self.device)
        except Exception as e:
            raise ValueError(f"[ERROR] Failed to compute class weights: {str(e)}")
    
    def compute_metrics(self, eval_preds) -> Dict[str, float]:
        """Compute metrics with error handling."""
        try:
            predictions, labels = eval_preds
            predictions = np.argmax(predictions, axis=1)
            
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            accuracy = accuracy_score(labels, predictions)
            precision, recall, _, _ = precision_recall_fscore_support(
                labels, predictions, average='weighted', zero_division=0
            )
            
            return {
                'accuracy': accuracy,
                'f1': f1,
                'precision': precision,
                'recall': recall
            }
        except Exception as e:
            print(f"[WARNING] compute_metrics failed: {str(e)}")
            return {'accuracy': 0.0, 'f1': 0.0}
    
    def _get_training_args(self, class_weights: Optional[torch.Tensor] = None) -> TrainingArguments:
        """
        Create TrainingArguments with version-aware filtering (transformers 4.57.6).
        """
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
            'fp16': torch.cuda.is_available()
        }
        
        # Filter to supported parameters in transformers 4.57.6
        supported = inspect.signature(TrainingArguments.__init__).parameters
        filtered = {k: v for k, v in ta_kwargs.items() if k in supported}
        ignored = set(ta_kwargs) - set(filtered)
        
        # Handle dependent parameters
        if 'evaluation_strategy' not in supported:
            for param in ['load_best_model_at_end', 'metric_for_best_model', 'greater_is_better']:
                filtered.pop(param, None)
        
        if self.debug and ignored:
            print(f"[DEBUG] Ignored unsupported parameters (v{transformers.__version__}): {ignored}")
        
        return TrainingArguments(**filtered)
    
    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
        class_weights: Optional[torch.Tensor] = None
    ) -> Dict:
        """Train model with comprehensive error handling."""
        try:
            os.makedirs(self.config.OUTPUT_DIR, exist_ok=True)
            
            training_args = self._get_training_args(class_weights)
            
            # Setup callbacks
            callbacks = []
            if getattr(training_args, 'metric_for_best_model', None):
                callbacks.append(EarlyStoppingCallback(early_stopping_patience=2))
            
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
                self.model = WeightedBertModel(self.model, class_weights)
                self.model.to(self.device)
                self.trainer.model = self.model
                if self.debug:
                    print("[DEBUG] Class weights applied")
            
            if self.debug:
                print("[DEBUG] Starting training...")
            
            train_result = self.trainer.train()
            
            if self.debug:
                print(f"[DEBUG] Training complete. Final loss: {train_result.training_loss:.4f}")
            
            return {
                'training_loss': train_result.training_loss,
                'model': self.model
            }
        
        except Exception as e:
            raise ValueError(f"[ERROR] Training failed: {str(e)}")
    
    def predict_proba(self, dataset: Dataset) -> np.ndarray:
        """Get probability predictions with error handling."""
        try:
            if self.trainer is None:
                raise ValueError("Model must be trained first")
            
            predictions = self.trainer.predict(dataset)
            logits = predictions.predictions
            
            # Ensure logits is numpy array
            if isinstance(logits, torch.Tensor):
                logits = logits.cpu().numpy()
            
            probas = nn.functional.softmax(torch.tensor(logits), dim=1).numpy()
            return probas
        
        except Exception as e:
            raise ValueError(f"[ERROR] predict_proba failed: {str(e)}")
    
    def predict(self, dataset: Dataset) -> np.ndarray:
        """Get class predictions."""
        try:
            probas = self.predict_proba(dataset)
            return np.argmax(probas, axis=1)
        except Exception as e:
            raise ValueError(f"[ERROR] predict failed: {str(e)}")
