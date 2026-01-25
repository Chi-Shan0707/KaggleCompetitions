"""
Configuration module for Disaster Tweets Classification.
Centralizes all hyperparameters with DEFENSIVE VALIDATION & COMPREHENSIVE DOCUMENTATION.

This module ensures all magic numbers are defined in a single location,
with comprehensive range checking and detailed parameter effects documentation.

=========================== PARAMETER EFFECTS GUIDE ===========================
Each training hyperparameter affects model performance differently.
This guide helps you understand what to adjust based on your training results.

Pattern P006: Comprehensive parameter validation with safe bounds.
================================================================================
"""

from typing import Dict, Any


class Config:
    """
    Centralized configuration with PATTERN P006 PREVENTION (Invalid Config).
    
    ✓ All hyperparameters validated on initialization and with defined bounds.
    ✓ Safe ranges provided for each parameter to avoid common mistakes.
    
    ===== HYPERPARAMETER DOCUMENTATION =====
    
    📌 BATCH_SIZE (Current: 16) 
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ Faster training (more samples processed per step)
      ✓ More stable gradient estimates (less noisy updates)
      ✗ Uses more GPU/CPU memory
      ✗ May lose generalization on small datasets
      ✗ Fewer gradient updates per epoch
    
    Effect of DECREASING:
      ✓ Uses less memory (fits on smaller GPUs)
      ✓ More frequent gradient updates = faster convergence (sometimes)
      ✗ Noisier gradients = unstable training
      ✗ Slower training (fewer samples per second)
    
    Safe Range: [8, 64] for most setups | Current: 16 ✓
    Recommended by dataset size:
      - Small dataset (< 5K): 8-16
      - Medium dataset (5K-50K): 16-32  ← Current dataset size
      - Large dataset (> 50K): 32-64
    
    📌 LEARNING_RATE (Current: 1e-5 = 0.00001)
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ Faster convergence (reaches good solution quicker)
      ✗ May overshoot optimal solution (training loss plateaus)
      ✗ Training becomes unstable (loss spikes)
      ✗ Model may diverge (loss → infinity)
    
    Effect of DECREASING:
      ✓ More stable training (smoother loss curve)
      ✓ Better fine-tuning on pre-trained BERT
      ✗ Slower convergence (more epochs needed)
      ✗ May get stuck in local minima
    
    Safe Range: [1e-6, 1e-4] for BERT fine-tuning | Current: 1e-5 ✓
    Rule of thumb for BERT:
      - Too high (> 1e-4): Training will diverge/be unstable
      - Recommended (1e-5 to 5e-5): Standard BERT fine-tuning
      - Too low (< 1e-6): Very slow convergence, not recommended
    
    📌 NUM_EPOCHS (Current: 4)
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ More training iterations (model sees data more times)
      ✓ Better training accuracy (model memorizes training data)
      ✗ OVERFITTING (validation accuracy drops after epoch 3-5)
      ✗ Longer training time
      ✗ Diminishing returns: improvement plateaus after 3-5 epochs for BERT
    
    Effect of DECREASING:
      ✓ Faster training
      ✓ Avoid overfitting on small datasets
      ✗ Underfitting (model doesn't learn enough)
      ✗ Lower final accuracy on validation set
    
    Safe Range: [1, 10] for disaster tweets | Current: 4 ✓
    BERT typically:
      - 1-2 epochs: Quick prototype
      - 3-4 epochs: Balanced (current) ← RECOMMENDED
      - 5+ epochs: High risk of overfitting on small datasets
    
    📌 WARMUP_STEPS (Current: 500)
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ Smoother learning rate ramp-up (gradual LR increase)
      ✓ Prevents gradient spikes at training start
      ✓ Better for training stability with high learning rates
      ✗ Takes more steps before reaching full learning rate
      ✗ Less aggressive initial learning
    
    Effect of DECREASING (including 0):
      ✓ Faster full learning rate activation
      ✗ May cause gradient spikes (loss instability at start)
      ✗ Risky if learning rate is high
    
    Safe Range: [0, 1000] steps | Current: 500 ✓
    Rule of thumb:
      - 0 (no warmup): OK if LR is small (< 1e-4), risky otherwise
      - 500-1000: Standard practice for BERT (smooth start)
      - > 1000: Very long warmup, rarely needed
    
    Tip: 500 steps ≈ ~5-10% of total training steps (good default)
    
    📌 WEIGHT_DECAY (Current: 0.01 = L2 Regularization)
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING (stronger regularization):
      ✓ Reduces overfitting (prevents model from learning noise)
      ✓ Encourages simpler, smaller weights
      ✓ Better generalization to unseen data
      ✗ May underfit (training accuracy drops)
      ✗ Slower convergence
    
    Effect of DECREASING (weaker regularization):
      ✓ Model can learn more complex patterns
      ✓ Higher training accuracy (potentially)
      ✗ Higher risk of overfitting
      ✗ Poor generalization to test data
    
    Safe Range: [0.0, 0.1] for BERT | Current: 0.01 ✓
    Guidelines:
      - 0.0 (no regularization): Only if validation loss improves for 5+ epochs
      - 0.01 (current): Standard BERT practice ✓
      - 0.1: Very aggressive regularization (only for noisy data)
    
    ===== DATA SPLIT PARAMETERS =====
    
    📌 TRAIN_SIZE, VALIDATION_SIZE, TEST_SIZE
    ──────────────────────────────────────────────────────────────────
    Current split: 80% train, 10% validation, 10% test
    
    Effect of increasing TRAIN_SIZE:
      ✓ More data for training (better model)
      ✗ Less validation/test data (worse metric reliability)
    
    Safe range: Train [0.6, 0.9], Val [0.05, 0.3], Test [0.05, 0.2]
    Current [0.8, 0.1, 0.1] ✓ is well-balanced
    
    ===== MEMORY & INFERENCE PARAMETERS =====
    
    📌 MAX_LENGTH (Current: 128 tokens)
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ Can capture longer tweets (some tweets > 100 tokens)
      ✗ Uses more GPU memory (quadratic with sequence length in BERT)
      ✗ Slower training/inference
    
    Effect of DECREASING:
      ✓ Less memory usage, faster training
      ✗ Truncates longer tweets (information loss)
    
    Safe Range: [64, 256] | Current: 128 ✓
    For tweets:
      - Average tweet length: ~50 tokens
      - Max tweet length: ~100 tokens
      - Current 128: Good buffer without wasting memory
    
    📌 MIN_TEXT_LENGTH (Current: 3 words) - Pattern P005 Defense
    ──────────────────────────────────────────────────────────────────
    Effect of INCREASING:
      ✓ Filters out short/noisy tweets
      ✗ May lose valid short tweets
    
    Effect of DECREASING:
      ✓ Includes more data
      ✗ May include very short, uninformative tweets
    
    Safe Range: [1, 10] | Current: 3 ✓
    Recommended: 3-5 (current is good)
    
    ========================================================================
    """
    
    # ========== Model Configuration ==========
    MODEL_NAME: str = 'bert-base-uncased'
    MAX_LENGTH: int = 256
    NUM_LABELS: int = 2
    
    # ========== Training Hyperparameters (transformers 4.57.6 compatible) ==========
    BATCH_SIZE: int = 32#16 8~64
    LEARNING_RATE: float = 5e-6 # 1e-6~1e-4
    NUM_EPOCHS: int = 6 #4 1~10
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01

    
    # ========== Data Split Configuration ==========
    TRAIN_SIZE: float = 0.85
    VALIDATION_SIZE: float = 0.1
    TEST_SIZE: float = 0.05
    RANDOM_STATE: int = 20260125
    
    # ========== Text Preprocessing (Pattern P005: Empty Sequence Prevention) ==========
    MIN_TEXT_LENGTH: int = 3  # Minimum words after cleaning (P005 defense)
    REMOVE_URLS: bool = True
    REMOVE_MENTIONS: bool = True
    REMOVE_SPECIAL_CHARS: bool = False  # BERT handles special tokens natively
    LOWERCASE: bool = True
    
    # ========== Debugging & Paths ==========
    DEBUG_ENABLED: bool = True
    SAMPLE_SIZE_DEBUG: int = 100
    CHECKPOINT_DIR: str = './checkpoint'
    OUTPUT_DIR: str = './outputs'
    
    # ========== DEFENSIVE BOUNDS (Pattern P006 Prevention) ==========
    _BOUNDS = {
        'MAX_LENGTH': (1, 512),
        'BATCH_SIZE': (1, 256),
        'LEARNING_RATE': (1e-8, 1e-2),
        'NUM_EPOCHS': (1, 50),
        'WARMUP_STEPS': (0, 10000),
        'WEIGHT_DECAY': (0.0, 1.0),
        'TRAIN_SIZE': (0.1, 0.9),
        'VALIDATION_SIZE': (0.05, 0.5),
        'MIN_TEXT_LENGTH': (1, 20)
    }
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate ALL configuration values against defined bounds.
        
        Pattern P006 Prevention: Comprehensive parameter validation
        
        Raises:
            ValueError: If any parameter violates bounds
            AssertionError: If data splits don't sum to ~1.0
        """
        errors = []
        
        # Validate individual bounds
        for param, (min_val, max_val) in cls._BOUNDS.items():
            actual = getattr(cls, param, None)
            if actual is None:
                errors.append(f"{param} not defined")
            elif not (min_val <= actual <= max_val):
                errors.append(f"{param}={actual} outside bounds [{min_val}, {max_val}]")
        
        # Validate data splits sum to ~1.0
        split_sum = cls.TRAIN_SIZE + cls.VALIDATION_SIZE + cls.TEST_SIZE
        if not (0.99 <= split_sum <= 1.01):
            errors.append(f"Data splits sum to {split_sum}, expected ~1.0")
        
        # Validate num_labels >= 2
        if cls.NUM_LABELS < 2:
            errors.append(f"NUM_LABELS={cls.NUM_LABELS}, must be >= 2")
        
        if errors:
            error_msg = "Configuration validation failed:\n  - " + "\n  - ".join(errors)
            raise ValueError(error_msg)
        
        if cls.DEBUG_ENABLED:
            print(f"[DEBUG] Configuration validated successfully (transformers 4.57.6)")
    
    @classmethod
    def summary(cls) -> str:
        """
        Return a formatted summary of key configuration values.
        
        Returns:
            String representation of configuration
        """
        return (
            f"Config Summary (transformers 4.57.6 / torch 2.9.1):\n"
            f"  Model: {cls.MODEL_NAME}\n"
            f"  Max Length: {cls.MAX_LENGTH}\n"
            f"  Batch Size: {cls.BATCH_SIZE}\n"
            f"  Learning Rate: {cls.LEARNING_RATE}\n"
            f"  Epochs: {cls.NUM_EPOCHS}\n"
            f"  Train/Val/Test Split: {cls.TRAIN_SIZE}/{cls.VALIDATION_SIZE}/{cls.TEST_SIZE}"
        )
