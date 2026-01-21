"""
Configuration module for Disaster Tweets Classification.
Centralizes all hyperparameters and settings.

This module ensures all magic numbers are defined in a single location,
making it easy to experiment with different configurations.
"""


class Config:
    """
    Centralized configuration for the disaster tweets pipeline.
    All hyperparameters and settings are defined as class variables.
    
    Attributes:
        MODEL_NAME: Pre-trained BERT model identifier
        MAX_LENGTH: Maximum token sequence length for BERT
        NUM_LABELS: Number of classification labels (binary: disaster/non-disaster)
        BATCH_SIZE: Training batch size
        LEARNING_RATE: Adam optimizer learning rate
        NUM_EPOCHS: Number of training epochs
        WARMUP_STEPS: Linear warmup steps for learning rate scheduler
        WEIGHT_DECAY: L2 regularization factor
        TRAIN_SIZE: Proportion of data for training
        VALIDATION_SIZE: Proportion of data for validation
        TEST_SIZE: Proportion of data for testing
        RANDOM_STATE: Seed for reproducibility
        MIN_TEXT_LENGTH: Minimum number of words after cleaning
        REMOVE_URLS: Whether to strip URLs from text
        REMOVE_MENTIONS: Whether to remove @mentions and #hashtags
        LOWERCASE: Whether to convert text to lowercase
        DEBUG_ENABLED: Whether to print debug information
        CHECKPOINT_DIR: Directory for model checkpoints
        OUTPUT_DIR: Directory for training outputs
    """
    
    # ========== Model Configuration ==========
    MODEL_NAME: str = 'bert-base-uncased'
    MAX_LENGTH: int = 128
    NUM_LABELS: int = 2
    
    # ========== Training Hyperparameters ==========
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 2e-5
    NUM_EPOCHS: int = 3
    WARMUP_STEPS: int = 500
    WEIGHT_DECAY: float = 0.01
    
    # ========== Data Split Configuration ==========
    TRAIN_SIZE: float = 0.8
    VALIDATION_SIZE: float = 0.1
    TEST_SIZE: float = 0.1
    RANDOM_STATE: int = 42
    
    # ========== Text Preprocessing ==========
    MIN_TEXT_LENGTH: int = 3
    REMOVE_URLS: bool = True
    REMOVE_MENTIONS: bool = True
    REMOVE_SPECIAL_CHARS: bool = False  # BERT handles special tokens natively
    LOWERCASE: bool = True
    
    # ========== Debugging & Paths ==========
    DEBUG_ENABLED: bool = True
    SAMPLE_SIZE_DEBUG: int = 100
    CHECKPOINT_DIR: str = './checkpoint'
    OUTPUT_DIR: str = './outputs'
    
    @classmethod
    def validate(cls) -> None:
        """
        Validate configuration values are sensible and within bounds.
        
        Raises:
            AssertionError: If any configuration value is invalid
        """
        assert cls.MAX_LENGTH > 0, "MAX_LENGTH must be positive"
        assert cls.BATCH_SIZE > 0, "BATCH_SIZE must be positive"
        assert 0 < cls.LEARNING_RATE < 1, "LEARNING_RATE should be between 0 and 1"
        assert 0 < cls.TRAIN_SIZE < 1, "TRAIN_SIZE should be between 0 and 1"
        if cls.DEBUG_ENABLED:
            print(f"[DEBUG] Configuration validated successfully")
    
    @classmethod
    def summary(cls) -> str:
        """
        Return a formatted summary of key configuration values.
        
        Returns:
            String representation of configuration
        """
        return (
            f"Config Summary:\n"
            f"  Model: {cls.MODEL_NAME}\n"
            f"  Max Length: {cls.MAX_LENGTH}\n"
            f"  Batch Size: {cls.BATCH_SIZE}\n"
            f"  Learning Rate: {cls.LEARNING_RATE}\n"
            f"  Epochs: {cls.NUM_EPOCHS}\n"
            f"  Train/Val/Test Split: {cls.TRAIN_SIZE}/{cls.VALIDATION_SIZE}/{cls.TEST_SIZE}"
        )
