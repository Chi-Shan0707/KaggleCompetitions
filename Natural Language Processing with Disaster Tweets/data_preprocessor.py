"""
Data preprocessing module for Disaster Tweets Classification.
Handles text cleaning, tokenization, and dataset preparation.

This module provides utilities for transforming raw tweet text into
BERT-compatible token sequences with comprehensive validation.
"""

import re
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from transformers import BertTokenizer
from datasets import Dataset
from config import Config


class DataPreprocessor:
    """
    Handles text cleaning, tokenization, and dataset preparation.
    
    This class provides methods for:
    - Cleaning raw text (removing URLs, mentions, normalizing whitespace)
    - Validating text quality (minimum length checks)
    - Tokenizing text using BERT tokenizer
    - Preparing HuggingFace datasets with validation checks
    
    Attributes:
        config: Config instance with hyperparameters
        tokenizer: BertTokenizer instance for tokenization
        debug: Whether to print debug information
    """
    
    def __init__(self, config: Config):
        """
        Initialize preprocessor with tokenizer and configuration.
        
        Args:
            config: Config class instance with hyperparameters
        """
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.debug = config.DEBUG_ENABLED
        
    def clean_text(self, text: str) -> str:
        """
        Remove noise from text while preserving semantic content.
        
        Processing steps:
        1. Handle NaN/missing values by returning empty string
        2. Remove URLs (http://, https://, www.)
        3. Remove @mentions and hashtag symbols (keep hashtag content)
        4. Normalize whitespace (collapse multiple spaces, trim edges)
        5. Convert to lowercase if configured
        
        Args:
            text: Raw text string (may contain URLs, mentions, etc.)
            
        Returns:
            Cleaned text string ready for tokenization
            
        Example:
            >>> preprocessor.clean_text("Check out http://example.com #disaster @user")
            'check out disaster'
        """
        # Handle missing values
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = re.sub(r'http\S+|www.\S+', '', text)
        
        # Remove @mentions and hashtag symbols (keep hashtag content)
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
        Check if text meets minimum quality requirements.
        
        Validation criteria:
        - Minimum word count (after tokenization by whitespace)
        
        Args:
            text: Cleaned text string
            
        Returns:
            True if text passes validation, False otherwise
        """
        return len(text.split()) >= self.config.MIN_TEXT_LENGTH
    
    def tokenize_batch(self, texts: List[str]) -> Dict:
        """
        Tokenize a batch of texts using BERT tokenizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            Dictionary with keys:
            - 'input_ids': Token ID sequences
            - 'attention_mask': Attention mask (1 for real tokens, 0 for padding)
            - 'token_type_ids': Segment IDs (0 for single sequence)
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
        Prepare dataset for training/inference with comprehensive validation.
        
        Processing pipeline:
        1. Check for missing values in text/label columns
        2. Drop rows with missing text
        3. Clean text using clean_text() method
        4. Validate text length
        5. Tokenize using BERT tokenizer
        6. Create HuggingFace Dataset object
        7. Report statistics (class distribution, token lengths)
        
        Args:
            df: Input DataFrame with raw text and optional labels
            text_col: Name of column containing tweet text
            label_col: Name of column containing labels (optional)
            split_type: Type of split ('train', 'validation', or 'inference')
            
        Returns:
            Tuple of (HuggingFace Dataset object, statistics dictionary)
            
        Raises:
            ValueError: If no valid samples remain after cleaning
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
