"""
Data preprocessing module with COMPREHENSIVE VALIDATION (Patterns P004 & P005).

Pattern P004: Validates CSV data completeness
Pattern P005: Handles empty sequences after cleaning with fallback
Pattern Prevention: Every function includes boundary & null checks
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
    Production-ready data preprocessor with DEFENSIVE PROGRAMMING.
    
    Patterns prevented:
    - P004: Missing/corrupted data detection
    - P005: Empty sequence handling with fallback
    """
    
    def __init__(self, config: Config):
        """Initialize with validation."""
        self.config = config
        self.tokenizer = BertTokenizer.from_pretrained(config.MODEL_NAME)
        self.debug = config.DEBUG_ENABLED
        
    def clean_text(self, text: str) -> str:
        """
        Clean text with P005 defense: handles None/NaN → fallback string.
        """
        # P005: Handle None/NaN gracefully
        if text is None or pd.isna(text):
            if self.debug:
                print(f"[DEBUG] Encountered NaN value, returning fallback")
            return "[EMPTY]"  # Fallback to avoid empty sequences
        
        text = str(text).strip()
        
        # P005: Check for already-empty input
        if not text or len(text) < 2:
            return "[EMPTY]"
        
        # Remove URLs
        if self.config.REMOVE_URLS:
            text = re.sub(r'http\S+|www\.\S+|ftp\S+', '', text)
        
        # Remove @mentions and hashtag symbols
        if self.config.REMOVE_MENTIONS:
            text = re.sub(r'@\w+', '', text)
            text = re.sub(r'#', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # P005: Check if cleaning left us with empty string
        if not text:
            return "[EMPTY]"
        
        # Lowercase
        if self.config.LOWERCASE:
            text = text.lower()
        
        return text
    
    def validate_text(self, text: str) -> bool:
        """
        Validate text meets minimum quality.
        Pattern P005: Explicit check for [EMPTY] fallback marker
        """
        if text == "[EMPTY]" or not text:
            return False
        
        word_count = len(text.split())
        return word_count >= self.config.MIN_TEXT_LENGTH
    
    def prepare_dataset(
        self,
        df: pd.DataFrame,
        text_col: str = 'text',
        label_col: str = 'target',
        split_type: str = 'train'
    ) -> Tuple[Dataset, Dict]:
        """
        Prepare dataset with COMPREHENSIVE DATA VALIDATION (Pattern P004).
        
        P004 Prevention:
        - Validate required columns exist
        - Check for missing values explicitly
        - Verify data types before processing
        - Report statistics for audit trail
        """
        # ===== STEP 1: VALIDATE INPUT DataFrame (P004) =====
        if df is None or df.empty:
            raise ValueError(f"[ERROR] Input DataFrame is empty or None (split_type={split_type})")
        
        if text_col not in df.columns:
            raise ValueError(f"[ERROR] Column '{text_col}' not found. Available: {df.columns.tolist()}")
        
        if label_col in df.columns and split_type != 'inference':
            pass  # Labels expected for train/val
        elif split_type != 'inference':
            if self.debug:
                print(f"[WARNING] No label column '{label_col}' for {split_type} split")
        
        df = df.copy()
        initial_count = len(df)
        
        # ===== STEP 2: REPORT INITIAL DATA STATE (P004 audit) =====
        if self.debug:
            print(f"\n[DATA_AUDIT] {split_type.upper()} SET INITIAL STATE:")
            print(f"  Shape: {df.shape}")
            print(f"  Text column null count: {df[text_col].isna().sum()} / {len(df)}")
            if label_col in df.columns:
                print(f"  Label column null count: {df[label_col].isna().sum()} / {len(df)}")
        
        # ===== STEP 3: DROP ROWS WITH MISSING TEXT (P004) =====
        df = df.dropna(subset=[text_col])
        dropped_na = initial_count - len(df)
        if dropped_na > 0 and self.debug:
            print(f"  Dropped {dropped_na} rows with missing text")
        
        if df.empty:
            raise ValueError(f"[ERROR] All rows removed due to missing text (split_type={split_type})")
        
        # ===== STEP 4: CLEAN TEXT + VALIDATE (P005) =====
        df['cleaned_text'] = df[text_col].apply(self.clean_text)
        
        # Mark valid rows
        df['is_valid'] = df['cleaned_text'].apply(self.validate_text)
        invalid_rows = (~df['is_valid']).sum()
        
        if invalid_rows > 0 and self.debug:
            print(f"  Marked {invalid_rows} rows as invalid (too short)")
        
        # Filter out invalid rows
        df = df[df['is_valid']].drop('is_valid', axis=1)
        
        if df.empty:
            raise ValueError(f"[ERROR] All rows removed after validation (split_type={split_type})")
        
        # ===== STEP 5: TOKENIZE WITH VALIDATION =====
        try:
            encodings = self.tokenizer(
                df['cleaned_text'].tolist(),
                max_length=self.config.MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors=None
            )
        except Exception as e:
            raise ValueError(f"[ERROR] Tokenization failed: {str(e)}")
        
        # Verify tokenization didn't fail
        if not encodings or 'input_ids' not in encodings:
            raise ValueError("[ERROR] Tokenizer returned invalid output")
        
        # ===== STEP 6: REPORT TOKEN DISTRIBUTION =====
        if self.debug:
            token_lens = [len(ids) for ids in encodings['input_ids']]
            print(f"  Token lengths - Min: {min(token_lens)}, Max: {max(token_lens)}, Mean: {np.mean(token_lens):.1f}")
        
        # ===== STEP 7: PREPARE DATASET DICT =====
        dataset_dict = {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'token_type_ids': encodings.get('token_type_ids', [[0]*len(ids) for ids in encodings['input_ids']]),
        }
        
        # Add labels if present
        if label_col in df.columns:
            try:
                labels = df[label_col].astype(int).tolist()
                dataset_dict['labels'] = labels
                
                if self.debug:
                    class_dist = pd.Series(labels).value_counts().sort_index().to_dict()
                    print(f"  Class distribution: {class_dist}")
            except Exception as e:
                if self.debug:
                    print(f"[WARNING] Could not process labels: {str(e)}")
        
        # ===== STEP 8: CREATE HF DATASET =====
        # Preserve original identifiers and indices to allow mapping back to the
        # original DataFrame ordering (useful for submission generation)
        if 'id' in df.columns:
            dataset_dict['id'] = df['id'].tolist()

        # Preserve original DataFrame index values for mapping
        dataset_dict['orig_index'] = df.index.tolist()

        hf_dataset = Dataset.from_dict(dataset_dict)
        
        stats = {
            'initial_rows': initial_count,
            'final_rows': len(hf_dataset),
            'rows_dropped': initial_count - len(hf_dataset),
            'split_type': split_type,
        }
        
        if self.debug:
            print(f"  Final dataset: {len(hf_dataset)} samples")
            print(f"[DATA_AUDIT] ✓ COMPLETE\n")
        
        return hf_dataset, stats
