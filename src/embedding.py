# -*- coding: utf-8 -*-
"""
Embedding Generation Module
Generates embeddings using Qwen2.5-1.5B
"""

import logging
import math
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generate embeddings from text chunks"""
    
    def __init__(
        self,
        data_dir: Path,
        model_id: str = "Qwen/Qwen2.5-1.5B",
        batch_size: int = 16,
        max_length: int = 512
    ):
        """
        Initialize embedding generator
        
        Args:
            data_dir: Directory containing data files
            model_id: HuggingFace model ID
            batch_size: Batch size for encoding
            max_length: Maximum sequence length
        """
        self.data_dir = Path(data_dir)
        self.model_id = model_id
        self.batch_size = batch_size
        self.max_length = max_length
        
        self.chunks_csv = self.data_dir / "lyrics_chunks_enriched.csv"
        self.emb_npy = self.data_dir / "qwen15b_embeddings_v2.npy"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        logger.info(f"Loading model: {model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            use_fast=True,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
        self.model.eval()
        
        if torch.cuda.is_available():
            torch.set_float32_matmul_precision("high")
        
        logger.info("Model loaded successfully")
    
    @torch.inference_mode()
    def encode_batch(self, batch: List[str]) -> np.ndarray:
        """
        Encode a batch of texts to embeddings
        
        Args:
            batch: List of text strings
        
        Returns:
            Numpy array of embeddings [batch_size, hidden_dim]
        """
        enc = self.tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length
        ).to(self.device)
        
        out = self.model(**enc)
        hidden = out.last_hidden_state  # [B, T, H]
        mask = enc["attention_mask"].unsqueeze(-1)
        
        # Mean pooling
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        
        # L2 normalization
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        
        return pooled.float().cpu().numpy()
    
    def generate_embeddings(self) -> np.ndarray:
        """
        Generate embeddings for all text chunks
        
        Returns:
            Numpy array of all embeddings
        """
        logger.info("Starting embedding generation")
        
        if not self.chunks_csv.exists():
            raise FileNotFoundError(
                f"Chunks CSV not found: {self.chunks_csv}. "
                "Run data preprocessing first."
            )
        
        df = pd.read_csv(self.chunks_csv)
        texts = df["text"].astype(str).tolist()
        
        logger.info(f"Processing {len(texts)} text chunks")
        
        vecs = []
        for i in tqdm(range(0, len(texts), self.batch_size), desc="Encoding"):
            batch = texts[i:i + self.batch_size]
            vecs.append(self.encode_batch(batch))
        
        Z = np.vstack(vecs).astype("float32")
        np.save(self.emb_npy, Z)
        
        logger.info(f"Saved embeddings: {self.emb_npy} with shape {Z.shape}")
        return Z
