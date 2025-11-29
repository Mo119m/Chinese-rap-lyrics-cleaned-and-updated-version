# -*- coding: utf-8 -*-
"""
Contrastive Learning Module
Implements SimCSE-based contrastive learning
"""

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ContrastiveLearner:
    """Contrastive learning for embeddings"""
    
    def __init__(
        self,
        data_dir: Path,
        epochs: int = 10,
        batch_size: int = 1024,
        temperature: float = 0.2,
        hidden_dim: int = 512,
        proj_dim: int = 128,
        learning_rate: float = 1e-3,
        noise_std: float = 0.05,
        dropout: float = 0.05
    ):
        """
        Initialize contrastive learner
        
        Args:
            data_dir: Directory containing data files
            epochs: Number of training epochs
            batch_size: Training batch size
            temperature: Temperature for NT-Xent loss
            hidden_dim: Hidden dimension for projection head
            proj_dim: Output projection dimension
            learning_rate: Learning rate
            noise_std: Standard deviation for noise augmentation
            dropout: Dropout probability for augmentation
        """
        self.data_dir = Path(data_dir)
        self.epochs = epochs
        self.batch_size = batch_size
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.proj_dim = proj_dim
        self.learning_rate = learning_rate
        self.noise_std = noise_std
        self.dropout = dropout
        
        self.emb_npy = self.data_dir / "qwen15b_embeddings_v2.npy"
        self.contrast_npy = self.data_dir / "Z_contrastive_v2.npy"
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
    
    def augment(self, x: np.ndarray) -> np.ndarray:
        """
        Apply augmentation to embeddings
        
        Args:
            x: Input embeddings
        
        Returns:
            Augmented embeddings
        """
        # Add Gaussian noise
        z = x + np.random.normal(0, self.noise_std, size=x.shape).astype("float32")
        
        # Random dropout
        if self.dropout > 0:
            mask = (np.random.rand(*x.shape) > self.dropout).astype("float32")
            z = z * mask
        
        return z
    
    def nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        NT-Xent (Normalized Temperature-scaled Cross Entropy) loss
        
        Args:
            z1: First view embeddings [B, D]
            z2: Second view embeddings [B, D]
        
        Returns:
            Loss scalar
        """
        B = z1.size(0)
        z = torch.cat([z1, z2], dim=0)  # [2B, D]
        
        # Compute similarity matrix
        sim = z @ z.T  # [2B, 2B]
        
        # Mask out self-similarity
        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim = sim.masked_fill(mask, -9e15)
        
        # Positive pairs
        pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # [2B]
        
        # Logits: [positive, all negatives]
        logits = torch.cat([pos.unsqueeze(1), sim], dim=1) / self.temperature
        
        # Labels: position 0 is the positive
        labels = torch.zeros(2 * B, dtype=torch.long, device=z.device)
        
        return F.cross_entropy(logits, labels)
    
    def train(self) -> np.ndarray:
        """
        Train contrastive learning model
        
        Returns:
            Learned embeddings
        """
        logger.info("Starting contrastive learning")
        
        if not self.emb_npy.exists():
            raise FileNotFoundError(
                f"Embeddings not found: {self.emb_npy}. "
                "Generate embeddings first."
            )
        
        # Load embeddings
        X = np.load(self.emb_npy).astype("float32")
        N, D = X.shape
        logger.info(f"Loaded embeddings: shape={X.shape}")
        
        # Projection head
        class Head(nn.Module):
            def __init__(self, in_dim, hidden, out_dim):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.ReLU(True),
                    nn.Linear(hidden, out_dim)
                )
            
            def forward(self, x):
                return F.normalize(self.net(x), dim=1)
        
        head = Head(D, self.hidden_dim, self.proj_dim).to(self.device)
        optimizer = torch.optim.Adam(head.parameters(), lr=self.learning_rate)
        
        # Training loop
        def iter_indices(n, shuffle=True):
            idx = np.arange(n)
            if shuffle:
                np.random.shuffle(idx)
            for i in range(0, n, self.batch_size):
                yield idx[i:i + self.batch_size]
        
        for epoch in range(1, self.epochs + 1):
            losses = []
            for idx in iter_indices(N):
                xb = X[idx]
                
                # Create two augmented views
                v1 = torch.tensor(self.augment(xb), device=self.device)
                v2 = torch.tensor(self.augment(xb), device=self.device)
                
                # Forward pass
                z1 = head(v1)
                z2 = head(v2)
                
                # Compute loss
                loss = self.nt_xent_loss(z1, z2)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                losses.append(loss.item())
            
            avg_loss = np.mean(losses)
            logger.info(f"Epoch {epoch:02d}/{self.epochs} | Loss: {avg_loss:.4f}")
        
        # Generate final embeddings
        logger.info("Generating contrastive embeddings")
        head.eval()
        outs = []
        with torch.no_grad():
            for idx in iter_indices(N, shuffle=False):
                xb = torch.tensor(X[idx], device=self.device)
                zb = head(xb).cpu().numpy()
                outs.append(zb)
        
        Z_contrast = np.vstack(outs).astype("float32")
        np.save(self.contrast_npy, Z_contrast)
        
        logger.info(f"Saved contrastive embeddings: {self.contrast_npy} "
                   f"with shape {Z_contrast.shape}")
        
        return Z_contrast
