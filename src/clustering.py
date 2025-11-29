# -*- coding: utf-8 -*-
"""
Clustering Module
Performs dimensionality reduction and clustering
"""

import logging
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import umap

logger = logging.getLogger(__name__)


class ClusterAnalyzer:
    """Clustering analysis with UMAP and KMeans"""
    
    def __init__(
        self,
        data_dir: Path,
        pca_components: int = 50,
        umap_neighbors: int = 15,
        umap_min_dist: float = 0.1,
        min_clusters: int = 2,
        max_clusters: int = 12
    ):
        """
        Initialize cluster analyzer
        
        Args:
            data_dir: Directory containing data files
            pca_components: Number of PCA components
            umap_neighbors: UMAP n_neighbors parameter
            umap_min_dist: UMAP min_dist parameter
            min_clusters: Minimum number of clusters to try
            max_clusters: Maximum number of clusters to try
        """
        self.data_dir = Path(data_dir)
        self.pca_components = pca_components
        self.umap_neighbors = umap_neighbors
        self.umap_min_dist = umap_min_dist
        self.min_clusters = min_clusters
        self.max_clusters = max_clusters
        
        self.contrast_npy = self.data_dir / "Z_contrastive_v2.npy"
        self.emb_npy = self.data_dir / "qwen15b_embeddings_v2.npy"
        self.umap_csv = self.data_dir / "umap_2d_v2.csv"
        self.clusters_csv = self.data_dir / "clusters_v2.csv"
        self.merged_csv = self.data_dir / "chunks_with_umap_clusters_enriched_v2.csv"
        self.chunks_csv = self.data_dir / "lyrics_chunks_enriched.csv"
    
    def reduce_dimensions(self) -> tuple:
        """
        Perform dimensionality reduction with PCA and UMAP
        
        Returns:
            Tuple of (2D UMAP coordinates, original embeddings)
        """
        logger.info("Starting dimensionality reduction")
        
        # Load embeddings (prefer contrastive if available)
        if self.contrast_npy.exists():
            Z = np.load(self.contrast_npy).astype("float32")
            logger.info(f"Using contrastive embeddings: {Z.shape}")
        else:
            Z = np.load(self.emb_npy).astype("float32")
            logger.info(f"Using base embeddings: {Z.shape}")
        
        # PCA reduction
        logger.info(f"Applying PCA to {self.pca_components} components")
        pca = PCA(n_components=self.pca_components, random_state=42)
        Z_pca = pca.fit_transform(Z)
        
        # UMAP reduction
        logger.info("Applying UMAP to 2D")
        um = umap.UMAP(
            n_components=2,
            n_neighbors=self.umap_neighbors,
            min_dist=self.umap_min_dist,
            metric="cosine",
            random_state=42
        )
        Z_2d = um.fit_transform(Z_pca)
        
        # Save 2D coordinates
        df_umap = pd.DataFrame({"x": Z_2d[:, 0], "y": Z_2d[:, 1]})
        df_umap.to_csv(self.umap_csv, index=False)
        
        logger.info(f"Saved UMAP coordinates: {self.umap_csv}")
        
        return Z_2d, Z
    
    def perform_clustering(self) -> pd.DataFrame:
        """
        Perform KMeans clustering with automatic K selection
        
        Returns:
            DataFrame with cluster labels
        """
        logger.info("Starting clustering analysis")
        
        # Load embeddings
        if self.contrast_npy.exists():
            Z = np.load(self.contrast_npy).astype("float32")
        else:
            Z = np.load(self.emb_npy).astype("float32")
        
        # Try different K values
        logger.info(f"Testing K from {self.min_clusters} to {self.max_clusters}")
        
        best_k, best_score, best_model = None, -1, None
        scores = []
        
        for k in range(self.min_clusters, self.max_clusters + 1):
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(Z)
            
            try:
                score = silhouette_score(Z, labels)
            except Exception as e:
                logger.warning(f"Failed to compute silhouette for k={k}: {e}")
                score = -1
            
            scores.append({"k": k, "silhouette": score})
            logger.info(f"K={k}: silhouette={score:.3f}")
            
            if score > best_score:
                best_k, best_score, best_model = k, score, km
        
        # Save scores
        pd.DataFrame(scores).to_csv(
            self.data_dir / "silhouette_scores_v2.csv",
            index=False
        )
        
        # Generate final labels
        labels = best_model.predict(Z)
        df_clusters = pd.DataFrame({"cluster": labels})
        df_clusters.to_csv(self.clusters_csv, index=False)
        
        logger.info(f"Best K={best_k} (silhouette={best_score:.3f})")
        logger.info(f"Saved clusters: {self.clusters_csv}")
        
        return df_clusters
    
    def merge_results(self) -> pd.DataFrame:
        """
        Merge chunks, UMAP coordinates, and cluster labels
        
        Returns:
            Merged DataFrame
        """
        logger.info("Merging results")
        
        df_chunks = pd.read_csv(self.chunks_csv)
        df_umap = pd.read_csv(self.umap_csv)
        df_clusters = pd.read_csv(self.clusters_csv)
        
        # Verify lengths match
        assert len(df_chunks) == len(df_umap) == len(df_clusters), \
            "Length mismatch between chunks, UMAP, and clusters"
        
        # Concatenate
        df_merged = pd.concat(
            [
                df_chunks.reset_index(drop=True),
                df_umap.reset_index(drop=True),
                df_clusters.reset_index(drop=True)
            ],
            axis=1
        )
        
        df_merged.to_csv(self.merged_csv, index=False)
        
        logger.info(f"Saved merged results: {self.merged_csv} ({len(df_merged)} rows)")
        
        return df_merged
