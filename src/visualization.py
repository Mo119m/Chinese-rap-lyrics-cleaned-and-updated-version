# -*- coding: utf-8 -*-
"""
Visualization Module
Creates visualizations for clustering results
"""

import logging
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


class Visualizer:
    """Create visualizations for analysis results"""
    
    def __init__(self, data_dir: Path):
        """
        Initialize visualizer
        
        Args:
            data_dir: Directory containing data files
        """
        self.data_dir = Path(data_dir)
        self.merged_csv = self.data_dir / "chunks_with_umap_clusters_enriched_v2.csv"
        self.output_dir = self.data_dir / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
    
    def plot_clusters(self):
        """Plot UMAP 2D visualization with cluster colors"""
        logger.info("Creating cluster visualization")
        
        df = pd.read_csv(self.merged_csv)
        df["cluster"] = df["cluster"].astype(int)
        
        plt.figure(figsize=(10, 8))
        scatter = plt.scatter(
            df["x"],
            df["y"],
            s=3,
            c=df["cluster"],
            cmap="tab10",
            alpha=0.6
        )
        plt.colorbar(scatter, label="Cluster")
        plt.title("UMAP 2D Visualization with Clusters", fontsize=14)
        plt.xlabel("UMAP Dimension 1")
        plt.ylabel("UMAP Dimension 2")
        plt.tight_layout()
        
        output = self.output_dir / "cluster_umap.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved cluster plot: {output}")
    
    def plot_artist_distributions(self, artists: list = None, top_n: int = 5):
        """
        Plot cluster distributions for specific artists
        
        Args:
            artists: List of artist names (None = top N by chunk count)
            top_n: Number of top artists to plot if artists is None
        """
        logger.info("Creating artist distribution plots")
        
        df = pd.read_csv(self.merged_csv)
        
        # Calculate artist cluster distributions
        ac = df.groupby(["artist", "cluster"]).size().unstack(fill_value=0)
        ac = ac.div(ac.sum(1), axis=0)  # Normalize
        
        # Select artists
        if artists is None:
            artist_counts = df["artist"].value_counts()
            artists = artist_counts.head(top_n).index.tolist()
        
        # Plot each artist
        for artist in artists:
            if artist not in ac.index:
                logger.warning(f"Artist '{artist}' not found in data")
                continue
            
            vec = ac.loc[artist].sort_index()
            
            plt.figure(figsize=(10, 4))
            plt.bar(vec.index.astype(int), vec.values, color='steelblue', alpha=0.7)
            plt.title(f"Cluster Distribution — {artist}", fontsize=14)
            plt.xlabel("Cluster")
            plt.ylabel("Ratio")
            plt.ylim(0, max(vec.values) * 1.1)
            plt.grid(axis='y', alpha=0.3)
            plt.tight_layout()
            
            # Clean filename
            safe_name = artist.replace("/", "_").replace("\\", "_")
            output = self.output_dir / f"artist_dist_{safe_name}.png"
            plt.savefig(output, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Saved artist distribution: {output}")
    
    def plot_cluster_sizes(self):
        """Plot cluster size distribution"""
        logger.info("Creating cluster size plot")
        
        df = pd.read_csv(self.merged_csv)
        cluster_counts = df["cluster"].value_counts().sort_index()
        
        plt.figure(figsize=(10, 6))
        plt.bar(cluster_counts.index, cluster_counts.values, color='coral', alpha=0.7)
        plt.title("Cluster Size Distribution", fontsize=14)
        plt.xlabel("Cluster")
        plt.ylabel("Number of Chunks")
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        output = self.output_dir / "cluster_sizes.png"
        plt.savefig(output, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Saved cluster sizes: {output}")
