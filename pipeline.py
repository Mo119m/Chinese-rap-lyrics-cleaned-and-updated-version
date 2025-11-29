#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Chinese Rap Lyrics Analysis Pipeline
Main orchestrator for the complete analysis workflow
"""

import argparse
import logging
from pathlib import Path
from typing import Optional

from src.data_processing import DataProcessor
from src.embedding import EmbeddingGenerator
from src.contrastive_learning import ContrastiveLearner
from src.clustering import ClusterAnalyzer
from src.downstream_analysis import DownstreamAnalyzer
from src.visualization import Visualizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RapLyricsPipeline:
    """Complete pipeline for Chinese rap lyrics analysis"""
    
    def __init__(self, base_dir: Path, config: Optional[dict] = None):
        """
        Initialize the pipeline
        
        Args:
            base_dir: Base directory for data storage
            config: Configuration dictionary for pipeline parameters
        """
        self.base_dir = Path(base_dir)
        self.data_dir = self.base_dir / "transformed_data"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Default configuration
        self.config = {
            'model_id': 'Qwen/Qwen2.5-1.5B',
            'max_length': 512,
            'batch_size': 16,
            'contrastive': {
                'epochs': 10,
                'batch_size': 1024,
                'temperature': 0.2,
                'hidden_dim': 512,
                'proj_dim': 128,
                'learning_rate': 1e-3,
                'noise_std': 0.05,
                'dropout': 0.05
            },
            'clustering': {
                'pca_components': 50,
                'umap_neighbors': 15,
                'umap_min_dist': 0.1,
                'min_clusters': 2,
                'max_clusters': 12
            },
            'downstream': {
                'top_artists': 10,
                'representative_samples': 20,
                'top_keywords': 30
            }
        }
        
        # Update with user config
        if config:
            self._update_config(config)
        
        # Initialize components
        self.data_processor = DataProcessor(self.data_dir)
        self.embedding_gen = EmbeddingGenerator(
            self.data_dir,
            model_id=self.config['model_id'],
            batch_size=self.config['batch_size'],
            max_length=self.config['max_length']
        )
        self.contrastive_learner = ContrastiveLearner(
            self.data_dir,
            **self.config['contrastive']
        )
        self.cluster_analyzer = ClusterAnalyzer(
            self.data_dir,
            **self.config['clustering']
        )
        self.downstream_analyzer = DownstreamAnalyzer(
            self.data_dir,
            **self.config['downstream']
        )
        self.visualizer = Visualizer(self.data_dir)
    
    def _update_config(self, config: dict):
        """Recursively update configuration"""
        for key, value in config.items():
            if isinstance(value, dict) and key in self.config:
                self.config[key].update(value)
            else:
                self.config[key] = value
    
    def run(self, 
            skip_preprocessing: bool = False,
            skip_embedding: bool = False,
            skip_contrastive: bool = False,
            skip_clustering: bool = False,
            skip_downstream: bool = False,
            skip_visualization: bool = False):
        """
        Run the complete pipeline
        
        Args:
            skip_preprocessing: Skip data preprocessing
            skip_embedding: Skip embedding generation
            skip_contrastive: Skip contrastive learning
            skip_clustering: Skip clustering
            skip_downstream: Skip downstream analysis
            skip_visualization: Skip visualization
        """
        logger.info("=" * 80)
        logger.info("Starting Chinese Rap Lyrics Analysis Pipeline")
        logger.info("=" * 80)
        
        # Step 1: Data Preprocessing
        if not skip_preprocessing:
            logger.info("\n[Step 1/6] Data Preprocessing")
            self.data_processor.parse_lyrics()
            self.data_processor.create_chunks()
        else:
            logger.info("\n[Step 1/6] Skipping data preprocessing")
        
        # Step 2: Embedding Generation
        if not skip_embedding:
            logger.info("\n[Step 2/6] Embedding Generation")
            self.embedding_gen.generate_embeddings()
        else:
            logger.info("\n[Step 2/6] Skipping embedding generation")
        
        # Step 3: Contrastive Learning
        if not skip_contrastive:
            logger.info("\n[Step 3/6] Contrastive Learning")
            self.contrastive_learner.train()
        else:
            logger.info("\n[Step 3/6] Skipping contrastive learning")
        
        # Step 4: Clustering
        if not skip_clustering:
            logger.info("\n[Step 4/6] Clustering Analysis")
            self.cluster_analyzer.reduce_dimensions()
            self.cluster_analyzer.perform_clustering()
            self.cluster_analyzer.merge_results()
        else:
            logger.info("\n[Step 4/6] Skipping clustering")
        
        # Step 5: Downstream Analysis
        if not skip_downstream:
            logger.info("\n[Step 5/6] Downstream Analysis")
            self.downstream_analyzer.analyze_top_artists()
            self.downstream_analyzer.find_representatives()
            self.downstream_analyzer.extract_keywords()
            self.downstream_analyzer.analyze_emotions()
            self.downstream_analyzer.create_song_vectors()
            self.downstream_analyzer.create_artist_vectors()
            self.downstream_analyzer.generate_report()
        else:
            logger.info("\n[Step 5/6] Skipping downstream analysis")
        
        # Step 6: Visualization
        if not skip_visualization:
            logger.info("\n[Step 6/6] Visualization")
            self.visualizer.plot_clusters()
            self.visualizer.plot_artist_distributions()
        else:
            logger.info("\n[Step 6/6] Skipping visualization")
        
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Chinese Rap Lyrics Analysis Pipeline'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        required=True,
        help='Base directory for data storage'
    )
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step'
    )
    parser.add_argument(
        '--skip-embedding',
        action='store_true',
        help='Skip embedding generation step'
    )
    parser.add_argument(
        '--skip-contrastive',
        action='store_true',
        help='Skip contrastive learning step'
    )
    parser.add_argument(
        '--skip-clustering',
        action='store_true',
        help='Skip clustering step'
    )
    parser.add_argument(
        '--skip-downstream',
        action='store_true',
        help='Skip downstream analysis step'
    )
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='Skip visualization step'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to custom configuration JSON file'
    )
    
    args = parser.parse_args()
    
    # Load custom config if provided
    config = None
    if args.config:
        import json
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Initialize and run pipeline
    pipeline = RapLyricsPipeline(args.base_dir, config)
    pipeline.run(
        skip_preprocessing=args.skip_preprocessing,
        skip_embedding=args.skip_embedding,
        skip_contrastive=args.skip_contrastive,
        skip_clustering=args.skip_clustering,
        skip_downstream=args.skip_downstream,
        skip_visualization=args.skip_visualization
    )


if __name__ == '__main__':
    main()
