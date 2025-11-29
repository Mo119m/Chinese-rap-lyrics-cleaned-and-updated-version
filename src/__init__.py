# -*- coding: utf-8 -*-
"""
Chinese Rap Lyrics Analysis Pipeline
"""

__version__ = "2.0.0"

from .data_processing import DataProcessor
from .embedding import EmbeddingGenerator
from .contrastive_learning import ContrastiveLearner
from .clustering import ClusterAnalyzer
from .downstream_analysis import DownstreamAnalyzer
from .visualization import Visualizer

__all__ = [
    'DataProcessor',
    'EmbeddingGenerator',
    'ContrastiveLearner',
    'ClusterAnalyzer',
    'DownstreamAnalyzer',
    'Visualizer',
]
