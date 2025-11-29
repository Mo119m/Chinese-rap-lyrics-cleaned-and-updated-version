# Chinese Rap Lyrics Analysis Pipeline

A comprehensive machine learning pipeline for semantic clustering and analysis of Chinese rap lyrics using large language models and contrastive learning.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Pipeline Stages](#pipeline-stages)
- [Configuration](#configuration)
- [Output Files](#output-files)
- [Advanced Usage](#advanced-usage)
- [Research Background](#research-background)
- [Citation](#citation)
- [License](#license)

## 🎯 Overview

This project implements a complete pipeline for analyzing Chinese rap lyrics through:

1. **Text Processing**: Parse and chunk raw lyrics into analyzable segments
2. **Embedding Generation**: Convert lyrics to high-dimensional semantic embeddings using Qwen2.5-1.5B
3. **Contrastive Learning**: Enhance embeddings using SimCSE-based contrastive learning
4. **Clustering**: Perform dimensionality reduction (PCA + UMAP) and semantic clustering (KMeans)
5. **Analysis**: Extract keywords, identify representative samples, analyze emotions, and create similarity search indices

**Dataset**: 15,000+ songs from 325+ Chinese rap artists

## ✨ Features

- **Modular Pipeline**: Clean, maintainable architecture with separate modules for each stage
- **Resume Capability**: Skip completed stages when re-running
- **Contrastive Learning**: SimCSE framework with NT-Xent loss for improved semantic clustering
- **Dimensionality Reduction**: PCA → UMAP pipeline for effective visualization
- **Comprehensive Analysis**:
  - Cluster-specific keyword extraction (c-TF-IDF)
  - Representative sample identification
  - Top artist analysis per cluster
  - Emotion/sentiment labeling
  - Song and artist similarity search
- **Visualization**: UMAP plots, artist distributions, cluster sizes
- **Configurable**: JSON-based configuration for hyperparameters

## 🏗️ Architecture

```
pipeline.py                    # Main orchestrator
├── src/
│   ├── data_processing.py     # Parse and chunk lyrics
│   ├── embedding.py           # Generate Qwen embeddings
│   ├── contrastive_learning.py # SimCSE contrastive learning
│   ├── clustering.py          # PCA + UMAP + KMeans
│   ├── downstream_analysis.py # Keywords, emotions, similarity
│   └── visualization.py       # Plot results
```

### Pipeline Flow

```
Raw Lyrics (TXT)
    ↓
[1] Data Processing → lyrics_chunks_enriched.csv
    ↓
[2] Embedding Generation → qwen15b_embeddings_v2.npy (1536-dim)
    ↓
[3] Contrastive Learning → Z_contrastive_v2.npy (128-dim)
    ↓
[4] Clustering → umap_2d_v2.csv + clusters_v2.csv
    ↓
[5] Downstream Analysis → keywords, emotions, representatives
    ↓
[6] Visualization → plots in visualizations/
```

## 📦 Installation

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended, CPU mode supported)
- 16GB+ RAM
- ~10GB disk space for models and data

### Setup

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/chinese-rap-lyrics-analysis.git
cd chinese-rap-lyrics-analysis
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Install jieba dictionary** (for Chinese tokenization):
```bash
python -c "import jieba; jieba.initialize()"
```

## 🚀 Quick Start

### 1. Prepare Your Data

Place your raw lyrics file in the data directory:

```
/path/to/data/
└── transformed_data/
    └── all_lyrics.txt
```

**Format**: The lyrics file should follow this structure:
```
歌手姓名：Artist Name
Song Title
作词：Lyricist Name
[Metadata lines...]

Lyrics content here...

歌手姓名：Next Artist
...
```

### 2. Run the Complete Pipeline

```bash
python pipeline.py --base-dir /path/to/data
```

This will execute all six stages sequentially.

### 3. View Results

Results will be saved in `/path/to/data/transformed_data/`:
- `chunks_with_umap_clusters_enriched_v2.csv` - Main results file
- `cluster_report_v2.md` - Comprehensive analysis report
- `visualizations/` - All plots and figures

## 📊 Pipeline Stages

### Stage 1: Data Processing

Parses raw lyrics and creates text chunks suitable for embedding.

```bash
# Run only this stage
python pipeline.py --base-dir /path/to/data \
    --skip-embedding --skip-contrastive \
    --skip-clustering --skip-downstream --skip-visualization
```

**Outputs**:
- `lyrics_master.csv` - Parsed songs with metadata
- `lyrics_chunks_enriched.csv` - Text chunks (15-1500 chars)

### Stage 2: Embedding Generation

Generates semantic embeddings using Qwen2.5-1.5B (1536-dimensional).

**Outputs**:
- `qwen15b_embeddings_v2.npy` - Shape: [N_chunks, 1536]

### Stage 3: Contrastive Learning

Applies SimCSE-based contrastive learning to enhance embeddings.

**Key Parameters**:
- Temperature: 0.2
- Projection dimension: 128
- Training epochs: 10
- Augmentation: Gaussian noise (σ=0.05) + dropout (p=0.05)

**Outputs**:
- `Z_contrastive_v2.npy` - Shape: [N_chunks, 128]

### Stage 4: Clustering

Reduces dimensions and performs clustering:
1. PCA: 1536 → 50 dimensions
2. UMAP: 50 → 2 dimensions
3. KMeans: Automatic K selection (2-12) via silhouette score

**Outputs**:
- `umap_2d_v2.csv` - 2D coordinates for visualization
- `clusters_v2.csv` - Cluster assignments
- `chunks_with_umap_clusters_enriched_v2.csv` - Merged results

### Stage 5: Downstream Analysis

Performs comprehensive cluster analysis:

**Outputs**:
- `cluster_keywords_v2.csv` - Top 30 keywords per cluster (c-TF-IDF)
- `cluster_top_artists_v2.csv` - Top 10 artists per cluster
- `cluster_representatives_v2.csv` - 20 representative samples per cluster
- `cluster_emotions_v2.csv` - Emotion analysis per cluster
- `song_vectors_v2.npy` - Song-level embeddings
- `artist_vectors_v2.npy` - Artist-level embeddings
- `cluster_report_v2.md` - Comprehensive markdown report

### Stage 6: Visualization

Creates plots and visualizations:

**Outputs** (in `visualizations/`):
- `cluster_umap.png` - UMAP 2D scatter plot with cluster colors
- `cluster_sizes.png` - Bar chart of cluster sizes
- `artist_dist_*.png` - Artist-specific cluster distributions

## ⚙️ Configuration

### Custom Configuration File

Create a JSON config file to override defaults:

```json
{
  "model_id": "Qwen/Qwen2.5-1.5B",
  "batch_size": 32,
  "max_length": 512,
  "contrastive": {
    "epochs": 15,
    "temperature": 0.15,
    "learning_rate": 5e-4
  },
  "clustering": {
    "pca_components": 64,
    "umap_neighbors": 20,
    "min_clusters": 3,
    "max_clusters": 15
  }
}
```

Use it:
```bash
python pipeline.py --base-dir /path/to/data --config config.json
```

### Default Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `Qwen/Qwen2.5-1.5B` | HuggingFace model ID |
| `batch_size` | 16 | Batch size for embedding |
| `max_length` | 512 | Max token length |
| `contrastive.epochs` | 10 | Training epochs |
| `contrastive.temperature` | 0.2 | NT-Xent temperature |
| `contrastive.proj_dim` | 128 | Projection dimension |
| `clustering.pca_components` | 50 | PCA dimensions |
| `clustering.umap_neighbors` | 15 | UMAP neighbors |
| `downstream.top_keywords` | 30 | Keywords per cluster |

## 📁 Output Files

### Main Results

| File | Description |
|------|-------------|
| `chunks_with_umap_clusters_enriched_v2.csv` | Complete results with text, coordinates, clusters |
| `cluster_report_v2.md` | Human-readable analysis report |

### Intermediate Files

| File | Description |
|------|-------------|
| `qwen15b_embeddings_v2.npy` | Base embeddings (1536-dim) |
| `Z_contrastive_v2.npy` | Contrastive embeddings (128-dim) |
| `umap_2d_v2.csv` | 2D UMAP coordinates |
| `clusters_v2.csv` | Cluster labels |

### Analysis Files

| File | Description |
|------|-------------|
| `cluster_keywords_v2.csv` | Keywords per cluster |
| `cluster_top_artists_v2.csv` | Top artists per cluster |
| `cluster_representatives_v2.csv` | Representative samples |
| `cluster_emotions_v2.csv` | Emotion analysis |
| `song_vectors_v2.npy` | Song-level embeddings |
| `artist_vectors_v2.npy` | Artist-level embeddings |

## 🔧 Advanced Usage

### Resume from Checkpoint

Skip completed stages to save time:

```bash
# Skip preprocessing and embedding if already done
python pipeline.py --base-dir /path/to/data \
    --skip-preprocessing --skip-embedding
```

### Run Specific Stages

```bash
# Only run clustering and visualization
python pipeline.py --base-dir /path/to/data \
    --skip-preprocessing --skip-embedding --skip-contrastive \
    --skip-downstream
```

### Use Without Contrastive Learning

For faster processing or baseline comparison:

```bash
python pipeline.py --base-dir /path/to/data --skip-contrastive
```

The pipeline will use base embeddings (`qwen15b_embeddings_v2.npy`) instead.

### Custom Stopwords

Add a custom stopwords file:

```
/path/to/data/transformed_data/stopwords-zh.txt
```

Format: one word per line.

### Similarity Search

Use the generated vectors for similarity search:

```python
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

# Load song vectors
songs_meta = pd.read_csv("songs_meta_v2.csv")
Z_songs = normalize(np.load("song_vectors_v2.npy"))

# Find similar songs
query_idx = 42  # Index of your query song
sims = Z_songs @ Z_songs[query_idx]
top_10 = sims.argsort()[::-1][1:11]  # Exclude self

print("Similar songs:")
print(songs_meta.iloc[top_10][["artist", "song_title"]])
```

## 🎓 Research Background

This pipeline is part of research on **Fine-tuning LLMs for Chinese Rap Lyrics Generation**.

### Key Findings

1. **Contrastive Learning Impact**: Enhanced clustering from 3 incoherent clusters (baseline) to 7 semantically meaningful clusters
2. **Silhouette Score Paradox**: Despite improved semantic quality, silhouette scores decreased - suggesting traditional metrics don't capture semantic improvements well
3. **Qwen2.5-1.5B Performance**: Small-scale LLMs can effectively understand regionally distinctive language when enhanced with domain-specific contrastive learning

### Methodology

- **Model**: Qwen2.5-1.5B (1.5B parameters)
- **Framework**: SimCSE contrastive learning with NT-Xent loss
- **Augmentation**: Gaussian noise + dropout in embedding space
- **Evaluation**: UMAP visualization + manual semantic validation

### Future Work

- Explore additional data augmentation techniques
- Manual labeling for supervised evaluation
- Scale to full dataset (600,000+ chunks vs current 60,000)
- Fine-tune LLM for Chinese rap lyrics generation

### Presented At

**2025 Research Bazaar**  
University of Wisconsin–Madison

**Authors**: 
- Moshi Fu (Statistics, Data Science, & Mathematics)
- Brendan C. Dowling (PhD, Chinese Language Sciences)

## 📖 Citation

If you use this pipeline in your research, please cite:

```bibtex
@misc{fu2025chinese,
  title={Fine-tuning LLMs for Chinese Rap Lyrics: 
         Embedding-based Clustering with Contrastive Learning},
  author={Fu, Moshi and Dowling, Brendan C.},
  year={2025},
  institution={University of Wisconsin--Madison}
}
```

## 📚 References

1. Sun, Y., et al. (2021). ERNIE 3.0: Large-scale Knowledge Enhanced Pre-training for Language Understanding and Generation.
2. Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-Based Clustering Based on Hierarchical Density Estimates.
3. Gao, T., Yao, X., & Chen, D. (2021). SimCSE: Simple Contrastive Learning of Sentence Embeddings. EMNLP 2021.

## 🙏 Acknowledgements

- Chris Endemenn
- Elaine Wu
- Finn Kuusisto
- Lao Kai
- Ryan Bemowski
- Google Gemini & OpenAI GPT-4o
- UW-Madison Research Computing

## 📄 License

MIT License - see [LICENSE](LICENSE) file for details.

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📧 Contact

For questions or collaborations:

- **Moshi Fu**: [your-email@wisc.edu]
- **Project**: [GitHub Repository URL]

---

**Note**: This pipeline requires significant computational resources. For the full dataset (600,000+ chunks), we recommend:
- GPU: NVIDIA A100 or equivalent
- RAM: 32GB+
- Disk: 50GB+ free space
