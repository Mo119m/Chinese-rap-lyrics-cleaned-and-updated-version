# Quick Reference Guide

## Common Commands

### Initial Setup
```bash
# Run setup script
bash setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Running the Pipeline

**Full pipeline:**
```bash
python pipeline.py --base-dir /path/to/data
```

**Resume from checkpoint:**
```bash
# Skip completed stages
python pipeline.py --base-dir /path/to/data \
    --skip-preprocessing --skip-embedding
```

**Run without contrastive learning:**
```bash
python pipeline.py --base-dir /path/to/data --skip-contrastive
```

**Custom configuration:**
```bash
python pipeline.py --base-dir /path/to/data --config my_config.json
```

### Utility Commands

**Similarity search (songs):**
```bash
# By index
python utils.py search --data-dir ./data/transformed_data \
    --type song --query 42 --top-n 10

# By name (partial match)
python utils.py search --data-dir ./data/transformed_data \
    --type song --query "夜色" --top-n 10
```

**Similarity search (artists):**
```bash
python utils.py search --data-dir ./data/transformed_data \
    --type artist --query "周士爵" --top-n 10
```

**Cluster summary:**
```bash
python utils.py summary --data-dir ./data/transformed_data --cluster 3
```

**Export cluster:**
```bash
python utils.py export --data-dir ./data/transformed_data \
    --cluster 3 --output cluster_3.csv
```

## File Locations

### Input Files
- Raw lyrics: `data/transformed_data/all_lyrics.txt`
- Stopwords (optional): `data/transformed_data/stopwords-zh.txt`
- Config (optional): `config.json`

### Key Output Files
- Main results: `data/transformed_data/chunks_with_umap_clusters_enriched_v2.csv`
- Report: `data/transformed_data/cluster_report_v2.md`
- Visualizations: `data/transformed_data/visualizations/`

### Embeddings
- Base: `data/transformed_data/qwen15b_embeddings_v2.npy`
- Contrastive: `data/transformed_data/Z_contrastive_v2.npy`
- Songs: `data/transformed_data/song_vectors_v2.npy`
- Artists: `data/transformed_data/artist_vectors_v2.npy`

### Analysis Results
- Keywords: `data/transformed_data/cluster_keywords_v2.csv`
- Top artists: `data/transformed_data/cluster_top_artists_v2.csv`
- Representatives: `data/transformed_data/cluster_representatives_v2.csv`
- Emotions: `data/transformed_data/cluster_emotions_v2.csv`

## Configuration Parameters

### Embedding Generation
- `model_id`: HuggingFace model (default: "Qwen/Qwen2.5-1.5B")
- `batch_size`: Batch size (default: 16)
- `max_length`: Max sequence length (default: 512)

### Contrastive Learning
- `epochs`: Training epochs (default: 10)
- `batch_size`: Training batch size (default: 1024)
- `temperature`: NT-Xent temperature (default: 0.2)
- `proj_dim`: Projection dimension (default: 128)
- `learning_rate`: Learning rate (default: 1e-3)
- `noise_std`: Augmentation noise (default: 0.05)
- `dropout`: Augmentation dropout (default: 0.05)

### Clustering
- `pca_components`: PCA dimensions (default: 50)
- `umap_neighbors`: UMAP n_neighbors (default: 15)
- `umap_min_dist`: UMAP min_dist (default: 0.1)
- `min_clusters`: Min K for KMeans (default: 2)
- `max_clusters`: Max K for KMeans (default: 12)

### Downstream Analysis
- `top_artists`: Top artists per cluster (default: 10)
- `representative_samples`: Representatives per cluster (default: 20)
- `top_keywords`: Keywords per cluster (default: 30)

## Troubleshooting

### CUDA Out of Memory
Reduce batch size:
```json
{
  "batch_size": 8,
  "contrastive": {
    "batch_size": 512
  }
}
```

### Too Many Clusters
Adjust range:
```json
{
  "clustering": {
    "min_clusters": 2,
    "max_clusters": 8
  }
}
```

### Poor Keyword Quality
1. Add more stopwords to `stopwords-zh.txt`
2. Adjust `min_df` in `src/downstream_analysis.py`

### Slow Embedding Generation
1. Use GPU if available
2. Reduce `max_length`:
```json
{
  "max_length": 256
}
```

## Performance Tips

### For Large Datasets (100K+ chunks)
1. Use GPU (A100 recommended)
2. Increase batch sizes:
   - Embedding: 32-64
   - Contrastive: 2048-4096
3. Use mixed precision (already enabled for CUDA)

### For Small Datasets (<10K chunks)
1. Reduce contrastive learning epochs: 5-8
2. Skip PCA: set `pca_components` equal to embedding dimension
3. Adjust UMAP parameters for smaller datasets:
   - `umap_neighbors`: 5-10
   - `umap_min_dist`: 0.0-0.05

## Python API Usage

```python
from pathlib import Path
from pipeline import RapLyricsPipeline

# Initialize pipeline
pipeline = RapLyricsPipeline(
    base_dir=Path("./data"),
    config={
        "contrastive": {"epochs": 15},
        "clustering": {"max_clusters": 10}
    }
)

# Run full pipeline
pipeline.run()

# Or run individual components
pipeline.data_processor.parse_lyrics()
pipeline.embedding_gen.generate_embeddings()
pipeline.contrastive_learner.train()
# etc.
```

## Common Workflows

### 1. Initial Analysis
```bash
# Full pipeline with default settings
python pipeline.py --base-dir ./data
```

### 2. Parameter Tuning
```bash
# Try different configurations
python pipeline.py --base-dir ./data --config config_1.json
python pipeline.py --base-dir ./data --config config_2.json \
    --skip-preprocessing --skip-embedding
```

### 3. Adding New Data
```bash
# Re-run preprocessing and embedding
python pipeline.py --base-dir ./data \
    --skip-contrastive --skip-clustering \
    --skip-downstream --skip-visualization

# Then resume
python pipeline.py --base-dir ./data \
    --skip-preprocessing --skip-embedding
```

### 4. Exploring Results
```bash
# Generate summary for each cluster
for i in {0..6}; do
    python utils.py summary --data-dir ./data/transformed_data --cluster $i
done

# Find similar artists
python utils.py search --data-dir ./data/transformed_data \
    --type artist --query "周士爵" --top-n 20
```

## Additional Resources

- Full documentation: `README.md`
- Example config: `config.example.json`
- Source code: `src/`
- Research poster: `poster_2_0.pdf`
