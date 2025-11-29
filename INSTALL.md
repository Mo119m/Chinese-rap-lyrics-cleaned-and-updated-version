# Installation & Usage Guide

## 🎯 What Is This?

This is a **production-ready machine learning pipeline** for analyzing Chinese rap lyrics. It transforms your original Colab notebook into a modular, maintainable system with ~1,800 lines of clean, documented code.

## 📦 What's Included

```
chinese-rap-pipeline/
├── README.md              ← Comprehensive documentation (250+ lines)
├── QUICKREF.md            ← Quick command reference
├── PROJECT_SUMMARY.md     ← This overview
├── pipeline.py            ← Main orchestrator (230 lines)
├── utils.py               ← Utility tools (180 lines)
├── setup.sh               ← One-command setup
├── requirements.txt       ← Python dependencies
├── config.example.json    ← Configuration template
├── .gitignore            ← Git configuration
└── src/                   ← Modular source code
    ├── __init__.py
    ├── data_processing.py       (170 lines)
    ├── embedding.py             (100 lines)
    ├── contrastive_learning.py  (150 lines)
    ├── clustering.py            (170 lines)
    ├── downstream_analysis.py   (370 lines)
    └── visualization.py         (100 lines)
```

## 🚀 Quick Installation

### Option 1: Automated Setup (Recommended)

```bash
# Navigate to the pipeline directory
cd chinese-rap-pipeline

# Run the setup script
bash setup.sh
```

This automatically:
- Creates Python virtual environment
- Installs all dependencies
- Initializes jieba dictionary
- Creates data directories

### Option 2: Manual Setup

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Initialize jieba
python -c "import jieba; jieba.initialize()"

# Create directories
mkdir -p data/transformed_data/visualizations
```

## 📝 Data Preparation

1. Place your raw lyrics file:
   ```
   data/transformed_data/all_lyrics.txt
   ```

2. (Optional) Add custom stopwords:
   ```
   data/transformed_data/stopwords-zh.txt
   ```

3. Format your lyrics file:
   ```
   歌手姓名：Artist Name
   Song Title
   作词：Lyricist Name
   
   Lyrics content here...
   
   歌手姓名：Next Artist
   ...
   ```

## ▶️ Running the Pipeline

### Basic Usage

```bash
# Activate environment (if not already active)
source venv/bin/activate

# Run full pipeline
python pipeline.py --base-dir ./data
```

This will:
1. Parse and chunk lyrics (→ CSV)
2. Generate embeddings (→ 1536-dim vectors)
3. Apply contrastive learning (→ 128-dim vectors)
4. Perform clustering (→ 2D visualization + labels)
5. Run downstream analysis (→ keywords, emotions, etc.)
6. Create visualizations (→ plots)

**Processing Time**: 
- Small dataset (1K chunks): ~10-15 minutes (CPU)
- Medium dataset (10K chunks): ~1-2 hours (CPU) or ~15-30 min (GPU)
- Large dataset (60K chunks): ~6-8 hours (CPU) or ~1-2 hours (GPU)

### Resume from Checkpoint

If interrupted, skip completed stages:

```bash
python pipeline.py --base-dir ./data \
    --skip-preprocessing \
    --skip-embedding
```

### Skip Contrastive Learning (Faster)

For baseline comparison:

```bash
python pipeline.py --base-dir ./data --skip-contrastive
```

## 📊 Viewing Results

### Main Output Files

After running, check:

```bash
cd data/transformed_data

# Main results file
head chunks_with_umap_clusters_enriched_v2.csv

# Analysis report (Markdown)
cat cluster_report_v2.md

# Visualizations
ls visualizations/
```

### Using Utility Tools

```bash
# View cluster summary
python utils.py summary \
    --data-dir ./data/transformed_data \
    --cluster 3

# Search similar songs
python utils.py search \
    --data-dir ./data/transformed_data \
    --type song \
    --query "夜色" \
    --top-n 10

# Export cluster data
python utils.py export \
    --data-dir ./data/transformed_data \
    --cluster 3 \
    --output cluster_3.csv
```

## ⚙️ Configuration

### Basic Configuration

Create `my_config.json`:

```json
{
  "batch_size": 32,
  "contrastive": {
    "epochs": 15,
    "temperature": 0.15
  },
  "clustering": {
    "max_clusters": 10
  }
}
```

Run with config:

```bash
python pipeline.py --base-dir ./data --config my_config.json
```

### Common Adjustments

**For faster processing:**
- Reduce `batch_size`: 8-16
- Reduce `max_length`: 256
- Reduce `contrastive.epochs`: 5

**For better quality:**
- Increase `batch_size`: 32-64 (with GPU)
- Increase `contrastive.epochs`: 15-20
- Adjust `contrastive.temperature`: 0.1-0.3

## 🔧 Troubleshooting

### CUDA Out of Memory

```json
{
  "batch_size": 8,
  "contrastive": {"batch_size": 512}
}
```

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

### Slow Processing

1. Verify GPU is being used:
   ```bash
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. Increase batch sizes if GPU available

### Bad Clusters

1. Check number of data points (need 100+ for good clustering)
2. Try different `temperature` values
3. Increase contrastive learning `epochs`

## 📚 Documentation

- **README.md** - Full documentation with examples
- **QUICKREF.md** - Command reference
- **Code comments** - Every function documented

## 🎓 Understanding the Pipeline

### Architecture Flow

```
Raw Text
   ↓
[Parse & Chunk] → CSV file
   ↓
[Qwen Embedding] → 1536-dim vectors
   ↓
[Contrastive Learning] → 128-dim enhanced vectors
   ↓
[PCA] → 50-dim
   ↓
[UMAP] → 2-dim for visualization
   ↓
[KMeans] → Cluster labels
   ↓
[Analysis] → Keywords, emotions, similarity
```

### Key Technologies

- **Qwen2.5-1.5B**: Chinese language model for embeddings
- **SimCSE**: Contrastive learning framework
- **UMAP**: Dimensionality reduction for visualization
- **KMeans**: Clustering algorithm
- **c-TF-IDF**: Keyword extraction
- **Jieba**: Chinese text segmentation

## 💡 Best Practices

1. **Start Small**: Test with 1,000 chunks first
2. **Use GPU**: 10-20x faster for embedding/training
3. **Iterate**: Try different parameters
4. **Validate**: Manually review cluster quality
5. **Document**: Keep track of what works

## 🔄 Workflow Example

### First Run

```bash
# 1. Setup
bash setup.sh

# 2. Prepare data
cp /path/to/your/all_lyrics.txt data/transformed_data/

# 3. Initial run
source venv/bin/activate
python pipeline.py --base-dir ./data

# 4. Review results
python utils.py summary --data-dir ./data/transformed_data --cluster 0
python utils.py summary --data-dir ./data/transformed_data --cluster 1
# ... check all clusters

# 5. View plots
open data/transformed_data/visualizations/cluster_umap.png
```

### Parameter Tuning

```bash
# Try different temperature
cat > config_temp_01.json << EOF
{"contrastive": {"temperature": 0.1}}
EOF

python pipeline.py --base-dir ./data \
    --config config_temp_01.json \
    --skip-preprocessing --skip-embedding

# Compare results
diff data/transformed_data/cluster_report_v2.md \
     data/transformed_data/cluster_report_v2.md.bak
```

## 📊 Output Files Guide

| File | Size | Description |
|------|------|-------------|
| `qwen15b_embeddings_v2.npy` | ~500MB | Base embeddings |
| `Z_contrastive_v2.npy` | ~50MB | Enhanced embeddings |
| `umap_2d_v2.csv` | ~1MB | 2D coordinates |
| `clusters_v2.csv` | <1MB | Cluster labels |
| `chunks_with_umap_clusters_enriched_v2.csv` | ~10MB | Complete results |
| `cluster_report_v2.md` | ~100KB | Analysis summary |
| `visualizations/*.png` | ~5MB | Plots |

## 🎯 Next Steps

1. Run the pipeline on your full dataset
2. Analyze results with utility tools
3. Tune parameters for better clusters
4. Explore similarity search
5. Extend for your specific needs

## 📞 Getting Help

1. Check **README.md** for detailed docs
2. Review **QUICKREF.md** for commands
3. Look at code comments
4. Start with small test dataset

## 🌟 Key Features Summary

✅ **Modular Design** - Clean, maintainable code  
✅ **Resume Support** - Skip completed stages  
✅ **Configurable** - JSON-based configuration  
✅ **Well Documented** - 250+ lines of docs  
✅ **Utility Tools** - Search, export, analyze  
✅ **Visualizations** - UMAP plots, distributions  
✅ **Production Ready** - Error handling, logging  
✅ **Research Validated** - Based on published work  

---

**Ready to start?** Run `bash setup.sh` and begin analyzing! 🚀
