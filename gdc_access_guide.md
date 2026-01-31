# GDC API Access Guide

## Overview

This project now supports **direct access to TCGA-BRCA data via the GDC API** - no manual downloads required! This feature enables:

- ✅ Automatic data fetching from GDC servers
- ✅ On-demand image downloading during training (streaming mode)
- ✅ Intelligent caching to avoid re-downloads
- ✅ Programmatic access to clinical data
- ✅ Scalable to entire TCGA-BRCA dataset (~1000 patients)

## Quick Start (30 seconds)

```bash
# 1. Install dependencies (if not already done)
pip install -r requirements.txt

# 2. Train with GDC streaming (no downloads needed!)
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 50

# That's it! The script will:
# - Fetch clinical data from GDC
# - Query available slide images
# - Download images on-demand during training
# - Cache everything for future use
```

## How It Works

### Architecture

```
┌─────────────────┐
│   GDC Servers   │  ← TCGA-BRCA Project
│  (NIH/NCI Data) │
└────────┬────────┘
         │ API Requests
         ↓
┌─────────────────┐
│   GDC Client    │  ← Our Python wrapper
│  (gdc_client.py)│
└────────┬────────┘
         │
         ├→ Clinical Data (CSV) ────────┐
         │                               │
         ├→ File Manifest (Image List)  │
         │                               │
         └→ Slide Images (On-Demand)    │
                                         ↓
                              ┌──────────────────┐
                              │ Local Cache      │
                              │ data/gdc_cache/  │
                              └──────────────────┘
                                         ↓
                              ┌──────────────────┐
                              │ Training Script  │
                              │ (PyTorch)        │
                              └──────────────────┘
```

### Streaming vs Download Modes

**Streaming Mode** (Recommended for exploration):
- Images downloaded as needed during training
- Low upfront time and storage
- Slight overhead per epoch (first time only)
- Perfect for experimenting with small subsets

**Download Mode** (Recommended for production):
- All images downloaded before training
- Higher upfront time and storage
- Faster training (no download overhead)
- Best for multiple training runs

## Usage Examples

### Example 1: Quick Experiment (10 samples)

```bash
# Train on 10 samples with streaming
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 10
```

**What happens:**
1. Fetches clinical data for 10 patients (~5 seconds)
2. Gets image file IDs for these patients (~3 seconds)
3. Starts training immediately
4. Downloads images during first epoch (~2-5 min per image)
5. Subsequent epochs use cached images (fast!)

**Storage required:** ~500MB - 2GB

### Example 2: Small Study (100 samples)

```bash
# Download first, then train (recommended)
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100 \
    --download-first
```

**What happens:**
1. Fetches metadata (~10 seconds)
2. Downloads all 100 images (~30-60 minutes)
3. Trains on downloaded images (fast)

**Storage required:** ~5-20GB

### Example 3: Full Dataset (~1000 samples)

```bash
# Use streaming for full dataset
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 1000
```

**What happens:**
1. Fetches all available data (~30 seconds)
2. Streams images during training
3. Gradually builds cache over epochs

**Storage required:** ~50-200GB (accumulates over time)

### Example 4: Programmatic Access

```python
from src.data.gdc_client import GDCDataManager

# Initialize manager
manager = GDCDataManager()

# Setup dataset (no download)
info = manager.setup_dataset(
    n_samples=50,
    download_images=False  # Streaming mode
)

# Get clinical data as pandas DataFrame
clinical_df = manager.get_matched_data()
print(clinical_df.head())

# Get specific columns
print(clinical_df[['patient_id', 'survival_time', 'vital_status']])

# Access GDC client directly
cases = manager.client.get_cases(size=10)
slides = manager.client.get_slide_files(size=10)
```

### Example 5: Custom Dataset Creation

```python
from src.data.gdc_streaming import create_gdc_dataset
from torch.utils.data import DataLoader

# Create streaming dataset
dataset = create_gdc_dataset(
    n_samples=200,
    streaming=True,
    cache_dir='data/my_cache',
    image_size=512
)

# Use with DataLoader as normal
loader = DataLoader(dataset, batch_size=16, shuffle=True)

# Train as usual
for batch in loader:
    images = batch['image']
    times = batch['time']
    events = batch['event']
    # ... train model
```

### Example 6: Multimodal with Clinical Features

```python
from src.data.gdc_streaming import create_gdc_dataset

# Create dataset with clinical features
dataset = create_gdc_dataset(
    n_samples=100,
    streaming=True,
    feature_columns=[
        'age_at_diagnosis',
        'tumor_stage',
        'tumor_grade',
        'gender',
        'race'
    ]
)

# Each sample includes clinical features
sample = dataset[0]
print(sample.keys())
# ['image', 'time', 'event', 'patient_id', 'clinical_features']
```

## Configuration Options

### gdc_streaming_config.yaml

```yaml
gdc:
  # Data source
  project_id: "TCGA-BRCA"
  cache_dir: "data/gdc_cache"
  
  # Streaming settings
  streaming: true              # Use streaming mode
  prefetch: true               # Prefetch images in background
  prefetch_factor: 10          # Number to prefetch ahead
  
  # Dataset size
  n_samples: 100               # Number of samples (null = all)
  
  # Download options
  download_immediately: false  # Download all before training
  max_download_size_gb: 50.0  # Safety limit
  max_workers: 4               # Parallel downloads
  
  # Caching
  cache_images_memory: true    # Keep in RAM (faster)
  cache_images_disk: true      # Save to disk (persistent)

advanced:
  max_download_retries: 3      # Retry failed downloads
  skip_failed_downloads: true  # Continue on errors
  clean_cache_on_exit: false   # Delete cache when done
```

## Available Data

### Clinical Variables

The GDC API provides access to:

**Survival Data:**
- `vital_status`: Alive/Dead
- `days_to_death`: Time to death event
- `days_to_last_follow_up`: Censoring time

**Diagnosis:**
- `age_at_diagnosis`: Age in days
- `tumor_stage`: I, II, III, IV
- `tumor_grade`: Differentiation grade
- `disease_type`: Cancer type
- `primary_site`: Anatomical location

**Demographics:**
- `gender`: Male/Female
- `race`: Racial category
- `ethnicity`: Ethnic background

**Additional:**
- `case_id`: Unique identifier
- `submitter_id`: Patient ID
- Various molecular markers (when available)

### Image Data

**Types Available:**
- Diagnostic slides (H&E stained)
- Tissue slides
- Frozen section slides

**Formats:**
- SVS (Aperio)
- TIFF
- Other WSI formats

**Typical Sizes:**
- 100MB - 2GB per slide
- Gigapixel resolution
- 40x magnification

## Caching System

### Cache Structure

```
data/gdc_cache/
├── clinical_data_TCGA-BRCA.csv      # Clinical data
├── manifest_TCGA-BRCA.csv           # File manifest
├── matched_data.csv                 # Matched clinical + images
├── images/                          # Downloaded images
│   ├── uuid-1.png
│   ├── uuid-2.png
│   └── ...
└── downloads/                       # Raw downloaded files
    ├── uuid-1
    └── uuid-2
```

### Cache Management

```python
from src.data.gdc_client import GDCClient

client = GDCClient(cache_dir='data/gdc_cache')

# Clear all cache
client.clear_cache()

# Check cache size
import shutil
cache_size = shutil.disk_usage(client.cache_dir).used
print(f"Cache size: {cache_size / 1e9:.2f} GB")
```

### Cache Best Practices

1. **Location**: Use fast SSD for cache if possible
2. **Size**: Monitor with `du -sh data/gdc_cache`
3. **Persistence**: Cache persists between runs
4. **Cleanup**: Delete when experimenting, keep for production

## Performance Tips

### Streaming Mode Optimization

```yaml
# Fast initial experiments
gdc:
  streaming: true
  cache_images_memory: true   # Keep in RAM
  prefetch_factor: 20         # Aggressive prefetching

data:
  num_workers: 4              # Parallel loading
  
training:
  batch_size: 16              # Moderate batch size
```

### Download Mode Optimization

```bash
# Download with maximum parallelism
python src/data/gdc_client.py \
    --n-samples 500 \
    --download-images \
    --cache-dir data/gdc_cache

# Then train
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml
```

### Network Optimization

```python
# Increase parallel downloads
client = GDCClient()
client.download_batch(
    file_ids=file_list,
    max_workers=8  # More parallel connections
)
```

## Troubleshooting

### Issue: Slow Downloads

**Solutions:**
1. Check internet connection
2. Increase `max_workers` in config
3. Use download mode for batch processing
4. Download during off-peak hours

### Issue: Connection Errors

**Solutions:**
```python
# Automatic retry is built-in
gdc:
  advanced:
    max_download_retries: 5  # More retries
```

Or manually:
```python
client = GDCClient()
try:
    client.download_file(file_id)
except Exception as e:
    print(f"Download failed: {e}")
    # Retry or skip
```

### Issue: Out of Disk Space

**Solutions:**
1. Reduce `n_samples`
2. Set `max_download_size_gb` limit
3. Use streaming mode without caching
4. Clean old cache periodically

```python
# Disable disk caching (keep in memory only)
config['gdc']['cache_images_disk'] = False
```

### Issue: Out of Memory

**Solutions:**
```yaml
gdc:
  cache_images_memory: false  # Don't cache in RAM

training:
  batch_size: 8  # Reduce batch size
  
data:
  num_workers: 2  # Fewer workers
```

### Issue: API Rate Limiting

GDC has rate limits. If you hit them:
- Add delays between requests
- Reduce `max_workers`
- Use exponential backoff (built-in)

## Comparison: Manual vs GDC API

### Manual Download Method

```bash
# Old way (manual)
1. Go to GDC Data Portal website
2. Select TCGA-BRCA project
3. Filter for slide images
4. Download manifest file
5. Install gdc-client tool
6. Download with: gdc-client download -m manifest.txt
7. Wait hours/days
8. Organize files
9. Create clinical data CSV manually
10. Match images to clinical data
```

**Time:** 4-48 hours  
**Complexity:** High  
**Reproducibility:** Low

### GDC API Method

```bash
# New way (automated)
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100
```

**Time:** < 5 minutes to start  
**Complexity:** Low  
**Reproducibility:** Perfect

## Advanced Usage

### Custom Queries

```python
from src.data.gdc_client import GDCClient

client = GDCClient()

# Query specific tumor stages
filters = {
    "op": "and",
    "content": [
        {"op": "=", "content": {"field": "project.project_id", "value": "TCGA-BRCA"}},
        {"op": "=", "content": {"field": "diagnoses.tumor_stage", "value": "stage iii"}}
    ]
}

# Get cases matching filter
cases = client._make_request(
    client.cases_endpoint,
    params={"filters": json.dumps(filters), "size": 100}
)
```

### Batch Processing

```python
# Process dataset in chunks
from src.data.gdc_client import GDCDataManager

manager = GDCDataManager()

# Process 100 samples at a time
for start_idx in range(0, 1000, 100):
    chunk_info = manager.setup_dataset(
        n_samples=100,
        download_images=True
    )
    
    # Train on this chunk
    # ... training code ...
    
    # Clear cache for next chunk
    manager.client.clear_cache()
```

### Integration with Existing Code

```python
# Modify existing datasets to use GDC
from src.data.gdc_streaming import GDCStreamingDataset
from src.data.gdc_client import GDCClient

# Replace manual dataset
# OLD: dataset = TCGADataset(image_dir, clinical_csv)

# NEW: 
client = GDCClient(cache_dir='data/cache')
dataset = GDCStreamingDataset(client, 'data/cache/matched_data.csv')

# Use exactly the same way
dataloader = DataLoader(dataset, ...)
```

## FAQ

**Q: Do I need GDC credentials?**  
A: No! TCGA data is public and doesn't require authentication.

**Q: Can I use this offline?**  
A: After caching, yes. Initial setup requires internet.

**Q: How much data is available?**  
A: ~1,000 TCGA-BRCA cases with slides and clinical data.

**Q: Can I filter by molecular subtype?**  
A: Yes! Use custom queries or filter the clinical DataFrame.

**Q: Does this work for other TCGA projects?**  
A: Yes! Change `project_id` to any TCGA project (TCGA-LUAD, etc.)

**Q: Is this faster than manual download?**  
A: For experiments, yes. For full dataset, similar speed but automated.

**Q: Can I resume interrupted downloads?**  
A: Yes! Cached files are skipped automatically.

## References

- **GDC API Docs**: https://docs.gdc.cancer.gov/API/
- **TCGA-BRCA Portal**: https://portal.gdc.cancer.gov/projects/TCGA-BRCA
- **Data User Guide**: https://docs.gdc.cancer.gov/Data/
- **API Examples**: https://docs.gdc.cancer.gov/API/Users_Guide/Python_Examples/

## Next Steps

1. **Try the quick start** with 10 samples
2. **Experiment** with different sample sizes
3. **Scale up** to full dataset as needed
4. **Integrate** with your custom models

**Happy experimenting! 🚀**
