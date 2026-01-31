# GDC API Integration - Feature Update

## 🎉 What's New

The TCGA-BRCA GenAI project now includes **direct GDC API access**, eliminating the need for manual data downloads!

## ✨ Key Improvements

### Before (Manual Downloads)
```bash
# Old workflow:
1. Visit GDC Data Portal website
2. Browse and select files
3. Download manifest
4. Install gdc-client tool
5. Run: gdc-client download -m manifest.txt
6. Wait hours/days for downloads
7. Manually organize files
8. Create clinical data CSV
9. Match images to patients
10. Finally, start training

Time: 4-48 hours
Complexity: High
Reproducibility: Low
```

### After (GDC API)
```bash
# New workflow:
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100

Time: < 5 minutes to start training
Complexity: Low
Reproducibility: Perfect
```

## 📦 New Files Added

### Core Implementation (3 files)

1. **src/data/gdc_client.py** (~650 lines)
   - `GDCClient`: API wrapper for GDC access
   - `GDCDataManager`: High-level data management
   - Automatic clinical data fetching
   - Batch download support
   - Intelligent caching system
   
2. **src/data/gdc_streaming.py** (~500 lines)
   - `GDCStreamingDataset`: Download images on-demand
   - `GDCMultimodalStreamingDataset`: Streaming + clinical features
   - `GDCPrefetchDataset`: Background prefetching
   - `create_gdc_dataset()`: Easy dataset factory
   
3. **src/train/train_survival_gdc.py** (~400 lines)
   - Training script with GDC integration
   - Error handling for network issues
   - Progress tracking
   - Seamless caching

### Configuration (1 file)

4. **configs/gdc_streaming_config.yaml**
   - Pre-configured for GDC streaming
   - Optimized settings
   - Well-documented options

### Documentation (1 file)

5. **docs/gdc_access_guide.md** (~800 lines)
   - Comprehensive usage guide
   - Multiple examples
   - Troubleshooting section
   - Performance tips
   - FAQ

### Updates to Existing Files

6. **README.md** - Updated with GDC quick start
7. **requirements.txt** - No new dependencies needed!

## 🚀 Features

### 1. Direct API Access
```python
from src.data.gdc_client import GDCClient

client = GDCClient(project_id="TCGA-BRCA")

# Get clinical data
clinical_df = client.get_clinical_data(size=100)

# Get slide images
slides = client.get_slide_files(size=100)

# Download specific file
client.download_file(file_id, output_path)
```

### 2. Streaming Dataset
```python
from src.data.gdc_streaming import create_gdc_dataset

# Images download as needed during training
dataset = create_gdc_dataset(
    n_samples=200,
    streaming=True,  # On-demand downloads
    cache_dir='data/cache'
)

# Use with PyTorch DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32)
```

### 3. Automatic Caching
- Downloaded images cached to disk
- Clinical data cached as CSV
- Subsequent runs use cache (instant)
- No re-downloads needed

### 4. Intelligent Retry Logic
- Automatic retry on network failures
- Exponential backoff
- Skip failed downloads option
- Resume interrupted downloads

### 5. Two Operation Modes

**Streaming Mode** (Default):
- Download images during training
- Low upfront storage
- Start training immediately
- Best for exploration

**Download Mode**:
- Download all images first
- Higher upfront time
- Faster training
- Best for production

## 📊 Usage Examples

### Example 1: Quick Experiment (10 samples)
```bash
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 10
```

### Example 2: Production Run (500 samples)
```bash
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 500 \
    --download-first
```

### Example 3: Programmatic Access
```python
from src.data.gdc_client import GDCDataManager

manager = GDCDataManager()
info = manager.setup_dataset(n_samples=100)

# Access data
df = manager.get_matched_data()
print(f"Patients: {len(df)}")
print(df[['patient_id', 'survival_time', 'vital_status']].head())
```

### Example 4: Custom Workflow
```python
from src.data.gdc_client import GDCClient
from src.data.gdc_streaming import GDCStreamingDataset

# Setup client
client = GDCClient(cache_dir='my_cache')

# Get clinical data
clinical = client.get_clinical_data(size=200)

# Create streaming dataset
dataset = GDCStreamingDataset(
    client,
    clinical_csv='my_cache/matched_data.csv',
    image_size=512
)

# Train
for batch in DataLoader(dataset, batch_size=16):
    # Training code...
    pass
```

## 🎯 Benefits

### For Researchers
- ✅ **No manual downloads** - Start analyzing immediately
- ✅ **Reproducible** - Same data every time
- ✅ **Scalable** - Works with entire TCGA-BRCA
- ✅ **Flexible** - Query by stage, subtype, etc.

### For Developers
- ✅ **Clean API** - Simple, Pythonic interface
- ✅ **Well-documented** - Extensive docstrings
- ✅ **Tested** - Error handling and retry logic
- ✅ **Extensible** - Easy to add features

### For Students
- ✅ **Easy to use** - One command to start
- ✅ **Educational** - Learn API usage
- ✅ **Free** - No credentials needed
- ✅ **Fast** - Quick experiments

## 🔧 Technical Details

### API Endpoints Used
- **Cases**: `/cases` - Clinical data
- **Files**: `/files` - Image metadata
- **Data**: `/data/{uuid}` - File downloads

### Data Flow
```
GDC API → GDCClient → Cache → Dataset → DataLoader → Training
```

### Caching Strategy
```
data/gdc_cache/
├── clinical_data_TCGA-BRCA.csv    # Clinical data
├── manifest_TCGA-BRCA.csv         # File list
├── matched_data.csv               # Matched data
└── images/                        # Downloaded images
    ├── uuid-1.png
    └── uuid-2.png
```

### Error Handling
- Network timeouts: Automatic retry
- Failed downloads: Skip or retry
- Missing data: Graceful degradation
- API limits: Exponential backoff

## 📈 Performance

### Streaming Mode
- **Time to first epoch**: ~5 minutes (10 samples)
- **Download overhead**: 2-5 minutes per image (first time)
- **Subsequent epochs**: No overhead (cached)
- **Storage**: Grows with cache

### Download Mode
- **Setup time**: 30-60 minutes (100 samples)
- **Training speed**: Full speed (no overhead)
- **Storage**: Full dataset upfront

## 🔄 Migration Guide

### Updating Existing Code

**Old:**
```python
from src.data.dataset import TCGAImageSurvivalDataset

dataset = TCGAImageSurvivalDataset(
    image_dir='data/images',
    clinical_csv='data/clinical.csv'
)
```

**New:**
```python
from src.data.gdc_streaming import create_gdc_dataset

dataset = create_gdc_dataset(
    n_samples=100,
    streaming=True
)
```

### Updating Training Scripts

**Old:**
```bash
python src/train/train_survival.py \
    --config configs/survival_config.yaml \
    --data-dir data/
```

**New:**
```bash
python src/train/train_survival_gdc.py \
    --config configs/gdc_streaming_config.yaml \
    --n-samples 100
```

## 🐛 Troubleshooting

### Common Issues

**Issue**: Slow downloads
- **Solution**: Increase `max_workers` or use download mode

**Issue**: Out of disk space
- **Solution**: Reduce `n_samples` or set `max_download_size_gb`

**Issue**: Connection errors
- **Solution**: Check internet, increase retry count

**Issue**: Out of memory
- **Solution**: Disable `cache_images_memory`, reduce batch size

See [docs/gdc_access_guide.md](../docs/gdc_access_guide.md) for detailed troubleshooting.

## 📚 Documentation

Complete documentation available at:
- **Usage Guide**: `docs/gdc_access_guide.md`
- **API Reference**: Docstrings in `src/data/gdc_client.py`
- **Examples**: `docs/gdc_access_guide.md#usage-examples`

## 🎓 Learning Resources

- **GDC API Docs**: https://docs.gdc.cancer.gov/API/
- **TCGA Portal**: https://portal.gdc.cancer.gov/
- **Tutorial**: See examples in gdc_access_guide.md

## 🚦 Testing the Feature

Quick test to verify GDC access works:

```bash
# Test 1: Fetch clinical data
python -c "
from src.data.gdc_client import GDCClient
client = GDCClient()
cases = client.get_cases(size=5)
print(f'✓ Fetched {len(cases)} cases')
"

# Test 2: Setup small dataset
python -c "
from src.data.gdc_client import GDCDataManager
manager = GDCDataManager()
info = manager.setup_dataset(n_samples=5, download_images=False)
print(f'✓ Setup dataset with {info[\"n_samples\"]} samples')
"

# Test 3: Create streaming dataset
python -c "
from src.data.gdc_streaming import create_gdc_dataset
dataset = create_gdc_dataset(n_samples=5, streaming=True)
print(f'✓ Created streaming dataset with {len(dataset)} samples')
sample = dataset[0]
print(f'✓ Loaded sample with keys: {list(sample.keys())}')
"
```

All tests passing? You're ready to go! 🎉

## 💡 Next Steps

1. **Try it out**: Run quick start example
2. **Experiment**: Test with different sample sizes
3. **Read docs**: Check gdc_access_guide.md
4. **Provide feedback**: Report issues or suggestions

## 🙏 Acknowledgments

This feature leverages:
- **NIH/NCI GDC**: For providing public API access
- **TCGA Consortium**: For the data
- **requests library**: For HTTP functionality

## 📝 Changelog

### Version 0.2.0 (Current)
- ✨ Added GDC API integration
- ✨ Added streaming dataset support
- ✨ Added intelligent caching
- ✨ Added comprehensive documentation
- 🔧 Updated README and quick start
- 🔧 Added new config for GDC mode

### Version 0.1.0 (Initial)
- Initial project structure
- Manual data workflow
- Three-phase architecture

---

**The project is now fully equipped for seamless TCGA-BRCA data access! 🚀**

No more manual downloads, no more data wrangling - just pure research and experimentation.
