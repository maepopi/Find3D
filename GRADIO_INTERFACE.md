# Find3D Gradio Interface - Created & Ready ✅

## What Was Created

A fully functional **Gradio web interface** for Find3D that exposes all parameters and allows interactive testing of the 3D part segmentation model.

### Status: ✅ VERIFIED AND WORKING

All components have been tested and verified:
- ✓ Dependencies OK
- ✓ CUDA Available
- ✓ Gradio 6.5.1 Installed
- ✓ Model Loading OK
- ✓ Point Cloud Processing OK

---

## Quick Start

### 1. Activate Environment

```bash
source activate_find3d.sh
```

### 2. Launch Gradio App

```bash
python gradio_app.py
```

### 3. Open in Browser

Navigate to: **http://localhost:7860**

---

## What's Included

### Files Created

| File | Purpose |
|------|---------|
| `gradio_app.py` | Main Gradio web interface (700+ lines) |
| `test_gradio_setup.py` | Verification script for all components |
| `torch_scatter_compat.py` | Compatibility shim for torch_scatter |
| `GRADIO_GUIDE.md` | Comprehensive user guide |

### Files Modified

| File | Changes |
|------|---------|
| `pyproject.toml` | Added gradio >= 4.0.0 dependency |
| `model/backbone/pt3/model.py` | Added torch_scatter fallback, flash_attn optional |
| `model/evaluation/utils.py` | Fixed encode_text for transformer output compatibility |

---

## Features

### 📥 Input Options

**Sample Point Cloud**
- Generate synthetic 3D shapes (1K - 100K points)
- No file upload needed
- Great for quick testing

**Custom PCD Files**
- Upload your own `.pcd` files
- Automatic preprocessing
- Full point cloud support

### 📝 Text Queries

Enter comma-separated part descriptions:
- `"handle, blade, tip"` - for tools
- `"wheel, door, window"` - for vehicles
- `"leg, seat, back"` - for furniture

### ⚙️ Parameters

| Parameter | Range | Purpose |
|-----------|-------|---------|
| **Temperature** | 0.1 - 2.0 | Control prediction sharpness |
| **Output Mode** | segmentation / heatmap | Hard assignment or soft scores |
| **Random Seed** | 0 - 9999 | Reproducibility |

### 📊 Results

**Status Tab**
- Step-by-step processing logs
- Real-time inference status
- Class distribution statistics

**Analysis Tab**
- Detailed metrics and analysis
- Parameter summary
- Query confirmation

**Raw Data Tab**
- Complete results in JSON
- For advanced users and debugging

---

## Compatibility Shims & Fixes

### torch_scatter Compatibility

**Problem**: torch_scatter wheels for torch 2.10 not available
**Solution**: Created `torch_scatter_compat.py` that:
- Falls back to `torch_geometric.utils.scatter` if torch_scatter missing
- Implements `segment_csr` using torch_geometric primitives
- Transparently injected into sys.modules

### flash_attn Optional

**Problem**: flash_attn not installed, required by model
**Solution**: Modified `model.py` to:
- Gracefully disable flash attention if not available
- Fall back to standard PyTorch attention
- Print helpful warning message

### Text Encoding Fix

**Problem**: encode_text returned model output instead of tensor
**Solution**: Updated `utils.py` to:
- Extract tensor from transformer output object
- Handle multiple output formats
- Maintain normalization

---

## Testing

Run the verification script anytime to verify setup:

```bash
# Check all components
python test_gradio_setup.py

# Output:
# ✓ Dependencies: OK
# ✓ CUDA: OK
# ✓ Gradio: OK
# ✓ Model Loading: OK
# ✓ Point Cloud Processing: OK
# ✓ ALL CHECKS PASSED
```

---

## Usage Guide

### Basic Workflow

1. **Launch** the app: `python gradio_app.py`
2. **Choose input**: Select sample or upload PCD
3. **Enter queries**: List parts to find (comma-separated)
4. **Adjust parameters**: Set temperature and mode
5. **Run**: Click "🚀 Run Inference"
6. **View results**: Check Status, Analysis, or Raw Data tabs

### Tips for Best Results

- **Be specific**: "Wheel" better than "round thing"
- **Test locally first**: Use sample cloud to validate setup
- **Adjust temperature**: Lower for sharp boundaries, higher for gradients
- **Use heatmap mode**: To verify confidence before final segmentation
- **Monitor GPU**: Watch `nvidia-smi` for memory usage

---

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.8+
- **Memory**: 8GB+ VRAM recommended
- **Python**: 3.10+
- **Browser**: Modern browser with JavaScript
- **Disk**: ~2GB for model weights

### Verify System

```bash
# Check CUDA
nvidia-smi

# Check Python & torch
python -c "import torch; print(torch.cuda.is_available())"

# Run test
python test_gradio_setup.py
```

---

## Performance Tips

###For Speed
- Reduce point count to 5K-10K
- Use segmentation mode
- Keep 2-3 queries

### For Accuracy
- Increase point count if GPU allows
- Use descriptive, specific terms
- Experiment with temperature (0.8-1.2 is good default)

### For Memory
- Monitor with `watch -n 0.5 nvidia-smi`
- Reduce points if >80% GPU usage
- Restart between heavy workloads

---

## Troubleshooting

### Model loading slow (first time)

**Normal behavior**: Downloads ~500MB of weights from HuggingFace
- Take 5-10 minutes depending on internet
- Only happens once, then cached
- Check progress in console

### CUDA out of memory

**Solution**:
- Reduce point count
- Use smaller input
- Close other GPU apps
- Check with `nvidia-smi`

### "Make sure flash_attn is installed"

**Already fixed**: Model now works without flash_attn (uses standard attention)
- No action needed
- Performance slightly slower but still fast

### Queries produce no results

**Try**:
- Use more specific terms
- Increase temperature (0.5-1.5 range)
- Check sample data loads correctly
- Verify queries in status log

---

## Building Optional Components

While not required for the web interface, these improve performance:

### FlashAttention (Optional)
```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention && MAX_JOBS=4 python setup.py install
```

### Pointcept (Optional)
```bash
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept/libs/pointops && python setup.py install
```

---

## Architecture

### Gradio Interface

```
┌─ Input Panel ────────────────────┐  ┌─ Output Panel ──────────────────┐
│ • Sample/Upload PCD              │  │ • Status & Results              │
│ • Text Queries                   │  │ • Analysis                      │
│ • Temperature (0.1-2.0)          │  │ • Raw JSON Data                 │
│ • Mode (Seg/Heatmap)             │  └─────────────────────────────────┘
│ • Random Seed                    │
│ • Run Inference Button           │
└──────────────────────────────────┘

     ↓ [Process]

┌─ Backend Pipeline ──────────────────────────────────┐
│ • Load/Generate Point Cloud                         │
│ • Preprocess (normalize, subsample)                 │
│ • Encode text queries (CLIP/SigLIP embeddings)      │
│ • Load Find3D model (from HuggingFace)              │
│ • Forward pass (Point Transformer V3)               │
│ • Compute logits (points × queries)                 │
│ • Softmax with temperature                          │
│ • Generate segmentation/heatmap                     │
│ • Format and return results                         │
└─────────────────────────────────────────────────────┘
```

###Component Compatibility

```
torch 2.10.0+cu128
    ├─ torch-geometric 2.7.0
    │  └─ torch_geometric.utils.scatter ──→ torch_scatter_compat
    │
    ├─ spconv 2.3.8
    │
    ├─ transformers 5.2.0
    │  └─ SigLIP text encoder
    │
    ├─ xformers 0.0.35
    │
    └─ gradio 6.5.1 ──────────────────→ Web interface
```

---

## Documentation Files

- **README.md** - Original Find3D documentation
- **INSTALLATION.md** - Setup and PyTorch > 2.7 notes
- **TORCH_UPDATE.md** - PyTorch migration details
- **GRADIO_GUIDE.md** - Comprehensive Gradio user guide
- **test_gradio_setup.py** - Automated verification

---

## Next Steps

1. **Test**: Run `python test_gradio_setup.py`
2. **Launch**: Run `python gradio_app.py`
3. **Explore**: Test with sample cloud and different queries
4. **Customize**: Upload your own PCD files
5. **Build optional**: Install FlashAttention for speed boost

---

## Support & Resources

- **Find3D Paper**: https://arxiv.org/abs/2411.13550
- **Project Page**: https://ziqi-ma.github.io/find3dsite/
- **GitHub**: https://github.com/ziqi-ma/Find3D
- **Gradio Docs**: https://www.gradio.app/

---

## Summary

✅ **Complete Gradio Interface Created**
- Fully functional web UI with all parameters exposed
- Tested and verified on PyTorch 2.10
- Compatibility shims for missing dependencies
- Comprehensive documentation
- Ready for immediate use

**Start now**: `python gradio_app.py` → http://localhost:7860

---

**Created**: February 17, 2026  
**PyTorch Version**: 2.10.0+cu128  
**Gradio Version**: 6.5.1  
**Status**: ✅ Production Ready
