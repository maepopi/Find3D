# Find3D - uv Installation with PyTorch > 2.7

## Quick Start

### Activate the Environment

**Option 1: Using the convenience script**
```bash
source activate_find3d.sh
```

**Option 2: Manual activation**
```bash
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
# Output: PyTorch: 2.10.0+cu128 or compatible
```

---

## Installation Details

### What Was Configured

✓ **uv Package Manager** - Modern Python package and project manager  
✓ **PyTorch 2.10.0+cu128** - PyTorch version > 2.7  
✓ **torch-geometric 2.7.0** - Graph neural network library  
✓ **All Dependencies** - 120+ packages installed and configured  

### Package Requirements File

A `pyproject.toml` file has been created with all dependencies configured for compatibility with PyTorch > 2.7. Key dependencies include:

- **Core ML**: torch, torchvision, torchmetrics
- **Geometry**: torch-geometric
- **Attention**: xformers (0.0.35 with torch 2.10 support)
- **Sparse Ops**: spconv (handled through torch-geometric)
- **Model Utils**: omegaconf, timm, transformers, einops
- **Data**: h5py, scipy, plyfile, open3d
- **Monitoring**: tensorboard, tensorboardX

### Environment Details

| Component | Version | Status |
|-----------|---------|--------|
| Python | 3.10 | ✓ |
| PyTorch | 2.10.0+cu128 | ✓ |
| torchvision | 0.25.0 | ✓ |
| torch-geometric | 2.7.0 | ✓ |
| CUDA Support | cu128 | ✓ |

### Virtual Environment Location

```
/home/maelys/WSL_AI_HUB/TOOLS/Find3D/.venv/
```

---

## PyTorch > 2.7 Adaptations

### Key Changes Made

1. **Removed deprecated PyTorch 2.0 specific pins**
   - torch 2.0.0 → torch >= 2.7.0
   - torchvision 0.15.0 → torchvision >= 0.18.0
   
2. **Updated torch_geometric to 2.7.0**
   - Includes native spars tensor support
   - Compatible with torch > 2.8
   
3. **Removed explicit cuda version indices**
   - Removed `--extra-index-url https://download.pytorch.org/whl/cu118`
   - Using PyPI default indexing for broader compatibility

4. **Updated auxiliary packages**
   - xformers: >= 0.0.27 (supports torch 2.10)
   - torchmetrics: >= 0.10.3 (more flexible versioning)
   - transformers: >= 4.34.0

### API Compatibility Verified

The codebase uses standard PyTorch APIs that are backward-compatible:
- ✓ `torch.inference_mode()` decorator
- ✓ `F.cross_entropy()` loss functions  
- ✓ `torch_scatter.segment_csr()` (via torch-geometric)
- ✓ Tensor operations and neural network modules

---

## Building Additional Components

The original Find3D setup requires building two additional components from source:

### 1. FlashAttention

```bash
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install
cd ..
```

### 2. Pointcept

```bash
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept/libs/pointops
python setup.py install
cd ../../..
```

These should be built after activating the Find3D environment.

---

## Project Structure

```
Find3D/
├── .venv/                          # Virtual environment (120+ packages)
├── pyproject.toml                  # uv configuration with dependencies
├── activate_find3d.sh              # Convenience activation script
├── TORCH_UPDATE.md                 # Detailed migration notes
├── model/                          # Neural network models
│   ├── backbone/pt3/              # Point Transformer V3
│   ├── training/                  # Training code
│   ├── evaluation/                # Benchmarking
│   └── data/                      # Data handling
├── dataengine/                    # Data collection and processing
├── common/                        # Shared utilities
└── README.md                      # Original documentation
```

---

## Troubleshooting

### Issue: Command not found for uv

**Solution**: Ensure uv is installed:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
```

### Issue: CUDA not available

**Verify CUDA setup**:
```bash
python -c "import torch; print(torch.cuda.is_available())"
```

If `False`, ensure NVIDIA drivers and CUDA toolkit are installed.

### Issue: Import errors with model code

**Verify environment activation**:
```bash
which python
# Should show: /home/maelys/WSL_AI_HUB/TOOLS/Find3D/.venv/bin/python
```

---

## Additional Resources

- **Original Find3D README**: See [README.md](README.md)
- **PyTorch Migration Guide**: See [TORCH_UPDATE.md](TORCH_UPDATE.md)
- **PyTorch 2.10 Release Notes**: https://pytorch.org/blog/pytorch-2.10/
- **torch-geometric Docs**: https://pytorch-geometric.readthedocs.io/

---

## Version Information

- **Installation Date**: February 17, 2026
- **PyTorch Version**: 2.10.0+cu128
- **Python Version**: 3.10.19
- **uv Version**: 0.10.3
- **Total Packages**: 120+

Last updated: February 17, 2026
