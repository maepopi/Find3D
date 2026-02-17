# Find3D Gradio Interface Guide

## Quick Start

### 1. Activate the Environment

```bash
# Option A: Using convenience script
source activate_find3d.sh

# Option B: Manual activation
export PATH="$HOME/.local/bin:$PATH"
source .venv/bin/activate
```

### 2. Run the Gradio App

```bash
python gradio_app.py
```

You should see output like:
```
🚀 Starting Find3D Gradio Interface
📍 Device: cuda
✓ CUDA available: True

Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://xxxx-xx-xxx-xxx.gradio.live
```

### 3. Access the Interface

- **Local**: Open `http://localhost:7860` in your browser
- **Remote**: Use the public URL shown in the output

---

## Features

### 📥 Input Methods

1. **Sample Point Cloud** (default)
   - Generates a synthetic 3D shape for testing
   - Adjustable point count (1,000 - 100,000 points)
   - No file upload needed

2. **Upload PCD File**
   - Uncheck "Use Sample Point Cloud"
   - Upload your own `.pcd` file
   - Automatically preprocessed

### 📝 Text Queries

Enter comma-separated descriptions of parts to find:

**Examples:**
- `handle, blade, tip` - for tools
- `wheel, door, window` - for cars
- `leg, seat, back rest` - for chairs
- `cup, handle` - for dishware

### ⚙️ Advanced Parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| **Temperature** | 0.1 - 2.0 | 1.0 | Controls prediction sharpness |
| **Output Mode** | segmentation, heatmap | segmentation | Hard assignment or soft scores |
| **Random Seed** | 0 - 9999 | 123 | Reproducibility |

### Temperature Guide

- **Low (0.1 - 0.5)**: Sharp boundaries, high confidence
- **Medium (0.8 - 1.2)**: Balanced results
- **High (1.5 - 2.0)**: Smooth gradients, uncertain

### Output Modes

**Segmentation**
- Each point assigned to one part or background
- Includes confidence levels
- Shows point counts and percentages per class

**Heatmap**
- Per-query confidence scores
- Shows min/max/mean confidence for each query
- Useful for understanding model confidence

---

## Results Interpretation

### 📊 Status & Results Tab

Shows step-by-step processing information:
```
✓ Created sample point cloud with 10,000 points
✓ Parsed 3 queries: handle, blade, tip
⏳ Preprocessing point cloud...
⏳ Encoding text queries...
⏳ Loading Find3D model...
⏳ Running inference...
✓ Inference complete!

📊 SEGMENTATION RESULTS
Total points: 10,000
Classes: 4 (0=background, 1-3=parts)
Confidence range: [0.9999]

Class distribution:
  [0] Background: 6,250 (62.5%)
  [1] handle: 2,100 (21.0%)
  [2] blade: 1,500 (15.0%)
  [3] tip: 150 (1.5%)
```

### 📈 Analysis Tab

Detailed statistics and parameters used

### 🔬 Raw Data Tab

Complete results in JSON format (for advanced users)

---

## Troubleshooting

### Issue: CUDA out of memory

**Solution**: 
- Reduce sample point count
- Use heatmap mode instead of segmentation
- Close other GPU-intensive applications

### Issue: Model loading fails

**Solution**:
```bash
# Ensure weights are downloaded
python -c "from model.evaluation.utils import load_model; load_model()"
```

### Issue: Slow inference

**Solution**:
- Reduce point count
- Use smaller sample size
- Ensure CUDA is being used (check "Device" in results)

### Issue: Queries produce no matches

**Possible causes**:
- Queries don't match object parts
- Language too complex or vague
- Temperature too low (sharp boundaries)

**Try**:
- Use simpler, more specific terms
- Increase temperature slightly
- Use sample cloud first to validate setup

---

## Performance Tips

### For Speed
- Start with 5,000-10,000 points
- Use segmentation mode (faster than heatmap inspection)
- Keep queries to 2-3 terms

### For Accuracy
- Increase point count if GPU allows
- Use descriptive, specific terms
- Experiment with temperature values
- Use heatmap mode to verify confidence

### For GPU Memory
- Monitor with `nvidia-smi` while running:
  ```bash
  watch -n 0.5 nvidia-smi
  ```
- Reduce point count if usage > 80%
- Restart interface after heavy workloads

---

## Advanced Usage

### Batch Processing

For multiple point clouds, you can script the interface:

```python
import gradio as gr
from gradio_app import run_inference

# Process multiple files
results_list = []
for pcd_file in ["file1.pcd", "file2.pcd"]:
    results, status = run_inference(
        file_input=pcd_file,
        use_sample=False,
        num_sample_points=10000,
        queries="wheel, door, window",
        mode="segmentation",
        temperature=1.0,
        seed=123
    )
    results_list.append(results)
```

### Custom Point Clouds

In `gradio_app.py`, modify `create_sample_point_cloud()`:

```python
def create_sample_point_cloud(num_points: int = 10000):
    # Your custom point cloud generation here
    xyz = ...  # n x 3
    rgb = ...  # n x 3, values 0-1
    normal = ...  # n x 3, normalized
    
    return xyz.astype(np.float32), rgb.astype(np.float32), normal.astype(np.float32)
```

---

## System Requirements

- **GPU**: NVIDIA GPU with CUDA 12.8+
- **Memory**: 8GB+ VRAM (for 10K+ points)
- **Python**: 3.10+
- **Browser**: Modern browser with JavaScript enabled

### Check System

```bash
# Check CUDA
nvidia-smi

# Check Python
python --version

# Verify PyTorch
python -c "import torch; print(torch.cuda.is_available())"
```

---

## Building Required Components

Before running heavy inference, build recommended components:

```bash
# FlashAttention (optional, for speed)
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
MAX_JOBS=4 python setup.py install
cd ..

# Pointcept (required for model backbone)
git clone https://github.com/Pointcept/Pointcept.git
cd Pointcept/libs/pointops
python setup.py install
cd ../../..
```

---

## Additional Resources

- **Find3D Paper**: https://arxiv.org/abs/2411.13550
- **Project Page**: https://ziqi-ma.github.io/find3dsite/
- **GitHub**: https://github.com/ziqi-ma/Find3D
- **Gradio Docs**: https://www.gradio.app/docs/

---

## Citation

If you use Find3D in your research, please cite:

```bibtex
@inproceedings{ma2024find3d,
  title={Find3D: Find Any Part in 3D},
  author={Ma, Ziqi and Yue, Yisong and Gkioxari, Georgia},
  booktitle={ICCV},
  year={2025}
}
```

---

**Last Updated**: February 17, 2026  
**Gradio Version**: 6.5.1+  
**PyTorch Version**: 2.10.0+
