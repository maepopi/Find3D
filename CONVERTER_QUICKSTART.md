# Find3D with 3D to Point Cloud Converter - Complete Setup Guide

## 🎯 What You Have

A complete Find3D setup with:
- **Find3D Inference Engine**: GPU-accelerated 3D part segmentation via natural language
- **3D Model Converter**: Transform any 3D file format into point clouds
- **Gradio Web Interface**: Easy-to-use UI combining both tools
- **Example Models**: Pre-made test shapes (cube, sphere, pyramid, torus)

---

## 🚀 Quick Start (2 Minutes)

### Launch the Web Interface

```bash
cd /home/maelys/WSL_AI_HUB/TOOLS/Find3D
python gradio_app.py
```

Open your browser to: **http://localhost:7860**

---

## 📊 Two-Tab Interface

### Tab 1: 🔄 3D → Point Cloud Converter

**Input**: 3D model files (.obj, .glb, .ply, .stl, .off, .gltf, etc.)
**Output**: Point cloud files (.pcd)

1. **Upload your 3D model** (drag & drop)
2. **Configure**:
   - Points to Sample: 10,000 typical (5K=fast, 30K=detailed)
   - Sampling Method: Poisson (uniform) or Random (fast)
   - Coloring: Height gradient (recommended) or Random
3. **Click "🚀 Convert to PCD"**
4. **Download** the `.pcd` file

#### Example Workflow

```bash
# Using provided example models
ls example_models/
# → cube.obj, sphere.obj, pyramid.obj, torus.obj
```

### Tab 2: 🧠 Inference

**Input**: Point cloud files (.pcd) + text queries
**Output**: Part segmentation or confidence heatmaps

1. **Upload point cloud** (or use sample)
2. **Enter queries**: "wheel, handle, frame" (comma-separated)
3. **Configure**:
   - Mode: Segmentation or Heatmap
   - Temperature: Control boundary sharpness (0.1=sharp, 2.0=soft)
   - Seed: For reproducibility
4. **Click "🚀 Run Inference"**
5. **View results**: Output, Analysis, Raw JSON data

---

## 📁 File Structure

```
Find3D/
├── gradio_app.py                 # Main web interface
├── convert_3d_to_pcd.py         # 3D to PCD converter
├── generate_example_models.py   # Create test shapes
│
├── example_models/              # Test data
│   ├── cube.obj
│   ├── sphere.obj
│   ├── pyramid.obj
│   └── torus.obj
│
├── converted/                   # Batch converted PCDs
│   ├── cube_pointcloud.pcd
│   ├── sphere_pointcloud.pcd
│   ├── pyramid_pointcloud.pcd
│   └── torus_pointcloud.pcd
│
├── 3D_CONVERTER_GUIDE.md        # Detailed converter docs
├── GRADIO_INTERFACE.md          # Interface documentation
├── model/                       # Find3D model code
├── common/                      # Shared utilities
└── .venv/                       # Python environment
```

---

## 🔧 Command Line Tools

### Convert Single File

```bash
# Basic (10,000 points, height coloring)
python convert_3d_to_pcd.py model.obj

# Custom parameters
python convert_3d_to_pcd.py model.glb \
  -o my_output.pcd \
  -n 50000 \
  -m poisson \
  -c height

# With visualization
python convert_3d_to_pcd.py model.ply --visualize
```

### Batch Convert

```bash
# Convert all models in a directory
python convert_3d_to_pcd.py \
  --batch ./my_models/ \
  --output ./point_clouds/ \
  -n 20000
```

### Generate Example Models

```bash
# Create test shapes again
python generate_example_models.py

# Custom location
python generate_example_models.py ./my_test_shapes/
```

---

## 💻 Python API

### Single Conversion

```python
from convert_3d_to_pcd import convert_3d_to_pcd

output_path = convert_3d_to_pcd(
    input_file="chair.glb",
    output_file="chair.pcd",
    num_points=20000,
    sampling_method="poisson",
    color_method="height",
    normalize=True
)

print(f"Saved: {output_path}")
```

### Batch Conversion

```python
from convert_3d_to_pcd import batch_convert

paths = batch_convert(
    input_dir="./models/",
    output_dir="./point_clouds/",
    num_points=15000
)

print(f"Converted {len(paths)} files")
```

### Find3D Inference

```python
from model.evaluation.utils import (
    load_model, 
    preprocess_pcd, 
    encode_text,
    read_pcd
)
import torch

# Load model
model = load_model()
model.eval()

# Load point cloud
xyz, rgb, normal = read_pcd("chair.pcd")
xyz_full, xyz_sub = preprocess_pcd(xyz, normal)

# Encode text queries
queries = ["legs", "backrest", "seat"]
with torch.no_grad():
    text_embeds = encode_text(queries)

# Run inference
with torch.no_grad():
    logits = model(xyz_sub, text_embeds, temperature=1.0)[0]
    predictions = logits.argmax(dim=1)
```

---

## 📋 Common Tasks

### Convert and Test in One Go

```bash
# 1. Convert a model
python convert_3d_to_pcd.py my_model.obj -n 20000

# 2. Launch interface
python gradio_app.py

# 3. In web UI:
#    - Go to "Inference" tab
#    - Upload "my_model_pointcloud.pcd"
#    - Enter queries
#    - Click "Run Inference"
```

### Test Example Models

```bash
# Convert examples
python convert_3d_to_pcd.py --batch example_models/

# Launch interface
python gradio_app.py

# Try different queries:
# - Cube: "upper face, lower face, side"
# - Sphere: "top hemisphere, bottom hemisphere"
# - Pyramid: "base, apex"
# - Torus: "outer surface, inner hole"
```

### Add Your Own 3D Models

```bash
# Place your files in a directory
mkdir my_models
cp /path/to/*.obj my_models/
cp /path/to/*.glb my_models/

# Convert all at once
python convert_3d_to_pcd.py --batch my_models/ --output my_pcds/

# Upload and test in web interface
python gradio_app.py
```

---

## 🎓 Understanding Point Clouds

A **point cloud** is simply 3D data represented as:
- **XYZ**: 3D coordinates (position in space)
- **RGB**: Colors (0-1 or 0-255)
- **Normals** (optional): Surface direction vectors

### Coloring Methods

| Method | Visual | Best For |
|--------|--------|----------|
| **height** | Blue (bottom) → Red (top) | Understanding orientation, debugging |
| **random** | Colorful variation | Visual interest when original colors unavailable |
| **vertex** | Original mesh colors | Preserving meaningful colors/textures |

---

## ⚡ Performance Tips

### Point Count vs Speed

| Points | Quality | Speed | Memory |
|--------|---------|-------|--------|
| 5,000 | Low | ⚡ Fast | Tiny |
| **10,000** | **Normal** | **Fast** | **Small** |
| 20,000 | High | Moderate | Medium |
| 50,000+ | Very High | Slow | Large |

**Recommendation**: Start with 10,000-15,000 for best balance

### Speeding Up Conversions

```bash
# Use random sampling instead of poisson
python convert_3d_to_pcd.py model.obj -m random -n 5000

# Skip normalization if not needed
python convert_3d_to_pcd.py model.obj --no-normalize
```

---

## 🐛 Troubleshooting

### "Could not load mesh"
- Check file format is supported (.obj, .glb, .ply, .stl, .off, .gltf)
- Verify file is not corrupted (open in Blender/MeshLab)
- Ensure file path is correct

### "Model loading failed" in Gradio
- Check CUDA is available: `nvidia-smi`
- Verify environment: `source .venv/bin/activate`
- Re-run verification: `python test_gradio_setup.py`

### Point cloud looks wrong
- Try different color method: `-c height` vs `-c random`
- Check original model orientation (might be upside down)
- Visualize before saving: `python convert_3d_to_pcd.py model.obj --visualize`

### Slow inference
- Reduce point count: `-n 5000`
- Check GPU usage: `nvidia-smi`
- Simplify model geometry before conversion

### "trimesh not installed"
- This is *optional* - not required for functionality
- Install if needed: `pip install trimesh`

---

## 📚 Documentation Files

- **3D_CONVERTER_GUIDE.md**: Detailed converter documentation
- **GRADIO_INTERFACE.md**: Web interface guide
- **GRADIO_README.md**: Additional Gradio setup notes

---

## 🔑 Key Concepts

### Find3D Processing Pipeline

```
Your 3D Model
    ↓
[3D Converter]  ← Convert to point cloud
    ↓
Point Cloud (.pcd)
    ↓
[Point Preprocessing]  ← Sample, normalize
    ↓
[Text Encoder]  ← Convert queries to embeddings
    ↓
[Find3D Model]  ← GPU inference
    ↓
Part Segmentation + Confidence Scores
```

### Query Tips

- ✅ **Specific**: "front wheel, back wheel, axle"
- ✅ **Clear**: "chair leg, seat surface, backrest"
- ❌ **Vague**: "parts of the object"
- ❌ **Mixed**: "wheel and the circular thing"

---

## 🌟 Next Steps

1. **Test with Examples**
   ```bash
   python gradio_app.py
   # Try the example models in both tabs
   ```

2. **Add Your Models**
   - Convert your own 3D files
   - Test different point counts and colors
   - Experiment with temperature values

3. **Batch Processing**
   ```bash
   python convert_3d_to_pcd.py --batch ./my_models/
   ```

4. **Integration**
   - Use converter in your Python code
   - Automate batch workflows
   - Build custom frontends

---

## 📞 Support

- **Find3D Paper**: https://arxiv.org/abs/2411.13550
- **Project Site**: https://ziqi-ma.github.io/find3dsite/
- **GitHub**: https://github.com/ziqi-ma/Find3D

---

## ✅ Verification Checklist

```bash
# Verify everything works
python test_gradio_setup.py

# Should output: ✓ ALL CHECKS PASSED

# Launch interface
python gradio_app.py

# Test converter
python convert_3d_to_pcd.py example_models/cube.obj

# Test batch
python convert_3d_to_pcd.py --batch example_models/
```

---

**Ready to find any part in any 3D object?** 🚀

Launch the interface: `python gradio_app.py`
