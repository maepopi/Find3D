# 3D to Point Cloud Converter Guide

**Convert your 3D models to point clouds for use with Find3D!**

## Overview

Find3D works with **point clouds** (3D data represented as a collection of points in space with colors and normals). This converter helps you transform 3D model files into the `.pcd` format that Find3D expects.

### Supported Input Formats
- **.obj** - Wavefront OBJ (most common)
- **.glb** / **.gltf** - glTF 2.0 (modern 3D exchange format)
- **.ply** - Polygon File Format
- **.stl** - Stereolithography (CAD models, 3D printing)
- **.off** - Object File Format
- **.fbx** - Autodesk FBX
- And more via Open3D and trimesh support

### Output Format
- **.pcd** - Point Cloud Data (Open3D format)

---

## Quick Start

### Option 1: Gradio Web Interface (Easiest)

1. **Launch the interface:**
   ```bash
   python gradio_app.py
   ```
   
2. **Navigate to the "🔄 3D → Point Cloud Converter" tab**

3. **Upload your 3D model** (drag and drop or click)

4. **Configure settings:**
   - **Points to Sample**: Higher = more detail but slower (10K-50K is typical)
   - **Sampling Method**: 
     - `poisson`: Uniform distribution (recommended)
     - `random`: Quick sampling
   - **Coloring**: 
     - `height`: Z-axis gradient (blue bottom → red top)
     - `random`: Colorful variation
     - `vertex`: Use original mesh colors if available

5. **Click "🚀 Convert to PCD"**

6. **Download** the resulting `.pcd` file

7. **Switch to the "🧠 Inference" tab** and upload your converted point cloud to run Find3D!

---

### Option 2: Command Line

```bash
# Basic conversion
python convert_3d_to_pcd.py model.obj

# With custom parameters
python convert_3d_to_pcd.py model.glb -o output.pcd -n 50000 -m poisson

# With visualization
python convert_3d_to_pcd.py model.ply --visualize

# Batch convert entire directory
python convert_3d_to_pcd.py --batch ./models/ --output ./pointclouds/
```

#### Command Line Options

```
usage: convert_3d_to_pcd.py [-h] [-o OUTPUT] [-n NUM_POINTS] [-m {poisson,random}]
                            [-c {vertex,height,random}] [--no-normalize]
                            [--visualize] [--batch] [-q]
                            [input]

Convert 3D models to point clouds (.pcd format)

positional arguments:
  input                    Input 3D model file or directory (for batch mode)

optional arguments:
  -h, --help              Show help message
  -o, --output OUTPUT     Output .pcd file (default: auto-generated)
  -n, --num-points NUM    Number of points to sample (default: 10000)
  -m, --method {poisson,random}
                         Sampling method (default: poisson)
  -c, --color {vertex,height,random}
                         Coloring method (default: height)
  --no-normalize         Skip centering and scaling
  --visualize            Show 3D visualization of result
  --batch                Batch convert all models in directory
  -q, --quiet            Suppress output messages
```

---

## Python API

Use the converter in your own Python code:

```python
from convert_3d_to_pcd import convert_3d_to_pcd

# Simple conversion
pcd_path = convert_3d_to_pcd(
    input_file="model.obj",
    output_file="model_pointcloud.pcd",
    num_points=10000,
    sampling_method="poisson",
    color_method="height"
)

print(f"Converted: {pcd_path}")
```

### Advanced Usage

```python
from convert_3d_to_pcd import (
    load_mesh,
    sample_point_cloud,
    colorize_point_cloud,
    normalize_point_cloud,
    visualize_point_cloud
)
import open3d as o3d

# Step by step control
mesh, loader = load_mesh("model.obj")
print(f"Loaded with {loader}")

pcd = sample_point_cloud(mesh, num_points=50000, method='poisson')
pcd = colorize_point_cloud(pcd, mesh, method='height')
pcd = normalize_point_cloud(pcd, scale=0.75)

# Visualize before saving
visualize_point_cloud(pcd, "My Point Cloud")

# Save
o3d.io.write_point_cloud("output.pcd", pcd)
```

### Batch Conversion

```python
from convert_3d_to_pcd import batch_convert

# Convert all 3D models in a directory
converted_files = batch_convert(
    input_dir="./models/",
    output_dir="./pointclouds/",
    num_points=10000,
    sampling_method="poisson"
)

print(f"Converted {len(converted_files)} files")
```

---

## Understanding Point Clouds

A point cloud is a collection of points in 3D space, each with:
- **XYZ coordinates**: Position in 3D space
- **RGB colors** (0-1 or 0-255): Color information
- **Normal vectors** (optional): Surface orientation (computed by converter)

Find3D uses this data to:
1. **Understand the shape**: XYZ positions form the 3D structure
2. **Process appearance**: RGB colors help identify distinct regions
3. **Compute geometry**: Normal vectors aid in understanding surfaces

### Coloring Methods Explained

| Method | What it Does | Best For |
|--------|-------------|----------|
| **height** | Colors gradient from blue (bottom) to red (top) based on Z-axis | Visual inspection, understanding orientation |
| **random** | Assigns random colors to add visual variation | Variety when original colors are unavailable |
| **vertex** | Preserves colors from the original 3D model | When input mesh has meaningful colors/textures |

---

## Typical Usage Workflow

1. **Prepare your 3D model**
   - Ensure the file is valid (import in Blender/Meshlab if unsure)
   - Consider model scale (normalization will handle this)
   - Optional: Clean mesh (remove holes, isolated vertices)

2. **Convert with appropriate settings**
   ```bash
   python convert_3d_to_pcd.py model.obj -n 30000 -m poisson
   ```

3. **Visualize (optional)**
   ```bash
   python convert_3d_to_pcd.py model.obj --visualize
   ```

4. **Use in Find3D**
   - Launch Gradio app: `python gradio_app.py`
   - Go to "🧠 Inference" tab
   - Uncheck "Use Sample Point Cloud"
   - Upload the `.pcd` file
   - Enter queries (e.g., "handle, blade, tip")
   - Run inference!

---

## Performance Considerations

### Point Count Impact

| Points | Quality | Speed | File Size |
|--------|---------|-------|-----------|
| 5,000 | Low | Very Fast | Small |
| **10,000** | **Standard** | **Fast** | **Medium** |
| 30,000 | High | Moderate | Large |
| 50,000+ | Very High | Slow | Very Large |

**Recommendation**: Start with 10,000-20,000 points. Increase if you need more detail for complex shapes, decrease for faster iteration.

### Sampling Methods

- **Poisson disk**: Produces more uniform point distribution. Slower but higher quality. ✓ **Recommended**
- **Random uniform**: Fast but can create clusters. Use when speed matters.

---

## Troubleshooting

### "Could not load mesh"
- **Check**: Is the file format supported? (obj, glb, ply, stl, off, gltf)
- **Check**: Is the file valid? Try opening in Blender or MeshLab
- **Check**: Are file permissions correct?

### "File not found"
- **Check**: Is the file path correct? Use absolute or relative from current directory
- **Check**: Does the file actually exist? (typo in filename?)

### Large file size or slow conversion
- **Solution**: Reduce point count (`-n 5000`or `--num-points 5000`)
- **Solution**: Use `random` sampling method instead of `poisson`
- **Note**: Normal processing takes 5-30 seconds depending on model complexity

### Point cloud looks wrong
- **Check**: Is the model right-side up? Try different color methods
- **Check**: Scale issues? Converter normalizesautomatically, but try `--no-normalize` to see original scale
- **Solution**: Visualize with `--visualize` to debug

### "trimesh not installed"
- This is **optional** - the converter uses Open3D by default
- If you want trimesh support: `pip install trimesh`

---

## Integration with Find3D Workflow

### Complete Pipeline

```
Your 3D Model
      ↓
[3D → PCD Converter] ← You are here
      ↓
Point Cloud (.pcd)
      ↓
[Find3D Inference] ← Upload to Gradio, run queries
      ↓
Part Segmentation / Heatmaps Results
```

### Example: Convert and Test

```bash
# Convert a model
python convert_3d_to_pcd.py chair.obj -n 20000

# This creates: chair_pointcloud.pcd

# In another terminal, run Gradio
python gradio_app.py

# Then in web UI:
# 1. Tab: "🔄 3D → Point Cloud Converter"
# 2. Tab: "🧠 Inference" 
# 3. Upload chair_pointcloud.pcd
# 4. Query: "legs, back, seat"
```

---

## Advanced Topics

### Mesh Preprocessing

If you need to clean or modify meshes before conversion:

```python
import open3d as o3d

# Load mesh
mesh = o3d.io.read_triangle_mesh("model.obj")

# Remove small isolated components
mesh = mesh.remove_degenerate_triangles()

# Simplify (reduce polygon count)
mesh = mesh.simplify_quadric_decimation(target_count=50000)

# Clean
mesh.remove_unreferenced_vertices()

# Save cleaned mesh
o3d.io.write_triangle_mesh("cleaned.obj", mesh)

# Now convert to point cloud
from convert_3d_to_pcd import convert_3d_to_pcd
convert_3d_to_pcd("cleaned.obj", num_points=30000)
```

### Custom Coloring

```python
import numpy as np
import open3d as o3d
from convert_3d_to_pcd import load_mesh, sample_point_cloud, normalize_point_cloud

# Load and sample
mesh, _ = load_mesh("model.obj")
pcd = sample_point_cloud(mesh, num_points=10000)
pcd = normalize_point_cloud(pcd)

# Custom coloring by distance from center
points = np.asarray(pcd.points)
distance = np.linalg.norm(points, axis=1)
distance = (distance - distance.min()) / (distance.max() - distance.min())

colors = np.zeros((len(points), 3))
colors[:, 0] = distance  # Red for far
colors[:, 2] = 1 - distance  # Blue for near

pcd.colors = o3d.utility.Vector3dVector(colors)
o3d.io.write_point_cloud("custom_colored.pcd", pcd)
```

---

## FAQ

**Q: What's the difference between .pcd and .ply point clouds?**
A: Both store point clouds, but Find3D expects `.pcd` format (Open3D's native format). The converter handles the conversion.

**Q: Can I edit the .pcd file after conversion?**
A: Yes! Use Open3D, CloudCompare, or other point cloud software. The format is text-based and editable.

**Q: Will normalization change my data?**
A: Yes, but it's beneficial! Normalization centers the cloud and scales it to a fixed size, which improves Find3D's processing. Use `--no-normalize` to preserve original coordinates.

**Q: How many points should I use?**
A: 10K-20K is a good balance. More points → better detail but slower processing. Fewer points → faster but may miss small parts.

**Q: Can I convert point clouds from other sources?**
A: Absolutely! Point clouds from scanners, photogrammetry, or other algorithms can be used. Just ensure they're in `.pcd` format (or convert using Open3D if in other formats like `.ply`).

---

## References

- [Open3D Documentation](http://www.open3d.org/)
- [Trimesh Documentation](https://trimesh.org/)
- [Point Cloud Data Format Spec](https://pointclouds.org/)
- [Find3D Paper](https://arxiv.org/abs/2411.13550)

---

**Need help?** Check the [gradio_app.py](gradio_app.py) interface or run in verbose mode for detailed logs.
