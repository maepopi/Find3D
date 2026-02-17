"""
3D Model to Point Cloud Converter
Converts various 3D mesh formats (.obj, .glb, .ply, .stl, etc.) to point clouds (.pcd)
"""

import os
import sys
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path

try:
    import trimesh
    TRIMESH_AVAILABLE = True
except ImportError:
    TRIMESH_AVAILABLE = False
    print("Warning: trimesh not installed. Install with: pip install trimesh")


def load_mesh(filepath):
    """
    Load a 3D mesh from various formats
    
    Supports: .obj, .glb, .ply, .stl, .off, .gltf, and more
    
    Args:
        filepath: Path to 3D model file
        
    Returns:
        mesh: Open3D TriangleMesh or trimesh Mesh object
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    suffix = filepath.suffix.lower()
    
    # Try open3d first (most formats)
    try:
        mesh = o3d.io.read_triangle_mesh(str(filepath))
        if mesh.has_vertices():
            return mesh, 'open3d'
    except Exception as e:
        print(f"Open3D failed: {e}")
    
    # Fallback to trimesh for additional formats
    if TRIMESH_AVAILABLE:
        try:
            mesh = trimesh.load(str(filepath))
            return mesh, 'trimesh'
        except Exception as e:
            print(f"Trimesh failed: {e}")
    
    raise ValueError(f"Could not load mesh from {filepath}. Supported formats: .obj, .glb, .ply, .stl, .off, .gltf")


def sample_point_cloud(mesh, num_points=10000, method='poisson'):
    """
    Sample points from a 3D mesh surface
    
    Args:
        mesh: Loaded mesh object
        num_points: Number of points to sample
        method: 'poisson' (uniform) or 'random'
        
    Returns:
        pcd: Open3D PointCloud object
    """
    
    mesh_type = type(mesh).__name__
    
    if mesh_type == 'TriangleMesh':  # Open3D mesh
        # Ensure mesh has normals
        if not mesh.has_vertex_normals():
            mesh.compute_vertex_normals()
        
        if method == 'poisson':
            # Poisson disk sampling (more uniform)
            pcd = mesh.sample_points_poisson_disk(num_points)
        else:
            # Random uniform sampling
            pcd = mesh.sample_points_uniformly(num_points)
    
    elif mesh_type == 'Trimesh':  # trimesh Mesh
        # Convert to Open3D for consistent output
        vertices = mesh.vertices
        faces = mesh.faces
        
        mesh_o3d = o3d.geometry.TriangleMesh()
        mesh_o3d.vertices = o3d.utility.Vector3dVector(vertices)
        mesh_o3d.triangles = o3d.utility.Vector3iVector(faces)
        mesh_o3d.compute_vertex_normals()
        
        if method == 'poisson':
            pcd = mesh_o3d.sample_points_poisson_disk(num_points)
        else:
            pcd = mesh_o3d.sample_points_uniformly(num_points)
    
    else:
        raise TypeError(f"Unknown mesh type: {mesh_type}")
    
    return pcd


def colorize_point_cloud(pcd, mesh=None, method='vertex'):
    """
    Add colors to point cloud
    
    Args:
        pcd: Point cloud object
        mesh: Original mesh (optional)
        method: 'vertex' (from mesh), 'height' (z-axis), 'random'
        
    Returns:
        pcd: Colored point cloud
    """
    points = np.asarray(pcd.points)
    
    if method == 'vertex' and mesh is not None:
        # Use vertex colors from mesh if available
        mesh_type = type(mesh).__name__
        if mesh_type == 'TriangleMesh' and mesh.has_vertex_colors():
            colors = np.asarray(mesh.vertex_colors)
            # Resample to match pcd points (approximate)
            colors = colors[np.random.randint(0, len(colors), len(points))]
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            # Use height-based coloring
            method = 'height'
    
    if method == 'height':
        # Color by z-coordinate (height)
        z = points[:, 2]
        z_min, z_max = z.min(), z.max()
        z_norm = (z - z_min) / (z_max - z_min + 1e-8)
        
        # Create colormap: blue (bottom) -> red (top)
        colors = np.zeros((len(points), 3))
        colors[:, 0] = z_norm  # Red channel
        colors[:, 2] = 1 - z_norm  # Blue channel
        
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    elif method == 'random':
        # Random colors
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(len(points), 3))
    
    return pcd


def normalize_point_cloud(pcd, scale=0.75):
    """
    Normalize point cloud (center and scale)
    
    Args:
        pcd: Point cloud object
        scale: Target size (0-1 range)
        
    Returns:
        pcd: Normalized point cloud
    """
    points = np.asarray(pcd.points)
    
    # Center
    center = points.mean(axis=0)
    points = points - center
    
    # Scale
    max_dist = np.linalg.norm(points, axis=1).max()
    if max_dist > 1e-8:
        points = points / max_dist * scale
    
    pcd.points = o3d.utility.Vector3dVector(points)
    
    return pcd


def visualize_point_cloud(pcd, title="Point Cloud"):
    """
    Visualize point cloud
    
    Args:
        pcd: Point cloud object
        title: Window title
    """
    print(f"Visualizing {title}...")
    print(f"  Points: {len(pcd.points):,}")
    print(f"  Colors: {pcd.has_colors()}")
    print(f"  Normals: {pcd.has_normals()}")
    print("\nControls:")
    print("  Left click + drag: Rotate")
    print("  Scroll: Zoom")
    print("  Right click + drag: Pan")
    print("  Q: Close window")
    
    o3d.visualization.draw_geometries([pcd], window_name=title)


def convert_3d_to_pcd(
    input_file,
    output_file=None,
    num_points=10000,
    sampling_method='poisson',
    color_method='height',
    normalize=True,
    visualize=False,
    verbose=True
):
    """
    Main conversion function
    
    Args:
        input_file: Path to input 3D model
        output_file: Path to output .pcd file (auto-generated if None)
        num_points: Number of points to sample
        sampling_method: 'poisson' or 'random'
        color_method: 'vertex', 'height', or 'random'
        normalize: Whether to center and scale
        visualize: Whether to show visualization
        verbose: Print progress messages
        
    Returns:
        output_path: Path to saved .pcd file
    """
    
    if verbose:
        print("\n" + "="*60)
        print("3D Model to Point Cloud Converter")
        print("="*60)
    
    input_path = Path(input_file)
    
    # Generate output path if not provided
    if output_file is None:
        output_file = input_path.stem + "_pointcloud.pcd"
    output_path = Path(output_file)
    
    # Step 1: Load mesh
    if verbose:
        print(f"\n[1/5] Loading mesh: {input_path}")
    mesh, loader = load_mesh(input_path)
    if verbose:
        print(f"      Loaded with {loader}")
        if hasattr(mesh, 'vertices'):
            print(f"      Vertices: {len(mesh.vertices):,}")
        if hasattr(mesh, 'triangles'):
            print(f"      Triangles: {len(mesh.triangles):,}")
        elif hasattr(mesh, 'faces'):
            print(f"      Faces: {len(mesh.faces):,}")
    
    # Step 2: Sample points
    if verbose:
        print(f"\n[2/5] Sampling {num_points:,} points ({sampling_method})...")
    pcd = sample_point_cloud(mesh, num_points, method=sampling_method)
    if verbose:
        print(f"      Sampled: {len(pcd.points):,} points")
    
    # Step 3: Add colors
    if verbose:
        print(f"\n[3/5] Colorizing ({color_method})...")
    pcd = colorize_point_cloud(pcd, mesh, method=color_method)
    
    # Step 4: Normalize
    if normalize:
        if verbose:
            print(f"\n[4/5] Normalizing point cloud...")
        pcd = normalize_point_cloud(pcd)
    else:
        if verbose:
            print(f"\n[4/5] Skipping normalization")
    
    # Step 5: Save
    if verbose:
        print(f"\n[5/5] Saving to: {output_path}")
    
    success = o3d.io.write_point_cloud(str(output_path), pcd)
    
    if success:
        if verbose:
            print(f"      ✓ Successfully saved!")
            print(f"\n      Points: {len(pcd.points):,}")
            print(f"      Colors: {pcd.has_colors()}")
            print(f"      Normals: {pcd.has_normals()}")
            print("\n" + "="*60)
        
        if visualize:
            visualize_point_cloud(pcd, f"Point Cloud: {input_path.stem}")
        
        return str(output_path)
    else:
        raise IOError(f"Failed to save point cloud to {output_path}")


def batch_convert(input_dir, output_dir=None, **kwargs):
    """
    Convert all 3D models in a directory to point clouds
    
    Args:
        input_dir: Directory containing 3D model files
        output_dir: Directory to save .pcd files (default: input_dir)
        **kwargs: Arguments for convert_3d_to_pcd()
        
    Returns:
        list: Paths to converted files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir or input_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Supported formats
    formats = ('*.obj', '*.glb', '*.ply', '*.stl', '*.off', '*.gltf')
    files = []
    for fmt in formats:
        files.extend(input_path.glob(fmt))
        files.extend(input_path.glob(fmt.upper()))
    
    if not files:
        print(f"No 3D model files found in {input_path}")
        return []
    
    print(f"\nFound {len(files)} 3D models")
    print("="*60)
    
    converted = []
    for i, file in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file.name}")
        try:
            output_file = output_path / (file.stem + "_pointcloud.pcd")
            result = convert_3d_to_pcd(
                str(file),
                str(output_file),
                **kwargs
            )
            print(f"  ✓ {output_file.name}")
            converted.append(result)
        except Exception as e:
            print(f"  ✗ Error: {e}")
    
    print("\n" + "="*60)
    print(f"Converted: {len(converted)}/{len(files)} files")
    
    return converted


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 3D models to point clouds (.pcd format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single file conversion
  python convert_3d_to_pcd.py model.obj
  python convert_3d_to_pcd.py model.glb -o output.pcd -n 50000
  
  # With visualization
  python convert_3d_to_pcd.py model.ply --visualize
  
  # Batch conversion
  python convert_3d_to_pcd.py --batch ./models/ --output ./pointclouds/
  
Supported formats: .obj, .glb, .ply, .stl, .off, .gltf
        """
    )
    
    parser.add_argument(
        "input",
        nargs='?',
        help="Input 3D model file or directory (for batch mode)"
    )
    parser.add_argument(
        "-o", "--output",
        help="Output .pcd file (default: auto-generated)"
    )
    parser.add_argument(
        "-n", "--num-points",
        type=int,
        default=10000,
        help="Number of points to sample (default: 10000)"
    )
    parser.add_argument(
        "-m", "--method",
        choices=['poisson', 'random'],
        default='poisson',
        help="Sampling method (default: poisson)"
    )
    parser.add_argument(
        "-c", "--color",
        choices=['vertex', 'height', 'random'],
        default='height',
        help="Coloring method (default: height)"
    )
    parser.add_argument(
        "--no-normalize",
        action='store_true',
        help="Skip normalization"
    )
    parser.add_argument(
        "--visualize",
        action='store_true',
        help="Show point cloud visualization"
    )
    parser.add_argument(
        "--batch",
        action='store_true',
        help="Batch convert all models in directory"
    )
    parser.add_argument(
        "-q", "--quiet",
        action='store_true',
        help="Suppress output messages"
    )
    
    args = parser.parse_args()
    
    if not args.input:
        parser.print_help()
        sys.exit(1)
    
    try:
        if args.batch:
            # Batch conversion
            converted = batch_convert(
                args.input,
                output_dir=args.output,
                num_points=args.num_points,
                sampling_method=args.method,
                color_method=args.color,
                normalize=not args.no_normalize,
                verbose=not args.quiet
            )
            sys.exit(0 if converted else 1)
        else:
            # Single file conversion
            output_file = convert_3d_to_pcd(
                args.input,
                output_file=args.output,
                num_points=args.num_points,
                sampling_method=args.method,
                color_method=args.color,
                normalize=not args.no_normalize,
                visualize=args.visualize,
                verbose=not args.quiet
            )
            print(f"\nOutput saved: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        if not args.quiet:
            traceback.print_exc()
        sys.exit(1)
