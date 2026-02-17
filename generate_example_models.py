"""
Generate example 3D models for testing the converter
Creates simple geometric shapes as .obj files
"""

import numpy as np
from pathlib import Path


def create_cube_obj(filepath, size=1.0):
    """Create a cube as OBJ file with triangulated faces"""
    s = size / 2
    vertices = [
        [-s, -s, -s],  # 0
        [s, -s, -s],   # 1
        [s, s, -s],    # 2
        [-s, s, -s],   # 3
        [-s, -s, s],   # 4
        [s, -s, s],    # 5
        [s, s, s],     # 6
        [-s, s, s],    # 7
    ]
    
    # Define triangular faces (2 triangles per side)
    faces = [
        [1, 2, 3],     # front 1
        [1, 3, 4],     # front 2
        [5, 8, 7],     # back 1
        [5, 7, 6],     # back 2
        [1, 5, 6],     # right 1
        [1, 6, 2],     # right 2
        [4, 7, 8],     # left 1
        [4, 8, 3],     # left 2
        [4, 3, 2],     # bottom 1
        [4, 2, 1],     # bottom 2
        [5, 6, 7],     # top 1
        [5, 7, 8],     # top 2
    ]
    
    with open(filepath, 'w') as f:
        f.write(f"# Cube\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {' '.join(str(i) for i in face)}\n")
    
    print(f"✓ Created {filepath}")


def create_sphere_obj(filepath, radius=1.0, sectors=20, stacks=20):
    """Create a sphere as OBJ file using UV sphere"""
    vertices = []
    faces = []
    
    # Generate vertices
    for i in range(stacks + 1):
        stack_angle = np.pi / 2 - i * np.pi / stacks
        xy = radius * np.cos(stack_angle)
        z = radius * np.sin(stack_angle)
        
        for j in range(sectors + 1):
            sector_angle = 2 * np.pi * j / sectors
            x = xy * np.cos(sector_angle)
            y = xy * np.sin(sector_angle)
            vertices.append([x, y, z])
    
    # Generate faces
    for i in range(stacks):
        k1 = i * (sectors + 1)
        k2 = k1 + sectors + 1
        
        for j in range(sectors):
            if i != 0:
                faces.append([k1 + 1, k1, k2])
            if i != (stacks - 1):
                faces.append([k1 + sectors + 2, k1 + 1, k2 + 1])
            k1 += 1
            k2 += 1
    
    with open(filepath, 'w') as f:
        f.write(f"# Sphere (radius={radius}, sectors={sectors}, stacks={stacks})\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"✓ Created {filepath}")


def create_torus_obj(filepath, major_radius=1.0, minor_radius=0.3, major_segs=30, minor_segs=20):
    """Create a torus as OBJ file"""
    vertices = []
    faces = []
    
    # Generate vertices
    for i in range(major_segs):
        theta = 2 * np.pi * i / major_segs
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        for j in range(minor_segs):
            phi = 2 * np.pi * j / minor_segs
            cos_phi = np.cos(phi)
            sin_phi = np.sin(phi)
            
            x = (major_radius + minor_radius * cos_phi) * cos_theta
            y = (major_radius + minor_radius * cos_phi) * sin_theta
            z = minor_radius * sin_phi
            
            vertices.append([x, y, z])
    
    # Generate faces
    for i in range(major_segs):
        for j in range(minor_segs):
            v1 = i * minor_segs + j
            v2 = ((i + 1) % major_segs) * minor_segs + j
            v3 = ((i + 1) % major_segs) * minor_segs + (j + 1) % minor_segs
            v4 = i * minor_segs + (j + 1) % minor_segs
            
            faces.append([v1 + 1, v2 + 1, v3 + 1])
            faces.append([v1 + 1, v3 + 1, v4 + 1])
    
    with open(filepath, 'w') as f:
        f.write(f"# Torus (major_radius={major_radius}, minor_radius={minor_radius})\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"✓ Created {filepath}")


def create_pyramid_obj(filepath, base=1.0, height=1.0):
    """Create a pyramid as OBJ file with triangulated faces"""
    s = base / 2
    h = height
    
    vertices = [
        [-s, -s, 0],   # 0: base corner 1
        [s, -s, 0],    # 1: base corner 2
        [s, s, 0],     # 2: base corner 3
        [-s, s, 0],    # 3: base corner 4
        [0, 0, h],     # 4: apex
    ]
    
    # All triangular faces
    faces = [
        [1, 3, 2],     # base 1
        [1, 4, 3],     # base 2
        [1, 5, 2],     # front
        [2, 5, 3],     # right
        [3, 5, 4],     # back
        [4, 5, 1],     # left
    ]
    
    with open(filepath, 'w') as f:
        f.write(f"# Pyramid (base={base}, height={height})\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in faces:
            f.write(f"f {' '.join(str(i) for i in face)}\n")
    
    print(f"✓ Created {filepath}")


def create_example_models(output_dir="example_models"):
    """Generate all example models"""
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Generating example 3D models in '{output_dir}/'...\n")
    
    create_cube_obj(output_path / "cube.obj", size=1.0)
    create_sphere_obj(output_path / "sphere.obj", radius=1.0)
    create_torus_obj(output_path / "torus.obj")
    create_pyramid_obj(output_path / "pyramid.obj")
    
    print(f"\n✓ All example models created!")
    print(f"\nTest with:")
    print(f"  python convert_3d_to_pcd.py {output_dir}/cube.obj")
    print(f"  python convert_3d_to_pcd.py {output_dir}/sphere.obj --visualize")
    print(f"  python convert_3d_to_pcd.py --batch {output_dir}/ --output converted/")


if __name__ == "__main__":
    import sys
    
    output_dir = sys.argv[1] if len(sys.argv) > 1 else "example_models"
    create_example_models(output_dir)
