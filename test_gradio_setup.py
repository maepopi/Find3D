#!/usr/bin/env python
"""
Test script to verify Find3D Gradio setup
Run this before launching the full web interface
"""

import sys
import torch
import numpy as np

def check_dependencies():
    """Check all required dependencies"""
    print("✓ Checking dependencies...\n")
    
    deps = {
        'torch': 'torch',
        'torch.cuda': 'torch',
        'torch_geometric': 'torch_geometric',
        'gradio': 'gradio',
        'transformers': 'transformers',
        'open3d': 'open3d',
        'numpy': 'numpy',
    }
    
    failed = []
    for name, module_name in deps.items():
        try:
            if module_name == 'torch':
                __import__(module_name)
            else:
                __import__(module_name)
            print(f"  ✓ {name}")
        except ImportError as e:
            print(f"  ✗ {name}: {e}")
            failed.append(name)
    
    return len(failed) == 0, failed


def check_cuda():
    """Check CUDA availability"""
    print("\n✓ Checking CUDA...\n")
    
    cuda_available = torch.cuda.is_available()
    print(f"  {'✓' if cuda_available else '✗'} CUDA Available: {cuda_available}")
    
    if cuda_available:
        print(f"  ✓ CUDA Version: {torch.version.cuda}")
        print(f"  ✓ GPU Count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"    - GPU {i}: {torch.cuda.get_device_name(i)}")
    
    return cuda_available


def check_model_loading():
    """Check if model can be loaded"""
    print("\n✓ Checking model loading...\n")
    
    try:
        from model.evaluation.utils import load_model, set_seed
        
        set_seed(123)
        print("  ⏳ Loading Find3D model (this may take a moment)...")
        
        # This will download from HF if needed
        model = load_model()
        model.eval()
        
        print(f"  ✓ Model loaded successfully")
        print(f"  ✓ Model device: {next(model.parameters()).device}")
        
        # Get temperature
        temp = np.exp(model.ln_logit_scale.item())
        print(f"  ✓ Model temperature: {temp:.4f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to load model: {e}")
        return False


def check_point_cloud_processing():
    """Check point cloud preprocessing"""
    print("\n✓ Checking point cloud processing...\n")
    
    try:
        from model.evaluation.utils import preprocess_pcd, encode_text
        
        # Create sample point cloud
        xyz = np.random.uniform(-0.5, 0.5, (1000, 3)).astype(np.float32)
        rgb = np.ones((1000, 3), dtype=np.float32) * 0.5
        normal = np.random.randn(1000, 3).astype(np.float32)
        normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
        
        xyz_t = torch.tensor(xyz)
        rgb_t = torch.tensor(rgb)
        normal_t = torch.tensor(normal)
        
        if torch.cuda.is_available():
            xyz_t = xyz_t.cuda()
            rgb_t = rgb_t.cuda()
            normal_t = normal_t.cuda()
        
        print("  ⏳ Preprocessing point cloud...")
        data = preprocess_pcd(xyz_t, rgb_t, normal_t)
        print(f"  ✓ Preprocessed to {len(data['coord'])} points")
        
        print("  ⏳ Encoding text queries...")
        embeds = encode_text(["handle", "blade", "tip"])
        print(f"  ✓ Encoded {len(embeds)} queries")
        print(f"  ✓ Embedding dimension: {embeds.shape[1]}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Failed to process point cloud: {e}")
        import traceback
        traceback.print_exc()
        return False


def check_gradio():
    """Check Gradio is installed"""
    print("\n✓ Checking Gradio...\n")
    
    try:
        import gradio as gr
        print(f"  ✓ Gradio version: {gr.__version__}")
        return True
    except Exception as e:
        print(f"  ✗ Gradio check failed: {e}")
        return False


def main():
    """Run all checks"""
    print("\n" + "="*60)
    print("Find3D Gradio Setup Verification")
    print("="*60 + "\n")
    
    # Check dependencies
    deps_ok, failed_deps = check_dependencies()
    
    # Check CUDA
    cuda_ok = check_cuda()
    
    # Check Gradio
    gradio_ok = check_gradio()
    
    # Check model loading
    model_ok = check_model_loading()
    
    # Check point cloud processing
    pc_ok = check_point_cloud_processing()
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60 + "\n")
    
    all_ok = deps_ok and cuda_ok and gradio_ok and model_ok and pc_ok
    
    if deps_ok:
        print("✓ Dependencies: OK")
    else:
        print(f"✗ Dependencies: MISSING ({', '.join(failed_deps)})")
    
    print(f"{'✓' if cuda_ok else '✗'} CUDA: {'OK' if cuda_ok else 'NOT AVAILABLE (CPU fallback enabled)'}")
    print(f"{'✓' if gradio_ok else '✗'} Gradio: {'OK' if gradio_ok else 'FAILED'}")
    print(f"{'✓' if model_ok else '✗'} Model Loading: {'OK' if model_ok else 'FAILED'}")
    print(f"{'✓' if pc_ok else '✗'} Point Cloud Processing: {'OK' if pc_ok else 'FAILED'}")
    
    print("\n" + "="*60)
    
    if all_ok:
        print("✓ ALL CHECKS PASSED - Ready to run Gradio app!")
        print("\nRun with: python gradio_app.py")
        print("\nThen open: http://localhost:7860")
        sys.exit(0)
    else:
        print("✗ Some checks failed - Fix issues above before running")
        if not model_ok:
            print("\nHint: First time model loading downloads weights (~500MB)")
            print("Make sure you have sufficient disk space and internet connection")
        sys.exit(1)


if __name__ == "__main__":
    main()
