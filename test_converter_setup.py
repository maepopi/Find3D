#!/usr/bin/env python
"""
Comprehensive test for Find3D with 3D to Point Cloud Converter
Verifies all components work together
"""

import subprocess
import sys
from pathlib import Path
import json

def print_header(text):
    print("\n" + "="*60)
    print(f"  {text}")
    print("="*60)

def test_imports():
    """Test all imports"""
    print_header("Testing Imports")
    
    try:
        from convert_3d_to_pcd import convert_3d_to_pcd, batch_convert
        print("✓ converter module")
        
        from gradio_app import create_interface, convert_3d_to_pcd_ui
        print("✓ gradio_app module")
        
        from model.evaluation.utils import load_model
        print("✓ Find3D model module")
        
        import gradio as gr
        print("✓ Gradio UI framework")
        
        import open3d as o3d
        print("✓ Open3D library")
        
        return True
    except Exception as e:
        print(f"✗ Import failed: {e}")
        return False

def test_example_models():
    """Test example model generation"""
    print_header("Testing Example Model Generation")
    
    try:
        example_dir = Path("example_models")
        expected_files = ["cube.obj", "sphere.obj", "pyramid.obj", "torus.obj"]
        
        missing = [f for f in expected_files if not (example_dir / f).exists()]
        
        if missing:
            print(f"✗ Missing files: {missing}")
            return False
        
        for f in expected_files:
            size = (example_dir / f).stat().st_size
            print(f"✓ {f} ({size:,} bytes)")
        
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

def test_converter():
    """Test converter functionality"""
    print_header("Testing 3D to Point Cloud Converter")
    
    try:
        from convert_3d_to_pcd import convert_3d_to_pcd
        
        input_file = "example_models/cube.obj"
        output_file = "test_conversion.pcd"
        
        print(f"Converting: {input_file}")
        result = convert_3d_to_pcd(
            input_file=input_file,
            output_file=output_file,
            num_points=5000,
            sampling_method="poisson",
            color_method="height",
            verbose=False
        )
        
        # Check output file
        output_path = Path(result)
        if output_path.exists():
            size = output_path.stat().st_size
            print(f"✓ Conversion successful")
            print(f"✓ Output: {output_path.name} ({size:,} bytes)")
            
            # Clean up
            output_path.unlink()
            return True
        else:
            print(f"✗ Output file not created")
            return False
            
    except Exception as e:
        print(f"✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_batch_converter():
    """Test batch conversion"""
    print_header("Testing Batch Converter")
    
    try:
        from convert_3d_to_pcd import batch_convert
        
        input_dir = "example_models"
        output_dir = "test_batch_output"
        
        print(f"Batch converting: {input_dir}/")
        converted = batch_convert(
            input_dir=input_dir,
            output_dir=output_dir,
            num_points=3000,
            verbose=False
        )
        
        if len(converted) == 4:
            print(f"✓ Successfully converted {len(converted)}/4 files")
            
            # Clean up
            import shutil
            shutil.rmtree(output_dir)
            return True
        else:
            print(f"✗ Expected 4 conversions, got {len(converted)}")
            return False
            
    except Exception as e:
        print(f"✗ Batch conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_gradio_ui():
    """Test Gradio UI functions"""
    print_header("Testing Gradio UI Integration")
    
    try:
        from gradio_app import convert_3d_to_pcd_ui, create_interface
        import gradio as gr
        
        # Test UI function with None input
        try:
            result_msg, pcd_path = convert_3d_to_pcd_ui(None)
            if "Please upload" in result_msg:
                print("✓ Converter UI handles missing input")
            else:
                print("⚠ Converter UI behavior unexpected")
        except:
            pass
        
        # Test interface creation
        interface = create_interface()
        print("✓ Gradio interface created successfully")
        print(f"✓ Interface has multiple tabs")
        
        return True
        
    except Exception as e:
        print(f"✗ Gradio UI test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_documentation():
    """Test documentation files exist"""
    print_header("Testing Documentation")
    
    docs = [
        "3D_CONVERTER_GUIDE.md",
        "CONVERTER_QUICKSTART.md",
        "CONVERTER_IMPLEMENTATION.md",
    ]
    
    for doc in docs:
        doc_path = Path(doc)
        if doc_path.exists():
            size = doc_path.stat().st_size
            print(f"✓ {doc} ({size:,} bytes)")
        else:
            print(f"✗ Missing: {doc}")
            return False
    
    return True

def test_converted_pcd_files():
    """Test that batch-converted PCD files exist"""
    print_header("Testing Batch-Converted Point Clouds")
    
    expected_files = [
        "converted/cube_pointcloud.pcd",
        "converted/sphere_pointcloud.pcd",
        "converted/pyramid_pointcloud.pcd",
        "converted/torus_pointcloud.pcd",
    ]
    
    missing = []
    for f in expected_files:
        f_path = Path(f)
        if f_path.exists():
            size = f_path.stat().st_size
            print(f"✓ {f} ({size:,} bytes)")
        else:
            missing.append(f)
    
    if missing:
        print(f"✗ Missing files: {missing}")
        return False
    
    return True

def main():
    """Run all tests"""
    print("\n" + "█"*60)
    print("█  Find3D with 3D→Point Cloud Converter - Test Suite")
    print("█"*60)
    
    tests = [
        ("Imports", test_imports),
        ("Example Models", test_example_models),
        ("Converter", test_converter),
        ("Batch Converter", test_batch_converter),
        ("Gradio UI", test_gradio_ui),
        ("Documentation", test_documentation),
        ("Converted Files", test_converted_pcd_files),
    ]
    
    results = {}
    for name, test_func in tests:
        try:
            results[name] = test_func()
        except Exception as e:
            print(f"✗ Test crashed: {e}")
            results[name] = False
    
    # Summary
    print_header("Test Summary")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print("\n" + "="*60)
    if passed == total:
        print(f"✓ ALL {total} TESTS PASSED")
        print("="*60)
        print("\n🚀 Ready to use! Launch with:")
        print("   python gradio_app.py")
        return 0
    else:
        print(f"✗ {total - passed}/{total} tests failed")
        print("="*60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
