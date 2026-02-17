"""
Find3D Gradio Interface
Interactive UI for testing 3D part segmentation with text queries
"""

import gradio as gr
import torch
import numpy as np
import os
import tempfile
from pathlib import Path
import json
from typing import Tuple, Dict, Optional

# Import Find3D utilities
from model.evaluation.utils import (
    load_model, 
    set_seed, 
    preprocess_pcd, 
    read_pcd,
    encode_text,
)

# Import 3D to PCD converter
from convert_3d_to_pcd import convert_3d_to_pcd

# Global model cache
_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model_cached():
    """Load model once and cache it"""
    global _model
    if _model is None:
        set_seed(123)
        try:
            _model = load_model()
            _model.eval()
        except Exception as e:
            raise gr.Error(f"Failed to load Find3D model: {str(e)}")
    return _model


def create_sample_point_cloud(num_points: int = 10000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a sample point cloud for testing"""
    np.random.seed(42)
    
    # Simple cube point cloud
    xyz = np.random.uniform(-0.5, 0.5, (num_points, 3))
    
    # Color based on position
    rgb = np.zeros((num_points, 3))
    
    # Red for top half (z > 0)
    top_mask = xyz[:, 2] > 0
    rgb[top_mask] = [1.0, 0.0, 0.0]
    
    # Blue for bottom half
    rgb[~top_mask] = [0.0, 0.0, 1.0]
    
    # Random normals
    normal = np.random.randn(num_points, 3)
    normal = normal / (np.linalg.norm(normal, axis=1, keepdims=True) + 1e-8)
    
    return xyz.astype(np.float32), rgb.astype(np.float32), normal.astype(np.float32)


def run_inference(
    file_input: Optional[str],
    use_sample: bool,
    num_sample_points: int,
    queries: str,
    mode: str,
    temperature: float,
    seed: int,
) -> Tuple[Dict, str]:
    """Run Find3D inference on point cloud"""
    
    set_seed(seed)
    results = {}
    
    try:
        # Load or create point cloud
        if use_sample:
            xyz, rgb, normal = create_sample_point_cloud(int(num_sample_points))
            status = f"✓ Created sample point cloud with {len(xyz):,} points\n"
        else:
            if file_input is None:
                return {}, "❌ Error: Please upload a PCD file or use sample point cloud"
            
            xyz, rgb, normal = read_pcd(file_input)
            status = f"✓ Loaded PCD file with {len(xyz):,} points\n"
        
        results['points_loaded'] = len(xyz)
        
        # Validate RGB values
        if np.max(rgb) > 1.0:
            rgb = rgb / 255.0
        
        # Parse queries
        query_list = [q.strip() for q in queries.split(",") if q.strip()]
        if not query_list:
            return results, "❌ Error: Please provide at least one query (comma-separated)"
        
        status += f"✓ Parsed {len(query_list)} queries: {', '.join(query_list)}\n"
        results['queries'] = query_list
        
        # Move to device
        if torch.cuda.is_available():
            xyz_tensor = torch.tensor(xyz, dtype=torch.float32).cuda()
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32).cuda()
            normal_tensor = torch.tensor(normal, dtype=torch.float32).cuda()
        else:
            xyz_tensor = torch.tensor(xyz, dtype=torch.float32)
            rgb_tensor = torch.tensor(rgb, dtype=torch.float32)
            normal_tensor = torch.tensor(normal, dtype=torch.float32)
        
        # Preprocess
        status += "⏳ Preprocessing point cloud...\n"
        data_dict = preprocess_pcd(xyz_tensor, rgb_tensor, normal_tensor)
        
        # Encode text
        status += "⏳ Encoding text queries...\n"
        label_embeds = encode_text(query_list)
        data_dict["label_embeds"] = label_embeds
        
        # Load model
        status += "⏳ Loading Find3D model...\n"
        model = load_model_cached()
        model_temp = np.exp(model.ln_logit_scale.item())
        
        # Inference
        status += "⏳ Running inference...\n"
        with torch.no_grad():
            for key in data_dict.keys():
                if isinstance(data_dict[key], torch.Tensor) and "full" not in key:
                    if torch.cuda.is_available():
                        data_dict[key] = data_dict[key].cuda(non_blocking=True)
            
            net_out = model(x=data_dict)
        
        xyz_full = data_dict["xyz_full"]
        text_embeds = data_dict['label_embeds']
        
        status += f"✓ Inference complete!\n\n"
        
        # Compute predictions
        logits = net_out @ text_embeds.T  # n_pts x n_queries
        
        if mode == "segmentation":
            # Prepend background class (0)
            logits_bg = torch.cat([torch.zeros(logits.shape[0], 1, device=logits.device), logits], dim=1)
            probs = torch.softmax(logits_bg * model_temp, dim=1)
            pred = probs.argmax(dim=1).cpu().numpy()
            
            # Statistics
            status += "📊 SEGMENTATION RESULTS\n"
            status += f"Total points: {len(pred):,}\n"
            status += f"Classes: {len(query_list) + 1} (0=background, 1-{len(query_list)}=parts)\n"
            status += f"Confidence range: [{probs.max().item():.4f}]\n\n"
            status += "Class distribution:\n"
            
            # Background
            bg_count = np.sum(pred == 0)
            bg_pct = 100 * bg_count / len(pred)
            status += f"  [0] Background: {bg_count:,} ({bg_pct:.1f}%)\n"
            
            # Parts
            for i, query in enumerate(query_list):
                count = np.sum(pred == i + 1)
                pct = 100 * count / len(pred)
                status += f"  [{i+1}] {query}: {count:,} ({pct:.1f}%)\n"
            
            results['predictions'] = pred
            results['mode'] = 'segmentation'
            
        else:  # heatmap
            probs = torch.softmax(logits * model_temp, dim=1)  # n_pts x n_queries
            
            status += "🔥 HEATMAP RESULTS\n"
            status += f"Total points: {len(xyz_full):,}\n"
            status += f"Heatmaps: {len(query_list)}\n\n"
            status += "Confidence ranges by query:\n"
            
            for i, query in enumerate(query_list):
                conf = probs[:, i].cpu().numpy()
                status += f"  {query}: [{conf.min():.4f} - {conf.max():.4f}] (avg: {conf.mean():.4f})\n"
            
            results['heatmaps'] = probs.cpu().numpy()
            results['mode'] = 'heatmap'
        
        results['status'] = 'success'
        results['model_temperature'] = float(model_temp)
        results['xyz_full'] = xyz_full
        results['text_embeds'] = text_embeds
        
        return results, status
        
    except Exception as e:
        import traceback
        error_msg = f"❌ Error: {str(e)}\n\n{traceback.format_exc()}"
        print(error_msg)
        return {}, error_msg


def create_stats_text(results: Dict) -> str:
    """Generate detailed statistics"""
    if not results or results.get('status') != 'success':
        return "No results available. Run inference first."
    
    text = "# 📊 Detailed Analysis\n\n"
    
    text += f"**Input Data**\n"
    text += f"- Points: {results['points_loaded']:,}\n"
    text += f"- Model Temperature: {results['model_temperature']:.4f}\n"
    text += f"- Mode: {results['mode'].capitalize()}\n\n"
    
    text += f"**Queries**\n"
    for i, q in enumerate(results['queries'], 1):
        text += f"{i}. {q}\n"
    
    return text


def convert_3d_to_pcd_ui(
    file_input,
    num_points: int = 10000,
    sampling_method: str = 'poisson',
    color_method: str = 'height'
) -> Tuple[str, str]:
    """Gradio wrapper for 3D to PCD converter"""
    
    if file_input is None:
        return "❌ Please upload a 3D model file", None
    
    try:
        status_msg = "⏳ Converting..."
        
        # Create temporary output file
        with tempfile.NamedTemporaryFile(suffix='.pcd', delete=False) as tmp:
            output_path = tmp.name
        
        # Convert
        result_path = convert_3d_to_pcd(
            input_file=file_input.name,
            output_file=output_path,
            num_points=int(num_points),
            sampling_method=sampling_method,
            color_method=color_method,
            normalize=True,
            visualize=False,
            verbose=False
        )
        
        # Read the converted file to show in UI
        pcd_info = f"✅ **Conversion Successful!**\n\n"
        pcd_info += f"- **Output**: {result_path}\n"
        pcd_info += f"- **Points**: {num_points:,}\n"
        pcd_info += f"- **Sampling**: {sampling_method}\n"
        pcd_info += f"- **Colors**: {color_method}-based\n\n"
        pcd_info += f"**Next Steps:**\n"
        pcd_info += f"1. Check the box \"Use Sample Point Cloud\" → unchecked\n"
        pcd_info += f"2. Upload the converted `.pcd` file in the **Inference** tab\n"
        pcd_info += f"3. Enter your text queries and run inference!\n\n"
        pcd_info += f"**File Ready**: `{Path(result_path).name}`"
        
        return pcd_info, result_path
        
    except Exception as e:
        error_msg = f"❌ Conversion failed: {str(e)}\n\nMake sure the file is a valid 3D model (obj, glb, ply, stl, etc.)"
        return error_msg, None


def create_interface():
    """Create Gradio interface"""
    
    with gr.Blocks(
        title="Find3D - 3D Part Segmentation",
    ) as demo:
        
        # Header
        gr.Markdown("""
        # 🔍 Find3D: Interactive 3D Part Segmentation
        
        **Find any part in any 3D object using natural language** 
        
        Convert your 3D models to point clouds, then describe the parts you want to find!
        """)
        
        with gr.Tabs():
            # Tab 1: Converter
            with gr.TabItem("🔄 3D → Point Cloud Converter", id="converter"):
                gr.Markdown("""
                ## Convert 3D Models to Point Clouds
                
                Transform your 3D models into point clouds (.pcd files) for use with Find3D.
                
                **Supported formats**: .obj, .glb, .ply, .stl, .off, .gltf, and more
                """)
                
                with gr.Row():
                    # Left panel - converter inputs
                    with gr.Column(scale=1, min_width=350):
                        gr.Markdown("### 📤 Upload & Configure")
                        
                        model_file = gr.File(
                            label="Upload 3D Model",
                            file_types=[".obj", ".glb", ".ply", ".stl", ".off", ".gltf", ".fbx"],
                            type="filepath"
                        )
                        
                        converter_points = gr.Slider(
                            label="Points to Sample",
                            value=10000,
                            minimum=1000,
                            maximum=100000,
                            step=1000,
                            info="More points = higher resolution but slower"
                        )
                        
                        converter_method = gr.Radio(
                            label="Sampling Method",
                            choices=["poisson", "random"],
                            value="poisson",
                            info="Poisson: uniform | Random: quick"
                        )
                        
                        converter_color = gr.Radio(
                            label="Coloring",
                            choices=["height", "random", "vertex"],
                            value="height",
                            info="Height: z-axis gradient | Random: random colors | Vertex: from original"
                        )
                        
                        convert_btn = gr.Button(
                            "🚀 Convert to PCD",
                            variant="primary",
                            size="lg"
                        )
                    
                    # Right panel - converter output
                    with gr.Column(scale=2):
                        gr.Markdown("### ✅ Results")
                        
                        converter_output = gr.Markdown("Upload a 3D model and click convert...")
                        
                        pcd_download = gr.File(
                            label="Download PCD",
                            visible=False,
                            interactive=False
                        )
                
                # Converter logic
                def run_converter(file_in, n_pts, method, color):
                    result_msg, pcd_path = convert_3d_to_pcd_ui(
                        file_input=file_in,
                        num_points=n_pts,
                        sampling_method=method,
                        color_method=color
                    )
                    
                    download_visible = pcd_path is not None
                    
                    return result_msg, gr.File(visible=download_visible, value=pcd_path if download_visible else None)
                
                convert_btn.click(
                    fn=run_converter,
                    inputs=[model_file, converter_points, converter_method, converter_color],
                    outputs=[converter_output, pcd_download],
                    show_progress="full"
                )
            
            # Tab 2: Inference
            with gr.TabItem("🧠 Inference", id="inference"):
                gr.Markdown("""
                ## Run Find3D Inference
                
                Use converted point clouds or samples to find parts using text descriptions.
                """)
                
                with gr.Row():
                    # Left panel - inputs
                    with gr.Column(scale=1, min_width=380):
                        gr.Markdown("### 📥 Input & Parameters")
                        
                        # Input selection
                        use_sample = gr.Checkbox(
                            label="Use Sample Point Cloud",
                            value=True,
                            info="Use built-in sample or upload your own .pcd file"
                        )
                        
                        points_input = gr.Number(
                            label="Sample Points",
                            value=10000,
                            minimum=1000,
                            maximum=100000,
                            step=1000,
                            visible=True,
                        )
                        
                        pcd_file = gr.File(
                            label="Upload PCD File",
                            file_types=[".pcd"],
                            visible=False
                        )
                        
                        # Toggle visibility
                        def toggle_inputs(use_sample_checkbox):
                            return [
                                gr.Number(visible=use_sample_checkbox),
                                gr.File(visible=not use_sample_checkbox),
                            ]
                        
                        use_sample.change(
                            toggle_inputs,
                            inputs=[use_sample],
                            outputs=[points_input, pcd_file]
                        )
                        
                        gr.Markdown("### 📝 Text Queries")
                        
                        queries = gr.Textbox(
                            label="What to find? (comma-separated)",
                            value="handle, blade, tip",
                            placeholder="e.g., 'wheel, door, window'",
                            lines=3,
                            info="Describe the parts you want to segment"
                        )
                        
                        gr.Markdown("### ⚙️ Advanced Options")
                        
                        mode = gr.Radio(
                            label="Output Mode",
                            choices=["segmentation", "heatmap"],
                            value="segmentation",
                            info="Segmentation: hard assignment | Heatmap: confidence scores"
                        )
                        
                        temperature = gr.Slider(
                            label="Temperature",
                            value=1.0,
                            minimum=0.1,
                            maximum=2.0,
                            step=0.1,
                            info="Higher = softer boundaries, Lower = sharper"
                        )
                        
                        seed = gr.Slider(
                            label="Random Seed",
                            value=123,
                            minimum=0,
                            maximum=9999,
                            step=1,
                        )
                        
                        # Submit button
                        submit_btn = gr.Button(
                            "🚀 Run Inference",
                            variant="primary",
                            size="lg",
                            scale=2
                        )
                    
                    # Right panel - outputs
                    with gr.Column(scale=2):
                        gr.Markdown("### 📊 Results")
                        
                        with gr.Tabs():
                            with gr.TabItem("Output", id="output"):
                                output_text = gr.Textbox(
                                    label="Status & Results",
                                    interactive=False,
                                    lines=25,
                                )
                            
                            with gr.TabItem("Analysis", id="analysis"):
                                analysis_text = gr.Markdown("Run inference to see analysis")
                            
                            with gr.TabItem("Raw Data", id="raw"):
                                raw_json = gr.JSON(
                                    label="Results (JSON)",
                                    visible=True
                                )
                
                # State for storing results
                results_state = gr.State({})
                
                # Inference logic
                def run_and_summarize(
                    file_up,
                    use_samp,
                    n_points,
                    quer,
                    mod,
                    temp,
                    rnd_seed
                ):
                    results, output = run_inference(
                        file_input=file_up,
                        use_sample=use_samp,
                        num_sample_points=int(n_points),
                        queries=quer,
                        mode=mod,
                        temperature=temp,
                        seed=int(rnd_seed),
                    )
                    
                    # Prepare display results (remove tensors)
                    display_results = {
                        k: v for k, v in results.items()
                        if k not in ['xyz_full', 'predictions', 'heatmaps', 'text_embeds']
                    }
                    
                    analysis = create_stats_text(results)
                    
                    return results, output, analysis, display_results
                
                # Connect submit button
                submit_btn.click(
                    fn=run_and_summarize,
                    inputs=[pcd_file, use_sample, points_input, queries, mode, temperature, seed],
                    outputs=[results_state, output_text, analysis_text, raw_json],
                    show_progress="full"
                )
        
        # Footer
        gr.Markdown("""
        ---
        
        ## About Find3D
        
        Find3D is a method for segmenting **any** part in **any** 3D object based on **any** text query.
        
        - **Paper**: [ICCV 2025 Highlight](https://arxiv.org/abs/2411.13550)
        - **Authors**: Ziqi Ma, Yisong Yue, Georgia Gkioxari
        - **Project**: [ziqi-ma.github.io/find3dsite](https://ziqi-ma.github.io/find3dsite/)
        - **Code**: [github.com/ziqi-ma/Find3D](https://github.com/ziqi-ma/Find3D)
        
        ### Tips for Best Results
        - Use specific part names (not vague descriptions)
        - Convert your 3D models with the **Converter** tab
        - Keep queries short and descriptive
        - Adjust temperature based on your needs
        - Test with the sample cloud first
        """)
    
    return demo


def main():
    """Main entry point for launching the Gradio interface"""
    print(f"🚀 Starting Find3D Gradio Interface")
    print(f"📍 Device: {_device}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    demo = create_interface()
    
    # Launch
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_api=False,
        show_error=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()

