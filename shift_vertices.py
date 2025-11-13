#!/usr/bin/env python3
"""
Script to shift vertices in vertices_backup.npy and render the result.

Usage:
    python shift_vertices.py <index1>:<x,y,z> [<index2>:<x,y,z> ...]
    
    Examples:
        python shift_vertices.py 0:0.5,0.7,0.5
        python shift_vertices.py 0:0.5,0.7,0.5 1:0.2,0.3,0.4 2:0.1,0.1,0.1
"""

import numpy as np
import torch
import os
import sys
import pickle
import argparse
import cv2
import trimesh
from pathlib import Path

# Add the amr module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr.utils.renderer import Renderer
from amr.configs import get_config

# Constants
LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)
COWS_DIR = "cows"

def load_data():
    """Load all necessary data from cows directory."""
    vertices_file = os.path.join(COWS_DIR, "vertices_backup.npy")
    boxes_file = os.path.join(COWS_DIR, "boxes.npy")
    cam_translations_file = os.path.join(COWS_DIR, "cam_translations.npy")
    inference_summary_file = os.path.join(COWS_DIR, "inference_summary.json")
    
    # Load vertices
    if not os.path.exists(vertices_file):
        raise FileNotFoundError(f"Could not find {vertices_file}")
    all_verts = np.load(vertices_file)
    
    # Load boxes
    if not os.path.exists(boxes_file):
        raise FileNotFoundError(f"Could not find {boxes_file}")
    boxes = np.load(boxes_file)
    
    # Load camera translations
    if not os.path.exists(cam_translations_file):
        raise FileNotFoundError(f"Could not find {cam_translations_file}")
    all_cam_t = np.load(cam_translations_file)
    
    # Load image info from inference summary
    import json
    if os.path.exists(inference_summary_file):
        with open(inference_summary_file, 'r') as f:
            summary = json.load(f)
        image_shape = summary.get("image_info", {}).get("composite_shape", [318, 500, 3])
    else:
        # Default image shape if summary doesn't exist
        image_shape = [318, 500, 3]
    
    print(f"Loaded {len(all_verts)} meshes")
    print(f"Vertices shape: {all_verts.shape}")
    print(f"Camera translations shape: {all_cam_t.shape}")
    print(f"Boxes shape: {boxes.shape}")
    print(f"Image shape: {image_shape}")
    
    return all_verts, all_cam_t, boxes, image_shape

def parse_shift(shift_str):
    """
    Parse a shift string in the format "index:x,y,z"
    
    Args:
        shift_str: String in format "index:x,y,z"
    
    Returns:
        tuple: (index, (x, y, z))
    """
    try:
        parts = shift_str.split(':')
        if len(parts) != 2:
            raise ValueError("Format must be 'index:x,y,z'")
        
        index = int(parts[0])
        shift_values = parts[1].split(',')
        
        if len(shift_values) != 3:
            raise ValueError("Shift values must be three numbers: x,y,z")
        
        x, y, z = [float(v.strip()) for v in shift_values]
        
        return index, (x, y, z)
    except ValueError as e:
        raise ValueError(f"Invalid shift format '{shift_str}': {e}")

def shift_vertices_multiple(vertices, shifts):
    """
    Shift multiple meshes by their respective (x, y, z) shifts.
    
    Args:
        vertices: Array of vertices (N, V, 3)
        shifts: Dictionary mapping index to (x, y, z) shift tuple
    
    Returns:
        Modified vertices array
    """
    vertices = vertices.copy()
    
    # Handle different input shapes
    if vertices.ndim == 2:
        # Single mesh (V, 3)
        if 0 not in shifts:
            raise ValueError("For single mesh, index must be 0")
        verts = vertices
        index = 0
        x_shift, y_shift, z_shift = shifts[0]
        verts[:, 0] += x_shift
        verts[:, 1] += y_shift
        verts[:, 2] += z_shift
        vertices = verts
        print(f"Shifted mesh {index} by ({x_shift:+.3f}, {y_shift:+.3f}, {z_shift:+.3f})")
        print(f"  New vertex ranges:")
        print(f"    X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
        print(f"    Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
        print(f"    Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
    else:
        # Multiple meshes (N, V, 3)
        for index, (x_shift, y_shift, z_shift) in shifts.items():
            if index < 0 or index >= len(vertices):
                raise ValueError(f"Index {index} out of range. Valid range: 0-{len(vertices)-1}")
            
            verts = vertices[index]
            verts[:, 0] += x_shift
            verts[:, 1] += y_shift
            verts[:, 2] += z_shift
            vertices[index] = verts
            
            print(f"Shifted mesh {index} by ({x_shift:+.3f}, {y_shift:+.3f}, {z_shift:+.3f})")
            print(f"  New vertex ranges:")
            print(f"    X: [{verts[:, 0].min():.3f}, {verts[:, 0].max():.3f}]")
            print(f"    Y: [{verts[:, 1].min():.3f}, {verts[:, 1].max():.3f}]")
            print(f"    Z: [{verts[:, 2].min():.3f}, {verts[:, 2].max():.3f}]")
    
    return vertices

def create_image_tensor(image_shape):
    """Create a white background image tensor."""
    h, w, c = image_shape
    # Create white image (1.0, 1.0, 1.0) in RGB
    img = np.ones((h, w, c), dtype=np.float32)
    # Convert to torch tensor (3, H, W) format
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_tensor

def setup_renderer():
    """Setup the renderer with model config and faces."""
    # Load model config
    path_model_cfg = 'data/hydra/config.yaml'
    model_cfg = get_config(path_model_cfg)
    
    # Load faces directly from SMAL pickle file
    smal_model_path = model_cfg.SMAL.MODEL_PATH
    with open(smal_model_path, 'rb') as f:
        smal_data = pickle.load(f, encoding="latin1")
    faces = np.array(smal_data['f'], dtype=np.int32)
    
    # Create renderer
    renderer = Renderer(model_cfg, faces=faces)
    
    return renderer

def render_meshes(renderer, vertices, cam_translations, img_tensor, boxes):
    """Render meshes using the renderer."""
    # Convert to lists if needed
    if isinstance(vertices, np.ndarray):
        if vertices.ndim == 2:
            vertices = [vertices]
        else:
            vertices = [v for v in vertices]
    
    if isinstance(cam_translations, np.ndarray):
        if cam_translations.ndim == 1:
            cam_translations = [cam_translations]
        else:
            cam_translations = [ct for ct in cam_translations]
    
    # Convert boxes to list of lists
    if isinstance(boxes, np.ndarray):
        boxes_list = [box.tolist() for box in boxes]
    else:
        boxes_list = boxes
    
    # Render
    regression_img = renderer(
        vertices,
        cam_translations,
        img_tensor,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        boxes=boxes_list,
    )
    
    # Convert to uint8
    regression_img = (regression_img * 255).astype(np.uint8)
    
    return regression_img

def save_mesh_obj(renderer, vertices, cam_translations, output_path):
    """Save mesh as OBJ file."""
    # Convert to lists if needed
    if isinstance(vertices, np.ndarray):
        if vertices.ndim == 2:
            vertices = [vertices]
        else:
            vertices = [v for v in vertices]
    
    if isinstance(cam_translations, np.ndarray):
        if cam_translations.ndim == 1:
            cam_translations = [cam_translations]
        else:
            cam_translations = [ct for ct in cam_translations]
    
    # Create trimeshes
    trimeshes = [
        renderer.vertices_to_trimesh(vvv, ttt.copy(), LIGHT_BLUE) 
        for vvv, ttt in zip(vertices, cam_translations)
    ]
    
    # Join meshes
    mesh = trimesh.util.concatenate(trimeshes)
    
    # Save mesh
    trimesh.exchange.export.export_mesh(mesh, output_path)
    print(f"Saved mesh to: {output_path}")

def create_filename_suffix(shifts):
    """Create a filename suffix from the shifts dictionary."""
    parts = []
    for index in sorted(shifts.keys()):
        x, y, z = shifts[index]
        parts.append(f"idx{index}_{x:+.3f}_{y:+.3f}_{z:+.3f}")
    return "_".join(parts)

def main():
    parser = argparse.ArgumentParser(
        description="Shift vertices and render the result",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Shift single mesh
  python shift_vertices.py 0:0.5,0.7,0.5
  
  # Shift multiple meshes
  python shift_vertices.py 0:0.5,0.7,0.5 1:0.2,0.3,0.4 2:0.1,0.1,0.1
  
  # Shift with negative values
  python shift_vertices.py 0:-0.5,0.7,-0.5 1:0.2,-0.3,0.4

Format: <index>:<x,y,z>
  - index: 0-based mesh index
  - x, y, z: shift amounts along each axis (comma-separated)
        """
    )
    parser.add_argument('shifts', nargs='+', 
                       help='Shifts in format "index:x,y,z" (e.g., "0:0.5,0.7,0.5")')
    parser.add_argument('--output-dir', type=str, default='shifted_output',
                       help='Output directory for rendered images and mesh (default: shifted_output)')
    
    args = parser.parse_args()
    
    # Parse all shifts
    shifts = {}
    print("="*60)
    print("Vertex Shifting and Rendering Script")
    print("="*60)
    print("Parsing shifts...")
    
    for shift_str in args.shifts:
        index, shift_xyz = parse_shift(shift_str)
        if index in shifts:
            print(f"Warning: Index {index} specified multiple times. Overwriting previous shift.")
        shifts[index] = shift_xyz
        print(f"  Mesh {index}: shift by ({shift_xyz[0]:+.3f}, {shift_xyz[1]:+.3f}, {shift_xyz[2]:+.3f})")
    
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    all_verts, all_cam_t, boxes, image_shape = load_data()
    
    # Validate indices
    max_index = len(all_verts) - 1
    for index in shifts.keys():
        if index < 0 or index > max_index:
            print(f"Error: Index {index} out of range. Valid range: 0-{max_index}")
            sys.exit(1)
    
    # Shift vertices
    print(f"\nShifting vertices...")
    shifted_verts = shift_vertices_multiple(all_verts, shifts)
    
    # Setup renderer
    print("\nSetting up renderer...")
    renderer = setup_renderer()
    
    # Create image tensor
    print("\nCreating image tensor...")
    img_tensor = create_image_tensor(image_shape)
    
    # Render meshes
    print("\nRendering meshes...")
    rendered_img = render_meshes(renderer, shifted_verts, all_cam_t, img_tensor, boxes)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create filename suffix
    filename_suffix = create_filename_suffix(shifts)
    
    # Save rendered image
    output_img_path = os.path.join(args.output_dir, f"rendered_shifted_{filename_suffix}.jpg")
    cv2.imwrite(output_img_path, cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
    print(f"Saved rendered image to: {output_img_path}")
    
    # Save mesh OBJ
    output_mesh_path = os.path.join(args.output_dir, f"mesh_shifted_{filename_suffix}.obj")
    save_mesh_obj(renderer, shifted_verts, all_cam_t, output_mesh_path)
    
    # Also save the shifted vertices
    output_vertices_path = os.path.join(args.output_dir, f"vertices_shifted_{filename_suffix}.npy")
    np.save(output_vertices_path, shifted_verts)
    print(f"Saved shifted vertices to: {output_vertices_path}")
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()