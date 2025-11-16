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

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.image as mpimg

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

def create_camera_view_visualization(renderer, vertices, cam_translations, output_dir, filename_suffix):
    """
    Create camera view visualization and render from modified objects.
    
    Args:
        renderer: Renderer instance
        vertices: Array of vertices (N, V, 3) or list of arrays
        cam_translations: Array of camera translations (N, 3) or list of arrays
        output_dir: Output directory for saving files
        filename_suffix: Suffix for output filenames
    """
    print("\n" + "="*60)
    print("Creating camera view visualization and render...")
    print("="*60)
    
    # Convert to lists if needed
    if isinstance(vertices, np.ndarray):
        if vertices.ndim == 2:
            vertices_list = [vertices]
        else:
            vertices_list = [v for v in vertices]
    else:
        vertices_list = vertices
    
    if isinstance(cam_translations, np.ndarray):
        if cam_translations.ndim == 1:
            cam_translations_list = [cam_translations]
        else:
            cam_translations_list = [ct for ct in cam_translations]
    else:
        cam_translations_list = cam_translations
    
    # Define colors for each mesh
    COLORS = [
        (1.0, 0.0, 0.0),      # Red
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 0.0),      # Yellow
        (0.0, 1.0, 0.0),      # Green
        (0.0, 1.0, 1.0),      # Cyan
        (1.0, 0.0, 1.0),      # Magenta
        (1.0, 0.647, 0.0),    # Orange
        (0.5, 0.0, 0.5),      # Purple
        (1.0, 0.752, 0.796),  # Pink
        (0.0, 0.5, 0.5),      # Teal
    ]
    
    # Create trimeshes with different colors
    trimeshes = []
    for i, (vvv, ttt) in enumerate(zip(vertices_list, cam_translations_list)):
        color = COLORS[i % len(COLORS)]
        mesh_obj = renderer.vertices_to_trimesh(vvv, ttt.copy(), color)
        # Set vertex colors for visualization
        vertex_colors = np.array([(*color, 1.0)] * len(vvv))
        mesh_obj.visual.vertex_colors = vertex_colors
        trimeshes.append(mesh_obj)
    
    # Combine all meshes
    mesh = trimesh.util.concatenate(trimeshes)
    
    # Get mesh vertices and bounds
    mesh_vertices = mesh.vertices
    bounds = mesh.bounds
    center = mesh.centroid
    
    print(f"\nMesh bounds (min, max):")
    print(f"  X: [{bounds[0][0]:.2f}, {bounds[1][0]:.2f}]")
    print(f"  Y: [{bounds[0][1]:.2f}, {bounds[1][1]:.2f}]")
    print(f"  Z: [{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
    print(f"Mesh center: {center}")
    
    # Create axis arrows (scale based on mesh size)
    axis_length = (bounds[1] - bounds[0]).max() * 0.3
    
    def create_axis_arrows(center, length):
        axes = []
        colors = ['red', 'green', 'blue']
        directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        labels = ['X (Right)', 'Y (Down)', 'Z (Into Screen)']
        
        for direction, color, label in zip(directions, colors, labels):
            end = center + np.array(direction) * length
            axes.append((center, end, color, label))
        
        return axes
    
    axis_arrows = create_axis_arrows(center, axis_length)
    
    # Calculate equal axis limits
    max_range = (bounds[1] - bounds[0]).max()
    mid_x = (bounds[0][0] + bounds[1][0]) / 2
    mid_y = (bounds[0][1] + bounds[1][1]) / 2
    mid_z = (bounds[0][2] + bounds[1][2]) / 2
    half_range = max_range / 2
    
    equal_xlim = [mid_x - half_range, mid_x + half_range]
    equal_ylim = [mid_y - half_range, mid_y + half_range]
    equal_zlim = [mid_z - half_range, mid_z + half_range]
    
    def set_equal_aspect(ax, xlim, ylim, zlim):
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_zlim(zlim)
        x_range = xlim[1] - xlim[0]
        y_range = ylim[1] - ylim[0]
        z_range = zlim[1] - zlim[0]
        ax.set_box_aspect([x_range, y_range, z_range])
    
    # Prepare colors for each vertex based on which mesh it belongs to
    vertex_colors_list = []
    for i, (vvv, ttt) in enumerate(zip(vertices_list, cam_translations_list)):
        color = COLORS[i % len(COLORS)]
        num_vertices = len(vvv)
        vertex_colors_list.extend([color] * num_vertices)
    
    vertex_colors_array = np.array(vertex_colors_list)
    
    # Calculate camera position (looking at the mesh center from a reasonable distance)
    camera_distance = (bounds[1] - bounds[0]).max() * 2.5
    camera_pos = center + np.array([0, 0, -camera_distance])
    
    # Render all meshes using the renderer
    render_resolution = [512, 512]
    print(f"\nRendering from camera view (resolution: {render_resolution})...")
    
    # Use render_rgba_multiple to render all meshes together
    rendered_image = renderer.render_rgba_multiple(
        vertices=vertices_list,
        cam_t=cam_translations_list,
        rot_axis=[1, 0, 0],
        rot_angle=0,
        mesh_base_color=COLORS[0],  # Base color (individual colors handled by trimesh)
        scene_bg_color=(1, 1, 1),  # White background
        render_res=render_resolution,
        focal_length=None,  # Use default
    )
    
    # Save rendered image
    rendered_output_path = os.path.join(output_dir, f"camera_view_render_{filename_suffix}.png")
    mpimg.imsave(rendered_output_path, rendered_image)
    print(f"Rendered image saved to '{rendered_output_path}'")
    
    # Create visualization figure with camera view and render
    fig = plt.figure(figsize=(16, 8))
    
    # Left: Camera view (what the camera sees in 3D space)
    ax1 = fig.add_subplot(121, projection='3d')
    # Show mesh from camera perspective
    ax1.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], 
               c=vertex_colors_array, s=0.1, alpha=0.6)
    
    # Draw camera position and viewing direction
    ax1.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]], 
               c='red', s=200, marker='^', label='Camera Position', edgecolors='black', linewidths=2)
    
    # Draw line from camera to mesh center (viewing direction)
    ax1.plot([camera_pos[0], center[0]], 
            [camera_pos[1], center[1]], 
            [camera_pos[2], center[2]], 
            'r--', linewidth=2, label='Viewing Direction')
    
    # Draw axis arrows
    for start, end, color, label in axis_arrows:
        ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
                color=color, linewidth=3)
    
    ax1.set_xlabel('X (Right)', fontsize=10)
    ax1.set_ylabel('Y (Down)', fontsize=10)
    ax1.set_zlabel('Z (Into Screen)', fontsize=10)
    ax1.set_title('Camera View Setup\n(Red triangle = camera position)', fontsize=12, fontweight='bold')
    
    # Set view to show camera position
    view_vector = center - camera_pos
    view_vector = view_vector / np.linalg.norm(view_vector)
    elevation = np.arcsin(view_vector[2]) * 180 / np.pi
    azimuth = np.arctan2(view_vector[1], view_vector[0]) * 180 / np.pi
    ax1.view_init(elev=elevation, azim=azimuth)
    ax1.legend()
    ax1.grid(True)
    set_equal_aspect(ax1, equal_xlim, equal_ylim, equal_zlim)
    
    # Right: Rendered image (what the camera actually sees)
    ax2 = fig.add_subplot(122)
    ax2.imshow(rendered_image)
    ax2.set_title('Rendered Camera View\n(Actual render from pyrender)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    
    plt.tight_layout()
    camera_view_path = os.path.join(output_dir, f"camera_view_and_render_{filename_suffix}.png")
    plt.savefig(camera_view_path, dpi=150, bbox_inches='tight')
    print(f"Camera view and render saved to '{camera_view_path}'")
    plt.close()
    
    # Also create individual mesh renders with different colors
    print("\nCreating individual mesh renders...")
    num_meshes = min(5, len(vertices_list))  # Limit to 5 meshes for visualization
    fig2 = plt.figure(figsize=(4 * num_meshes, 4))
    
    for i in range(num_meshes):
        vvv = vertices_list[i]
        ttt = cam_translations_list[i]
        color = COLORS[i % len(COLORS)]
        
        rendered_single = renderer.render_rgba(
            vertices=vvv,
            cam_t=ttt,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=color,
            scene_bg_color=(1, 1, 1),
            render_res=render_resolution,
            focal_length=None,
        )
        
        ax = fig2.add_subplot(1, num_meshes, i+1)
        ax.imshow(rendered_single)
        ax.set_title(f'Mesh {i}\n{COLORS[i % len(COLORS)][:3]}', fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    individual_renders_path = os.path.join(output_dir, f"individual_mesh_renders_{filename_suffix}.png")
    plt.savefig(individual_renders_path, dpi=150, bbox_inches='tight')
    print(f"Individual mesh renders saved to '{individual_renders_path}'")
    plt.close()
    
    print("="*60)

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
    
    # Create camera view visualization and render
    create_camera_view_visualization(renderer, shifted_verts, all_cam_t, args.output_dir, filename_suffix)
    
    print("\n" + "="*60)
    print("Done!")
    print("="*60)

if __name__ == "__main__":
    main()