#!/usr/bin/env python3
"""
Visualization utilities for vertex-to-renderer coordinate mapping.

This module provides functions to visualize the relationship between
3D vertex coordinates and their 2D projections in rendered images.
"""

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Union, List
import os
import sys

# Add the amr module to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from amr.utils.geometry import perspective_projection


def visualize_vertex_to_renderer_mapping(
    renderer,
    vertices: Union[np.ndarray, List[np.ndarray]],
    camera_translations: Union[np.ndarray, List[np.ndarray]],
    faces: np.ndarray,
    image_shape: Tuple[int, int],
    boxes: Optional[List[List[int]]] = None,
    output_path: str = None,
    num_sample_vertices: int = 100,
    focal_length: Optional[float] = None,
    camera_center: Optional[Tuple[float, float]] = None,
    img_res: int = 256,
):
    """
    Visualize the relationship between 3D vertex coordinates and their 2D projections.
    
    Creates a visualization showing:
    1. 3D vertices in world space
    2. Camera position and viewing direction
    3. Projected 2D points on the rendered image
    4. Correspondence lines connecting 3D vertices to their 2D projections
    
    Args:
        renderer: Renderer instance
        vertices: Array of vertices (N, V, 3) or list of arrays
        camera_translations: Array of camera translations (N, 3) or list of arrays
        faces: Array of shape (F, 3) containing the mesh faces
        image_shape: Tuple of (height, width) for the output image
        boxes: Optional list of bounding boxes [x1, y1, x2, y2] for each mesh
        output_path: Path to save the visualization
        num_sample_vertices: Number of vertices to sample for visualization (to avoid clutter)
        focal_length: Focal length for projection. If None, uses renderer default
        camera_center: Camera center (cx, cy). If None, uses image center
        img_res: Reference resolution for scaling
    """
    # Convert to lists if needed
    if isinstance(vertices, np.ndarray):
        if vertices.ndim == 2:
            vertices_list = [vertices]
        else:
            vertices_list = [v for v in vertices]
    else:
        vertices_list = vertices
    
    if isinstance(camera_translations, np.ndarray):
        if camera_translations.ndim == 1:
            camera_translations_list = [camera_translations]
        else:
            camera_translations_list = [ct for ct in camera_translations]
    else:
        camera_translations_list = camera_translations
    
    # Set default focal length
    if focal_length is None:
        focal_length = renderer.focal_length
    
    height, width = image_shape
    
    # Set default camera center
    if camera_center is None:
        camera_center = (width / 2., height / 2.)
    
    # Render the image first
    print("Rendering image for visualization...")
    if boxes is not None:
        # Create a white background image tensor
        img_tensor = torch.ones((3, height, width), dtype=torch.float32)
        rendered_img, _ = renderer(
            vertices_list,
            camera_translations_list,
            img_tensor,
            mesh_base_color=(0.85882353, 0.74117647, 0.65098039),
            scene_bg_color=(1, 1, 1),
            boxes=boxes,
        )
    else:
        img_tensor = torch.ones((3, height, width), dtype=torch.float32)
        rendered_img, _ = renderer(
            vertices_list,
            camera_translations_list,
            img_tensor,
            mesh_base_color=(0.85882353, 0.74117647, 0.65098039),
            scene_bg_color=(1, 1, 1),
        )
    
    # Project vertices to 2D for each mesh
    all_3d_points = []
    all_2d_points = []
    all_mesh_indices = []
    
    for mesh_idx, (verts, cam_trans) in enumerate(zip(vertices_list, camera_translations_list)):
        # Apply the same transformations as the renderer
        # 1. Apply camera translation (mesh is in camera space)
        # 2. Apply 180° rotation around X-axis
        # 3. Project to 2D
        
        # Transform vertices: apply 180° rotation around X-axis
        rot_x_180 = trimesh.transformations.rotation_matrix(np.radians(180), [1, 0, 0])
        verts_transformed = trimesh.transformations.transform_points(verts, rot_x_180)
        
        # Apply camera translation (negate x component as in renderer)
        cam_trans_copy = cam_trans.copy()
        cam_trans_copy[0] *= -1.
        verts_camera_space = verts_transformed + cam_trans_copy
        
        # Sample vertices for visualization
        if len(verts_camera_space) > num_sample_vertices:
            sample_indices = np.linspace(0, len(verts_camera_space)-1, num_sample_vertices).astype(int)
            sampled_verts = verts_camera_space[sample_indices]
        else:
            sampled_verts = verts_camera_space
            sample_indices = np.arange(len(verts_camera_space))
        
        # Project to 2D using perspective projection
        # Scale focal length based on image/box size
        if boxes is not None and mesh_idx < len(boxes):
            box = boxes[mesh_idx]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            box_width = x2 - x1
            box_height = y2 - y1
            scale_factor = max(box_height, box_width) / img_res
            scaled_focal_length = focal_length * scale_factor
            box_camera_center = (box_width / 2., box_height / 2.)
        else:
            scale_factor = max(height, width) / img_res
            scaled_focal_length = focal_length * scale_factor
            box_camera_center = camera_center
        
        # Convert to torch tensors for projection
        points_3d = torch.from_numpy(sampled_verts).float().unsqueeze(0)  # (1, N, 3)
        translation = torch.from_numpy(cam_trans_copy).float().unsqueeze(0)  # (1, 3)
        focal = torch.tensor([[scaled_focal_length, scaled_focal_length]]).float()  # (1, 2)
        cam_center = torch.tensor([[box_camera_center[0], box_camera_center[1]]]).float()  # (1, 2)
        
        # Project to 2D
        points_2d = perspective_projection(
            points_3d,
            translation,
            focal,
            cam_center
        )  # (1, N, 2)
        
        points_2d = points_2d.squeeze(0).numpy()  # (N, 2)
        
        # Adjust 2D coordinates if boxes are used
        if boxes is not None and mesh_idx < len(boxes):
            box = boxes[mesh_idx]
            x1, y1, x2, y2 = [int(coord) for coord in box]
            points_2d[:, 0] += x1
            points_2d[:, 1] += y1
        
        all_3d_points.append(sampled_verts)
        all_2d_points.append(points_2d)
        all_mesh_indices.extend([mesh_idx] * len(sampled_verts))
    
    # Combine all points
    all_3d_points = np.vstack(all_3d_points)
    all_2d_points = np.vstack(all_2d_points)
    
    # Filter out points outside image bounds
    valid_mask = (
        (all_2d_points[:, 0] >= 0) & (all_2d_points[:, 0] < width) &
        (all_2d_points[:, 1] >= 0) & (all_2d_points[:, 1] < height)
    )
    all_3d_points = all_3d_points[valid_mask]
    all_2d_points = all_2d_points[valid_mask]
    all_mesh_indices = np.array(all_mesh_indices)[valid_mask]
    
    # Create visualization
    fig = plt.figure(figsize=(20, 10))
    
    # Define colors for each mesh
    COLORS = [
        (1.0, 0.0, 0.0),      # Red
        (0.0, 0.0, 1.0),      # Blue
        (1.0, 1.0, 0.0),      # Yellow
        (0.0, 1.0, 0.0),      # Green
        (0.0, 1.0, 1.0),      # Cyan
        (1.0, 0.0, 1.0),      # Magenta
    ]
    
    # Left: 3D view showing vertices and camera
    ax1 = fig.add_subplot(121, projection='3d')
    
    # Plot 3D vertices colored by mesh
    for mesh_idx in range(len(vertices_list)):
        mask = all_mesh_indices == mesh_idx
        if mask.sum() > 0:
            color = COLORS[mesh_idx % len(COLORS)]
            ax1.scatter(
                all_3d_points[mask, 0],
                all_3d_points[mask, 1],
                all_3d_points[mask, 2],
                c=[color], s=10, alpha=0.6, label=f'Mesh {mesh_idx}'
            )
    
    # Draw camera position (at origin after transformations)
    camera_pos = np.array([0, 0, 0])
    ax1.scatter([camera_pos[0]], [camera_pos[1]], [camera_pos[2]],
               c='red', s=300, marker='^', label='Camera', edgecolors='black', linewidths=2)
    
    # Draw viewing direction (negative Z axis)
    view_end = camera_pos + np.array([0, 0, -5])
    ax1.plot([camera_pos[0], view_end[0]],
            [camera_pos[1], view_end[1]],
            [camera_pos[2], view_end[2]],
            'r--', linewidth=2, label='Viewing Direction')
    
    ax1.set_xlabel('X (Right)', fontsize=10)
    ax1.set_ylabel('Y (Down)', fontsize=10)
    ax1.set_zlabel('Z (Into Screen)', fontsize=10)
    ax1.set_title('3D Vertex Positions\n(in camera space after transformations)', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True)
    
    # Right: 2D rendered image with projected points overlaid
    ax2 = fig.add_subplot(122)
    ax2.imshow(rendered_img)
    
    # Plot projected 2D points
    for mesh_idx in range(len(vertices_list)):
        mask = all_mesh_indices == mesh_idx
        if mask.sum() > 0:
            color = COLORS[mesh_idx % len(COLORS)]
            ax2.scatter(
                all_2d_points[mask, 0],
                all_2d_points[mask, 1],
                c=[color], s=20, alpha=0.8, marker='o',
                edgecolors='white', linewidths=0.5, label=f'Mesh {mesh_idx} vertices'
            )
    
    ax2.set_title('2D Projection on Rendered Image\n(Colored dots = projected vertices)', fontsize=12, fontweight='bold')
    ax2.axis('off')
    ax2.legend(loc='upper right')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Vertex-to-renderer mapping visualization saved to '{output_path}'")
    else:
        plt.show()
    
    plt.close()
    
    # Print statistics
    print(f"\n{'='*60}")
    print("Vertex Projection Statistics")
    print(f"{'='*60}")
    print(f"Total vertices sampled: {len(all_3d_points)}")
    print(f"Valid projections (within image): {valid_mask.sum()}")
    print(f"3D vertex range:")
    print(f"  X: [{all_3d_points[:, 0].min():.2f}, {all_3d_points[:, 0].max():.2f}]")
    print(f"  Y: [{all_3d_points[:, 1].min():.2f}, {all_3d_points[:, 1].max():.2f}]")
    print(f"  Z: [{all_3d_points[:, 2].min():.2f}, {all_3d_points[:, 2].max():.2f}]")
    print(f"2D projection range:")
    print(f"  X: [{all_2d_points[:, 0].min():.1f}, {all_2d_points[:, 0].max():.1f}]")
    print(f"  Y: [{all_2d_points[:, 1].min():.1f}, {all_2d_points[:, 1].max():.1f}]")
    print(f"{'='*60}\n")
