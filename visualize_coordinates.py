import numpy as np
import trimesh
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
import sys
import pickle

# Add the amr module to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from amr.utils.renderer import Renderer
from amr.configs import get_config

# Load the data from cows directory
cows_dir = "cows"
vertices_file = os.path.join(cows_dir, "vertices.npy")
cam_translations_file = os.path.join(cows_dir, "cam_translations.npy")
mesh_file = os.path.join(cows_dir, "mesh.obj")

# Load data
all_verts = np.load(vertices_file)
all_cam_t = np.load(cam_translations_file)

print(f"Loaded {len(all_verts)} meshes")
print(f"Vertices shape: {all_verts.shape}")
print(f"Camera translations shape: {all_cam_t.shape}")
print(f"\nFirst camera translation: {all_cam_t[0]}")
print(f"First vertex sample (first 5 vertices):\n{all_verts[0][:5]}")

# Load model config (only needed for Renderer initialization)
path_model_cfg = 'data/hydra/config.yaml'
model_cfg = get_config(path_model_cfg)

# Load faces directly from SMAL pickle file (much faster than loading full checkpoint!)
smal_model_path = model_cfg.SMAL.MODEL_PATH
with open(smal_model_path, 'rb') as f:
    smal_data = pickle.load(f, encoding="latin1")
faces = np.array(smal_data['f'], dtype=np.int32)

print(f"\nLoaded faces directly from SMAL model: {smal_model_path}")
print(f"Faces shape: {faces.shape}")

# Create renderer
renderer = Renderer(model_cfg, faces=faces)

# Create the mesh using the same transformations as in app.py
LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)

# Define color cycle: red, blue, yellow, green, cyan, magenta, orange, purple, etc.
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
for i, (vvv, ttt) in enumerate(zip(all_verts, all_cam_t)):
    color = COLORS[i % len(COLORS)]  # Cycle through colors
    mesh_obj = renderer.vertices_to_trimesh(vvv, ttt.copy(), color)
    # Set vertex colors for visualization
    vertex_colors = np.array([(*color, 1.0)] * len(vvv))
    mesh_obj.visual.vertex_colors = vertex_colors
    trimeshes.append(mesh_obj)

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

# Function to create axis arrows
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

# Calculate equal axis limits (same scale for all axes)
max_range = (bounds[1] - bounds[0]).max()
mid_x = (bounds[0][0] + bounds[1][0]) / 2
mid_y = (bounds[0][1] + bounds[1][1]) / 2
mid_z = (bounds[0][2] + bounds[1][2]) / 2
half_range = max_range / 2

# Equal limits for all axes
equal_xlim = [mid_x - half_range, mid_x + half_range]
equal_ylim = [mid_y - half_range, mid_y + half_range]
equal_zlim = [mid_z - half_range, mid_z + half_range]

# Helper function to set equal aspect ratio
def set_equal_aspect(ax, xlim, ylim, zlim):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    # Calculate box aspect based on ranges
    x_range = xlim[1] - xlim[0]
    y_range = ylim[1] - ylim[0]
    z_range = zlim[1] - zlim[0]
    ax.set_box_aspect([x_range, y_range, z_range])

# Create visualization
fig = plt.figure(figsize=(18, 6))

# Prepare colors for each vertex based on which mesh it belongs to
vertex_colors_list = []
vertex_to_mesh_idx = []
for i, (vvv, ttt) in enumerate(zip(all_verts, all_cam_t)):
    color = COLORS[i % len(COLORS)]
    num_vertices = len(vvv)
    vertex_colors_list.extend([color] * num_vertices)
    vertex_to_mesh_idx.extend([i] * num_vertices)

vertex_colors_array = np.array(vertex_colors_list)

# View 1: Front view (looking down -Z axis)
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], 
           c=vertex_colors_array, s=0.1, alpha=0.6)
for start, end, color, label in axis_arrows:
    ax1.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=3, label=label)
ax1.set_xlabel('X (Right)', fontsize=10)
ax1.set_ylabel('Y (Down)', fontsize=10)
ax1.set_zlabel('Z (Into Screen)', fontsize=10)
ax1.set_title('Front View\n(Looking down -Z axis)', fontsize=12, fontweight='bold')
ax1.view_init(elev=0, azim=0)
ax1.legend()
ax1.grid(True)
set_equal_aspect(ax1, equal_xlim, equal_ylim, equal_zlim)

# View 2: Side view (looking down -X axis)
ax2 = fig.add_subplot(132, projection='3d')
ax2.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], 
           c=vertex_colors_array, s=0.1, alpha=0.6)
for start, end, color, label in axis_arrows:
    ax2.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=3)
ax2.set_xlabel('X (Right)', fontsize=10)
ax2.set_ylabel('Y (Down)', fontsize=10)
ax2.set_zlabel('Z (Into Screen)', fontsize=10)
ax2.set_title('Side View\n(Looking down -X axis)', fontsize=12, fontweight='bold')
ax2.view_init(elev=0, azim=90)
ax2.grid(True)
set_equal_aspect(ax2, equal_xlim, equal_ylim, equal_zlim)

# View 3: Top view (looking down +Y axis)
ax3 = fig.add_subplot(133, projection='3d')
ax3.scatter(mesh_vertices[:, 0], mesh_vertices[:, 1], mesh_vertices[:, 2], 
           c=vertex_colors_array, s=0.1, alpha=0.6)
for start, end, color, label in axis_arrows:
    ax3.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=3)
ax3.set_xlabel('X (Right)', fontsize=10)
ax3.set_ylabel('Y (Down)', fontsize=10)
ax3.set_zlabel('Z (Into Screen)', fontsize=10)
ax3.set_title('Top View\n(Looking down +Y axis)', fontsize=12, fontweight='bold')
ax3.view_init(elev=90, azim=0)
ax3.grid(True)
set_equal_aspect(ax3, equal_xlim, equal_ylim, equal_zlim)

plt.tight_layout()
plt.savefig('coordinate_system_visualization.png', dpi=150, bbox_inches='tight')
print("\nVisualization saved to 'coordinate_system_visualization.png'")
plt.show()

# Also create a wireframe visualization
fig2 = plt.figure(figsize=(12, 12))
ax4 = fig2.add_subplot(221, projection='3d')
ax5 = fig2.add_subplot(222, projection='3d')
ax6 = fig2.add_subplot(223, projection='3d')
ax7 = fig2.add_subplot(224, projection='3d')

# Sample vertices for wireframe (too many points otherwise)
sample_indices = np.linspace(0, len(mesh_vertices)-1, min(5000, len(mesh_vertices))).astype(int)
sample_vertices = mesh_vertices[sample_indices]
sample_colors = vertex_colors_array[sample_indices]

# Isometric view
ax4.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2], 
           c=sample_colors, s=1, alpha=0.5)
for start, end, color, label in axis_arrows:
    ax4.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=4, label=label)
ax4.set_xlabel('X (Right)', fontsize=12)
ax4.set_ylabel('Y (Down)', fontsize=12)
ax4.set_zlabel('Z (Into Screen)', fontsize=12)
ax4.set_title('Isometric View', fontsize=14, fontweight='bold')
ax4.view_init(elev=20, azim=45)
ax4.legend(loc='upper left')
ax4.grid(True)
set_equal_aspect(ax4, equal_xlim, equal_ylim, equal_zlim)

# Front
ax5.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2], 
           c=sample_colors, s=1, alpha=0.5)
for start, end, color, label in axis_arrows:
    ax5.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=4)
ax5.set_xlabel('X (Right)', fontsize=12)
ax5.set_ylabel('Y (Down)', fontsize=12)
ax5.set_zlabel('Z (Into Screen)', fontsize=12)
ax5.set_title('Front View (elev=0°, azim=0°)', fontsize=12)
ax5.view_init(elev=0, azim=0)
ax5.grid(True)
set_equal_aspect(ax5, equal_xlim, equal_ylim, equal_zlim)

# Side
ax6.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2], 
           c=sample_colors, s=1, alpha=0.5)
for start, end, color, label in axis_arrows:
    ax6.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=4)
ax6.set_xlabel('X (Right)', fontsize=12)
ax6.set_ylabel('Y (Down)', fontsize=12)
ax6.set_zlabel('Z (Into Screen)', fontsize=12)
ax6.set_title('Side View (elev=0°, azim=90°)', fontsize=12)
ax6.view_init(elev=0, azim=90)
ax6.grid(True)
set_equal_aspect(ax6, equal_xlim, equal_ylim, equal_zlim)

# Top
ax7.scatter(sample_vertices[:, 0], sample_vertices[:, 1], sample_vertices[:, 2], 
           c=sample_colors, s=1, alpha=0.5)
for start, end, color, label in axis_arrows:
    ax7.plot([start[0], end[0]], [start[1], end[1]], [start[2], end[2]], 
            color=color, linewidth=4)
ax7.set_xlabel('X (Right)', fontsize=12)
ax7.set_ylabel('Y (Down)', fontsize=12)
ax7.set_zlabel('Z (Into Screen)', fontsize=12)
ax7.set_title('Top View (elev=90°, azim=0°)', fontsize=12)
ax7.view_init(elev=90, azim=0)
ax7.grid(True)
set_equal_aspect(ax7, equal_xlim, equal_ylim, equal_zlim)

plt.tight_layout()
plt.savefig('coordinate_system_detailed.png', dpi=150, bbox_inches='tight')
print("Detailed visualization saved to 'coordinate_system_detailed.png'")
plt.show()

# Print coordinate system summary
print("\n" + "="*60)
print("COORDINATE SYSTEM SUMMARY")
print("="*60)
print(f"Origin (0, 0, 0): World space origin")
print(f"X-axis (Red): Points RIGHT (+X = right)")
print(f"Y-axis (Green): Points DOWN (+Y = down, flipped from camera space)")
print(f"Z-axis (Blue): Points INTO SCREEN/AWAY FROM VIEWER (+Z = away)")
print(f"\nMesh is positioned at centroid: {center}")
print(f"Mesh extent: X=[{bounds[0][0]:.2f}, {bounds[1][0]:.2f}], "
      f"Y=[{bounds[0][1]:.2f}, {bounds[1][1]:.2f}], "
      f"Z=[{bounds[0][2]:.2f}, {bounds[1][2]:.2f}]")
print("\nNote: The 180° rotation around X-axis flips Y and Z:")
print("  - Y: up → down (positive Y points down)")
print("  - Z: forward → backward (positive Z points away)")
print("="*60)