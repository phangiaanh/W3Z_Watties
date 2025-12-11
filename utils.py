import numpy as np
from typing import List, Tuple, Dict
import torch
import trimesh
from amr.utils.renderer import Renderer

LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)

def compare_masks(mask1: np.ndarray, mask2: np.ndarray, name1: str = "Mask 1", name2: str = "Mask 2"):
    """
    Compare two binary masks in terms of scale (area/size) and position (centroid, bounding box).
    
    Args:
        mask1: Binary mask array, shape (H, W) or (H, W, 1)
        mask2: Binary mask array, shape (H, W) or (H, W, 1)
        name1: Name for mask1 (for printing)
        name2: Name for mask2 (for printing)
    
    Returns:
        dict: Dictionary containing comparison metrics
    """
    # Ensure masks are 2D and binary
    if mask1.ndim == 3:
        mask1 = mask1.squeeze()
    if mask2.ndim == 3:
        mask2 = mask2.squeeze()
    
    mask1 = (mask1 > 0).astype(np.float32)
    mask2 = (mask2 > 0).astype(np.float32)
    
    # Scale comparison: area (number of pixels)
    area1 = np.sum(mask1)
    area2 = np.sum(mask2)
    area_ratio = area2 / area1 if area1 > 0 else 0
    area_diff = area2 - area1
    
    # Position comparison: centroid (center of mass)
    y_coords, x_coords = np.mgrid[0:mask1.shape[0], 0:mask1.shape[1]]
    
    if area1 > 0:
        centroid1_y = np.sum(y_coords * mask1) / area1
        centroid1_x = np.sum(x_coords * mask1) / area1
    else:
        centroid1_y, centroid1_x = 0, 0
    
    if area2 > 0:
        centroid2_y = np.sum(y_coords * mask2) / area2
        centroid2_x = np.sum(x_coords * mask2) / area2
    else:
        centroid2_y, centroid2_x = 0, 0
    
    centroid_diff_y = centroid2_y - centroid1_y
    centroid_diff_x = centroid2_x - centroid1_x
    centroid_distance = np.sqrt(centroid_diff_y**2 + centroid_diff_x**2)
    
    # Bounding box comparison
    def get_bbox(mask):
        if np.sum(mask) == 0:
            return None
        coords = np.argwhere(mask > 0)
        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)
        return {
            'x_min': x_min, 'y_min': y_min,
            'x_max': x_max, 'y_max': y_max,
            'width': x_max - x_min + 1,
            'height': y_max - y_min + 1,
            'center_x': (x_min + x_max) / 2,
            'center_y': (y_min + y_max) / 2
        }
    
    bbox1 = get_bbox(mask1)
    bbox2 = get_bbox(mask2)
    
    # IoU (Intersection over Union)
    intersection = np.sum(mask1 * mask2)
    union = np.sum((mask1 + mask2) > 0)
    iou = intersection / union if union > 0 else 0
    
    # Compile results
    results = {
        'area1': area1,
        'area2': area2,
        'area_ratio': area_ratio,
        'area_diff': area_diff,
        'centroid1': (centroid1_x, centroid1_y),
        'centroid2': (centroid2_x, centroid2_y),
        'centroid_diff': (centroid_diff_x, centroid_diff_y),
        'centroid_distance': centroid_distance,
        'bbox1': bbox1,
        'bbox2': bbox2,
        'iou': iou
    }
    
    # Print comparison
    # print(f"\n{'='*60}")
    # print(f"Mask Comparison: {name1} vs {name2}")
    # print(f"{'='*60}")
    # print(f"\nScale (Area):")
    # print(f"  {name1}: {area1:.2f} pixels")
    # print(f"  {name2}: {area2:.2f} pixels")
    # print(f"  Ratio ({name2}/{name1}): {area_ratio:.4f}")
    # print(f"  Difference: {area_diff:.2f} pixels")
    
    # print(f"\nPosition (Centroid):")
    # print(f"  {name1}: ({centroid1_x:.2f}, {centroid1_y:.2f})")
    # print(f"  {name2}: ({centroid2_x:.2f}, {centroid2_y:.2f})")
    # print(f"  Difference: ({centroid_diff_x:.2f}, {centroid_diff_y:.2f})")
    # print(f"  Distance: {centroid_distance:.2f} pixels")
    
    # if bbox1 and bbox2:
    #     print(f"\nBounding Box:")
    #     print(f"  {name1}: [{bbox1['x_min']}, {bbox1['y_min']}] to [{bbox1['x_max']}, {bbox1['y_max']}] "
    #           f"(size: {bbox1['width']}x{bbox1['height']})")
    #     print(f"  {name2}: [{bbox2['x_min']}, {bbox2['y_min']}] to [{bbox2['x_max']}, {bbox2['y_max']}] "
    #           f"(size: {bbox2['width']}x{bbox2['height']})")
    #     bbox_center_diff_x = bbox2['center_x'] - bbox1['center_x']
    #     bbox_center_diff_y = bbox2['center_y'] - bbox1['center_y']
    #     bbox_size_diff_w = bbox2['width'] - bbox1['width']
    #     bbox_size_diff_h = bbox2['height'] - bbox1['height']
    #     print(f"  Center difference: ({bbox_center_diff_x:.2f}, {bbox_center_diff_y:.2f})")
    #     print(f"  Size difference: ({bbox_size_diff_w:.2f}, {bbox_size_diff_h:.2f})")
    
    # print(f"\nOverlap:")
    # print(f"  IoU (Intersection over Union): {iou:.4f}")
    # print(f"{'='*60}\n")
    
    return results

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0

def extract_mesh_depths(depth: np.ndarray, valid_mask: np.ndarray) -> List[float]:
    """
    Extract synthesized depth information for each mesh based on valid_mask.
    
    For each mesh, masks the depth map using the mesh's valid_mask and computes
    the median depth value of the masked region.
    
    The depth map is automatically resized to match valid_mask dimensions if they differ.
    
    Args:
        depth: Depth map array, shape (H, W)
        valid_mask: Valid mask array, shape (I, H, W, 1) or (I, H, W) where I is number of meshes
    
    Returns:
        List of N floats, where N is the number of meshes (I), containing the median
        depth value for each mesh. Returns np.nan if a mesh has no valid pixels.
    """
    from scipy.ndimage import zoom
    
    # Ensure depth is 2D
    if depth.ndim != 2:
        raise ValueError(f"Depth must be 2D (H, W), got shape {depth.shape}")
    
    # Ensure valid_mask is in correct format
    if valid_mask.ndim == 4:
        # Shape: (I, H, W, 1) -> squeeze to (I, H, W)
        valid_mask = valid_mask.squeeze(axis=-1)
    elif valid_mask.ndim == 3:
        # Shape: (I, H, W) - already correct
        pass
    else:
        raise ValueError(f"valid_mask must be 3D (I, H, W) or 4D (I, H, W, 1), got shape {valid_mask.shape}")
    
    # Get target spatial dimensions from valid_mask
    target_h, target_w = valid_mask.shape[1:3]
    depth_h, depth_w = depth.shape
    
    # Resize depth to match valid_mask dimensions if they differ
    if (depth_h, depth_w) != (target_h, target_w):
        print(f"Resizing depth from {depth_h}x{depth_w} to {target_h}x{target_w} to match valid_mask")
        scale_h = target_h / depth_h
        scale_w = target_w / depth_w
        depth = zoom(depth, (scale_h, scale_w), order=1)  # Bilinear interpolation
    
    # Binarize valid_mask (threshold > 0)
    valid_mask = (valid_mask > 0).astype(bool)
    
    num_meshes = valid_mask.shape[0]
    median_depths = []
    
    for i in range(num_meshes):
        # Get mask for this mesh
        mask = valid_mask[i]  # Shape: (H, W)
        
        # Apply mask to depth map
        masked_depth = depth[mask]
        
        # Compute median depth for this mesh
        if len(masked_depth) > 0:
            median_depth = np.mean(masked_depth)
            median_depths.append(float(median_depth))
        else:
            # No valid pixels for this mesh
            median_depths.append(np.nan)
    
    return median_depths

def shift_vertices_by_translation(vertices, translation, scale=1.0):
    """
    Shift vertices along a translation vector and optionally scale around centroid.
    
    Args:
        vertices: Array of vertices with shape (V, 3) for a single mesh,
                  or (N, V, 3) for multiple meshes
        translation: Translation vector with shape (3,) containing [x, y, z] shift
        scale: Scale factor to apply around the centroid (default: 1.0)
    
    Returns:
        Shifted and scaled vertices array with the same shape as input vertices
    
    Example:
        >>> vertices = np.array([[0, 0, 0], [1, 1, 1]])  # Shape: (2, 3)
        >>> translation = np.array([0.5, 0.3, 0.2])  # Shape: (3,)
        >>> shifted = shift_vertices_by_translation(vertices, translation, scale=2.0)
        >>> # Result: vertices scaled by 2x around centroid, then translated
    """
    vertices = vertices.copy()
    translation = np.asarray(translation)
    
    if translation.shape != (3,):
        raise ValueError(f"Translation must have shape (3,), got {translation.shape}")
    
    # Scale around centroid if scale != 1.0
    if scale != 1.0:
        if vertices.ndim == 2:
            # Single mesh: (V, 3)
            centroid = vertices.mean(axis=0)
            vertices = (vertices - centroid) * scale + centroid
        elif vertices.ndim == 3:
            # Multiple meshes: (N, V, 3)
            centroid = vertices.mean(axis=1, keepdims=True)  # Shape: (N, 1, 3)
            vertices = (vertices - centroid) * scale + centroid
        else:
            raise ValueError(f"Vertices must have 2 or 3 dimensions, got {vertices.ndim}")
    
    # Apply translation
    if vertices.ndim == 2:
        # Single mesh: (V, 3)
        vertices += translation
    elif vertices.ndim == 3:
        # Multiple meshes: (N, V, 3)
        vertices += translation[np.newaxis, np.newaxis, :]
    else:
        raise ValueError(f"Vertices must have 2 or 3 dimensions, got {vertices.ndim}")
    
    return vertices

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
    regression_img, valid_mask = renderer(
        vertices,
        cam_translations,
        img_tensor,
        mesh_base_color=LIGHT_BLUE,
        scene_bg_color=(1, 1, 1),
        boxes=boxes_list,
    )
    
    # Convert to uint8
    regression_img = (regression_img * 255).astype(np.uint8)
    return regression_img, valid_mask

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

def create_image_tensor(image_shape: Tuple[int, int]):
    """Create a white background image tensor with full image shape."""
    height, width = image_shape
    # Create white image (1.0, 1.0, 1.0) in RGB
    img = np.ones((height, width, 3), dtype=np.float32)
    # Convert to torch tensor (3, H, W) format
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_tensor

def optimize_camera_translation(vertices: np.ndarray, 
                                initial_camera_translation: np.ndarray,
                                target_mask: np.ndarray,
                                renderer: Renderer,
                                img_tensor: torch.Tensor,
                                boxes: List[List[int]],
                                initial_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                method: str = 'L-BFGS-B',
                                not_anchor: bool = False,
                                size_ratio_threshold: float = 1.01,
                                # Stage 2 hyperparameters
                                initial_step_ratio: float = 0.01,
                                min_step_size: float = 0.01,
                                adaptive_multiplier: float = 0.3,
                                step_reduction_factor: float = 0.7,
                                centroid_convergence_threshold: float = 0.5,
                                max_outer_iterations: int = 50,
                                max_inner_iterations: int = 5,
                                iou_tolerance: float = 1e-5,
                                # Linear relationship parameters (from linearity test)
                                slope_x: float = 79.4157,
                                slope_y: float = 86.1128,
                                use_linear_optimization: bool = True) -> Dict:
    """
    Hierarchical optimization of camera translation offset (tx, ty, sz):
    1. First optimize z (depth) if target mask is smaller than rendered mask
    2. Then optimize x-y (translation) once sizes are approximately equal
    3. Final refinement of all parameters together
    
    If not_anchor=True, scales vertices directly instead of scaling via camera pose.
    
    Args:
        vertices: Vertices array, shape (V, 3) for single mesh
        initial_camera_translation: Initial camera translation, shape (3,)
        target_mask: Target mask to match, shape (H, W) or (H, W, 1)
        renderer: Renderer instance
        img_tensor: Image tensor for rendering, shape (3, H, W)
        boxes: Bounding boxes for rendering
        initial_offset: Initial guess for offset (tx, ty, sz) to add to camera_translation
        method: Optimization method (default: 'L-BFGS-B')
        not_anchor: If True, scale vertices directly instead of scaling via camera pose
        size_ratio_threshold: Ratio threshold for size matching (e.g., 1.01 means within 1% difference)
        initial_step_ratio: Initial step size as ratio of image dimension (default: 0.01 = 1%)
        min_step_size: Minimum step size for fine-tuning (default: 0.01)
        adaptive_multiplier: Multiplier for adaptive step based on distance (default: 0.3)
        step_reduction_factor: Factor to reduce step when no improvement (default: 0.7)
        centroid_convergence_threshold: Pixel threshold for centroid convergence (default: 0.5)
        max_outer_iterations: Maximum outer loop iterations (default: 50)
        max_inner_iterations: Maximum inner loop iterations per coordinate (default: 5)
        iou_tolerance: IoU error tolerance for convergence (default: 1e-5)
    
    Returns:
        Dictionary containing:
            - 'camera_translation': Optimized camera translation (initial + offset)
            - 'offset': Optimized offset (tx, ty, sz)
            - 'min_difference': Minimized difference value
            - 'optimized_mask': Final rendered mask
            - 'iou': IoU between optimized mask and target mask
            - 'pixel_diff': Pixel-wise difference count
    """
    # Ensure target mask is 2D and binary
    if target_mask.ndim == 3:
        target_mask = target_mask.squeeze()
    target_mask = (target_mask > 0).astype(np.float32)
    h, w = target_mask.shape
    
    # Get image shape from img_tensor
    img_h, img_w = img_tensor.shape[1], img_tensor.shape[2]
    if (h, w) != (img_h, img_w):
        print(f"Warning: Target mask shape {(h, w)} != image shape {(img_h, img_w)}. Using image shape.")
        h, w = img_h, img_w
    
    # Ensure vertices is in correct format
    if vertices.ndim == 2:
        vertices_list = [vertices]
    else:
        vertices_list = [v for v in vertices]
    
    # Get vertices center for scaling (if not_anchor mode)
    vertices_center = None
    if not_anchor:
        vertices_center = vertices_list[0].mean(axis=0)
    
    # Ensure boxes is in correct format
    if isinstance(boxes, np.ndarray):
        if boxes.ndim == 1:
            boxes_list = [boxes.tolist()]
        else:
            boxes_list = [box.tolist() for box in boxes]
    elif isinstance(boxes, list):
        boxes_list = boxes
    else:
        boxes_list = [boxes]
    
    # If no boxes provided, use full image
    if len(boxes_list) == 0:
        boxes_list = [[0, 0, w, h]]
    
    # Compute target mask area
    target_mask_area = np.sum(target_mask)
    opt_scale = 1.0
    
    # Helper function to render and get mask area
    def render_and_get_area(tx, ty, tz, ts):
        
    # Convert array to scalar if needed
        if isinstance(tz, np.ndarray):
            tz = tz[0] if tz.size > 0 else tz.item()

        new_camera_translation = initial_camera_translation + np.array([tx, ty, tz])
        transformed_vertices = shift_vertices_by_translation(vertices_list[0], initial_camera_translation + np.array([tx, ty, tz]), scale=ts)
        try:
            _, rendered_mask = render_meshes(
                renderer,
                np.array([transformed_vertices]),
                np.array([[0, 0, 0]]),
                img_tensor,
                np.array([boxes_list[0]])
            )
            
            rendered_mask = np.array(rendered_mask)
            if rendered_mask.ndim == 4:
                rendered_mask = rendered_mask[0]
            if rendered_mask.ndim == 3:
                rendered_mask = rendered_mask.squeeze()
            rendered_mask = (rendered_mask > 0).astype(np.float32)
            
            if rendered_mask.shape != target_mask.shape:
                from scipy.ndimage import zoom
                scale_h = target_mask.shape[0] / rendered_mask.shape[0]
                scale_w = target_mask.shape[1] / rendered_mask.shape[1]
                rendered_mask = zoom(rendered_mask, (scale_h, scale_w), order=1)
                rendered_mask = (rendered_mask > 0.5).astype(np.float32)
            
            rendered_area = np.sum(rendered_mask)
            return rendered_mask, rendered_area
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            return None, 0
    
    # STAGE 1: Optimize z (depth) if target mask is smaller
    # Get initial rendered mask area
    _, initial_rendered_mask = render_meshes(
        renderer,
        np.array([shift_vertices_by_translation(vertices_list[0], initial_camera_translation)]),
        np.array([[0, 0, 0]]),
        img_tensor,
        np.array([boxes_list[0]])
    )
    initial_rendered_mask = np.array(initial_rendered_mask)
    if initial_rendered_mask.ndim == 4:
        initial_rendered_mask = initial_rendered_mask[0]
    if initial_rendered_mask.ndim == 3:
        initial_rendered_mask = initial_rendered_mask.squeeze()
    initial_rendered_mask = (initial_rendered_mask > 0).astype(np.float32)
    initial_rendered_area = np.sum(initial_rendered_mask)
    
    # Check if we need to optimize z first
    size_ratio = initial_rendered_area / target_mask_area if target_mask_area > 0 else 1.0
    opt_tx, opt_ty, opt_sz = initial_offset
    
    if abs(size_ratio - size_ratio_threshold) > 0.01:  # Rendered mask is larger than target
        print(f"Stage 1: Optimizing z (depth) with binary search. Size ratio: {size_ratio:.4f}")
        
        z_initial = opt_scale if not_anchor else opt_sz
        
        # Establish search bounds based on size ratio
        # For depth (z): larger z = further away = smaller rendered area
        # For scale: larger scale = larger rendered area
        if size_ratio > 1.0:  # Rendered area is too large
            if not_anchor:
                # Need to decrease scale to make area smaller
                z_low = 0.1
                z_high = z_initial
            else:
                # Need to increase z (move further away) to make area smaller
                z_low = z_initial
                z_high = z_initial + 30.0
        else:  # Rendered area is too small
            if not_anchor:
                # Need to increase scale to make area larger
                z_low = z_initial
                z_high = 5.0
            else:
                # Need to decrease z (move closer) to make area larger
                z_low = max(z_initial - 30.0, -30.0)
                z_high = z_initial
        
        # Binary search parameters
        tolerance = 1e-5
        max_iter = 30
        best_z = z_initial
        best_error = float('inf')
        
        # Binary search
        for iteration in range(max_iter):
            z_mid = (z_low + z_high) / 2.0
            
            # Test midpoint
            if not_anchor:
                rendered_mask, rendered_area = render_and_get_area(opt_tx, opt_ty, opt_sz, z_mid)
            else:
                rendered_mask, rendered_area = render_and_get_area(opt_tx, opt_ty, z_mid, opt_scale)
            
            if rendered_mask is None:
                # If rendering fails, narrow to the working side
                if z_mid > z_initial:
                    z_high = z_mid
                else:
                    z_low = z_mid
                continue
            
            current_ratio = rendered_area / target_mask_area if target_mask_area > 0 else 1.0
            error = abs(current_ratio - 1.0)
            
            # Track best solution
            if error < best_error:
                best_error = error
                best_z = z_mid
            
            # Check convergence
            if error < tolerance:
                best_z = z_mid
                break
            
            # Binary search logic: determine which side to search
            if current_ratio > 1.0:  # Rendered area is too large
                if not_anchor:
                    z_high = z_mid  # Decrease scale
                else:
                    z_low = z_mid  # Increase z (move further away)
            else:  # Rendered area is too small
                if not_anchor:
                    z_low = z_mid  # Increase scale
                else:
                    z_high = z_mid  # Decrease z (move closer)
            
            # Check if bounds are too close
            if abs(z_high - z_low) < tolerance:
                break
        
        # Use best found value
        z = best_z
        if not_anchor:
            opt_scale = z
        else:
            opt_sz = z
        print(f"Stage 1 complete. Optimized sz: {opt_sz:.4f}, scale: {opt_scale:.4f} "
              f"(iterations: {iteration+1}, final error: {best_error:.6f})")
    
    # STAGE 2: Optimize x-y once sizes are approximately equal
    # Check current size ratio with optimized z
    current_mask, current_area = render_and_get_area(opt_tx, opt_ty, opt_sz, opt_scale)
    if current_mask is not None:
        # Precompute target mask centroid (constant throughout optimization)
        c0 = centroid(target_mask)
        if c0 is None:
            print("Skipping Stage 2: Target mask has no valid pixels")
            opt_tx, opt_ty = opt_tx, opt_ty
        else:
            if use_linear_optimization:
                print(f"Stage 2: Optimizing x-y translation using LINEAR relationship. Target centroid: {c0}")
                print(f"  Using slopes: slope_x={slope_x:.4f}, slope_y={slope_y:.4f}")
                
                # Get initial rendered mask centroid
                cz = centroid(current_mask)
                if cz is None:
                    print("Skipping Stage 2: Rendered mask has no valid pixels")
                    opt_tx, opt_ty = opt_tx, opt_ty
                else:
                    # Compute required centroid shifts
                    dx = c0[0] - cz[0]  # positive → rendered is left → move tx right
                    dy = c0[1] - cz[1]  # positive → rendered is up → move ty down
                    
                    # Use linear relationship to compute required translation deltas
                    # delta_tx = (centroid_x_shift) / slope_x
                    # delta_ty = (centroid_y_shift) / slope_y
                    delta_tx = dx / slope_x
                    delta_ty = dy / slope_y
                    
                    # Apply the computed deltas
                    opt_tx = opt_tx + delta_tx
                    opt_ty = opt_ty + delta_ty
                    
                    # HYPERPARAMETER TUNING GUIDE:
                    # - initial_step_ratio: Increase (0.02-0.05) if optimization is too slow or far from target
                    #                        Decrease (0.005-0.01) if optimization overshoots or oscillates
                    # - min_step_size: Increase (0.02-0.05) for faster convergence, decrease (0.005-0.01) for precision
                    # - adaptive_multiplier: Increase (0.4-0.6) for larger steps when far, decrease (0.2-0.3) for stability
                    # - step_reduction_factor: Increase (0.8-0.9) for slower reduction, decrease (0.5-0.7) for faster
                    # - centroid_convergence_threshold: Increase (1.0-2.0) for faster convergence, decrease (0.1-0.5) for precision
                    # - max_outer_iterations: Increase (100-200) if not converging, decrease (20-30) for speed
                    # - max_inner_iterations: Increase (8-10) for more refinement per coordinate, decrease (3-5) for speed
                    
                    # Track previous residuals to detect oscillation
                    prev_dx = float('inf')
                    prev_dy = float('inf')
                    damping_factor = 1.0  # Start with full step
                    
                    # Track best state in case we need to recover
                    best_tx, best_ty = opt_tx, opt_ty
                    best_mask = current_mask
                    best_iou = iou(current_mask, target_mask) if current_mask is not None else 0.0
                    best_cz = cz
                    
                    # Track previous centroid for out-of-frame estimation
                    prev_cz = cz  # Last known valid centroid
                    prev_tx, prev_ty = opt_tx, opt_ty  # Last known valid position
                    
                    for linear_iter in range(max_inner_iterations):
                        # Render to check current state
                        final_mask, _ = render_and_get_area(opt_tx, opt_ty, opt_sz, opt_scale)
                        if final_mask is None:
                            # Rendering failed, but continue with estimated position
                            print(f"  Warning: Rendering failed at iteration {linear_iter+1}, using estimated position")
                            # Estimate centroid based on translation from last known position
                            if prev_cz is not None:
                                # Estimate where centroid would be based on translation delta
                                tx_delta = opt_tx - prev_tx
                                ty_delta = opt_ty - prev_ty
                                # Use linear relationship to estimate centroid movement
                                estimated_cz_x = prev_cz[0] + tx_delta * slope_x
                                estimated_cz_y = prev_cz[1] + ty_delta * slope_y
                                final_cz = (estimated_cz_x, estimated_cz_y)
                            else:
                                # No previous centroid, use target as fallback
                                final_cz = c0
                        else:
                            final_cz = centroid(final_mask)
                            if final_cz is None:
                                # Empty mask - object is out of frame
                                print(f"  Object out of frame at iteration {linear_iter+1}, using estimated position")
                                # Estimate centroid based on translation from last known position
                                if prev_cz is not None:
                                    tx_delta = opt_tx - prev_tx
                                    ty_delta = opt_ty - prev_ty
                                    # Use linear relationship to estimate centroid movement
                                    estimated_cz_x = prev_cz[0] + tx_delta * slope_x
                                    estimated_cz_y = prev_cz[1] + ty_delta * slope_y
                                    final_cz = (estimated_cz_x, estimated_cz_y)
                                else:
                                    # No previous centroid, use target as fallback
                                    final_cz = c0
                            else:
                                # Valid centroid - update tracking
                                prev_cz = final_cz
                                prev_tx, prev_ty = opt_tx, opt_ty
                        
                        # Now we always have a centroid (either real or estimated)
                        final_dx = c0[0] - final_cz[0]
                        final_dy = c0[1] - final_cz[1]
                        
                        # Compute IoU if we have a valid mask
                        if final_mask is not None:
                            final_iou = iou(final_mask, target_mask)
                            # Track best state only if mask is valid
                            if final_iou > best_iou:
                                best_iou = final_iou
                                best_tx, best_ty = opt_tx, opt_ty
                                best_mask = final_mask
                                best_cz = final_cz
                        else:
                            # No valid mask, use IoU of 0 but continue optimizing
                            final_iou = 0.0
                        
                        # Check convergence (only if we have a valid mask)
                        if final_mask is not None and abs(final_dx) <= centroid_convergence_threshold and abs(final_dy) <= centroid_convergence_threshold:
                            print(f"  Converged after {linear_iter+1} iterations")
                            break
                        
                        # Detect oscillation: check if residual sign flipped OR magnitude increased significantly
                        oscillation_detected = False
                        if linear_iter > 0:
                            # Sign flip detection (more reliable)
                            if (prev_dx * final_dx < 0 and abs(final_dx) > abs(prev_dx) * 0.5) or \
                               (prev_dy * final_dy < 0 and abs(final_dy) > abs(prev_dy) * 0.5):
                                oscillation_detected = True
                            # Magnitude increase detection (for any magnitude)
                            elif (abs(final_dx) > abs(prev_dx) * 1.2 and abs(prev_dx) > 1.0) or \
                                 (abs(final_dy) > abs(prev_dy) * 1.2 and abs(prev_dy) > 1.0):
                                oscillation_detected = True
                            
                            if oscillation_detected:
                                # Oscillation detected, reduce step size aggressively
                                damping_factor *= 0.3  # More aggressive reduction
                                if damping_factor < 0.05:
                                    damping_factor = 0.05
                                print(f"  Oscillation detected at iteration {linear_iter+1}, damping={damping_factor:.3f}")
                                # Restore previous position and try again with smaller step
                                opt_tx = opt_tx - (final_dx / slope_x) * (1.0 - damping_factor)
                                opt_ty = opt_ty - (final_dy / slope_y) * (1.0 - damping_factor)
                                continue  # Re-render with corrected position
                        
                        # Adaptive damping: reduce step size as we get closer
                        residual_magnitude = np.sqrt(final_dx**2 + final_dy**2)
                        if residual_magnitude < 10.0:
                            # When close, use smaller steps to avoid overshooting
                            adaptive_damping = min(1.0, residual_magnitude / 10.0)
                            damping_factor = min(damping_factor, adaptive_damping)
                        elif residual_magnitude > 100.0:
                            # When very far, also use smaller steps to avoid huge jumps
                            adaptive_damping = min(1.0, 100.0 / residual_magnitude)
                            damping_factor = min(damping_factor, adaptive_damping)
                        
                        # Print progress
                        out_of_frame_marker = " [OUT]" if final_mask is None or (final_mask is not None and np.sum(final_mask) == 0) else ""
                        if linear_iter == 0 or abs(final_dx) > 10 or abs(final_dy) > 10 or linear_iter % 5 == 0:
                            print(f"  Iteration {linear_iter+1}: tx={opt_tx:.4f}, ty={opt_ty:.4f}, residual dx={final_dx:.2f}, dy={final_dy:.2f}, damping={damping_factor:.3f}{out_of_frame_marker}")
                        
                        # Refine using linear relationship with damping
                        refinement_delta_tx = (final_dx / slope_x) * damping_factor
                        refinement_delta_ty = (final_dy / slope_y) * damping_factor
                        
                        # Limit maximum step size to prevent huge jumps
                        max_step = 2.0  # Maximum translation step
                        if abs(refinement_delta_tx) > max_step:
                            refinement_delta_tx = np.sign(refinement_delta_tx) * max_step
                        if abs(refinement_delta_ty) > max_step:
                            refinement_delta_ty = np.sign(refinement_delta_ty) * max_step
                        
                        opt_tx = opt_tx + refinement_delta_tx
                        opt_ty = opt_ty + refinement_delta_ty
                        
                        # Store previous residuals for oscillation detection
                        prev_dx = final_dx
                        prev_dy = final_dy
                    
                    # Use best state if we have one and it's better than current
                    if best_mask is not None and best_iou > final_iou:
                        opt_tx, opt_ty = best_tx, best_ty
                        final_mask = best_mask
                        final_cz = best_cz
                        if final_cz is not None:
                            final_dx = c0[0] - final_cz[0]
                            final_dy = c0[1] - final_cz[1]
                        final_iou = best_iou
                    elif final_cz is None:
                        # If we still don't have a valid centroid, use best or previous
                        if best_cz is not None:
                            final_cz = best_cz
                            final_dx = c0[0] - final_cz[0]
                            final_dy = c0[1] - final_cz[1]
                        elif prev_cz is not None:
                            final_cz = prev_cz
                            final_dx = c0[0] - final_cz[0]
                            final_dy = c0[1] - final_cz[1]
                    print(f"Stage 2 complete (LINEAR). Optimized tx: {opt_tx:.4f}, ty: {opt_ty:.4f}")
                    if final_cz is not None:
                        print(f"  Final centroid: {final_cz}, target: {c0}, residual: dx={final_dx:.2f}, dy={final_dy:.2f}")
                        print(f"  Final IoU: {final_iou:.6f}")
                    else:
                        print(f"  Warning: Could not compute final centroid")
            else:
                # Original iterative optimization (fallback)
                print(f"Stage 2: Optimizing x-y translation with coordinate descent. Target centroid: {c0}")
                
                # HYPERPARAMETER TUNING GUIDE:
                # - initial_step_ratio: Increase (0.02-0.05) if optimization is too slow or far from target
                #                        Decrease (0.005-0.01) if optimization overshoots or oscillates
                # - min_step_size: Increase (0.02-0.05) for faster convergence, decrease (0.005-0.01) for precision
                # - adaptive_multiplier: Increase (0.4-0.6) for larger steps when far, decrease (0.2-0.3) for stability
                # - step_reduction_factor: Increase (0.8-0.9) for slower reduction, decrease (0.5-0.7) for faster
                # - centroid_convergence_threshold: Increase (1.0-2.0) for faster convergence, decrease (0.1-0.5) for precision
                # - max_outer_iterations: Increase (100-200) if not converging, decrease (20-30) for speed
                # - max_inner_iterations: Increase (8-10) for more refinement per coordinate, decrease (3-5) for speed
                
                tx, ty = opt_tx, opt_ty
                tolerance = iou_tolerance
                min_step = min_step_size
                max_iterations = max_outer_iterations
                
                # Initial step sizes (adaptive based on image size)
                # Use larger steps for larger images, but ensure minimum step size
                step_tx = max(min_step * 10, w * initial_step_ratio)  # Default: 1% of image width
                step_ty = max(min_step * 10, h * initial_step_ratio)  # Default: 1% of image height
                
                best_tx, best_ty = tx, ty
                best_iou = 0.0
                prev_error = float('inf')
                no_improvement_count = 0
                
                # Coordinate descent: optimize x and y separately
                for iteration in range(max_iterations):
                    # Optimize x direction
                    for x_iter in range(max_inner_iterations):  # Max iterations per coordinate
                        rendered_mask, _ = render_and_get_area(tx, ty, opt_sz, opt_scale)
                        if rendered_mask is None:
                            break
                        
                        cz = centroid(rendered_mask)
                        if cz is None:
                            break
                        
                        dx = c0[0] - cz[0]  # positive → rendered is left → move tx right
                        
                        # Check convergence for x
                        if abs(dx) < centroid_convergence_threshold:  # Pixel difference threshold
                            break
                        
                        # Adaptive step: larger steps when far away, smaller when close
                        adaptive_step_x = min(step_tx, abs(dx) * adaptive_multiplier)
                        adaptive_step_x = max(min_step, adaptive_step_x)
                        
                        # Move in direction of target centroid
                        tx_new = tx + (1 if dx > 0 else -1) * adaptive_step_x
                        
                        # Test new position
                        test_mask, _ = render_and_get_area(tx_new, ty, opt_sz, opt_scale)
                        if test_mask is None:
                            break
                        
                        test_iou = iou(test_mask, target_mask)
                        current_iou = iou(rendered_mask, target_mask)
                        
                        # If improvement, accept; otherwise reduce step
                        if test_iou > current_iou:
                            tx = tx_new
                            if test_iou > best_iou:
                                best_iou = test_iou
                                best_tx, best_ty = tx_new, ty
                        else:
                            step_tx *= step_reduction_factor  # Reduce step if no improvement
                            if step_tx < min_step:
                                break
                    
                    # Optimize y direction
                    for y_iter in range(max_inner_iterations):  # Max iterations per coordinate
                        rendered_mask, _ = render_and_get_area(tx, ty, opt_sz, opt_scale)
                        if rendered_mask is None:
                            break
                        
                        cz = centroid(rendered_mask)
                        if cz is None:
                            break
                        
                        dy = c0[1] - cz[1]  # positive → rendered is up → move ty down
                        
                        # Check convergence for y
                        if abs(dy) < centroid_convergence_threshold:  # Pixel difference threshold
                            break
                        
                        # Adaptive step: larger steps when far away, smaller when close
                        adaptive_step_y = min(step_ty, abs(dy) * adaptive_multiplier)
                        adaptive_step_y = max(min_step, adaptive_step_y)
                        
                        # Move in direction of target centroid
                        ty_new = ty + (1 if dy > 0 else -1) * adaptive_step_y
                        
                        # Test new position
                        test_mask, _ = render_and_get_area(tx, ty_new, opt_sz, opt_scale)
                        if test_mask is None:
                            break
                        
                        test_iou = iou(test_mask, target_mask)
                        current_iou = iou(rendered_mask, target_mask)
                        
                        # If improvement, accept; otherwise reduce step
                        if test_iou > current_iou:
                            ty = ty_new
                            if test_iou > best_iou:
                                best_iou = test_iou
                                best_tx, best_ty = tx, ty_new
                        else:
                            step_ty *= step_reduction_factor  # Reduce step if no improvement
                            if step_ty < min_step:
                                break
                    
                    # Check overall convergence
                    rendered_mask, _ = render_and_get_area(tx, ty, opt_sz, opt_scale)
                    if rendered_mask is None:
                        break
                    
                    current_iou = iou(rendered_mask, target_mask)
                    error = 1.0 - current_iou
                    
                    # Track best solution
                    if current_iou > best_iou:
                        best_iou = current_iou
                        best_tx, best_ty = tx, ty
                        no_improvement_count = 0
                    else:
                        no_improvement_count += 1
                    
                    # Early termination conditions
                    if error < tolerance:
                        break
                    
                    # If no improvement for several iterations, use best found
                    if no_improvement_count >= 5:
                        tx, ty = best_tx, best_ty
                        break
                    
                    # If error increased, restore best and reduce steps
                    if error > prev_error:
                        step_tx *= 0.5
                        step_ty *= 0.5
                        if step_tx < min_step and step_ty < min_step:
                            tx, ty = best_tx, best_ty
                            break
                    
                    prev_error = error
                
                # Use best found solution
                opt_tx, opt_ty = best_tx, best_ty
                final_mask, _ = render_and_get_area(opt_tx, opt_ty, opt_sz, opt_scale)
                final_iou = iou(final_mask, target_mask) if final_mask is not None else 0.0
                print(f"Stage 2 complete. Optimized tx: {opt_tx:.4f}, ty: {opt_ty:.4f} "
                      f"(iterations: {iteration+1}, final IoU: {final_iou:.6f})")
    else:
        print(f"Skipping Stage 2: Rendered mask is None")

    opt_camera_translation = initial_camera_translation + np.array([opt_tx, opt_ty, opt_sz])
    
    return {
        'camera_translation': opt_camera_translation,
        'scale': opt_scale,
    }