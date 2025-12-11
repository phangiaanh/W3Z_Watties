import numpy as np
from typing import Tuple, List, Dict
import os
import pyrender
import pickle
import trimesh
import torch
from scipy.optimize import minimize
from scipy.ndimage import affine_transform
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from amr.configs import get_config
from amr.utils.renderer import Renderer
from utils import optimize_camera_translation
import cv2

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

renderer = setup_renderer()

LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)



def optimize_meshes(vertices: np.ndarray, camera_translations: np.ndarray, image_shape: Tuple[int, int], boxes: List[List[int]], valid_mask: np.ndarray, median_depths: List[float]):
    """
    Optimize the vertices and camera translations to minimize the error between the projected vertices and the valid mask.
    Args:
        vertices: np.ndarray, shape (N, V, 3)
        camera_translations: np.ndarray, shape (N, 3)
        image_shape: Tuple[int, int] - (height, width)
        boxes: List[List[int]]
        valid_mask: np.ndarray, shape (I, H, W, 1)
    Returns:
        vertices: np.ndarray, shape (N, V, 3)
        camera: Intrinsic camera matrix, shape (I, 3, 3)
    """
    trimeshes = [
        renderer.vertices_to_trimesh(vvv, ttt.copy(), LIGHT_BLUE) 
        for vvv, ttt in zip(vertices, camera_translations)
    ]

    penalties = []
    final_meshes = []
    final_camera_translations = []

    for i in trimeshes:
        penalties.append(i.centroid)
    img_tensor = create_image_tensor(image_shape)

    # Use this as anchor
    best_index = np.argmin(median_depths)
    best_vertices = vertices[best_index]
    best_camera_translations = camera_translations[best_index]


    print(f"Current index: {best_index}")
    optimized_results = optimize_camera_translation(
        vertices=shift_vertices_by_translation(best_vertices, best_camera_translations, scale=1),
        initial_camera_translation=[0,0,0],
        target_mask=valid_mask[best_index],
        renderer=renderer,
        img_tensor=img_tensor,
        boxes=[[0, 0, image_shape[1], image_shape[0]]],
        initial_offset=(0.0, 0.0, 0.0),
    )

    cv2.imwrite(f"output/mask_{best_index}.png", valid_mask[best_index] * 255)

    rendered_mask, _ = render_meshes(
        renderer,
        np.array([shift_vertices_by_translation(best_vertices, best_camera_translations + optimized_results['camera_translation'], scale=optimized_results['scale'])]),
        np.array([[0, 0, 0]]),
        img_tensor,
        np.array([[0, 0, image_shape[1], image_shape[0]]])
    )
    cv2.imwrite(f"output/rendered_mask_before_{best_index}.png", rendered_mask)
    
    anchor = best_camera_translations + optimized_results['camera_translation']
    final_meshes.append(shift_vertices_by_translation(best_vertices, anchor))
    final_camera_translations.append(camera_translations[best_index])
    rendered_mask, _ = render_meshes(
        renderer,
        np.array([final_meshes[0]]),
        np.array([0, 0, 0]),
        img_tensor,
        np.array([[0, 0, image_shape[1], image_shape[0]]])
    )
    cv2.imwrite(f"output/rendered_mask_after_{best_index}.png", rendered_mask)
    print(f"Optimized camera translation and scale: {optimized_results['camera_translation']}, {optimized_results['scale']}")

    for index, value in enumerate(vertices):
        if index == best_index:
            continue

        print(f"Current index: {index}")
        item_scale = median_depths[index] * anchor[2] / median_depths[best_index]
        cam_translation = camera_translations[index]

        optimized_results = optimize_camera_translation(
            vertices=shift_vertices_by_translation(value, cam_translation, scale=1),
            initial_camera_translation=[0,0,item_scale],
            target_mask=valid_mask[index],
            renderer=renderer,
            img_tensor=img_tensor,
            boxes=[[0, 0, image_shape[1], image_shape[0]]],
            initial_offset=(0.0, 0.0, 0.0),
            not_anchor=True
        )
        optimized_camera_translation = optimized_results['camera_translation']
        optimized_scale = optimized_results['scale']
        # optimized_camera_translation = [item_scale * optimized_results['camera_translation'][0], optimized_results['camera_translation'][1], optimized_results['camera_translation'][2]]
        print(f"Optimized camera translation ans scale for index {index}: {optimized_results['camera_translation']}, {optimized_results['scale']}")
        final_meshes.append(shift_vertices_by_translation(value, anchor + optimized_camera_translation, scale=optimized_scale))
        final_camera_translations.append(camera_translations[index])
        rendered_mask, _ = render_meshes(
            renderer,
            np.array([shift_vertices_by_translation(value, optimized_camera_translation, scale=optimized_scale)]),
            np.array([0, 0, 0]),
            img_tensor,
            np.array([[0, 0, image_shape[1], image_shape[0]]])
        )
        cv2.imwrite(f"output/mask_{index}.png", valid_mask[index] * 255)
        cv2.imwrite(f"output/rendered_mask_after_{index}.png", rendered_mask)

    save_mesh_obj(renderer, final_meshes, final_camera_translations, "mesh_optimized.obj")

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

def compute_bbox(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0: return None
    return np.min(xs), np.max(xs), np.min(ys), np.max(ys)

def centroid(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return np.mean(xs), np.mean(ys)

def iou(mask1, mask2):
    inter = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return inter / union if union > 0 else 0.0



def create_image_tensor(image_shape: Tuple[int, int]):
    """Create a white background image tensor with full image shape."""
    height, width = image_shape
    # Create white image (1.0, 1.0, 1.0) in RGB
    img = np.ones((height, width, 3), dtype=np.float32)
    # Convert to torch tensor (3, H, W) format
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float()
    return img_tensor

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

def optimize_camera_translation1(vertices: np.ndarray, 
                                initial_camera_translation: np.ndarray,
                                target_mask: np.ndarray,
                                renderer: Renderer,
                                img_tensor: torch.Tensor,
                                boxes: List[List[int]],
                                initial_offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
                                method: str = 'L-BFGS-B',
                                not_anchor: bool = False,
                                size_ratio_threshold: float = 1.01) -> Dict:
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
        # print(f"Rendering with tx: {tx}, ty: {ty}, sz: {sz}")
        # print(f"Initial camera translation: {initial_camera_translation}")
        
        # if not_anchor:
        #     # Keep z unchanged, only apply tx and ty to camera translation
        #     new_camera_translation = initial_camera_translation + np.array([tx, ty, 0])
        #     # Use sz as the scale parameter for shift_vertices_by_translation
        #     scale = 1.0 + sz * 0.1
        #     transformed_vertices = shift_vertices_by_translation(vertices_list[0], new_camera_translation, scale=scale)
        # else:
        #     new_camera_translation = initial_camera_translation + np.array([tx, ty, sz])
        #     transformed_vertices = shift_vertices_by_translation(vertices_list[0], new_camera_translation)
        
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
        print(f"Stage 1: Optimizing z (depth). Size ratio: {size_ratio:.4f}")
        z = opt_scale if not_anchor else opt_sz
        step = 1
        min_step = 0.01
        prev_error = None
        prev_sign = None

        for _ in range(100):
            rendered_mask, rendered_area = render_and_get_area(opt_tx, opt_ty, opt_sz, z) if not_anchor else render_and_get_area(opt_tx, opt_ty, z, opt_scale)
            # print(f"Rendering with z: {z}, rendered_area: {rendered_area}, target_mask_area: {target_mask_area}")
            if rendered_mask is None:
                break
            current_ratio = rendered_area / target_mask_area if target_mask_area > 0 else 1.0
            error = current_ratio - 1.0
            sign = 1 if error > 0 else -1
            if prev_sign is not None and sign != prev_sign:
                step *= 0.5                        # shrink step
                # print(" ** Toggle detected → reducing step to", step)
            prev_error = error
            prev_sign = sign
            if abs(error) < 1e-5:
                break
            if sign > 0:
                z += step * (-1 if not_anchor else 1)
            else:
                z -= step * (-1 if not_anchor else 1)
            if step < min_step:
                step = min_step

            prev_sign = sign
            prev_error = error
        if not_anchor:
            opt_scale = z
        else:
            opt_sz = z
        print(f"Stage 1 complete. Optimized sz and scale: {opt_sz:.4f}, {opt_scale:.4f}")
    
    # STAGE 2: Optimize x-y once sizes are approximately equal
    # Check current size ratio with optimized z
    current_mask, current_area = render_and_get_area(opt_tx, opt_ty, opt_sz, opt_scale) if not_anchor else render_and_get_area(opt_tx, opt_ty, opt_sz, opt_scale)
    if current_mask is not None:
        tx, ty = opt_tx, opt_ty
        prev_error = None
        step_tx, step_ty = 0.2, 0.2
        min_step = 0.01
        alpha = 0.2
        prev_dir_x = None
        prev_dir_y = None
        for _ in range(100):
            rendered_mask, rendered_area = render_and_get_area(tx, ty, opt_sz, opt_scale) if not_anchor else render_and_get_area(tx, ty, opt_sz, opt_scale)
            current_iou = iou(rendered_mask, target_mask)
            error = 1.0 - current_iou
            # print(f"Rendering with tx: {tx}, ty: {ty}, opt_sz: {opt_sz}, error: {error}")

            if error < 1e-5:
                break

            c0 = centroid(target_mask)
            cz = centroid(rendered_mask)
            # print(f"Centroid of target mask: {c0}, centroid of rendered mask: {cz}")
            if c0 is None or cz is None:
                break

            dx = c0[0] - cz[0]   # positive → maskz is left → move tx right
            dy = c0[1] - cz[1]   # positive → maskz is up → move ty down

            dir_x = 1 if dx > 0 else -1
            dir_y = 1 if dy > 0 else -1

            # --- Oscillation detection ---
            if prev_dir_x is not None and dir_x != prev_dir_x:
                step_tx = max(min_step, step_tx * 0.5)
            if prev_dir_y is not None and dir_y != prev_dir_y:
                step_ty = max(min_step, step_ty * 0.5)

            prev_dir_x = dir_x
            prev_dir_y = dir_y

            # --- Apply step movement ---
            tx += dir_x * step_tx
            ty += dir_y * step_ty

            # # --- Adaptive step sizes ---
            # if prev_error is not None:
            #     # If error increases → step too big → shrink
            #     if error > prev_error:
            #         step_x = max(min_step, step_x * 0.5)
            #         step_y = max(min_step, step_y * 0.5)

            #     # If error decreases strongly → can use larger steps
            #     elif abs(prev_error - error) > 0.05:
            #         step_x *= 1.1
            #         step_y *= 1.1

            # prev_error = error


        # if current_ratio <= size_ratio_threshold:
        #     print(f"Stage 2: Optimizing x-y (translation). Size ratio: {current_ratio:.4f}")
            
        #     # Loss function for x-y optimization (z is fixed)
        #     def loss_function_xy(params):
        #         tx, ty = params
        #         sz = opt_sz  # Fix z from stage 1
                
        #         rendered_mask, _ = render_and_get_area(tx, ty, sz)
                
        #         if rendered_mask is None:
        #             return 1e6
                
        #         # Focus on IoU for position alignment
        #         intersection = np.sum(rendered_mask * target_mask)
        #         union = np.sum((rendered_mask + target_mask) > 0)
        #         iou = intersection / union if union > 0 else 0
                
        #         loss = -iou  # Negative IoU (minimize this = maximize IoU)
                
        #         # Small penalty for large x-y offsets
        #         loss += 0.001 * (tx**2 + ty**2)
                
        #         return loss
            
        #     # Optimize x-y
        #     result_xy = minimize(
        #         loss_function_xy,
        #         [opt_tx, opt_ty],
        #         method=method,
        #         bounds=[(-w, w), (-h, h)],
        #         options={'maxiter': 30, 'disp': False}
        #     )
            
        #     opt_tx, opt_ty = result_xy.x
        opt_tx, opt_ty = tx, ty 
        print(f"Stage 2 complete. Optimized tx: {opt_tx:.4f}, ty: {opt_ty:.4f}")
    else:
        print(f"Skipping Stage 2: Rendered mask is None")

    opt_camera_translation = initial_camera_translation + np.array([opt_tx, opt_ty, opt_sz])
    # STAGE 3: Final optimization: refine all parameters together
    # print("Stage 3: Final refinement of all parameters")
    
    # def loss_function_final(params):
    #     tx, ty, sz = params
    #     rendered_mask, _ = render_and_get_area(tx, ty, sz)
        
    #     if rendered_mask is None:
    #         return 1e6
        
    #     intersection = np.sum(rendered_mask * target_mask)
    #     union = np.sum((rendered_mask + target_mask) > 0)
    #     iou = intersection / union if union > 0 else 0
        
    #     loss = -iou
    #     loss += 0.001 * (tx**2 + ty**2 + sz**2)
        
    #     return loss
    
    # result = minimize(
    #     loss_function_final,
    #     [opt_tx, opt_ty, opt_sz],
    #     method=method,
    #     bounds=[(-w, w), (-h, h), (-10.0, 10.0)],
    #     options={'maxiter': 20, 'disp': True}
    # )
    
    # # Extract optimized parameters
    # opt_tx, opt_ty, opt_sz = result.x
    # opt_offset = (opt_tx, opt_ty, opt_sz)
    # opt_camera_translation = initial_camera_translation + np.array(opt_offset)
    
    # # Get final transformed vertices
    # if not_anchor:
    #     # Scale vertices around their center
    #     scale = 1.0 + opt_sz * 0.1  # Same mapping as in loss function
    #     scaled_vertices = (vertices_list[0] - vertices_center) * scale + vertices_center
    #     # Then shift by camera translation
    #     final_vertices = shift_vertices_by_translation(scaled_vertices, opt_camera_translation)
    # else:
    #     # Shift vertices by camera translation
    #     final_vertices = shift_vertices_by_translation(vertices_list[0], opt_camera_translation)
    
    # # Get final rendered mask
    # _, final_rendered_mask = render_meshes(
    #     renderer,
    #     np.array([final_vertices]),
    #     np.array([[0, 0, 0]]),
    #     img_tensor,
    #     np.array([boxes_list[0]])
    # )
    
    # # Process final mask
    # final_rendered_mask = np.array(final_rendered_mask)
    # if final_rendered_mask.ndim == 4:
    #     final_rendered_mask = final_rendered_mask[0]
    # if final_rendered_mask.ndim == 3:
    #     final_rendered_mask = final_rendered_mask.squeeze()
    # final_rendered_mask = (final_rendered_mask > 0).astype(np.float32)
    
    # # Ensure same shape
    # if final_rendered_mask.shape != target_mask.shape:
    #     from scipy.ndimage import zoom
    #     scale_h = target_mask.shape[0] / final_rendered_mask.shape[0]
    #     scale_w = target_mask.shape[1] / final_rendered_mask.shape[1]
    #     final_rendered_mask = zoom(final_rendered_mask, (scale_h, scale_w), order=1)
    #     final_rendered_mask = (final_rendered_mask > 0.5).astype(np.float32)
    
    # # Compute final metrics
    # intersection = np.sum(final_rendered_mask * target_mask)
    # union = np.sum((final_rendered_mask + target_mask) > 0)
    # final_iou = intersection / union if union > 0 else 0
    # pixel_diff = np.sum(final_rendered_mask != target_mask)
    
    # print(f"\nOptimization Results:")
    # if not_anchor:
    #     scale = 1.0 + opt_sz * 0.1
    #     print(f"  Optimized offset: ({opt_tx:.4f}, {opt_ty:.4f}, {opt_sz:.4f})")
    #     print(f"  Vertex scale factor: {scale:.4f}")
    #     print(f"  Optimized camera translation: {opt_camera_translation}")
    # else:
    #     print(f"  Optimized offset: ({opt_tx:.4f}, {opt_ty:.4f}, {opt_sz:.4f})")
    #     print(f"  Optimized camera translation: {opt_camera_translation}")
    # print(f"  Final IoU: {final_iou:.4f}")
    # print(f"  Pixel difference: {pixel_diff} pixels")
    # print(f"  Success: {result.success}")
    
    return {
        'camera_translation': opt_camera_translation,
        'scale': opt_scale,
        # 'offset': opt_offset,
        # 'min_difference': 1.0 - final_iou,
        # 'iou': final_iou,
        # 'pixel_diff': pixel_diff,
        # 'optimized_mask': final_rendered_mask,
        # 'optimization_success': result.success,
        # 'optimization_message': result.message
    }

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
    # valid_mask = (valid_mask * 255).astype(np.uint8)
    
    return regression_img, valid_mask

def test_translation_linearity(vertices: np.ndarray,
                              initial_camera_translation: np.ndarray,
                              renderer,
                              img_tensor: torch.Tensor,
                              boxes: List[List[int]],
                              opt_sz: float = 0.0,
                              opt_scale: float = 1.0,
                              delta_range: float = 5.0,
                              num_samples: int = 20):
    """
    Test if there's a linear relationship between delta_tx/delta_ty and 
    the resulting centroid shift in the rendered mask.
    
    Args:
        vertices: Mesh vertices
        initial_camera_translation: Base camera translation
        renderer: Renderer instance
        img_tensor: Image tensor
        boxes: Bounding boxes
        opt_sz: Current optimized z translation
        opt_scale: Current optimized scale
        delta_range: Range of delta values to test (e.g., ±5.0)
        num_samples: Number of samples to test
    
    Returns:
        dict: Linearity analysis results including correlation, R², and linear fit
    """
    from scipy.ndimage import zoom
    
    # Helper function to render and get centroid
    def render_and_get_centroid(tx, ty, tz, ts):
        if isinstance(tz, np.ndarray):
            tz = tz[0] if tz.size > 0 else tz.item()
        
        new_camera_translation = initial_camera_translation + np.array([tx, ty, tz])
        transformed_vertices = shift_vertices_by_translation(
            vertices, initial_camera_translation + np.array([tx, ty, tz]), scale=ts
        )
        
        try:
            _, rendered_mask = render_meshes(
                renderer,
                np.array([transformed_vertices]),
                np.array([[0, 0, 0]]),
                img_tensor,
                np.array([boxes[0]])
            )
            
            rendered_mask = np.array(rendered_mask)
            if rendered_mask.ndim == 4:
                rendered_mask = rendered_mask[0]
            if rendered_mask.ndim == 3:
                rendered_mask = rendered_mask.squeeze()
            rendered_mask = (rendered_mask > 0).astype(np.float32)
            
            # Get target mask shape from first render
            if not hasattr(render_and_get_centroid, 'target_shape'):
                render_and_get_centroid.target_shape = rendered_mask.shape
            
            if rendered_mask.shape != render_and_get_centroid.target_shape:
                scale_h = render_and_get_centroid.target_shape[0] / rendered_mask.shape[0]
                scale_w = render_and_get_centroid.target_shape[1] / rendered_mask.shape[1]
                rendered_mask = zoom(rendered_mask, (scale_h, scale_w), order=1)
                rendered_mask = (rendered_mask > 0.5).astype(np.float32)
            
            c = centroid(rendered_mask)
            return c
        except Exception as e:
            print(f"Warning: Rendering failed: {e}")
            return None
    
    # Get baseline centroid
    print("Rendering baseline to get initial centroid...")
    baseline_centroid = render_and_get_centroid(0.0, 0.0, opt_sz, opt_scale)
    if baseline_centroid is None:
        return {"error": "Failed to render baseline"}
    
    print(f"Baseline centroid: {baseline_centroid}")
    
    # Test delta_tx
    print(f"\nTesting delta_tx linearity (range: ±{delta_range}, samples: {num_samples})...")
    delta_tx_values = np.linspace(-delta_range, delta_range, num_samples)
    centroid_x_shifts = []
    valid_deltas_tx = []
    
    for i, delta_tx in enumerate(delta_tx_values):
        print(f"  Testing delta_tx={delta_tx:.2f} ({i+1}/{num_samples})", end='\r')
        c = render_and_get_centroid(delta_tx, 0.0, opt_sz, opt_scale)
        if c is not None:
            centroid_x_shift = c[0] - baseline_centroid[0]
            centroid_x_shifts.append(centroid_x_shift)
            valid_deltas_tx.append(delta_tx)
    print()  # New line after progress
    
    valid_deltas_tx = np.array(valid_deltas_tx)
    centroid_x_shifts = np.array(centroid_x_shifts)
    
    # Test delta_ty
    print(f"\nTesting delta_ty linearity (range: ±{delta_range}, samples: {num_samples})...")
    delta_ty_values = np.linspace(-delta_range, delta_range, num_samples)
    centroid_y_shifts = []
    valid_deltas_ty = []
    
    for i, delta_ty in enumerate(delta_ty_values):
        print(f"  Testing delta_ty={delta_ty:.2f} ({i+1}/{num_samples})", end='\r')
        c = render_and_get_centroid(0.0, delta_ty, opt_sz, opt_scale)
        if c is not None:
            centroid_y_shift = c[1] - baseline_centroid[1]
            centroid_y_shifts.append(centroid_y_shift)
            valid_deltas_ty.append(delta_ty)
    print()  # New line after progress
    
    valid_deltas_ty = np.array(valid_deltas_ty)
    centroid_y_shifts = np.array(centroid_y_shifts)
    
    # Analyze linearity for delta_tx -> centroid_x_shift
    results = {}
    
    if len(valid_deltas_tx) > 2:
        # Pearson correlation
        corr_x, p_value_x = pearsonr(valid_deltas_tx, centroid_x_shifts)
        
        # Linear regression
        reg_x = LinearRegression()
        reg_x.fit(valid_deltas_tx.reshape(-1, 1), centroid_x_shifts)
        r2_x = r2_score(centroid_x_shifts, reg_x.predict(valid_deltas_tx.reshape(-1, 1)))
        slope_x = reg_x.coef_[0]
        intercept_x = reg_x.intercept_
        
        results['delta_tx'] = {
            'correlation': float(corr_x),
            'p_value': float(p_value_x),
            'r2': float(r2_x),
            'slope': float(slope_x),
            'intercept': float(intercept_x),
            'num_samples': len(valid_deltas_tx),
            'is_linear': abs(corr_x) > 0.95 and r2_x > 0.90,  # Threshold for "linear"
            'delta_values': valid_deltas_tx.tolist(),
            'centroid_shifts': centroid_x_shifts.tolist()
        }
        
        print(f"\n{'='*60}")
        print(f"delta_tx -> centroid_x_shift Analysis:")
        print(f"{'='*60}")
        print(f"  Correlation: {corr_x:.4f} (p={p_value_x:.4e})")
        print(f"  R²: {r2_x:.4f}")
        print(f"  Slope: {slope_x:.4f} pixels per unit delta_tx")
        print(f"  Intercept: {intercept_x:.4f}")
        print(f"  Is Linear: {results['delta_tx']['is_linear']}")
        print(f"  Valid samples: {len(valid_deltas_tx)}/{num_samples}")
    
    # Analyze linearity for delta_ty -> centroid_y_shift
    if len(valid_deltas_ty) > 2:
        # Pearson correlation
        corr_y, p_value_y = pearsonr(valid_deltas_ty, centroid_y_shifts)
        
        # Linear regression
        reg_y = LinearRegression()
        reg_y.fit(valid_deltas_ty.reshape(-1, 1), centroid_y_shifts)
        r2_y = r2_score(centroid_y_shifts, reg_y.predict(valid_deltas_ty.reshape(-1, 1)))
        slope_y = reg_y.coef_[0]
        intercept_y = reg_y.intercept_
        
        results['delta_ty'] = {
            'correlation': float(corr_y),
            'p_value': float(p_value_y),
            'r2': float(r2_y),
            'slope': float(slope_y),
            'intercept': float(intercept_y),
            'num_samples': len(valid_deltas_ty),
            'is_linear': abs(corr_y) > 0.95 and r2_y > 0.90,  # Threshold for "linear"
            'delta_values': valid_deltas_ty.tolist(),
            'centroid_shifts': centroid_y_shifts.tolist()
        }
        
        print(f"\n{'='*60}")
        print(f"delta_ty -> centroid_y_shift Analysis:")
        print(f"{'='*60}")
        print(f"  Correlation: {corr_y:.4f} (p={p_value_y:.4e})")
        print(f"  R²: {r2_y:.4f}")
        print(f"  Slope: {slope_y:.4f} pixels per unit delta_ty")
        print(f"  Intercept: {intercept_y:.4f}")
        print(f"  Is Linear: {results['delta_ty']['is_linear']}")
        print(f"  Valid samples: {len(valid_deltas_ty)}/{num_samples}")
    
    results['baseline_centroid'] = baseline_centroid
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    if 'delta_tx' in results:
        print(f"  delta_tx is {'LINEAR' if results['delta_tx']['is_linear'] else 'NON-LINEAR'}")
        if results['delta_tx']['is_linear']:
            print(f"    → Can use slope {results['delta_tx']['slope']:.4f} to estimate delta_tx from centroid_x shift")
    if 'delta_ty' in results:
        print(f"  delta_ty is {'LINEAR' if results['delta_ty']['is_linear'] else 'NON-LINEAR'}")
        if results['delta_ty']['is_linear']:
            print(f"    → Can use slope {results['delta_ty']['slope']:.4f} to estimate delta_ty from centroid_y shift")
    
    return results

def main(vertices: np.ndarray, camera_translations: np.ndarray, image_shape: Tuple[int, int], boxes: List[List[int]], valid_mask: np.ndarray, median_depths: List[float]):
    """
    Test linearity between delta_tx/delta_ty and rendered mask centroid shifts.
    """
    renderer = setup_renderer()
    
    # Choose the best index (same logic as before)
    best_index = np.argmin(median_depths)
    print(f"Testing linearity for mesh index: {best_index}")
    
    # Get best vertices and camera translation
    best_vertices = vertices[best_index]
    best_camera_translations = camera_translations[best_index]
    
    # Create image tensor with full image shape (3, H, W)
    img_tensor = create_image_tensor(image_shape)
    
    # First, run a quick optimization to get reasonable opt_sz and opt_scale
    print("Running quick optimization to get initial parameters...")
    optimized_results = optimize_camera_translation(
        vertices=shift_vertices_by_translation(best_vertices, best_camera_translations, scale=1),
        initial_camera_translation=[0, 0, 0],
        target_mask=valid_mask[best_index],
        renderer=renderer,
        img_tensor=img_tensor,
        boxes=[[0, 0, image_shape[1], image_shape[0]]],
        initial_offset=(0.0, 0.0, 0.0),
    )
    
    opt_sz = optimized_results['camera_translation'][2]
    opt_scale = optimized_results['scale']
    
    print(f"\nUsing optimized parameters: opt_sz={opt_sz:.4f}, opt_scale={opt_scale:.4f}")
    
    # Now test linearity
    print("\n" + "="*60)
    print("TESTING TRANSLATION LINEARITY")
    print("="*60)
    
    linearity_results = test_translation_linearity(
        vertices=shift_vertices_by_translation(best_vertices, best_camera_translations, scale=1),
        initial_camera_translation=[0, 0, 0],
        renderer=renderer,
        img_tensor=img_tensor,
        boxes=[[0, 0, image_shape[1], image_shape[0]]],
        opt_sz=opt_sz,
        opt_scale=opt_scale,
        delta_range=5.0,  # Test ±5.0 units
        num_samples=20   # 20 samples per direction
    )
    
    # Save results
    import json
    output_file = os.path.join("output", "linearity_test_results.json")
    os.makedirs("output", exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(linearity_results, f, indent=2)
    print(f"\nResults saved to: {output_file}")
    
    return linearity_results

if __name__ == "__main__":
    # directory = "/home/watermelon/demo_out/horseanddog"

    directory = "/home/watermelon/demo_out/dogs"
    # directory = "/home/watermelon/demo_out/lions"
    # directory = "/home/watermelon/Downloads/inference_20251120_220046_539"
    # directory = "/home/watermelon/Downloads/inference_20251120_220009_593"
    # directory = "/home/watermelon/Downloads/inference_20251123_051111_548"
    # directory = "/home/watermelon/Downloads/inference_20251123_052035_926"
    vertices = np.load(os.path.join(directory, "vertices.npy"))
    camera_translations = np.load(os.path.join(directory, "cam_translations.npy"))
    boxes = np.load(os.path.join(directory, "boxes.npy"))
    valid_mask = np.load(os.path.join(directory, "valid_mask.npy"))
    depth = np.load(os.path.join(directory, "depth_map.npy"))
    # Binarize valid_mask for each layer (threshold > 0)
    valid_mask = (valid_mask > 0).astype(valid_mask.dtype)
    # After loading depth and valid_mask
    median_depths = extract_mesh_depths(depth, valid_mask)
    print(f"Median depths for each mesh: {median_depths}")
    save_mesh_obj(renderer, vertices, camera_translations, os.path.join(directory, "mesh.obj"))
    # main(vertices, camera_translations, valid_mask.shape[1:3], boxes, valid_mask, median_depths)
    optimize_meshes(vertices, camera_translations, valid_mask.shape[1:3], boxes, valid_mask, median_depths)