from pathlib import Path
import torch
import os
import cv2
import numpy as np
import tempfile
from tqdm import tqdm
import torch.utils
import trimesh
import torch.utils.data
import gradio as gr
from typing import Union, List, Tuple, Dict
from amr.models import AMR
from amr.configs import get_config
from amr.utils import recursive_to
from amr.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from amr.utils.renderer import Renderer, cam_crop_to_full
from huggingface_hub import snapshot_download
from ultralytics import YOLO

import torch.nn.functional as F

from depthfm import DepthFM
import matplotlib.pyplot as plt
import json
import pickle
from datetime import datetime


LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)

# Load model config
path_model_cfg = 'data/hydra/config.yaml'
model_cfg = get_config(path_model_cfg)

# Load model
local_dir = "data/"
PATH_CHECKPOINT = os.path.join(local_dir, "checkpoint.ckpt")
model = AMR.load_from_checkpoint(checkpoint_path=PATH_CHECKPOINT, map_location="cpu",
                                 cfg=model_cfg, strict=False, weights_only=True)
depth_model = DepthFM(os.path.join(local_dir, "depthfm-v1.ckpt"))
yolo_model = YOLO("yolo11n.pt")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
depth_model = depth_model.to(device)
model.eval()
depth_model.eval()

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smal.faces)

# Make output directory if it does not exist
OUTPUT_FOLDER = "demo_out"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def resize_max_res_tensor(
    img: torch.Tensor, 
    max_edge_resolution: int, 
    mode: str = "bilinear"
) -> tuple[torch.Tensor, tuple[int, int]]:
    
    # Ensure shape (1, C, H, W)
    if img.ndim == 3:
        img = img.unsqueeze(0)
    elif img.ndim != 4:
        raise ValueError("img must have shape (C, H, W) or (1, C, H, W)")

    _, C, H, W = img.shape

    # Downscale factor
    downscale_factor = min(max_edge_resolution / W, max_edge_resolution / H)
    new_W = int(W * downscale_factor)
    new_H = int(H * downscale_factor)

    # Round to multiples of 64
    new_W = max(64, round(new_W / 64) * 64)
    new_H = max(64, round(new_H / 64) * 64)

    print(f"Resizing tensor from {W}x{H} to {new_W}x{new_H}")

    # Resize with interpolation
    resized = F.interpolate(img, size=(new_H, new_W), mode=mode, align_corners=False)

    # Return to (1, C, H, W)
    return resized, (W, H)

def predict(im):
    return im["composite"]

def inference(img: Dict)-> Tuple[Union[np.ndarray|None], List[str]]:
    orin_img = np.array(img["background"])[:, :, :-1]
    img = np.array(img["composite"])[:, :, :-1]

    # Generate unique timestamp for this inference
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # Include milliseconds
    inference_id = f"inference_{timestamp}"

    results = yolo_model(img)

    boxes = []
    if len(results) > 0 and results[0].boxes is not None:
        # Get all detected boxes
        for box in results[0].boxes:
            # Convert from xywh to xyxy format
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            boxes.append([x1, y1, x2, y2])
    
    # If no detections, fall back to full image
    if len(boxes) == 0:
        boxes = np.array([[0, 0, img.shape[1], img.shape[0]]])  # x1, y1, x2, y2
        print("No YOLO detections found, using full image")
    else:
        boxes = np.array(boxes)
        print(f"Found {len(boxes)} YOLO detections")

    # Save bounding boxes info
    boxes_info = {
        "boxes": boxes.tolist(),
        "num_detections": len(boxes),
        "image_shape": img.shape
    }

    annotated_img = img.copy()
    for box in boxes:
        x1, y1, x2, y2 = box.astype(int)
        # Draw bounding box
        cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # Add label
        cv2.putText(annotated_img, "Detected", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    print(f"img.shape: {img.shape}")
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0  # Convert to (3, H, W) and normalize to [0,1]
    img_tensor = img_tensor.to(device)

    
    depth_tensor = torch.from_numpy(orin_img).permute(2, 0, 1).float()[None] / 127.5 - 1
    depth_tensor, _ = resize_max_res_tensor(depth_tensor, max_edge_resolution=max(depth_tensor.shape))
    depth_tensor = depth_tensor.to(device)
    print(f"img.shape: {depth_tensor.shape}")

    # Run AniMer on the crop image
    dataset = ViTDetDataset(model_cfg, img, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=10)
    all_verts = []
    all_cam_t = []
    all_model_outputs = []  # Store all model outputs
    temp_name = next(tempfile._get_candidate_names())
    
    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        # Store model output
        model_output = {
            "pred_vertices": out['pred_vertices'].detach().cpu().numpy(),
            "pred_cam": out['pred_cam'].detach().cpu().numpy(),
            "batch_info": {
                "box_center": batch["box_center"].detach().cpu().numpy(),
                "box_size": batch["box_size"].detach().cpu().numpy(),
                "img_size": batch["img_size"].detach().cpu().numpy()
            }
        }
        all_model_outputs.append(model_output)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                           scaled_focal_length).detach().cpu().numpy()

        # Render the result
        input_patch = (batch['img'][0].cpu() * 255 * (DEFAULT_STD[:, None, None]) + (
                    DEFAULT_MEAN[:, None, None])) / 255.
        input_patch = input_patch.permute(1, 2, 0).numpy()

        verts = out['pred_vertices'][0].detach().cpu().numpy()
        cam_t = pred_cam_t_full[0]
        # all_verts.append(verts)
        # all_cam_t.append(cam_t)
        all_verts = [verts.detach().cpu().numpy() for verts in out['pred_vertices']]
        all_cam_t = out['pred_cam_t']
        print(f"batch['img'][0].shape: {batch['img'][0].shape}")
        regression_img = renderer([x.detach().cpu().numpy() for x in out['pred_vertices']],
                                  [x.detach().cpu().numpy() for x in out['pred_cam_t']],
                                  img_tensor,
                                  mesh_base_color=LIGHT_BLUE,
                                  scene_bg_color=(1, 1, 1),
                                  boxes=boxes,
                                )
        regression_img = (regression_img * 255).astype(np.uint8)
        print(f"regression_img.shape: {regression_img.shape}")

        with torch.autocast(device_type="cuda"):
            depth = depth_model.predict_depth(depth_tensor, num_steps=2, ensemble_size=4)
        depth = depth.squeeze(0).squeeze(0).cpu().numpy() 
        depth_colored = plt.get_cmap('magma')(depth, bytes=True)[..., :3]

        # Save all inference data
        inference_data = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "model_outputs": all_model_outputs,
            "boxes_info": boxes_info,
            "depth_output": {
                "depth_map": depth,  # Raw depth values
                "depth_colored": depth_colored,  # Colored depth for visualization
                "depth_tensor_shape": depth_tensor.shape
            },
            "final_vertices": all_verts,
            "final_cam_translations": all_cam_t,
            "image_info": {
                "original_shape": orin_img.shape,
                "composite_shape": img.shape
            }
        }

        # Create inference-specific directory
        inference_dir = os.path.join(OUTPUT_FOLDER, inference_id)
        os.makedirs(inference_dir, exist_ok=True)

        # Save as pickle for complete data preservation
        with open(os.path.join(inference_dir, "inference_data.pkl"), "wb") as f:
            pickle.dump(inference_data, f)

        # Save as JSON for human-readable format (excluding large arrays)
        json_data = {
            "inference_id": inference_id,
            "timestamp": timestamp,
            "boxes_info": boxes_info,
            "image_info": inference_data["image_info"],
            "num_model_outputs": len(all_model_outputs),
            "num_vertices": len(all_verts),
            "depth_shape": depth.shape
        }
        with open(os.path.join(inference_dir, "inference_summary.json"), "w") as f:
            json.dump(json_data, f, indent=2)

        # Save individual components
        np.save(os.path.join(inference_dir, "depth_map.npy"), depth)
        np.save(os.path.join(inference_dir, "boxes.npy"), boxes)
        np.save(os.path.join(inference_dir, "vertices.npy"), np.array(all_verts))
        np.save(os.path.join(inference_dir, "cam_translations.npy"), np.array(all_cam_t))

        print(f"Saved inference data to: {inference_dir}")

        if len(all_verts):
            # Return mesh path
            trimeshes = [renderer.vertices_to_trimesh(vvv, ttt.copy(), LIGHT_BLUE) for vvv,ttt in zip(all_verts, all_cam_t)]
            # Join meshes
            mesh = trimesh.util.concatenate(trimeshes)
            # Save mesh to file
            mesh_name = os.path.join(inference_dir, "mesh.obj")
            trimesh.exchange.export.export_mesh(mesh, mesh_name)

            return ([regression_img, annotated_img], mesh_name, depth_colored)
        else:
            return (None, [], None)

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Interface(
        fn=inference,
        analytics_enabled=False,
        inputs=gr.ImageEditor(
            sources=("upload", "clipboard"),
            brush=False,
            eraser=False,
            # crop_size="1:1",
            layers=False,
            placeholder="Upload an image or select from the examples.",
        ),
        outputs=[
            gr.Gallery(label="Output images"),
            gr.Model3D(display_mode="wireframe", label="3D Mesh"),
            gr.Image(label="Depth image"),
        ],
        title="Watties: 3D Quadruped Animal Pose and Shape Estimation",
        description="""
        Project page: https://github.com/phangiaanh  

        Author: pganh.sdh221

        ## Usage
        1. **Input**: Select an example image or upload your own.
        2. **Output**:
        - Animal detection and bounding box on the original image
        - Multiple 2D mesh overlays on the original image
        - Interactive 3D model visualization
        - Depth estimation visualization 

        The demo is for academic purposes only.

        """,

        examples=[
            'example_data/cow.jpg',
            'example_data/dog.jpg',
            'example_data/hippo.jpg',
            'example_data/horse.jpg',
            'example_data/tiger.jpg',
        ],
    )

demo.launch(share=True)