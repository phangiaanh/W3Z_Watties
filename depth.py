
import gradio as gr
import torch
import os
import cv2
import numpy as np
from typing import Union, List, Tuple, Dict
import torch.nn.functional as F


import einops
from PIL import Image
from PIL.Image import Resampling


from depthfm import DepthFM
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

local_dir = "data/"
depth_model = DepthFM(os.path.join(local_dir, "depthfm-v1.ckpt"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
depth_model = depth_model.to(device)
depth_model.eval()

def get_dtype_from_str(dtype_str):
    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype_str]

def resize_max_res_tensor(
    img: torch.Tensor, 
    max_edge_resolution: int, 
    mode: str = "bilinear"
) -> tuple[torch.Tensor, tuple[int, int]]:
    """
    Resize tensor image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`torch.Tensor`):
            Image tensor with shape (C, H, W) or (1, C, H, W).
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        mode (`str`):
            Interpolation method (bilinear, bicubic, nearest...).

    Returns:
        (`torch.Tensor`, `(int, int)`):
            Resized tensor with shape (C, H_new, W_new), 
            and original resolution (W, H).
    """
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

def resize_max_res(
    img: Image.Image, max_edge_resolution: int, resample_method=Resampling.BILINEAR
) -> Image.Image:
    """
    Resize image to limit maximum edge length while keeping aspect ratio.

    Args:
        img (`Image.Image`):
            Image to be resized.
        max_edge_resolution (`int`):
            Maximum edge length (pixel).
        resample_method (`PIL.Image.Resampling`):
            Resampling method used to resize images.

    Returns:
        `Image.Image`: Resized image.
    """
    original_width, original_height = img.size
    downscale_factor = min( max_edge_resolution / original_width, max_edge_resolution / original_height)

    new_width  = int(original_width * downscale_factor)
    new_height = int(original_height * downscale_factor)

    new_width  = round(new_width / 64) * 64
    new_height = round(new_height / 64) * 64

    print(f"Resizing image from {original_width}x{original_height} to {new_width}x{new_height}")

    resized_img = img.resize((new_width, new_height), resample=resample_method)
    resized_img = resized_img.unsqueeze(0)             # (1, C, H, W)
    return resized_img, (original_width, original_height)

def load_im(fp, processing_res=-1):
    assert os.path.exists(fp), f"File not found: {fp}"
    im = Image.open(fp).convert('RGB')
    if processing_res < 0:
        processing_res = max(im.size)
    #im, orig_res = resize_max_res(im, processing_res)
    x = np.array(im)
    x = einops.rearrange(x, 'h w c -> c h w')
    x = x / 127.5 - 1
    x = torch.tensor(x, dtype=torch.float32)[None]
    return x, None


def inference(img: Dict)-> Tuple[Union[np.ndarray|None], List[str]]:
    orin_img = np.array(img["background"])[:, :, :-1]
    
    depth_tensor = torch.from_numpy(orin_img).permute(2, 0, 1).float()[None] / 127.5 - 1
    depth_tensor, _ = resize_max_res_tensor(depth_tensor, max_edge_resolution=max(depth_tensor.shape))
    depth_tensor = depth_tensor.to(device)
    print(f"img.shape: {depth_tensor.shape}")

    '''
    im, orig_res = load_im("example_data/hippo.jpg", -1)
    im = im.to(device)

    # Run AniMer on the crop image
    dtype = get_dtype_from_str("fp32")
    depth_model.model.dtype = dtype
    '''
    with torch.autocast(device_type="cuda", dtype=torch.float32):
        depth = depth_model.predict_depth(depth_tensor, num_steps=2, ensemble_size=4)
    depth = depth.squeeze(0).squeeze(0).cpu().numpy() 
    depth = plt.get_cmap('magma')(depth, bytes=True)[..., :3]

        

    return (depth)


demo = gr.Interface(
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
        gr.Image(label="Depth image"),
    ],
    title="Watties: 3D Quadruped Animal Pose and Shape Estimation",
    description="""
    Project page: https://github.com/phangiaanh
    Author: pganh.sdh221

    ## Usage
    1. **Input**: Select an example image or upload your own.
    2. **Processing**: Crop the image to a square.
    3. **Output**:
    - 2D mesh overlay on the original image
    - Interactive 3D model visualization
    
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