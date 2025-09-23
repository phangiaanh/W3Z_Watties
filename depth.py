
import gradio as gr
import torch
import os
import cv2
import numpy as np
from typing import Union, List, Tuple, Dict

from depthfm import DepthFM
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

local_dir = "data/"
depth_model = DepthFM(os.path.join(local_dir, "depthfm-v1.ckpt"))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
depth_model = depth_model.to(device)
depth_model.eval()


def inference(img: Dict)-> Tuple[Union[np.ndarray|None], List[str]]:
    orin_img = np.array(img["background"])[:, :, :-1]
    
    depth_tensor = torch.from_numpy(orin_img).permute(2, 0, 1).float()[None] / 127.5 - 1
    depth_tensor = depth_tensor.to(device)
    print(f"img.shape: {depth_tensor.shape}")

    # Run AniMer on the crop image

    with torch.autocast(device_type="cuda"):
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