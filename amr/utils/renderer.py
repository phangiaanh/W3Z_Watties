import os

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
import numpy as np
import pyrender
import trimesh
import cv2
from yacs.config import CfgNode
from typing import List, Optional, Union


def cam_crop_to_full(cam_bbox, box_center, box_size, img_size, focal_length=5000.):
    # Convert cam_bbox to full image
    img_w, img_h = img_size[:, 0], img_size[:, 1]
    cx, cy, b = box_center[:, 0], box_center[:, 1], box_size
    w_2, h_2 = img_w / 2., img_h / 2.
    bs = b * cam_bbox[:, 0] + 1e-9
    tz = 2 * focal_length / bs
    tx = (2 * (cx - w_2) / bs) + cam_bbox[:, 1]
    ty = (2 * (cy - h_2) / bs) + cam_bbox[:, 2]
    full_cam = torch.stack([tx, ty, tz], dim=-1)
    return full_cam


def get_light_poses(n_lights=5, elevation=np.pi / 3, dist=12):
    # get lights in a circle around origin at elevation
    thetas = elevation * np.ones(n_lights)
    phis = 2 * np.pi * np.arange(n_lights) / n_lights
    poses = []
    trans = make_translation(torch.tensor([0, 0, dist]))
    for phi, theta in zip(phis, thetas):
        rot = make_rotation(rx=-theta, ry=phi, order="xyz")
        poses.append((rot @ trans).numpy())
    return poses


def make_translation(t):
    return make_4x4_pose(torch.eye(3), t)


def make_rotation(rx=0, ry=0, rz=0, order="xyz"):
    Rx = rotx(rx)
    Ry = roty(ry)
    Rz = rotz(rz)
    if order == "xyz":
        R = Rz @ Ry @ Rx
    elif order == "xzy":
        R = Ry @ Rz @ Rx
    elif order == "yxz":
        R = Rz @ Rx @ Ry
    elif order == "yzx":
        R = Rx @ Rz @ Ry
    elif order == "zyx":
        R = Rx @ Ry @ Rz
    elif order == "zxy":
        R = Ry @ Rx @ Rz
    return make_4x4_pose(R, torch.zeros(3))


def make_4x4_pose(R, t):
    """
    :param R (*, 3, 3)
    :param t (*, 3)
    return (*, 4, 4)
    """
    dims = R.shape[:-2]
    pose_3x4 = torch.cat([R, t.view(*dims, 3, 1)], dim=-1)
    bottom = (
        torch.tensor([0, 0, 0, 1], device=R.device)
        .reshape(*(1,) * len(dims), 1, 4)
        .expand(*dims, 1, 4)
    )
    return torch.cat([pose_3x4, bottom], dim=-2)


def rotx(theta):
    return torch.tensor(
        [
            [1, 0, 0],
            [0, np.cos(theta), -np.sin(theta)],
            [0, np.sin(theta), np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def roty(theta):
    return torch.tensor(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=torch.float32,
    )


def rotz(theta):
    return torch.tensor(
        [
            [np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta), np.cos(theta), 0],
            [0, 0, 1],
        ],
        dtype=torch.float32,
    )


def create_raymond_lights() -> List[pyrender.Node]:
    """
    Return raymond light nodes for the scene.
    """
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def print_parameter_shapes(vertices, camera_translation, image, full_frame, imgname, 
                          side_view, rot_angle, mesh_base_color, scene_bg_color, 
                          return_rgba, boxes):
    """
    Print shapes and dimensions of all parameters for debugging purposes.
    """
    print("=== Parameter Shapes/Dimensions ===")
    
    # vertices
    print(f"vertices: {type(vertices)}")
    if isinstance(vertices, np.ndarray):
        print(f"  - Shape: {vertices.shape}")
        print(f"  - Data type: {vertices.dtype}")
    elif isinstance(vertices, list):
        print(f"  - List length: {len(vertices)}")
        for i, v in enumerate(vertices):
            print(f"  - vertices[{i}] shape: {v.shape}, dtype: {v.dtype}")
    
    # camera_translation
    print(f"camera_translation: {type(camera_translation)}")
    if isinstance(camera_translation, np.ndarray):
        print(f"  - Shape: {camera_translation.shape}")
        print(f"  - Data type: {camera_translation.dtype}")
    elif isinstance(camera_translation, list):
        print(f"  - List length: {len(camera_translation)}")
        for i, ct in enumerate(camera_translation):
            print(f"  - camera_translation[{i}] shape: {ct.shape}, dtype: {ct.dtype}")
    
    # image
    print(f"image: {type(image)}")
    print(f"  - Shape: {image.shape}")
    print(f"  - Data type: {image.dtype}")
    print(f"  - Device: {image.device}")
    
    # full_frame
    print(f"full_frame: {type(full_frame)} = {full_frame}")
    
    # imgname
    print(f"imgname: {type(imgname)} = {imgname}")
    
    # side_view
    print(f"side_view: {type(side_view)} = {side_view}")
    
    # rot_angle
    print(f"rot_angle: {type(rot_angle)} = {rot_angle}")
    
    # mesh_base_color
    print(f"mesh_base_color: {type(mesh_base_color)} = {mesh_base_color}")
    if isinstance(mesh_base_color, (list, tuple)):
        print(f"  - Length: {len(mesh_base_color)}")
    
    # scene_bg_color
    print(f"scene_bg_color: {type(scene_bg_color)} = {scene_bg_color}")
    if isinstance(scene_bg_color, (list, tuple)):
        print(f"  - Length: {len(scene_bg_color)}")
    
    # return_rgba
    print(f"return_rgba: {type(return_rgba)} = {return_rgba}")
    
    # boxes
    print(f"boxes: {type(boxes)}")
    if boxes is not None:
        print(f"  - List length: {len(boxes)}")
        for i, box in enumerate(boxes):
            print(f"  - boxes[{i}]: {box} (length: {len(box)})")
    else:
        print("  - None")
    
    print("===================================")


class Renderer:

    def __init__(self, cfg: CfgNode, faces: np.array):
        """
        Wrapper around the pyrender renderer to render MANO meshes.
        Args:
            cfg (CfgNode): Model config file.
            faces (np.array): Array of shape (F, 3) containing the mesh faces.
        """
        self.cfg = cfg
        self.focal_length = 1000. if faces.shape[0] == 7774 else 2167.
        self.img_res = cfg.MODEL.IMAGE_SIZE

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        # Check if faces is already a numpy array or a PyTorch tensor
        if isinstance(faces, torch.Tensor):
            self.faces = faces.cpu().numpy()
        else:
            self.faces = faces.copy() if isinstance(faces, np.ndarray) else np.array(faces)

    def __call__(self,
             vertices: Union[np.array, List[np.array]],
             camera_translation: Union[np.array, List[np.array]],
             image: torch.Tensor,
             full_frame: bool = False,
             imgname: Optional[str] = None,
             side_view=False, rot_angle=90,
             mesh_base_color=(1.0, 1.0, 0.9),
             scene_bg_color=(0, 0, 0),
             return_rgba=False,
             boxes: Optional[List[List[int]]] = None,
             ) -> np.array:
        """
        Render meshes on input image
        Args:
            vertices (Union[np.array, List[np.array]]): Single array of shape (V, 3) or list of arrays for multiple meshes.
            camera_translation (Union[np.array, List[np.array]]): Single array of shape (3,) or list of arrays for multiple meshes.
            image (torch.Tensor): Tensor of shape (3, H, W) containing the image crop with normalized pixel values.
            full_frame (bool): If True, then render on the full image.
            imgname (Optional[str]): Contains the original image filenamee. Used only if full_frame == True.
            boxes (Optional[List[List[int]]]): List of bounding boxes [x1, y1, x2, y2] for each mesh. 
                                             If provided, each mesh will be rendered in its corresponding box area.
        """
        
        # Print parameter shapes for debugging
        # print_parameter_shapes(vertices, camera_translation, image, full_frame, imgname,
        #                       side_view, rot_angle, mesh_base_color, scene_bg_color,
        #                       return_rgba, boxes)
        
        # Convert single inputs to lists for uniform processing
        if isinstance(vertices, np.ndarray):
            vertices = [vertices]
        if isinstance(camera_translation, np.ndarray):
            camera_translation = [camera_translation]
        
        # Handle colors - if single color provided, use for all meshes
        if isinstance(mesh_base_color, tuple):
            mesh_base_colors = [mesh_base_color] * len(vertices)
        else:
            mesh_base_colors = mesh_base_color
    
        if full_frame:
            image = cv2.imread(imgname).astype(np.float32)[:, :, ::-1] / 255.
        else:
            #image = (image.clone()) * (torch.tensor(self.cfg.MODEL.IMAGE_STD, device=image.device).reshape(3, 1, 1))
            #image = image + torch.tensor(self.cfg.MODEL.IMAGE_MEAN, device=image.device).reshape(3, 1, 1)
            image = image.permute(1, 2, 0).cpu().numpy()
    
        # If boxes are provided, render each mesh in its corresponding box
        if boxes is not None:
            if len(boxes) != len(vertices):
                raise ValueError(f"Number of boxes ({len(boxes)}) must match number of meshes ({len(vertices)})")
            
            output_img = image.copy()
            valid_mask_full = []  # Initialize as list to store separate masks for each mesh
            
            for i, (verts, cam_trans, box, color) in enumerate(zip(vertices, camera_translation, boxes, mesh_base_colors)):
                # print(f"Box {i}: {box} (type: {type(box)})")
                x1, y1, x2, y2 = [int(coord) for coord in box]
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Create renderer for this specific box
                renderer = pyrender.OffscreenRenderer(viewport_width=box_width,
                                                      viewport_height=box_height,
                                                      point_size=1.0)
                
                # Create material for this mesh
                material = pyrender.MetallicRoughnessMaterial(
                    metallicFactor=0.0,
                    alphaMode='OPAQUE',
                    baseColorFactor=(*color, 1.0))
    
                cam_trans = cam_trans.copy()
                cam_trans[0] *= -1.
    
                # Create mesh
                mesh = trimesh.Trimesh(verts.copy(), self.faces.copy())
                if side_view:
                    rot = trimesh.transformations.rotation_matrix(
                        np.radians(rot_angle), [0, 1, 0])
                    mesh.apply_transform(rot)
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(180), [1, 0, 0])
                mesh.apply_transform(rot)
                mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
                # Create scene
                scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                                       ambient_light=(0.3, 0.3, 0.3))
                scene.add(mesh, f'mesh_{i}')
    
                # Set up camera for this box
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = cam_trans
                
                # Calculate camera center relative to the box
                camera_center = [box_width / 2., box_height / 2.]
                
                # Scale focal length based on box size
                scale_factor = max(box_height, box_width) / self.img_res
                scaled_focal_length = self.focal_length * scale_factor
                
                camera = pyrender.IntrinsicsCamera(
                    fx=scaled_focal_length, 
                    fy=scaled_focal_length,
                    cx=camera_center[0], 
                    cy=camera_center[1], 
                    zfar=1e12
                )
                scene.add(camera, pose=camera_pose)
    
                # Add lighting
                light_nodes = create_raymond_lights()
                for node in light_nodes:
                    scene.add_node(node)
    
                # Render this mesh
                color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
                color = color.astype(np.float32) / 255.0
                renderer.delete()
    
                if return_rgba:
                    # For RGBA mode, return the last rendered result
                    output_img = color
                    # Convert RGB to grayscale and threshold to binary mask
                    rgb = color[:, :, :3]
                    grayscale = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
                    # Create full-size mask for this mesh
                    mask_full = np.zeros_like(image[:, :, :1])
                    mask_full[y1:y2, x1:x2] = (grayscale > 0.01).astype(np.float32)[:, :, np.newaxis]
                    valid_mask_full.append(mask_full)
                else:
                    # Composite this mesh onto the output image
                    valid_mask = (color[:, :, -1])[:, :, np.newaxis]
                    # Create full-size mask for this mesh and store separately
                    mask_full = np.zeros_like(image[:, :, :1])
                    mask_full[y1:y2, x1:x2] = valid_mask
                    valid_mask_full.append(mask_full)
                    if not side_view:
                        rendered_mesh = color[:, :, :3] * valid_mask + (1 - valid_mask) * image[y1:y2, x1:x2]
                    else:
                        rendered_mesh = color[:, :, :3]
                    
                    # Place the rendered mesh in the correct box area
                    output_img[y1:y2, x1:x2] = rendered_mesh
            
            return output_img.astype(np.float32), valid_mask_full
        
        else:
            # Original single mesh rendering logic
            renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                                  viewport_height=image.shape[0],
                                                  point_size=1.0)
            material = pyrender.MetallicRoughnessMaterial(
                metallicFactor=0.0,
                alphaMode='OPAQUE',
                baseColorFactor=(*mesh_base_colors[0], 1.0))
    
            camera_translation[0] *= -1.
    
            mesh = trimesh.Trimesh(vertices[0].copy(), self.faces.copy())
            if side_view:
                rot = trimesh.transformations.rotation_matrix(
                    np.radians(rot_angle), [0, 1, 0])
                mesh.apply_transform(rot)
            rot = trimesh.transformations.rotation_matrix(
                np.radians(180), [1, 0, 0])
            mesh.apply_transform(rot)
            mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    
            scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(mesh, 'mesh')
    
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = camera_translation[0]
            camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
            # Scale focal length based on image size
            image_height, image_width = image.shape[:2]
            scale_factor = max(image_height, image_width) / self.img_res
            scaled_focal_length = self.focal_length * scale_factor
            camera = pyrender.IntrinsicsCamera(
                fx=scaled_focal_length, 
                fy=scaled_focal_length,
                cx=camera_center[0], 
                cy=camera_center[1], 
                zfar=1e12
            )
            scene.add(camera, pose=camera_pose)
    
            light_nodes = create_raymond_lights()
            for node in light_nodes:
                scene.add_node(node)
    
            color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32) / 255.0
            renderer.delete()
    
            if return_rgba:
                valid_mask = (color[:, :, -1])[:, :, np.newaxis]
                return color, valid_mask
    
            valid_mask = (color[:, :, -1])[:, :, np.newaxis]
            if not side_view:
                output_img = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image)
            else:
                output_img = color[:, :, :3]
    
            output_img = output_img.astype(np.float32)
            return output_img, valid_mask

    def vertices_to_trimesh(self, vertices, camera_translation, mesh_base_color=(1.0, 1.0, 0.9),
                            rot_axis=[1, 0, 0], rot_angle=0):
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))
        # vertex_colors = np.array([(*mesh_base_color, 1.0)] * vertices.shape[0])
        mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy())
        # mesh = trimesh.Trimesh(vertices.copy() + camera_translation, self.faces.copy(), vertex_colors=vertex_colors)
        # mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())

        rot = trimesh.transformations.rotation_matrix(
            np.radians(rot_angle), rot_axis)
        mesh.apply_transform(rot)

        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        return mesh

    def render_rgba(
            self,
            vertices: np.array,
            cam_t=None,
            rot=None,
            rot_axis=[1, 0, 0],
            rot_angle=0,
            camera_z=3,
            # camera_translation: np.array,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        focal_length = focal_length if focal_length is not None else self.focal_length

        if cam_t is not None:
            camera_translation = cam_t.copy()
            camera_translation[0] *= -1.
        else:
            camera_translation = np.array([0, 0, camera_z * focal_length / render_res[1]])

        mesh = self.vertices_to_trimesh(vertices, np.array([0, 0, 0]), mesh_base_color, rot_axis, rot_angle,
                                        )
        mesh = pyrender.Mesh.from_trimesh(mesh)
        # mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def render_rgba_multiple(
            self,
            vertices: List[np.array],
            cam_t: List[np.array],
            rot_axis=[1, 0, 0],
            rot_angle=0,
            mesh_base_color=(1.0, 1.0, 0.9),
            scene_bg_color=(0, 0, 0),
            render_res=[256, 256],
            focal_length=None,
    ):

        renderer = pyrender.OffscreenRenderer(viewport_width=render_res[0],
                                              viewport_height=render_res[1],
                                              point_size=1.0)
        # material = pyrender.MetallicRoughnessMaterial(
        #     metallicFactor=0.0,
        #     alphaMode='OPAQUE',
        #     baseColorFactor=(*mesh_base_color, 1.0))

        mesh_list = [pyrender.Mesh.from_trimesh(
            self.vertices_to_trimesh(vvv, ttt.copy(), mesh_base_color, rot_axis, rot_angle)) for
                     vvv, ttt in zip(vertices, cam_t)]

        scene = pyrender.Scene(bg_color=[*scene_bg_color, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        for i, mesh in enumerate(mesh_list):
            scene.add(mesh, f'mesh_{i}')

        camera_pose = np.eye(4)
        # camera_pose[:3, 3] = camera_translation
        camera_center = [render_res[0] / 2., render_res[1] / 2.]
        focal_length = focal_length if focal_length is not None else self.focal_length
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1], zfar=1e12)

        # Create camera node and add it to pyRender scene
        camera_node = pyrender.Node(camera=camera, matrix=camera_pose)
        scene.add_node(camera_node)
        self.add_point_lighting(scene, camera_node)
        self.add_lighting(scene, camera_node)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        renderer.delete()

        return color

    def add_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses()
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            node = pyrender.Node(
                name=f"light-{i:02d}",
                light=pyrender.DirectionalLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)

    def add_point_lighting(self, scene, cam_node, color=np.ones(3), intensity=1.0):
        # from phalp.visualize.py_renderer import get_light_poses
        light_poses = get_light_poses(dist=0.5)
        light_poses.append(np.eye(4))
        cam_pose = scene.get_pose(cam_node)
        for i, pose in enumerate(light_poses):
            matrix = cam_pose @ pose
            # node = pyrender.Node(
            #     name=f"light-{i:02d}",
            #     light=pyrender.DirectionalLight(color=color, intensity=intensity),
            #     matrix=matrix,
            # )
            node = pyrender.Node(
                name=f"plight-{i:02d}",
                light=pyrender.PointLight(color=color, intensity=intensity),
                matrix=matrix,
            )
            if scene.has_node(node):
                continue
            scene.add_node(node)



