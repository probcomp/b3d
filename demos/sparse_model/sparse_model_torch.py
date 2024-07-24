import os

import b3d
import pytorch3d.transforms
import rerun as rr
import torch
import trimesh
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

<<<<<<< HEAD
import os
=======
>>>>>>> main

mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32) * 10.0
object_vertices = vertices - torch.mean(vertices, axis=0)
object_vertices = object_vertices[torch.randperm(object_vertices.shape[0])[:500]]

fx, fy = 100.0, 100.0
cx, cy = 50, 50
image_height, image_width = 100, 100
rr.init("demo")
rr.connect("127.0.0.1:8812")


positions = torch.zeros((100, 3), requires_grad=True)
quaternions = torch.ones((100, 4), requires_grad=True)

<<<<<<< HEAD
positions = torch.zeros((100, 3), requires_grad=True)
quaternions = torch.ones((100, 4), requires_grad=True)

=======
>>>>>>> main

class Pose:
    def __init__(self, positions, quaternions):
        self.positions = positions
        self.quaternions = quaternions

    def apply(self, xyz):
        quaternions = self.quaternions
        positions = self.positions
        Rs = pytorch3d.transforms.quaternion_to_matrix(quaternions)
        return torch.einsum("...ij,aj->...ai", Rs, xyz) + positions[:, None, ...]


def xyz_to_pixel_coordinates(xyz, fx, fy, cx, cy):
    x = fx * xyz[..., 0] / xyz[..., 2] + cx
    y = fy * xyz[..., 1] / xyz[..., 2] + cy
    return torch.stack([y, x], axis=-1)


def model(positions, quaternions, xyz):
    poses = Pose(positions, quaternions)
    transformed_vertices = poses.apply(xyz)
    pixel_coordinates = xyz_to_pixel_coordinates(transformed_vertices, fx, fy, cx, cy)
    return pixel_coordinates


def viz_pixel_coordinates(pixel_coordinates):
    img = torch.zeros((len(pixel_coordinates), image_height, image_width))
    rounded_pixel_coordinates = torch.round(pixel_coordinates).long().reshape(-1, 2)
    image_index = torch.arange(len(pixel_coordinates)).repeat_interleave(
        pixel_coordinates.shape[1]
    )
    keypoint_index = (
        torch.arange(pixel_coordinates.shape[1]).repeat(len(pixel_coordinates)).float()
    )

    valid = (
        (rounded_pixel_coordinates[..., 0] >= 0)
        & (rounded_pixel_coordinates[..., 0] < image_height)
        & (rounded_pixel_coordinates[..., 1] >= 0)
        & (rounded_pixel_coordinates[..., 1] < image_width)
    )
    img[
        image_index[valid],
        rounded_pixel_coordinates[valid, 0],
        rounded_pixel_coordinates[valid, 1],
    ] = keypoint_index[valid] + 1
    return img


center_position = torch.tensor([0.0, 0.0, 3.0], requires_grad=True)
quaternion = torch.tensor([0.0, 0.0, 0.0, 1.0], requires_grad=True)


num_views = 10
object_positions = torch.normal(
    torch.tile(center_position[None, ...], (num_views, 1)), std=0.1
)
object_quaternions = torch.normal(
    torch.tile(quaternion[None, ...], (num_views, 1)), std=2.2
)
gt_pixel_coordinates = model(
    object_positions, object_quaternions, object_vertices
).detach()
viz_img_gt = viz_pixel_coordinates(gt_pixel_coordinates)


positions = torch.ones((num_views, 3), requires_grad=True)
quaternions = torch.ones((num_views, 4), requires_grad=True)
xyz = torch.normal(0.0, 0.1, size=object_vertices.shape, requires_grad=True)
rr.set_time_sequence("frame", 0)
rr.log(
    "gt", rr.DepthImage(viz_pixel_coordinates(gt_pixel_coordinates)[0]), timeless=True
)
rr.log(
    "overlay",
    rr.DepthImage(viz_pixel_coordinates(gt_pixel_coordinates)[0]),
    timeless=True,
)
rr.log("xyz", rr.Points3D(object_vertices), timeless=True)


optimizer = torch.optim.Adam(
    [
        {"params": positions, "lr": 0.1},
        {"params": quaternions, "lr": 0.1},
        {"params": xyz, "lr": 0.5},
    ]
)

pbar = tqdm(range(2000))
for t in pbar:
    optimizer.zero_grad()
    pixel_coordinates = model(positions, quaternions, xyz)
    loss = torch.mean((pixel_coordinates - gt_pixel_coordinates) ** 2)
    loss.backward()
    optimizer.step()
    pbar.set_description(f"loss: {loss.item()}")

    reconstruction = viz_pixel_coordinates(pixel_coordinates)[0]
    rr.set_time_sequence("frame", t)
    rr.log("reconstruction", rr.DepthImage(reconstruction))
    rr.log("overlay/reconstruction", rr.DepthImage(reconstruction))
    rr.log("xyz/overlay", rr.Points3D(xyz.detach().cpu().numpy()))
