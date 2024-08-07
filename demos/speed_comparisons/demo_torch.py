#!/usr/bin/env python
import os
import time

import b3d
import b3d.nvdiffrast_original.torch as dr
import b3d.torch
import b3d.torch.renderutils
import pytorch3d.transforms
import rerun as rr
import torch
import trimesh
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_default_device("cuda")
rr.init("demo")
rr.connect("127.0.0.1:8812")

torch.set_default_device("cuda")

height = 100
width = 100
fx = 200.0
fy = 200.0
cx = 50.0
cy = 50.0
near = 0.001
far = 6.0

Proj = b3d.torch.projection_matrix_from_intrinsics(
    width, height, fx, fy, cx, cy, near, far
)


mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32) * 10.0
faces = torch.tensor(mesh.faces, dtype=torch.int32)
object_vertices = vertices - torch.mean(vertices, axis=0)
vertex_colors = torch.tensor(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0


glctx = dr.RasterizeGLContext()  # if use_opengl else dr.RasterizeCudaContext()


def from_translation(translations):
    M = torch.eye(4).tile(*(*translations.shape[:-1], 1, 1))
    M[..., :3, 3] = translations
    return M


def from_quaternion(quaternions):
    M = torch.eye(4).tile(*(*quaternions.shape[:-1], 1, 1))
    M[..., :3, :3] = pytorch3d.transforms.quaternion_to_matrix(quaternions)
    return M


def from_position_and_quaternion(positions, quaternions):
    M = torch.eye(4).tile(*(*positions.shape[:-1], 1, 1))
    M[..., :3, :3] = pytorch3d.transforms.quaternion_to_matrix(quaternions)
    M[..., :3, 3] = positions
    return M


def from_position_and_rotation_matrix(positions, rotation_matrix):
    M = torch.eye(4).tile(*(*positions.shape[:-1], 1, 1))
    M[..., :3, :3] = rotation_matrix
    M[..., :3, 3] = positions
    return M


def from_look_at(position, target, up=None):
    if up is None:
        up = torch.tensor([0.0, 0.0, 1.0])
    z = target - position
    z = z / torch.linalg.norm(z)
    x = torch.cross(z, up)
    x = x / torch.linalg.norm(x)
    y = torch.cross(z, x)
    y = y / torch.linalg.norm(y)
    rotation_matrix = torch.hstack(
        [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)]
    )
    return from_position_and_rotation_matrix(position, rotation_matrix)


def identity_quaternion():
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


N = 1
translations = torch.stack(
    torch.meshgrid(
        torch.linspace(-0.02, 0.02, N),
        torch.linspace(-0.02, 0.02, N),
        torch.linspace(-0.02, 0.02, N),
    ),
    dim=-1,
).reshape(-1, 3)
translation_poses = from_translation(translations)

rotation_poses = from_quaternion(
    torch.normal(0.0, 0.02, (N * N * N, 4), device=device) + identity_quaternion()
)

pose = torch.linalg.inv(
    from_look_at(
        torch.tensor([0.0, 6.3, 0.5]),
        torch.tensor([0.0, 0.0, 0.0]),
        torch.tensor([0.0, 0.0, 1.0]),
    )
)

delta_pose = from_position_and_quaternion(
    torch.normal(torch.zeros(3), 0.02), torch.normal(identity_quaternion(), 0.02)
)

images = []
poses = []
for t in range(100):
    pose = pose @ delta_pose
    poses.append(pose)

    clip_space = b3d.torch.renderutils.xfm_points(
        vertices[None, ...], Proj @ pose[None, ...]
    )
    rast_out, _ = dr.rasterize(glctx, clip_space, faces, resolution=[height, width])
    color, _ = dr.interpolate(vertex_colors, rast_out, faces)

    images.append(color[0])
    rr.set_time_sequence("frame", t)
    rr.log("img", rr.Image(color.cpu().numpy()[0, ...]))


def update(pose_estimate, gt_image):
    for deltas in [translation_poses, rotation_poses]:
        potential_poses = pose_estimate @ deltas
        clip_space = b3d.torch.renderutils.xfm_points(
            vertices[None, ...], Proj @ potential_poses
        )
        rast_out, _ = dr.rasterize(glctx, clip_space, faces, resolution=[height, width])
        color, _ = dr.interpolate(vertex_colors, rast_out, faces)
        error = torch.mean((color - gt_image) ** 2, dim=(1, 2, 3))
        pose_estimate = potential_poses[torch.argmin(error)]
    return pose_estimate


pose_estimate = torch.tensor(poses[0], requires_grad=True)


sum_total = 0.0
pose_estimates = []
start = time.time()
for t in tqdm(range(len(images))):
    pose_estimate = update(pose_estimate, images[t])
    pose_estimates.append(pose_estimate)
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", len(images) / (end - start))

for t in range(len(images)):
    rr.set_time_sequence("frame", t)
    clip_space = b3d.torch.renderutils.xfm_points(
        vertices[None, ...], Proj @ pose_estimates[t][None, ...]
    )
    rast_out, _ = dr.rasterize(glctx, clip_space, faces, resolution=[height, width])
    color, _ = dr.interpolate(vertex_colors, rast_out, faces)
    rr.log("img/reconstruct", rr.Image(color.detach().cpu().numpy()[0, ...]))
