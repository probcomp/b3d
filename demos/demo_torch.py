#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
import rerun as rr
from tqdm import tqdm
import torch
import torchvision


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rr.init("demo")
rr.connect("127.0.0.1:8812")

torch.set_default_device("cuda")

# # Load date
# path = os.path.join(
#     b3d.get_root_path(),
#     "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
# )
# with open(path, "rb") as f:
#     data = np.load(f)
#     rgb=np.array(data["rgb"])
#     xyz=np.array(data["xyz"])
#     camera_positions=np.array(data["camera_positions"])
#     camera_quaternions=np.array(data["camera_quaternions"])
#     camera_intrinsics_rgb=np.array(data["camera_intrinsics_rgb"])
#     camera_intrinsics_depth=np.array(data["camera_intrinsics_depth"])
    

# # Get intrinsics
# width, height, fx, fy, cx, cy, near, far = np.array(
#     camera_intrinsics_depth
# )
# width, height = int(width), int(height)
# fx, fy, cx, cy, near, far = (
#     float(fx),
#     float(fy),
#     float(cx),
#     float(cy),
#     float(near),
#     float(far),
# )

height=100
width=100
fx=200.0
fy=200.0
cx=50.0
cy=50.0
near=0.001
far=6.0


def projection_matrix_from_intrinsics(w, h, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = torch.eye(4)
    view[1:3] = (view[1:3] * -1)

    # see http://ksimek.github.io/2013/06/03/calibrated_cameras_in_opengl/
    persp = torch.zeros((4, 4))
    persp = torch.tensor(
        [
            [fx, 0.0, -cx, 0.0],
            [0.0, -fy, -cy, 0.0],
            [0.0, 0.0, -near + far, near * far],
            [0.0, 0.0, -1, 0.0],
        ]
    )

    left, right, bottom, top = -0.5, w - 0.5, -0.5, h - 0.5
    orth = torch.tensor(
        [
            (2 / (right - left), 0, 0, -(right + left) / (right - left)),
            (0, 2 / (top - bottom), 0, -(top + bottom) / (top - bottom)),
            (0, 0, -2 / (far - near), -(far + near) / (far - near)),
            (0, 0, 0, 1),
        ]
    )
    return orth @ persp @ view

P = projection_matrix_from_intrinsics(width, height, fx, fy, cx, cy, near, far)

# Get RGBS and Depth
# rgbs = torch.tensor(rgb[::4] / 255.0, device=device)
# xyzs = torch.tensor(xyz[::4], device=device)

# # Resize rgbs to be same size as depth.
# rgbs_resized  = torch.clip(
#     torchvision.transforms.Resize(
#        size=(xyzs.shape[1], xyzs.shape[2])
#     )(rgbs.permute(0,3,1,2)),
#     0.0,
#     1.0,
# ).permute(0,2,3,1)

import pytorch3d.transforms
import pytorch3d.renderer

class Pose:
    def __init__(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion
    

    def unit_quaternion(device="cuda"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    def identity():
        return Pose(torch.zeros(3), Pose.unit_quaternion())

    def from_translation(translation):
        return Pose(translation, torch.tile(Pose.unit_quaternion(), (*translation.shape[:-1], 1)))

    def apply(self, xyz):
        quaternion = self.quaternion
        position = self.position
        Rs = pytorch3d.transforms.quaternion_to_matrix(quaternion)
        if len(Rs.shape) == 2:
            return torch.einsum(
                "ij,...j->...i",
                Rs,
                xyz,
            ) + position
        else:
            return torch.einsum(
                "bij,...j->b...i",
                Rs,
                xyz,
            ) + position.reshape((len(position), *(1,)*(len(xyz.shape)-1), -1))

    def compose(self, pose):
        return Pose(
            self.apply(pose.position[None,...])[0],
            pytorch3d.transforms.quaternion_multiply(self.quaternion, pose.quaternion),
        )

    def from_look_at(position, target, up=torch.tensor([0.0, 0.0, 1.0])):
        z = target - position
        z = z / torch.linalg.norm(z)

        x = torch.cross(z, up)
        x = x / torch.linalg.norm(x)

        y = torch.cross(z, x)
        y = y / torch.linalg.norm(y)

        rotation_matrix = torch.hstack([x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)])
        return Pose(position, pytorch3d.transforms.matrix_to_quaternion(rotation_matrix))

    def inv(self):
        R = pytorch3d.transforms.quaternion_to_matrix(self.quaternion)
        return Pose(-R.T @ self.position, pytorch3d.transforms.quaternion_invert(self.quaternion))

    def __str__(self):
        return f"Pose(position={repr(self.position)}, quaternion={repr(self.quaternion)})"

    def __repr__(self):
        return self.__str__()

    def __call__(self, vec):
        """Apply pose to vectors."""
        return self.apply(vec)

    def __matmul__(self, pose):
        """Compose with other poses."""
        # TODO: Add test, in particular to lock in matmul vs mul.
        return self.compose(pose)

    def __getitem__(self, index):
        return Pose(self.position[index], self.quaternion[index])


def apply_projection(P, points):
    points = torch.cat([points, torch.ones_like(points[..., :1])], dim=-1)
    return torch.einsum("ij,...j->...i", P, points)

import os
mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32) * 10.0
faces = torch.tensor(mesh.faces, dtype=torch.int32)
object_vertices = vertices - torch.mean(vertices, axis=0)
vertex_colors = torch.tensor(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0

import b3d.nvdiffrast_original.torch as dr
glctx = dr.RasterizeGLContext() #if use_opengl else dr.RasterizeCudaContext()




def render(pose, vertices, faces, vertex_colors, P, height, width):
    clip_space = apply_projection(P, pose.apply(vertices))
    rast_out, _ = dr.rasterize(glctx, clip_space.contiguous(), faces, resolution=[height, width])
    color   , _ = dr.interpolate(vertex_colors, rast_out, faces)
    return color



pose = Pose.from_look_at(
    torch.tensor([0.0, 6.3, 0.5]),
    torch.tensor([0.0, 0.0, 0.0]),
).inv()

delta_pose = Pose.identity()
delta_pose.position = torch.normal(delta_pose.position, 0.02)
delta_pose.quaternion = torch.normal(delta_pose.quaternion, 0.02)

images = []
poses = []
for t in range(100):
    pose = pose.compose(delta_pose)
    print(pose)
    poses.append(pose)
    color = render(pose[None,...], vertices, faces, vertex_colors, P, height, width)
    images.append(color)
    rr.set_time_sequence("frame", t)
    rr.log(f"img", rr.Image(color.cpu().numpy()[0, ...]))

translation_deltas = Pose(
    torch.stack(
        torch.meshgrid(
            torch.linspace(-0.01, 0.01, 11),
            torch.linspace(-0.01, 0.01, 11),
            torch.linspace(-0.01, 0.01, 11),
        ),
        dim=-1,
    ).reshape(-1, 3),
    torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), (11 * 11 * 11, 1)),
)
rotation_deltas = Pose(
    torch.zeros((11 * 11 * 11, 3), device=device),
    torch.normal(0.0, 0.03, (11 * 11 * 11, 4), device=device) + Pose.unit_quaternion()
)

import torch.jit as jit

def update(pose_estimate, gt_image):
    pose_hypotheses = pose_estimate.compose(translation_deltas)
    color = render(pose_hypotheses, vertices, faces, vertex_colors, P, height, width)
    error = torch.mean((color - gt_image)**2, dim=(1,2,3))
    pose_estimate = pose_hypotheses[torch.argmin(error)]
    pose_hypotheses = pose_estimate.compose(rotation_deltas)
    color = render(pose_hypotheses, vertices, faces, vertex_colors, P, height, width)
    error = torch.mean((color - gt_image)**2, dim=(1,2,3))
    pose_estimate = pose_hypotheses[torch.argmin(error)]
    return pose_estimate

pose_estimate = poses[0]

for t in tqdm(range(100)):
    pose_estimate= update(pose_estimate, images[t])
    rr.set_time_sequence("frame", t)
    color = render(pose_estimate[None,...], vertices, faces, vertex_colors, P, height, width)
    rr.log(f"img/overlay", rr.Image(color.cpu().numpy()[0, ...]))



