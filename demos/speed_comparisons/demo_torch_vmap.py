#!/usr/bin/env python
import os
import time

import b3d.nvdiffrast_original.torch as dr
import pytorch3d.transforms
import rerun as rr
import torch
import torch.utils._pytree as pytree_utils
import trimesh
from tqdm import tqdm

import b3d

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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


def projection_matrix_from_intrinsics(w, h, fx, fy, cx, cy, near, far):
    # transform from cv2 camera coordinates to opengl (flipping sign of y and z)
    view = torch.eye(4)
    view[1:3] = view[1:3] * -1

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


class Pose:
    def __init__(self, position, quaternion):
        self.position = position
        self.quaternion = quaternion

    def unit_quaternion(device="cuda"):
        return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)

    def identity():
        return Pose(torch.zeros(3), Pose.unit_quaternion())

    def from_translation(translation):
        return Pose(
            translation,
            torch.tile(Pose.unit_quaternion(), (*translation.shape[:-1], 1)),
        )

    def apply_single(self, xyz):
        quaternion = self.quaternion
        position = self.position
        Rs = pytorch3d.transforms.quaternion_to_matrix(quaternion)

        def _single_R_and_p(R, p):
            return R @ xyz + p

        if len(Rs.shape) == 2:
            return _single_R_and_p(Rs, position)
        else:
            return torch.vmap(_single_R_and_p)(Rs, position)

    def apply(self, xyzs):
        if xyzs.dim() == 1:
            return self.apply_single(xyzs)
        else:
            return torch.vmap(self.apply)(xyzs)

    def compose(self, pose):
        return Pose(
            self.apply(pose.position),
            pytorch3d.transforms.quaternion_multiply(self.quaternion, pose.quaternion),
        )

    def from_look_at(position, target, up=torch.tensor([0.0, 0.0, 1.0])):
        z = target - position
        z = z / torch.linalg.norm(z)

        x = torch.cross(z, up, dim=0)
        x = x / torch.linalg.norm(x)

        y = torch.cross(z, x, dim=0)
        y = y / torch.linalg.norm(y)

        rotation_matrix = torch.hstack(
            [x.reshape(-1, 1), y.reshape(-1, 1), z.reshape(-1, 1)]
        )
        return Pose(
            position, pytorch3d.transforms.matrix_to_quaternion(rotation_matrix)
        )

    def inv(self):
        R = pytorch3d.transforms.quaternion_to_matrix(self.quaternion)
        return Pose(
            -R.T @ self.position,
            pytorch3d.transforms.quaternion_invert(self.quaternion),
        )

    def __str__(self):
        return (
            f"Pose(position={repr(self.position)}, quaternion={repr(self.quaternion)})"
        )

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


pytree_utils.register_pytree_node(
    Pose,
    lambda pose: ((pose.position, pose.quaternion), None),
    lambda pos_and_quat, _: Pose(*pos_and_quat),
)


def apply_single_projection(P, xyz):
    xyzw = torch.cat([xyz, torch.ones(1, dtype=xyz.dtype)])
    return P @ xyzw


apply_projection = torch.vmap(apply_single_projection, in_dims=(None, 0))


mesh_path = os.path.join(
    b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
)
mesh = trimesh.load(mesh_path)
vertices = torch.tensor(mesh.vertices, dtype=torch.float32) * 10.0
faces = torch.tensor(mesh.faces, dtype=torch.int32)
object_vertices = vertices - torch.mean(vertices, axis=0)
vertex_colors = torch.tensor(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0


glctx = dr.RasterizeGLContext()  # if use_opengl else dr.RasterizeCudaContext()

pose = Pose.from_look_at(
    torch.tensor([0.0, 6.3, 0.5]),
    torch.tensor([0.0, 0.0, 0.0]),
).inv()


def render_single(pose, vertices, faces, vertex_colors, P, height, width):
    # since the original renderer was implemented with batch support, we'll
    # have to use add/remove a fake leading batch dimension
    clip_space = apply_projection(P, pose.apply(vertices))
    rast_out, _ = dr.rasterize(
        glctx, clip_space.contiguous()[None, ...], faces, resolution=[height, width]
    )
    color, _ = dr.interpolate(vertex_colors, rast_out, faces)
    return color[0]


render_batch = torch.vmap(
    render_single, in_dims=(0, None, None, None, None, None, None)
)

# # sanity checks
# image = render_single(pose, vertices, faces, vertex_colors, P, height, width)
# images = render_batch(
#     pose[None, ...], vertices, faces, vertex_colors, P, height, width
# )


delta_pose = Pose.identity()
delta_pose.position = torch.normal(delta_pose.position, 0.02)
delta_pose.quaternion = torch.normal(delta_pose.quaternion, 0.02)

images = []
poses = []
for t in range(100):
    pose = pose.compose(delta_pose)
    # print(pose)
    poses.append(pose)
    color = render_single(pose, vertices, faces, vertex_colors, P, height, width)
    images.append(color)
    rr.set_time_sequence("frame", t)
    rr.log("img", rr.Image(color.cpu().numpy()))

translation_deltas = Pose(
    torch.stack(
        torch.meshgrid(
            torch.linspace(-0.01, 0.01, 5),
            torch.linspace(-0.01, 0.01, 5),
            torch.linspace(-0.01, 0.01, 5),
            indexing="xy",
        ),
        dim=-1,
    ).reshape(-1, 3),
    torch.tile(torch.tensor([1.0, 0.0, 0.0, 0.0], device=device), (5 * 5 * 5, 1)),
)
rotation_deltas = Pose(
    torch.zeros((100, 3), device=device),
    torch.normal(0.0, 0.03, (100, 4), device=device) + Pose.unit_quaternion(),
)


def score_single(pose, gt_image):
    color = render_single(pose, vertices, faces, vertex_colors, P, height, width)
    error = torch.mean((color - gt_image) ** 2)
    return error


score_batch = torch.vmap(score_single, in_dims=(0, None))
# sanity checks
# error = score_single(pose, image)
# errors = score_batch(pose.compose(translation_deltas), image)


@torch.compile  # <- this should run, but it doesn't seem to improve the performance
def update(pose_estimate, gt_image):
    pose_hypotheses = pose_estimate.compose(translation_deltas)
    errors = score_batch(pose_hypotheses, gt_image)
    pose_estimate = pose_hypotheses[torch.argmin(errors)]

    pose_hypotheses = pose_estimate.compose(rotation_deltas)
    errors = score_batch(pose_hypotheses, gt_image)
    pose_estimate = pose_hypotheses[torch.argmin(errors)]
    return pose_estimate


pose_estimate = poses[0]
pose_estimate = update(pose_estimate, images[t])


pose_estimate = poses[0]

start = time.time()
for t in tqdm(range(len(images))):
    pose_estimate = update(pose_estimate, images[t])
end = time.time()
print("Time elapsed:", end - start)
print("FPS:", len(images) / (end - start))
