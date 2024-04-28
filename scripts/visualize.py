import b3d
import os
import jax.numpy as jnp
import jax
import fire
from b3d import Pose


def visualize_video_input(path):
    video_input = b3d.VideoInput.load(path)

    import rerun as rr

    PORT = 8812
    rr.init("asdf223ff3")
    rr.connect(addr=f"127.0.0.1:{PORT}")

    import numpy as np

    image_width, image_height, fx, fy, cx, cy, near, far = np.array(
        video_input.camera_intrinsics_depth
    )
    image_width, image_height = int(image_width), int(image_height)
    fx, fy, cx, cy, near, far = (
        float(fx),
        float(fy),
        float(cx),
        float(cy),
        float(near),
        float(far),
    )

    rgbs = video_input.rgb / 255.0
    # Resize rgbs to be same size as depth.
    rgbs_resized = jnp.clip(
        jax.vmap(jax.image.resize, in_axes=(0, None, None))(
            rgbs, (video_input.xyz.shape[1], video_input.xyz.shape[2], 3), "linear"
        ),
        0.0,
        1.0,
    )

    TIME_FOR_MESH = 0
    point_cloud_for_mesh = video_input.xyz[TIME_FOR_MESH].reshape(-1, 3)
    colors_for_mesh = rgbs_resized[TIME_FOR_MESH].reshape(-1, 3)

    _vertices, faces, vertex_colors, face_colors = (
        b3d.make_mesh_from_point_cloud_and_resolution(
            point_cloud_for_mesh, colors_for_mesh, point_cloud_for_mesh[:, 2] / fx
        )
    )
    object_pose = Pose.from_translation(_vertices.mean(0))  # CAMERA frame object pose
    vertices = object_pose.inverse().apply(_vertices)  # WORLD frame vertices

    renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)

    img, depth = renderer.render_attribute(
        object_pose.as_matrix()[None, ...],
        vertices,
        faces,
        jnp.array([[0, len(faces)]]),
        vertex_colors,
    )
    rr.log("img", rr.Image(img))
    rr.log("depth", rr.DepthImage(depth))

    for t in range(len(video_input.xyz)):
        rr.set_time_sequence("frame", t)
        rr.log(
            f"/img",
            rr.Points3D(
                video_input.xyz[t].reshape(-1, 3),
                colors=(rgbs_resized[t].reshape(-1, 3) * 255).astype(jnp.uint8),
            ),
        )
        rr.log(f"/rgb", rr.Image(rgbs_resized[t]))


if __name__ == "__main__":
    fire.Fire(visualize_video_input)
