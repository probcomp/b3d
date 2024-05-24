import jax
import jax.numpy as jnp
import genjax
import b3d
import b3d.tessellation as t
import b3d.utils as u
import os
import rerun as rr

rr.init("tessellation")
rr.connect("127.0.0.1:8812")

path = os.path.join(
    b3d.get_root_path(),
    "assets/shared_data_bucket/input_data/shout_on_desk.r3d.video_input.npz",
)
video_input = b3d.VideoInput.load(path)
image_width, image_height, fx, fy, cx, cy, near, far = jnp.array(
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

rgbs = video_input.rgb[::4] / 255.0
vertices_2D, faces, triangle_colors = t.generate_tessellated_2D_mesh_from_rgb_image(rgbs[0])
key = jax.random.PRNGKey(0)
depth = genjax.uniform.sample(key, 0.2, 200.0)
vertices_3D = jnp.hstack(((vertices_2D  - jnp.array([cx, cy])) * depth / fx , jnp.ones((vertices_2D.shape[0], 1)) * depth))

_v, _f, _vertex_colors = u.triangle_color_mesh_to_vertex_color_mesh(vertices_3D, faces, triangle_colors)
rr.log("3D/tessellated_poster_board", rr.Mesh3D(
    vertex_positions=_v, indices=_f, vertex_colors=_vertex_colors
))


rr.log("3D/camera", rr.Pinhole(
    focal_length = fx,
    width = image_width,
    height = image_height,
    principal_point = jnp.array([cx, cy])
))

renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, near, far)
rendering, _ = renderer.render_attribute(
    b3d.Pose.identity()[None, ...], _v, _f, jnp.array([[0, len(_f)]]), _vertex_colors
)

rr.log("rendering", rr.Image(rendering))