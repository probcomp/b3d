import b3d
import rerun as rr
import os
import jax.numpy as jnp
import jax
import trimesh
import genjax
from b3d import Pose
from tqdm import tqdm

PORT = 8812
rr.init("mug sm2c inference")
rr.connect(addr=f"127.0.0.1:{PORT}")

import pickle

    # Load a pickle file
    path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/input_data/aidan_panda_dataset/mug_backwards.pkl",
    )
    with open(path, "rb") as f:
        data = pickle.load(f)

    d = data[5]
    print(d.keys())
    K = d["camera_image"]["camera_matrix"][0]
    fx,fy = K[0,0], K[1,1]
    cx,cy = K[0,2], K[1,2]
    rgb = d["camera_image"]["rgbPixels"]
    depth = d["camera_image"]["depthPixels"]
    camera_pose = d["camera_image"]["camera_pose"]
    image_height, image_width = rgb.shape[:2]

    video_input = b3d.VideoInput(
        rgb=jnp.array(rgb)[None,...],
        xyz=b3d.xyz_from_depth(jnp.array(depth), fx,fy, cx,cy)[None,...],
        camera_intrinsics_rgb=jnp.array([image_width, image_height, fx, fy, cx, cy, 0.01, 10.0]),
        camera_intrinsics_depth=jnp.array([image_width, image_height, fx, fy, cx, cy, 0.01, 10.0]),
        camera_positions=jnp.zeros((1,3)),
        camera_quaternions=jnp.zeros((1,4)),
    )
    T =  0
    scaling_factor = 5


    image_width, image_height, fx, fy, cx, cy, near, far = (
        jnp.array(video_input.camera_intrinsics_depth) / scaling_factor
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

    _rgb = video_input.rgb[T].astype(jnp.float32) / 255.0
    _depth = video_input.xyz[T].astype(jnp.float32)[..., 2]
    rgb = jnp.clip(
        jax.image.resize(_rgb, (image_height, image_width, 3), "nearest"), 0.0, 1.0
    )
    depth = jax.image.resize(_depth, (image_height, image_width), "nearest")

    rr.log(
        "xyz", rr.Points3D(b3d.xyz_from_depth(jnp.array(depth), fx, fy, cx, cy).reshape(-1, 3))
    )

    object_library = b3d.MeshLibrary.make_empty_library()
    mesh_path = os.path.join(
        b3d.get_root_path(),
        "assets/shared_data_bucket/ycb_video_models/models/025_mug/textured_simple.obj",
    )
    object_library.add_trimesh(trimesh.load(mesh_path))

    renderer = None
    color_error, depth_error = (60.0, 2.1)
    inlier_score, outlier_prob = (5.0, 0.00001)
    color_multiplier, depth_multiplier = (10000.0, 500.0)
    model_args = b3d.ModelArgs(
        color_error,
        depth_error,
        inlier_score,
        outlier_prob,
        color_multiplier,
        depth_multiplier,
    )

    key = jax.random.PRNGKey(1000)
    if renderer is None:
        renderer = b3d.Renderer(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)
    else:
        renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, 0.01, 10.0)
    model = b3d.model_multiobject_gl_factory(renderer, b3d.rgbd_sensor_model)



    importance_jit = jax.jit(model.importance)
    key = jax.random.PRNGKey(110)


    point_cloud = b3d.xyz_from_depth(depth, fx, fy, cx, cy).reshape(-1, 3)

    vertex_colors = object_library.attributes
    rgb_object_samples = vertex_colors[
        jax.random.choice(jax.random.PRNGKey(0), jnp.arange(len(vertex_colors)), (10,))
    ]
    distances = jnp.abs(rgb[..., None] - rgb_object_samples.T).sum([-1, -2]).reshape(-1)
    distances = distances  + (point_cloud[:,2] < 1e-10) * 1000
    # rr.log("image/distances", rr.DepthImage(distances))
    # rr.log("img", rr.Image(rgb))

    object_center_hypothesis = point_cloud[distances.argmin()]



    key = jax.random.split(key, 2)[-1]
    trace, _ = model.importance(
        jax.random.PRNGKey(0),
        genjax.choice_map(
            {
                "camera_pose": Pose.identity(),
                "object_pose_0": Pose.sample_gaussian_vmf_pose(
                    key, Pose.from_translation(object_center_hypothesis), 0.001, 0.01
                ),
                "object_0": 0,
                "observed_rgb_depth": (rgb, depth),
            }
        ),
        (jnp.arange(1), model_args, object_library),
    )
    b3d.rerun_visualize_trace_t(trace, 0)


    params = jnp.array([0.02, 1.0])
    skips = 0
    counter = 1
    for t in tqdm(range(30)):
        (
            trace2,
            key,
        ) = b3d.gvmf_and_sample(
            trace, key, params[0], params[1], genjax.Pytree.const("object_pose_0"), 10000
        )
        if trace2.get_score() > trace.get_score():
            trace = trace2
            b3d.rerun_visualize_trace_t(trace, counter)
            counter += 1
        else:
            params = jnp.array([params[0] * 0.5, params[1] * 2.0])
            skips += 1
            print(f"shrinking")
            if skips > 5:
                print(f"skip {t}")
                break


