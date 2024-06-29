import b3d
import os
import jax.numpy as jnp
import rerun as rr
import jax
import trimesh

PORT = 8812
rr.init("real")
rr.connect(addr=f"127.0.0.1:{PORT}")

def test_resolution_invariance(renderer):
    import trimesh

    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)

    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)

    image_width = 200
    image_height = 200
    fx = 200.0
    fy = 200.0
    cx = 100.0
    cy = 100.0
    near = 0.001
    far = 16.0
    renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

    near_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.3, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    ).inv()

    rgb_near, depth_near = renderer.render_attribute(
        near_pose[None, ...],
        object_library.vertices,
        object_library.faces,
        object_library.ranges,
        object_library.attributes,
    )

    color_error, depth_error = (50.0, 0.01)
    inlier_score, outlier_prob = (4.0, 0.000001)
    color_multiplier, depth_multiplier = (10000.0, 1.0)
    model_args = b3d.ModelArgs(
        color_error,
        depth_error,
        inlier_score,
        outlier_prob,
        color_multiplier,
        depth_multiplier,
    )

    logpdf = b3d.rgbd_sensor_model.logpdf(
        (rgb_near, depth_near), rgb_near, depth_near, model_args, fx, fy, image_height, image_width
    )

    for SCALING_FACTOR in [2,3,4,5,6,7,8]:
        rgb_resized = jax.image.resize(
            rgb_near, 
            (rgb_near.shape[0] * SCALING_FACTOR, rgb_near.shape[1] * SCALING_FACTOR, 3),
            "nearest"
        )
        depth_resized = jax.image.resize(
            depth_near, 
            (depth_near.shape[0] * SCALING_FACTOR, depth_near.shape[1] * SCALING_FACTOR),
            "nearest"
        )
        scaled_up_logpdf = b3d.rgbd_sensor_model.logpdf(
            #(rgb_resized, depth_resized), rgb_resized, depth_resized, model_args, fx , fy * SCALING_FACTOR, 1.0
            (rgb_resized, depth_resized), rgb_resized, depth_resized, model_args, fx * SCALING_FACTOR, fy * SCALING_FACTOR, image_height * SCALING_FACTOR, image_width * SCALING_FACTOR
        )
        assert jnp.isclose(logpdf, scaled_up_logpdf, rtol=0.01)

def test_distance_to_camera_invarance(renderer):

    mesh_path = os.path.join(
        b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
    )
    mesh = trimesh.load(mesh_path)
    mesh.vertices = mesh.vertices - mesh.vertices.mean(axis=0)
    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_trimesh(mesh)


    object_library = b3d.MeshLibrary.make_empty_library()
    occluder = trimesh.creation.box(extents=jnp.array([0.15, 0.1, 0.1]))
    occluder_colors = jnp.tile(jnp.array([0.8, 0.8, 0.8])[None,...], (occluder.vertices.shape[0], 1))
    object_library = b3d.MeshLibrary.make_empty_library()
    object_library.add_object(occluder.vertices, occluder.faces, attributes=occluder_colors)

    image_width = 200
    image_height = 200
    fx = 200.0
    fy = 200.0
    cx = 100.0
    cy = 100.0
    near = 0.001
    far = 16.0
    renderer.set_intrinsics(image_width, image_height, fx, fy, cx, cy, near, far)

    near_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.3, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    ).inv()

    far_pose = b3d.Pose.from_position_and_target(
        jnp.array([0.9, 0.0, 0.0]), jnp.array([0.0, 0.0, 0.0]), jnp.array([0.0, 0.0, 1.0])
    ).inv()

    rgb_near, depth_near = renderer.render_attribute(
        near_pose[None, ...],
        object_library.vertices,
        object_library.faces,
        object_library.ranges,
        object_library.attributes,
    )

    rgb_far, depth_far = renderer.render_attribute(
        far_pose[None, ...],
        object_library.vertices,
        object_library.faces,
        object_library.ranges,
        object_library.attributes,
    )


    color_error, depth_error = (50.0, 0.01)
    inlier_score, outlier_prob = (4.0, 0.000001)
    color_multiplier, depth_multiplier = (100.0, 1.0)
    model_args = b3d.ModelArgs(
        color_error,
        depth_error,
        inlier_score,
        outlier_prob,
        color_multiplier,
        depth_multiplier,
    )

    from genjax.generative_functions.distributions import ExactDensity
    import genjax


    rr.log("img_near", rr.Image(rgb_near))
    rr.log("img_far", rr.Image(rgb_far))



    area_near = ((depth_near / fx) * (depth_near / fy)).sum()
    area_far = ((depth_far / fx) * (depth_far / fy)).sum()
    print(area_near, area_far)

    near_score = (
        b3d.rgbd_sensor_model.logpdf(
            (rgb_near, depth_near), rgb_near, depth_near, model_args, fx, fy, image_height, image_width
        )
    )

    far_score = (
        b3d.rgbd_sensor_model.logpdf(
            (rgb_near, depth_near), rgb_near, depth_near, model_args, fx, fy, image_height, image_width
        )
    )
    print(near_score, far_score)
    print(b3d.normalize_log_scores(jnp.array([near_score, far_score])))


    assert jnp.isclose(near_score, far_score, rtol=0.03)

