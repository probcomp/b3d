import b3d
import matplotlib.pyplot as plt
import jax.numpy as jnp
import jax
from b3d import Pose, Mesh
import rerun as rr
import genjax
import os
import genjax
from b3d.modeling_utils import uniform_discrete, uniform_pose, gaussian_vmf
from collections import namedtuple
from genjax import Pytree
import b3d
from b3d.bayes3d.enumerative_proposals import gvmf_and_select_best_move
from tqdm import tqdm

import importlib
importlib.reload(b3d.mesh)
importlib.reload(b3d.io.data_loader)
importlib.reload(b3d.utils)
importlib.reload(b3d.renderer.renderer_original)

FRAME_RATE = 50

ycb_dir = os.path.join(b3d.get_assets_path(), "bop/ycbv")

for scene_id in range(48, 60):
    print(f"Scene {scene_id}")
    num_scenes = b3d.io.data_loader.get_ycbv_num_test_images(ycb_dir, scene_id)
    all_data = b3d.io.get_ycbv_test_images(ycb_dir, scene_id, range(1, num_scenes, FRAME_RATE))

    meshes = [
        Mesh.from_obj_file(os.path.join(ycb_dir, f'models/obj_{f"{id + 1}".rjust(6, "0")}.ply')).scale(0.001)
        for id in all_data[0]["object_types"]
    ]

    height, width = all_data[0]["rgbd"].shape[:2]
    fx,fy,cx,cy = all_data[0]["camera_intrinsics"]
    scaling_factor = 0.3
    renderer = b3d.renderer.renderer_original.RendererOriginal(
        width * scaling_factor, height * scaling_factor, fx * scaling_factor, fy * scaling_factor, cx * scaling_factor, cy * scaling_factor, 0.01, 2.0
    )
    IDX = 0
    camera_pose = all_data[0]["camera_pose"]
    object_poses = all_data[0]["object_poses"]
    meshes[IDX].rr_visualize("mesh")
    mesh_at_pose = meshes[IDX]
    mesh_at_pose = mesh_at_pose.transform(camera_pose.inv() @ object_poses[IDX])
    rgbd = renderer.render_rgbd(
        mesh_at_pose.vertices,
        mesh_at_pose.faces,
        mesh_at_pose.vertex_attributes,
    )
    b3d.viz_rgb(rgbd[...,:3])

    import b3d.chisight.dense.likelihoods.image_likelihood
    import b3d.chisight.dense.likelihoods.simple_likelihood
    intermediate_likelihood_func = b3d.chisight.dense.likelihoods.simple_likelihood.simple_likelihood
    image_likelihood = b3d.chisight.dense.likelihoods.image_likelihood.make_image_likelihood(
        intermediate_likelihood_func,
        renderer
    )

    @genjax.gen
    def dense_multiobject_model(num_objects, meshes, likelihood_args):
        all_poses = []
        for i in range(num_objects.const):
            object_pose = uniform_pose(jnp.ones(3)*-100.0, jnp.ones(3)*100.0) @ f"object_pose_{i}"
            all_poses.append(object_pose)

        all_poses = Pose.stack_poses(all_poses)
        scene_mesh = Mesh.transform_and_merge_meshes(meshes, all_poses)
        image = image_likelihood(scene_mesh, likelihood_args) @ "image"
        return {"scene_mesh": scene_mesh, "image": image}

    def rerun_visualize_trace_t(trace, t):
        rr.set_time_sequence("time", t)
        observed_rgbd = trace.get_retval()["image"]

        intermediate_info = intermediate_likelihood_func(
            observed_rgbd, 
            trace.get_retval()["scene_mesh"],
            renderer,
            trace.get_args()[2]
        )
        rendered_rgbd = intermediate_info["rendered_rgbd"]

        rr.log("rgb", rr.Image(observed_rgbd[...,:3]))
        rr.log("rgb/rendering", rr.Image(rendered_rgbd[...,:3]))
        rr.log("rgb/depth", rr.DepthImage(observed_rgbd[...,3]))
        rr.log("rgb/depth_rendering", rr.DepthImage(rendered_rgbd[...,3]))
        info_string = f"# Score : {trace.get_score()}"
        
        for add in ["is_match", "is_mismatched", "color_match", "depth_match", "is_hypothesized"]:
            rr.log(f"/rgb/overlay/{add}", rr.DepthImage(intermediate_info[add] * 1.0))
        rr.log("/info", rr.TextDocument(info_string))

    importance_jit = jax.jit(dense_multiobject_model.importance)


    initial_camera_pose = all_data[0]["camera_pose"]
    initial_object_poses = all_data[0]["object_poses"]


    likelihood_args= {
        "inlier_score": 20.0,
        "color_tolerance": 20.0,
        "depth_tolerance": 0.01,
        "outlier_prob": 0.000001,
        "multiplier": 10000.0,
        "bounds": jnp.array([70.0, 45.0, 45.0, 0.005]),
        "variances" : jnp.zeros(4)
    }


    for IDX in range(len(initial_object_poses)):
        print(f"Object {IDX + 1} out of {len(initial_object_poses)}")

        pose = camera_pose.inv() @ all_data[0]["object_poses"][IDX]
        choicemap = genjax.ChoiceMap.d(
            dict(
                [
                    ("object_pose_0",  pose),
                    ("image", 
                        b3d.utils.resize_image(
                            all_data[0]["rgbd"], renderer.height, renderer.width
                        )
                    )
                ]
            )
        )

        trace, _ = importance_jit(
            jax.random.PRNGKey(2),
            choicemap,
            (Pytree.const(1), [meshes[IDX]], likelihood_args),
        )
        trace0 = trace
        # rerun_visualize_trace_t(trace, 0)
        print(trace.get_score())
        key = jax.random.PRNGKey(100)


        trace = trace0
        tracking_results = {}
        for t in tqdm(range(len(all_data))):
        # for t in tqdm(range(10)):
            trace = b3d.update_choices_jit(trace, jax.random.PRNGKey(0), ("image","pose_prior"),
                b3d.utils.resize_image(all_data[t]["rgbd"], renderer.height, renderer.width),
                trace.get_choices()["object_pose_0"]
            )
            saved_trace = trace
            potential_traces = []
            for var in [0.02, 0.01, 0.005]:
                trace = saved_trace
                trace, key = gvmf_and_select_best_move(trace, key, var, 700.0, Pytree.const("object_pose_0"), 700)
                trace, key = gvmf_and_select_best_move(trace, key, var, 700.0, Pytree.const("object_pose_0"), 700)
                trace, key = gvmf_and_select_best_move(trace, key, var, 700.0, Pytree.const("object_pose_0"), 700)
                trace, key = gvmf_and_select_best_move(trace, key, var, 700.0, Pytree.const("object_pose_0"), 700)
                potential_traces.append(trace)
            scores = jnp.array([t.get_score() for t in potential_traces])
            trace = potential_traces[scores.argmax()]
            tracking_results[t] = trace
            # print(t, trace.get_score())
            # rerun_visualize_trace_t(trace, t)
        
        inferred_poses = Pose.stack_poses(
            [tracking_results[t].get_choices()["object_pose_0"] for t in range(len(all_data))]
        )
        jnp.save(f"SCENE_{scene_id}_OBJECT_INDEX_{IDX}_POSES.npy", inferred_poses)

        trace = tracking_results[len(all_data) - 1]
        intermediate_info = intermediate_likelihood_func(
            trace.get_choices()["image"], 
            trace.get_retval()["scene_mesh"],
            renderer,
            trace.get_args()[2]
        )
        rendered_rgbd = intermediate_info["rendered_rgbd"]


        a = b3d.viz_rgb(
            trace.get_choices()["image"][...,:3],
        )
        b = b3d.viz_rgb(
            rendered_rgbd[...,:3],
        )
        b3d.multi_panel(
            [
                a,b, b3d.overlay_image(a,b)
            ]
        ).save(f"SCENE_{scene_id}_OBJECT_INDEX_{IDX}_POSES.png")