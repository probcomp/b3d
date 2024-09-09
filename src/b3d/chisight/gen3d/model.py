import genjax
import jax.numpy as jnp
import rerun as rr

import b3d

# TODOs
# 1. Tests of drift kernels
# 2. Test of choicemap creation, and model updating


@genjax.gen
def dynamic_object_generative_model(hyperparams, previous_state):
    hyperparams["vertices"]

    pose_kernel = hyperparams["pose_kernel"]
    color_kernel = hyperparams["color_kernel"]
    visibility_prob_kernel = hyperparams["visibility_prob_kernel"]
    depth_nonreturn_prob_kernel = hyperparams["depth_nonreturn_prob_kernel"]
    depth_scale_kernel = hyperparams["depth_scale_kernel"]
    color_scale_kernel = hyperparams["color_scale_kernel"]

    pose = pose_kernel(previous_state["pose"]) @ "pose"
    colors = color_kernel.vmap()(previous_state["colors"]) @ "colors"
    visibility_prob = (
        visibility_prob_kernel.vmap()(previous_state["visibility_prob"])
        @ "visibility_prob"
    )
    depth_nonreturn_prob = (
        depth_nonreturn_prob_kernel.vmap()(previous_state["depth_nonreturn_prob"])
        @ "depth_nonreturn_prob"
    )
    depth_scale = depth_scale_kernel(previous_state["depth_scale"]) @ "depth_scale"
    color_scale = color_scale_kernel(previous_state["color_scale"]) @ "color_scale"

    new_state = {
        "pose": pose,
        "colors": colors,
        "visibility_prob": visibility_prob,
        "depth_nonreturn_prob": depth_nonreturn_prob,
        "depth_scale": depth_scale,
        "color_scale": color_scale,
    }

    rgbd = None
    likelihood_args = None
    # if not ("image_likelihood" in hyperparams):

    # else:
    #     likelihood_args = {
    #         "fx": hyperparams["fx"],
    #         "fy": hyperparams["fy"],
    #         "cx": hyperparams["cx"],
    #         "cy": hyperparams["cy"],
    #         "image_width": hyperparams["image_width"],
    #         "image_height": hyperparams["image_height"],
    #         "vertices": hyperparams["vertices"],
    #         "colors": colors,
    #         "pose": pose,
    #         "color_scale": color_scale,
    #         "depth_scale": depth_scale,
    #         "visibility": visibility_prob,
    #         "depth_nonreturn": depth_nonreturn_prob,
    #     }
    #     rgbd = hyperparams["image_likelihood"](likelihood_args) @ "rgbd"

    return {
        "new_state": new_state,
        "rgbd": rgbd,
        "likelihood_args": likelihood_args,
    }


### Viz ###
def viz_trace(trace, t=0, ground_truth_vertices=None, ground_truth_pose=None):
    b3d.rr_set_time(t)
    hyperparams, _ = trace.get_args()
    new_state = trace.get_retval()["new_state"]

    pose = new_state["pose"]
    colors = new_state["colors"]
    visibility_prob = new_state["visibility_prob"]
    depth_nonreturn_prob = new_state["depth_nonreturn_prob"]

    vertices = hyperparams["vertices"]
    b3d.rr_log_cloud(
        vertices,
        "object/model",
        colors,
    )
    b3d.rr_log_cloud(
        vertices,
        "object/visibility_prob",
        jnp.array([[1.0, 0.0, 0.0]]) * visibility_prob[..., None],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/depth_nonreturn_prob",
        jnp.array([[0.0, 1.0, 0.0]]) * depth_nonreturn_prob[..., None],
    )

    rr.log(
        "info",
        rr.TextDocument(
            f"""
            depth_scale: {new_state["depth_scale"]}
            color_scale: {new_state["color_scale"]}
            """.strip(),
            media_type=rr.MediaType.MARKDOWN,
        ),
    )

    vertices_transformed = pose.apply(vertices)
    b3d.rr_log_cloud(
        vertices_transformed,
        "scene/model",
        colors,
    )

    # output = trace.get_retval()
    # if output["rgbd"] is not None:
    #     info = info_from_trace(trace)
    #     b3d.rr_log_rgb(output["rgbd"][..., :3], "image")
    #     b3d.rr_log_rgb(output["rgbd"][..., :3], "image/rgb/observed")
    #     b3d.rr_log_depth(output["rgbd"][..., 3], "image/depth/observed")

    #     latent_rgbd = info["latent_rgbd"]
    #     b3d.rr_log_rgb(latent_rgbd[..., :3], "image/rgb/latent")
    #     b3d.rr_log_depth(latent_rgbd[..., 3], "image/depth/latent")

    #     likelihood_args = trace.get_retval()["likelihood_args"]
    #     fx, fy, cx, cy = (
    #         likelihood_args["fx"],
    #         likelihood_args["fy"],
    #         likelihood_args["cx"],
    #         likelihood_args["cy"],
    #     )
    #     b3d.rr_log_cloud(
    #         b3d.xyz_from_depth(
    #             output["rgbd"][..., 3],
    #             fx,
    #             fy,
    #             cx,
    #             cy,
    #         ),
    #         "scene/observed",
    #         output["rgbd"][..., :3].reshape(-1, 3),
    #     )

    # if ground_truth_vertices is not None:
    #     b3d.rr_log_cloud(
    #         trace.get_choices()["pose"].apply(ground_truth_vertices),
    #         "scene/full_object_model",
    #     )

    #     if ground_truth_pose:
    #         b3d.rr_log_cloud(
    #             ground_truth_pose.apply(ground_truth_vertices),
    #             "scene/ground_truth_object_mesh",
    #         )
