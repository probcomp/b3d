import genjax
import jax
import jax.numpy as jnp
import rerun as rr
import rerun.blueprint as rrb
from genjax import ChoiceMapBuilder as C

import b3d
from b3d.chisight.gen3d.image_kernel import PixelsPointsAssociation

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
    color_for_each_latent_point = (
        color_kernel.vmap()(previous_state["colors"]) @ "colors"
    )
    visibility_prob_for_each_latent_point = (
        visibility_prob_kernel.vmap()(previous_state["visibility_prob"])
        @ "visibility_prob"
    )
    depth_nonreturn_prob_for_each_latent_point = (
        depth_nonreturn_prob_kernel.vmap()(previous_state["depth_nonreturn_prob"])
        @ "depth_nonreturn_prob"
    )
    global_depth_scale = (
        depth_scale_kernel(previous_state["depth_scale"]) @ "depth_scale"
    )
    global_color_scale = (
        color_scale_kernel(previous_state["color_scale"]) @ "color_scale"
    )

    new_state = {
        "pose": pose,
        "colors": color_for_each_latent_point,
        "visibility_prob": visibility_prob_for_each_latent_point,
        "depth_nonreturn_prob": depth_nonreturn_prob_for_each_latent_point,
        "depth_scale": global_depth_scale,
        "color_scale": global_color_scale,
    }

    if "image_kernel" not in hyperparams:
        rgbd = None
    else:
        rgbd = hyperparams["image_kernel"](new_state, hyperparams) @ "rgbd"

    return {
        "new_state": new_state,
        "rgbd": rgbd,
    }


### Helpers ###
def make_colors_choicemap(colors):
    return jax.vmap(lambda idx: C["colors", idx].set(colors[idx]))(
        jnp.arange(len(colors))
    )


def make_visibility_prob_choicemap(visibility_prob):
    return jax.vmap(lambda idx: C["visibility_prob", idx].set(visibility_prob[idx]))(
        jnp.arange(len(visibility_prob))
    )


def make_depth_nonreturn_prob_choicemap(depth_nonreturn_prob):
    return jax.vmap(
        lambda idx: C["depth_nonreturn_prob", idx].set(depth_nonreturn_prob[idx])
    )(jnp.arange(len(depth_nonreturn_prob)))


def get_hypers(trace):
    return trace.get_args()[0]


def get_prev_state(trace):
    return trace.get_args()[1]


def get_new_state(trace):
    return trace.get_retval()["new_state"]


def get_n_vertices(trace):
    return get_hypers(trace)["vertices"].shape[0]


def get_observed_rgbd(trace):
    return trace.get_retval()["rgbd"]


### Visualization Code ###


def viz_trace(
    trace,
    t=0,
    ground_truth_vertices=None,
    ground_truth_pose=None,
    log_blueprint=True,
):
    b3d.rr_set_time(t)
    hyperparams, _ = trace.get_args()
    new_state = trace.get_retval()["new_state"]

    pose = new_state["pose"]
    colors = new_state["colors"]
    visibility_prob = new_state["visibility_prob"]
    depth_nonreturn_prob = new_state["depth_nonreturn_prob"]

    vertices = hyperparams["vertices"]
    b3d.rr_log_cloud(
        vertices[visibility_prob > 0.1],
        "object/model",
        colors[visibility_prob > 0.1],
    )
    b3d.rr_log_cloud(
        vertices,
        "object/visibility_prob",
        jnp.array([[1.0, 0.0, 0.0]]) * visibility_prob[..., None],
    )
    b3d.rr_log_cloud(
        vertices[visibility_prob > 0.1],
        "object/depth_nonreturn_prob",
        (jnp.array([[0.0, 1.0, 0.0]]) * depth_nonreturn_prob[..., None])[
            visibility_prob > 0.1
        ],
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

    # rr.log("color_scale", rr.Scalar(new_state["color_scale"]))
    # rr.log("depth_scale", rr.Scalar(new_state["depth_scale"]))

    vertices_transformed = pose.apply(vertices)
    b3d.rr_log_cloud(
        vertices_transformed,
        "scene/model",
        colors,
    )

    output = trace.get_retval()
    if output["rgbd"] is not None:
        observed_rgbd = output["rgbd"]
        b3d.rr_log_rgb(observed_rgbd[..., :3], "image")
        b3d.rr_log_rgb(observed_rgbd[..., :3], "image/rgb/observed")
        b3d.rr_log_depth(observed_rgbd[..., 3], "image/depth/observed")

        pixel_point_association = PixelsPointsAssociation.from_points_and_intrinsics(
            vertices_transformed,
            hyperparams["intrinsics"],
        )
        pixel_latent_rgb = jnp.clip(
            pixel_point_association.get_pixel_attributes(new_state["colors"]), 0.0, 1.0
        )
        pixel_latent_depth = jnp.clip(
            pixel_point_association.get_pixel_attributes(vertices_transformed[..., 2]),
            0.0,
            10.0,
        )

        b3d.rr_log_rgb(pixel_latent_rgb, "image/rgb/latent")
        b3d.rr_log_depth(pixel_latent_depth, "image/depth/latent")

        # TODO: should we add in a way to visualize a noise-free projection
        # of the points to the camera plane?

        fx, fy, cx, cy = (
            hyperparams["intrinsics"]["fx"],
            hyperparams["intrinsics"]["fy"],
            hyperparams["intrinsics"]["cx"],
            hyperparams["intrinsics"]["cy"],
        )
        b3d.rr_log_cloud(
            b3d.xyz_from_depth(
                output["rgbd"][..., 3],
                fx,
                fy,
                cx,
                cy,
            ),
            "scene/observed",
            output["rgbd"][..., :3].reshape(-1, 3),
        )

    if ground_truth_vertices is not None:
        b3d.rr_log_cloud(
            trace.get_choices()["pose"].apply(ground_truth_vertices),
            "scene/full_object_model",
        )

        if ground_truth_pose:
            b3d.rr_log_cloud(
                ground_truth_pose.apply(ground_truth_vertices),
                "scene/ground_truth_object_mesh",
            )

            b3d.rr_log_pose(ground_truth_pose, "scene/ground_truth_pose")
            b3d.rr_log_pose(trace.get_choices()["pose"], "scene/inferred_pose")

    if not b3d.get_blueprint_logged() and log_blueprint:
        rr.send_blueprint(get_blueprint())
        b3d.set_blueprint_logged(True)


def get_blueprint():
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="scene/"),
                rrb.Horizontal(
                    rrb.Spatial2DView(origin="image/rgb/"),
                    rrb.Spatial2DView(origin="image/depth/"),
                ),
            ),
            rrb.Horizontal(
                rrb.Spatial3DView(origin="object/model"),
                rrb.Spatial3DView(origin="object/visibility_prob"),
                rrb.Spatial3DView(origin="object/depth_nonreturn_prob"),
                rrb.TextDocumentView(origin="info"),
            ),
        )
    )
