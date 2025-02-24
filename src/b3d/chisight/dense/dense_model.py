import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree
import warp as wp
from warp.jax_experimental.ffi import jax_callable

import b3d
import b3d.chisight.dense.likelihoods.image_likelihood
from b3d import Mesh, Pose


def get_hypers(trace):
    return trace.get_args()[0]


def get_prev_state(trace):
    return trace.get_args()[1]


def get_new_state(trace):
    return trace.get_retval()["new_state"]



# @wp.kernel
# def scale_kernel(a: wp.array(dtype=float), s: wp.array(dtype=wp.transform), output: wp.array(dtype=float)):
#     tid = wp.tid()
#     output[tid] = a[tid]


# def example_func(
#     # inputs
#     a: wp.array(dtype=float),
#     s: wp.array(dtype=wp.transform),
#     # outputs
#     b: wp.array(dtype=float),
# ):
#     # launch multiple kernels
#     wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[b])

# jax_func = jax_callable(example_func, num_outputs=1)


# @jax.jit
# def my_jax_func(model, state):
#     jax_func(model.body_mass, state.body_q)


@wp.kernel
def scale_kernel(a: wp.array(dtype=float), s: float, output: wp.array(dtype=float)):
    tid = wp.tid()
    output[tid] = a[tid] * s

def example_func(
    # inputs
    a: wp.array(dtype=float),
    s: float,
    # outputs
    b: wp.array(dtype=float),
):
    # launch multiple kernels
    if s < 0:
        jax.debug.print("launching")
        wp.launch(scale_kernel, dim=a.shape, inputs=[a, s], outputs=[b])
    else:
        jax.debug.print("not launching")
        b = a

jax_func = jax_callable(example_func)

@jax.jit
def my_jax_func(pose):
    s = 2.0
    c = jax_func(pose.pos, s)
    return c

def make_dense_multiobject_dynamics_model(renderer, likelihood_func, sample_func=None):
    if sample_func is None:

        def f(key, likelihood_args):
            return jnp.zeros(
                (
                    likelihood_args["image_height"].unwrap(),
                    likelihood_args["image_width"].unwrap(),
                    4,
                )
            )

        sample_func = f

    @Pytree.dataclass
    class ImageLikelihood(genjax.ExactDensity):
        def sample(self, key, likelihood_args):
            return sample_func(key, likelihood_args)

        def logpdf(self, observed_rgbd, likelihood_args):
            return likelihood_func(observed_rgbd, likelihood_args)["score"]

    image_likelihood = ImageLikelihood()

    @genjax.gen
    def dense_multiobject_model(
        hyperparams,
        previous_state,
    ):
        background = hyperparams["background"][previous_state["t"]]
        meshes = hyperparams["meshes"].values()
        likelihood_args = hyperparams["likelihood_args"]
        object_ids = hyperparams["object_ids"]
        pose_kernel = hyperparams["pose_kernel"]

        # stepped_model, stepped_state = Physics.step(previous_info["prev_model"], previous_info["prev_state"])
        # c = my_jax_func(previous_state["prev_model"], previous_state["prev_state"])
        all_poses = {}
        # all_scales = {}
        # scaled_meshes = []
        for o_id in object_ids.unwrap():
            object_pose = (
                pose_kernel(previous_state[f"object_pose_{o_id}"])
                @ f"object_pose_{o_id}"
            )
            c = my_jax_func(previous_state[f"object_pose_{o_id}"])
            jax.debug.print("before {v}", v=previous_state[f"object_pose_{o_id}"].pos)
            jax.debug.print("after {v}", v=c)
            # # top = 0.0
            # all_comp_poses = [Pose.from_translation(jnp.array([0.0, 0, 0.0]))]
            # all_comp_meshes = mesh_composite
            # # for i, component in enumerate(mesh_composite):
            # #     object_scale = (
            # #         uniform_scale(jnp.ones(3) * 0.01, jnp.ones(3) * 10.0)
            # #         @ f"object_scale_{o_id}_{i}"
            # #     )
            # #     # all_scales[f"object_scale_{o_id}_{i}"] = object_scale
            # #     scaled_comp = component.scale(object_scale)
            # #     all_comp_meshes.append(scaled_comp)
            # #     pose = Pose.from_translation(jnp.array([0.0, top, 0.0]))
            # #     all_comp_poses.append(pose)
            # #     top += scaled_comp.vertices[:, 1].max()
            # merged_mesh = Mesh.transform_and_merge_meshes(
            #     all_comp_meshes, all_comp_poses
            # )
            # scaled_meshes.append(merged_mesh)
            all_poses[f"object_pose_{o_id}"] = object_pose

        camera_pose = hyperparams["camera_pose"]
        # (
        #     uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ "camera_pose"
        # )

        scene_mesh = Mesh.transform_and_merge_meshes(
            list(meshes), Pose.stack_poses(list(all_poses.values()))
        ).transform(camera_pose.inv())

        likelihood_args["scene_mesh"] = [
            Mesh.transform_mesh(mesh, pose)
            for mesh, pose in zip(
                meshes, Pose.stack_poses(list(all_poses.values()))
            )
        ]

        # depth_noise_variance = genjax.uniform(0.0001, 100000.0) @ "depth_noise_variance"
        # likelihood_args["depth_noise_variance"] = hyperparams["depth_noise_variance"]
        # color_noise_variance = genjax.uniform(0.0001, 100000.0) @ "color_noise_variance"
        # likelihood_args["color_noise_variance"] = hyperparams["color_noise_variance"]

        # outlier_probability = genjax.uniform(0.01, 1.0) @ "outlier_probability"
        # likelihood_args["outlier_probability"] = hyperparams["outlier_probability"]

        if renderer is not None:
            rasterize_results = renderer.rasterize(
                scene_mesh.vertices, scene_mesh.faces
            )
            latent_rgbd = renderer.interpolate(
                jnp.concatenate(
                    [scene_mesh.vertex_attributes, scene_mesh.vertices[..., -1:]],
                    axis=-1,
                ),
                rasterize_results,
                scene_mesh.faces,
            )

            # add distractor and occluders
            latent_rgbd = jnp.flip(latent_rgbd, 1)
            bg_rgb = background[..., :3]
            bg_d = background[..., 3:]
            latent_rgb = jnp.where(
                bg_rgb == jnp.array([jnp.inf, jnp.inf, jnp.inf]),
                latent_rgbd[..., 0:3],
                bg_rgb,
            )
            latent_d = jnp.minimum(
                jnp.where(latent_rgbd[..., 3:] == 0.0, 10, latent_rgbd[..., 3:]), bg_d
            )

            likelihood_args["latent_rgbd"] = jnp.concatenate(
                [latent_rgb, latent_d], axis=-1
            )
            likelihood_args["rasterize_results"] = rasterize_results

        image = image_likelihood(likelihood_args) @ "rgbd"
        return {
            "likelihood_args": likelihood_args,
            "rgbd": image,
            "new_state": all_poses | {"t": previous_state["t"] + 1},
        }

    @jax.jit
    def info_from_trace(trace):
        return likelihood_func(
            trace.get_choices()["rgbd"],
            trace.get_retval()["likelihood_args"],
        )

    def viz_trace(trace, t=0, cloud=True):
        info = info_from_trace(trace)
        b3d.utils.rr_set_time(t)
        likelihood_args = trace.get_retval()["likelihood_args"]
        fx, fy, cx, cy = (
            likelihood_args["fx"],
            likelihood_args["fy"],
            likelihood_args["cx"],
            likelihood_args["cy"],
        )

        rr.log("image", rr.Image(trace.get_choices()["rgbd"][..., :3]))
        b3d.rr_log_rgb(trace.get_choices()["rgbd"][..., :3], "image/rgb/observed")
        b3d.rr_log_rgb(info["latent_rgbd"][..., :3], "image/rgb/latent")
        b3d.rr_log_depth(trace.get_choices()["rgbd"][..., 3], "image/depth/observed")
        b3d.rr_log_depth(info["latent_rgbd"][..., 3], "image/depth/latent")
        rr.log("image/overlay/pixelwise_score", rr.DepthImage(info["pixelwise_score"]))

        if cloud:
            b3d.rr_log_cloud(
                b3d.xyz_from_depth(
                    info["latent_rgbd"][..., 3],
                    fx,
                    fy,
                    cx,
                    cy,
                ),
                "latent",
            )
            b3d.rr_log_cloud(
                b3d.xyz_from_depth(
                    trace.get_retval()["rgbd"][..., 3],
                    fx,
                    fy,
                    cx,
                    cy,
                ),
                "observed",
            )
        # rr.log("rgb/is_match", rr.DepthImage(intermediate_info["is_match"] * 1.0))
        # rr.log("rgb/color_match", rr.DepthImage(intermediate_info["color_match"] * 1.0))

    return dense_multiobject_model, viz_trace
