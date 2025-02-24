import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

import b3d
import b3d.chisight.dense.likelihoods.image_likelihood
from b3d import Mesh, Pose
# from b3d.physics.physics_utils import step

def get_hypers(trace):
    return trace.get_args()[0]


def get_prev_state(trace):
    return trace.get_args()[1]


def get_new_state(trace):
    return trace.get_retval()["new_state"]


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
        previous_info,
    ):
        background = hyperparams["background"][previous_info["t"]]
        meshes = hyperparams["meshes"].values()
        likelihood_args = hyperparams["likelihood_args"]
        object_ids = hyperparams["object_ids"]
        pose_kernel = hyperparams["pose_kernel"]

        # stepped_model, stepped_state = step(previous_info["prev_model"], previous_info["prev_state"], hyperparams["sim_dt"])
        all_poses = {}
        for o_id in object_ids.unwrap():
            object_pose = (
                pose_kernel(previous_info[f"object_pose_{o_id}"])
                @ f"object_pose_{o_id}"
            )
            all_poses[f"object_pose_{o_id}"] = object_pose

        camera_pose = hyperparams["camera_pose"]
        scene_mesh = Mesh.transform_and_merge_meshes(
            list(meshes), Pose.stack_poses(list(all_poses.values()))
        ).transform(camera_pose.inv())

        likelihood_args["scene_mesh"] = [
            Mesh.transform_mesh(mesh, pose)
            for mesh, pose in zip(
                meshes, Pose.stack_poses(list(all_poses.values()))
            )
        ]

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
            "new_state": all_poses | {"t": previous_info["t"] + 1},
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
            
    return dense_multiobject_model, viz_trace
