import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

import b3d
import b3d.chisight.dense.likelihoods.image_likelihood
from b3d import Mesh, Pose
from b3d.modeling_utils import uniform_pose


def make_dense_multiobject_model(renderer, likelihood_func, sample_func=None):
    if sample_func is None:

        def f(key, rendered_rgbd, likelihood_args):
            return rendered_rgbd

        sample_func = f

    @Pytree.dataclass
    class ImageLikelihood(genjax.ExactDensity):
        def sample(self, key, rendered_rgbd, likelihood_args):
            return sample_func(key, rendered_rgbd, likelihood_args)

        def logpdf(self, observed_rgbd, rendered_rgbd, likelihood_args):
            return likelihood_func(observed_rgbd, rendered_rgbd, likelihood_args)[
                "score"
            ]

    image_likelihood = ImageLikelihood()

    @genjax.gen
    def dense_multiobject_model(args_dict):
        meshes = args_dict["meshes"]
        likelihood_args = args_dict["likelihood_args"]
        num_objects = args_dict["num_objects"]

        blur = genjax.uniform(0.0001, 100000.0) @ "blur"
        likelihood_args["blur"] = blur

        all_poses = []
        for i in range(num_objects.const):
            object_pose = (
                uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0)
                @ f"object_pose_{i}"
            )

            outlier_probability = genjax.uniform(0.0, 1.0) @ f"outlier_probability_{i}"
            lightness_variance = genjax.uniform(0.0001, 1.0) @ f"lightness_variance_{i}"
            color_variance = genjax.uniform(0.0001, 100.0) @ f"color_variance_{i}"
            depth_variance = genjax.uniform(0.0001, 100.0) @ f"depth_variance_{i}"

            likelihood_args[f"outlier_probability_{i}"] = outlier_probability
            likelihood_args[f"lightness_variance_{i}"] = lightness_variance
            likelihood_args[f"color_variance_{i}"] = color_variance
            likelihood_args[f"depth_variance_{i}"] = depth_variance

            all_poses.append(object_pose)
        all_poses = Pose.stack_poses(all_poses)

        camera_pose = (
            uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ "camera_pose"
        )

        scene_mesh = Mesh.transform_and_merge_meshes(meshes, all_poses).transform(
            camera_pose.inv()
        )
        latent_rgbd = renderer.render_rgbd_from_mesh(scene_mesh)

        image = image_likelihood(latent_rgbd, likelihood_args) @ "rgbd"
        return {
            "likelihood_args": likelihood_args,
            "scene_mesh": scene_mesh,
            "latent_rgbd": latent_rgbd,
            "rgbd": image,
        }

    @jax.jit
    def info_from_trace(trace):
        return likelihood_func(
            trace.get_choices()["rgbd"],
            trace.get_retval()["latent_rgbd"],
            trace.get_retval()["likelihood_args"],
        )

    def viz_trace(trace, t=0):
        rr.set_time_sequence("time", t)
        likelihood_args = trace.get_retval()["likelihood_args"]
        fx, fy, cx, cy = (
            likelihood_args["fx"],
            likelihood_args["fy"],
            likelihood_args["cx"],
            likelihood_args["cy"],
        )

        info = info_from_trace(trace)
        rr.log("image", rr.Image(trace.get_choices()["rgbd"][..., :3]))
        b3d.rr_log_rgb(trace.get_choices()["rgbd"][..., :3], "image/rgb/observed")
        b3d.rr_log_rgb(trace.get_retval()["latent_rgbd"][..., :3], "image/rgb/latent")
        b3d.rr_log_depth(trace.get_choices()["rgbd"][..., 3], "image/depth/observed")
        b3d.rr_log_depth(
            trace.get_retval()["latent_rgbd"][..., 3], "image/depth/latent"
        )
        rr.log("image/overlay/pixelwise_score", rr.DepthImage(info["pixelwise_score"]))

        b3d.rr_log_cloud(
            b3d.xyz_from_depth(
                trace.get_retval()["latent_rgbd"][..., 3],
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

    return dense_multiobject_model, viz_trace, info_from_trace
