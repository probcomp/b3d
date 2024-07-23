from b3d.modeling_utils import uniform_discrete, uniform_pose, gaussian_vmf
import genjax
import b3d
from b3d import Pose, Mesh
import jax
import jax.numpy as jnp
import b3d.chisight.dense.likelihoods.image_likelihood
from genjax import Pytree
import rerun as rr


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

        outlier_probability = genjax.uniform(0.0001, 0.9999) @ "outlier_probability"
        lightness_variance = genjax.uniform(0.0001, 1.0) @ "lightness_variance"
        color_variance = genjax.uniform(0.0001, 1.0) @ "color_variance"
        depth_variance = genjax.uniform(0.0001, 1.0) @ "depth_variance"

        likelihood_args["outlier_probability"] = outlier_probability
        likelihood_args["lightness_variance"] = lightness_variance
        likelihood_args["color_variance"] = color_variance
        likelihood_args["depth_variance"] = depth_variance

        all_poses = []
        for i in range(num_objects.const):
            object_pose = (
                uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0)
                @ f"object_pose_{i}"
            )
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

    def viz_trace(trace, t=0):
        rr.set_time_sequence("time", t)
        intermediate_info = likelihood_func(
            trace.get_choices()["rgbd"],
            trace.get_retval()["latent_rgbd"],
            trace.get_retval()["likelihood_args"],
        )

        rr.log("rgb", rr.Image(trace.get_choices()["rgbd"][..., :3]))
        rr.log("rgb/depth/observed", rr.DepthImage(trace.get_choices()["rgbd"][..., 3]))
        rr.log(
            "rgb/depth/latent", rr.DepthImage(trace.get_retval()["latent_rgbd"][..., 3])
        )
        rr.log("rgb/latent", rr.Image(trace.get_retval()["latent_rgbd"][..., :3]))
        # rr.log("rgb/is_match", rr.DepthImage(intermediate_info["is_match"] * 1.0))
        # rr.log("rgb/color_match", rr.DepthImage(intermediate_info["color_match"] * 1.0))

    return dense_multiobject_model, viz_trace
