from b3d.modeling_utils import uniform_discrete, uniform_pose, gaussian_vmf
import genjax
import b3d
from b3d import Pose, Mesh
import jax
import jax.numpy as jnp
import b3d.chisight.dense.likelihoods.image_likelihood


def make_dense_multiobject_model(renderer, image_likelihood):
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

    return dense_multiobject_model
