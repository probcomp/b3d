import genjax
import jax
import jax.numpy as jnp
import rerun as rr
from genjax import Pytree

import b3d
import b3d.chisight.dense.likelihoods.image_likelihood
from b3d import Mesh, Pose
from b3d.modeling_utils import uniform_pose, uniform_scale


def make_dense_multiobject_model(renderer, likelihood_func, sample_func=None):
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
        object_ids,
        meshes,
        likelihood_args,
    ):
        blur = genjax.uniform(0.0001, 100000.0) @ "blur"
        likelihood_args["blur"] = blur

        all_poses = []
        scaled_meshes = []
        for o_id, mesh_composite in zip(object_ids.unwrap(), meshes):
            object_pose = (
                uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0)
                @ f"object_pose_{o_id}"
            )
            top = 0.0
            all_comp_poses = []
            all_comp_meshes = []
            for i, component in enumerate(mesh_composite):
                object_scale = (
                    uniform_scale(jnp.ones(3) * 0.01, jnp.ones(3) * 10.0)
                    @ f"object_scale_{o_id}_{i}"
                )
                scaled_comp = component.scale(object_scale)
                all_comp_meshes.append(scaled_comp)
                pose = Pose.from_translation(jnp.array([0.0, top, 0.0]))
                all_comp_poses.append(pose)
                top += scaled_comp.vertices[:, 1].max()
            merged_mesh = Mesh.transform_and_merge_meshes(
                all_comp_meshes, all_comp_poses
            )
            scaled_meshes.append(merged_mesh)
            all_poses.append(object_pose)
        all_poses = Pose.stack_poses(all_poses)

        camera_pose = (
            uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ "camera_pose"
        )

        scene_mesh = Mesh.transform_and_merge_meshes(
            scaled_meshes, all_poses
        ).transform(camera_pose.inv())

        likelihood_args["scene_mesh"] = [
            Mesh.transform_mesh(mesh, pose)
            for mesh, pose in zip(scaled_meshes, all_poses)
        ]

        depth_noise_variance = genjax.uniform(0.0001, 100000.0) @ "depth_noise_variance"
        likelihood_args["depth_noise_variance"] = depth_noise_variance
        color_noise_variance = genjax.uniform(0.0001, 100000.0) @ "color_noise_variance"
        likelihood_args["color_noise_variance"] = color_noise_variance

        outlier_probability = genjax.uniform(0.01, 1.0) @ "outlier_probability"
        likelihood_args["outlier_probability"] = outlier_probability

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

            likelihood_args["latent_rgbd"] = jnp.flip(latent_rgbd, 1)
            likelihood_args["rasterize_results"] = rasterize_results

        image = image_likelihood(likelihood_args) @ "rgbd"
        return {
            "likelihood_args": likelihood_args,
            "rgbd": image,
        }

    @jax.jit
    def info_from_trace(trace):
        return likelihood_func(
            trace.get_choices()["rgbd"],
            trace.get_retval()["likelihood_args"],
        )

    def viz_trace(trace, t=0, cloud=False):
        info = info_from_trace(trace)
        b3d.utils.rr_set_time(t)
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

    return dense_multiobject_model, viz_trace, info_from_trace