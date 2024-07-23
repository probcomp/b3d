import genjax
import jax.numpy as jnp
import b3d
from b3d.pose import Pose
import rerun as rr
from collections import namedtuple
from b3d.modeling_utils import uniform_discrete, uniform_pose

ModelArgs = namedtuple(
    "ModelArgs",
    [
        "color_tolerance",
        "depth_tolerance",
        "inlier_score",
        "outlier_prob",
        "color_multiplier",
        "depth_multiplier",
    ],
)


def get_rgb_depth_inliers_from_trace(trace):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    model_args = trace.get_args()[1]
    return get_rgb_depth_inliers_from_observed_rendered_args(
        observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args
    )


def get_rgb_depth_inliers_from_observed_rendered_args(
    observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args
):
    observed_lab = b3d.colors.rgb_to_lab(observed_rgb)
    rendered_lab = b3d.colors.rgb_to_lab(rendered_rgb)
    error = jnp.linalg.norm(
        observed_lab[..., 1:3] - rendered_lab[..., 1:3], axis=-1
    ) + jnp.abs(observed_lab[..., 0] - rendered_lab[..., 0])

    valid_data_mask = rendered_rgb.sum(-1) != 0.0

    color_inliers = (error < model_args.color_tolerance) * valid_data_mask
    depth_inliers = (
        jnp.abs(observed_depth - rendered_depth) < model_args.depth_tolerance
    ) * valid_data_mask
    inliers = color_inliers * depth_inliers
    outliers = jnp.logical_not(inliers) * valid_data_mask
    undecided = jnp.logical_not(inliers) * jnp.logical_not(outliers)
    return (inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask)


@genjax.Pytree.dataclass
class RGBDSensorModel(genjax.ExactDensity):
    def sample(self, key, rendered_rgb, rendered_depth, model_args, fx, fy, far):
        return (rendered_rgb, rendered_depth)

    def logpdf(self, observed, rendered_rgb, rendered_depth, model_args, fx, fy, far):
        observed_rgb, observed_depth = observed

        inliers, color_inliers, depth_inliers, outliers, undecided, valid_data_mask = (
            get_rgb_depth_inliers_from_observed_rendered_args(
                observed_rgb, rendered_rgb, observed_depth, rendered_depth, model_args
            )
        )

        inlier_score = model_args.inlier_score
        outlier_prob = model_args.outlier_prob
        multiplier = model_args.color_multiplier

        corrected_depth = rendered_depth + (rendered_depth == 0.0) * far
        areas = (corrected_depth / fx) * (corrected_depth / fy)

        return (
            jnp.log(
                # This is leaving out a 1/A (which does depend upon the scene)
                inlier_score * jnp.sum(inliers * areas)
                + 1.0 * jnp.sum(undecided * areas)
                + outlier_prob * jnp.sum(outliers * areas)
            )
            * multiplier
        )


rgbd_sensor_model = RGBDSensorModel()


def model_multiobject_gl_factory(renderer, image_likelihood=rgbd_sensor_model):
    @genjax.gen
    def model(
        _num_obj_arr,  # new
        model_args,
        object_library,
    ):
        object_poses = Pose(jnp.zeros((0, 3)), jnp.zeros((0, 4)))
        object_indices = jnp.empty((0,), dtype=int)
        camera_pose = (
            uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0) @ "camera_pose"
        )

        for i in range(_num_obj_arr.shape[0]):
            object_identity = (
                uniform_discrete(jnp.arange(-1, len(object_library.ranges)))
                @ f"object_{i}"
            )
            object_indices = jnp.concatenate(
                (object_indices, jnp.array([object_identity]))
            )

            object_pose = (
                uniform_pose(jnp.ones(3) * -100.0, jnp.ones(3) * 100.0)
                @ f"object_pose_{i}"
            )
            object_poses = Pose.concatenate_poses(
                [object_poses, camera_pose.inv() @ object_pose[None, ...]]
            )

        rendered_rgb, rendered_depth = renderer.render_attribute(
            object_poses,
            object_library.vertices,
            object_library.faces,
            object_library.ranges[object_indices]
            * (object_indices >= 0).reshape(-1, 1),
            object_library.attributes,
        )
        observed_rgb, observed_depth = (
            image_likelihood(
                rendered_rgb, rendered_depth, model_args, renderer.fx, renderer.fy, 1.0
            )
            @ "observed_rgb_depth"
        )
        return (observed_rgb, rendered_rgb), (observed_depth, rendered_depth)

    return model


def get_rendered_rgb_depth_from_trace(trace):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval
    return (rendered_rgb, rendered_depth)


def get_poses_from_trace(trace):
    return Pose.stack_poses(
        [trace[f"object_pose_{i}"] for i in range(len(trace.get_args()[0]))]
    )


def get_object_ids_from_trace(trace):
    return jnp.array([trace[f"object_{i}"] for i in range(len(trace.get_args()[0]))])


def rerun_visualize_trace_t(trace, t, modes=["rgb", "depth", "inliers"]):
    (observed_rgb, rendered_rgb), (observed_depth, rendered_depth) = trace.get_retval()
    rr.set_time_sequence("frame", t)

    if "rgb" in modes:
        rr.log("/image", rr.Image(observed_rgb))
        rr.log("/image/rgb_rendering", rr.Image(rendered_rgb))

    if "depth" in modes:
        rr.log("/image/depth/", rr.DepthImage(observed_depth))
        rr.log("/image/depth/rendering", rr.DepthImage(rendered_depth))

    info_string = f"# Score : {trace.get_score()}"

    if "inliers" in modes:
        (
            inliers,
            color_inliers,
            depth_inliers,
            outliers,
            undecided,
            valid_data_mask,
        ) = get_rgb_depth_inliers_from_trace(trace)
        rr.log("/image/overlay/inliers", rr.DepthImage(inliers * 1.0))
        rr.log("/image/overlay/outliers", rr.DepthImage(outliers * 1.0))
        rr.log("/image/overlay/undecided", rr.DepthImage(undecided * 1.0))
        info_string += f"\n # Inliers : {jnp.sum(inliers)}"
        info_string += f"\n # Outliers : {jnp.sum(outliers)}"
        info_string += f"\n # Undecided : {jnp.sum(undecided)}"
    rr.log("/info", rr.TextDocument(info_string))

    if "3d" in modes:
        poses = get_poses_from_trace(trace)
        ids = get_object_ids_from_trace(trace)
        object_library = trace.get_args()[2]
        for idx, (i, pose) in enumerate(zip(ids, poses)):
            mask = object_library.vertex_index_to_object == i
            vertices = object_library.vertices[mask]
            attributes = object_library.attributes[mask]
            rr.log(
                f"object_{idx}",
                rr.Points3D(
                    pose.apply(vertices),
                    colors=(attributes * 255).astype(jnp.uint8),
                ),
            )
