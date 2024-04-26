#### Resolution invariance test for likelihoods
# Test likelihoods for invariance to the resolution of the image space.
# The inferred posterior distribution of an object's pose
# should NOT become peakier when the scene/observed images gain resolution.


import jax.numpy as jnp
import jax
import genjax
import matplotlib.pyplot as plt
import os
import trimesh
import b3d
from b3d import Pose
import rerun as rr
from tqdm import tqdm
import unittest


class UpsamplingRenderer(b3d.Renderer):
    """
    Renderer that upsamples rendered image to a desired resolution.
    Designed for image invariance resolution test, to express images that
    have more pixels but equal amount of "information"
    """

    def __init__(self, *init_args):
        super().__init__(*init_args)
        self.IMAGE_WIDTH = self.width
        self.IMAGE_HEIGHT = self.height

    def render_attribute(self, *render_inputs):  ## overload
        rgb, depth = super().render_attribute(*render_inputs)
        return (
            jax.image.resize(rgb, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT, 3), "nearest"),
            jax.image.resize(depth, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), "nearest"),
        )


class TestImgResolutionInvariance(unittest.TestCase):
    """
    Assert that the posterior over poses has same landscape (i.e. no significant change to variance)
    across changes in image resolutions.

    To debug, enable `rerun` (visualizes posterior traces) or `plot` (visualizes importance weights on grid)
    """

    def setUp(self, rerun=False, plot=False):  # TODO pass in mesh path
        self.rerun = rerun
        self.plot = plot

        ## load desired mesh and add to library
        MESH_PATH = os.path.join(
            b3d.get_root_path(), "assets/shared_data_bucket/025_mug/textured.obj"
        )
        mesh = trimesh.load(MESH_PATH)

        vertices = jnp.array(mesh.vertices) * 5.0
        vertices = vertices - vertices.mean(0)
        faces = jnp.array(mesh.faces)
        vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
        vertex_colors = jnp.array(mesh.visual.to_color().vertex_colors)[..., :3] / 255.0
        ranges = jnp.array([[0, len(faces)]])
        self.object_library = b3d.MeshLibrary.make_empty_library()
        self.object_library.add_object(vertices, faces, vertex_colors)
        print(f"{self.object_library.get_num_objects()} object(s) in library")

        #####
        # resolutions to test
        #####
        self.TEST_RESOLUTIONS = [
            (320, 320),
            (160, 160),
            (80, 80),
        ]

        ####
        # setup renderer
        ####
        RENDERER_IMAGE_WIDTH, RESOLUTION_IMAGE_HEIGHT = self.TEST_RESOLUTIONS[
            -1
        ]  # lowest resolution (will upsample for image reoslution varying)
        fx, fy, cx, cy, near, far = (
            RENDERER_IMAGE_WIDTH * 2,
            RESOLUTION_IMAGE_HEIGHT * 2,
            RENDERER_IMAGE_WIDTH / 2,
            RESOLUTION_IMAGE_HEIGHT / 2,
            0.01,
            10.0,
        )

        self.renderer = UpsamplingRenderer(
            RENDERER_IMAGE_WIDTH, RESOLUTION_IMAGE_HEIGHT, fx, fy, cx, cy, near, far
        )

    def test_peaky(self):
        """
        Test case 1:
            Object is positioned such that there is no uncertainty in the pose (i.e. z axis angle).
            We expect a posterior peaked near the GT pose.
        See _test_common() for the invariances asserted.
        """
        angle_var_bound = 1e-4
        self._test_common(-jnp.pi, angle_var_bound)

    def test_uncertain(self, rerun=False, plot=False):
        """
        Test case 2:
            Object is positioned such that there exists certainty
            in the pose (i.e. z axis angle) due to self-occlusion, etc.
            We expect a smoother posterior that increases in the region of the GT pose.
        See _test_common() for the invariances asserted.
        """
        angle_var_bound = 1e-3
        self._test_common(-jnp.pi * 0.55, angle_var_bound)

    def _test_common(self, gt_rotation_angle_z, angle_var_bound=1e-4):
        """
        The testing procedure common to all test cases.

        For a specified set of image resolutions (self.TEST_RESOLUTIONS),
        Given the GT z axis angle and a uniform gridding of poses, acquires
        (1) importance sampling scores from each element of the uniform grid and
        (2) posterior samples from the IS weights.


        For a given scene (i.e. the cam/obj poses held constant),
        asserts that, across varying image resolutions:
        - IS scores/likelihoods on grid remain the same.
        """
        test_resolutions = self.TEST_RESOLUTIONS
        #############
        # get posterior per resolution
        ##############
        object_id = 0
        color_error, depth_error = (40.0, 0.02)
        inlier_score, outlier_prob = (5.0, 0.00001)
        color_multiplier, depth_multiplier = (10000.0, 500.0)
        num_x_tr, num_y_tr, num_x_rot, num_z_rot = 11, 11, 5, 80

        samples_variances = []
        samples_means = []

        for IMAGE_WIDTH, IMAGE_HEIGHT in test_resolutions:
            print(f"========TESTING RESOLUTION ({IMAGE_WIDTH}, {IMAGE_HEIGHT})========")

            self.renderer.IMAGE_WIDTH = IMAGE_WIDTH
            self.renderer.IMAGE_HEIGHT = IMAGE_HEIGHT

            ###########
            # Setup test image (no self-occlusion)
            ###########
            camera_pose = Pose.from_position_and_target(
                jnp.array([0.0, 3.0, 0.0]),
                jnp.array([0.0, 0.0, 0.0]),
                up=jnp.array([0, 0, 1]),
            )
            _gt_translation = jnp.array([-0.005, 0.01, 0])
            _gt_rotation_angle_z = gt_rotation_angle_z
            _gt_rotation_z = b3d.Rot.from_euler(
                "z", _gt_rotation_angle_z, degrees=False
            ).quat
            gt_pose_cam = Pose(_gt_translation, _gt_rotation_z)  # camera frame pose

            gt_img, gt_depth = self.renderer.render_attribute(
                (camera_pose.inv() @ gt_pose_cam)[None, ...],
                self.object_library.vertices,
                self.object_library.faces,
                self.object_library.ranges[jnp.array([object_id])],
                self.object_library.attributes,
            )
            if self.plot:
                _, axes = plt.subplots(1, 1, figsize=(10, 10))
                axes.imshow(gt_img)
                plt.title(
                    f"GT image ({IMAGE_WIDTH}, {IMAGE_HEIGHT}):\ntr {_gt_translation}, z rotation (-0.55pi), x rotation (0.03pi)"
                )
                plt.savefig(
                    f"{IMAGE_HEIGHT}_{IMAGE_WIDTH}_gt_{gt_rotation_angle_z}.png"
                )

            samples, scores, pose_enums = self.get_gridding_posterior(
                camera_pose,
                gt_pose_cam,
                gt_img,
                gt_depth,
                color_error,
                depth_error,
                inlier_score,
                outlier_prob,
                color_multiplier,
                depth_multiplier,
                num_x_tr,
                num_y_tr,
                num_x_rot,
                num_z_rot,
            )

            if self.plot:
                f = self._generate_plot(
                    samples, scores, pose_enums, IMAGE_WIDTH, IMAGE_HEIGHT
                )
                f.savefig(
                    f"{IMAGE_WIDTH}_{IMAGE_HEIGHT}_viz_{gt_rotation_angle_z}.png",
                    bbox_inches="tight",
                )

            samples_variance = jnp.var(pose_enums[samples], axis=0)
            samples_variances.append(samples_variance)

        samples_variances, samples_means = (
            jnp.asarray(samples_variances),
            jnp.asarray(samples_means),
        )

        #############
        # Asserts on sample statistics
        #############

        # The variance of sampled angles does not vary significantly.
        #     (i.e. variances are similar across image resolution)
        assert jnp.allclose(
            samples_variances[0, 2], samples_variances[:, 2], atol=angle_var_bound
        )

    def get_gridding_posterior(
        self,
        camera_pose,
        gt_pose_cam,
        gt_img,
        gt_depth,
        color_error,
        depth_error,
        inlier_score,
        outlier_prob,
        color_multiplier,
        depth_multiplier,
        num_x_tr=11,
        num_y_tr=11,
        num_x_rot=5,
        num_z_rot=81,
    ):
        """
        Given the GT camera/obj poses and the object pose hypotheses,
        evaluates the importance sample weights, and
        acquires posterior samples from those scores.
        """

        model = b3d.model_multiobject_gl_factory(self.renderer)
        key = jax.random.PRNGKey(0)

        ## sampling grid
        cp_to_pose = lambda cp: Pose(
            jnp.array([cp[0], cp[1], 0.0]),
            b3d.Rot.from_rotvec(jnp.array([cp[2], 0.0, cp[3]])).as_quat(),
        )

        delta_cps = jnp.stack(
            jnp.meshgrid(
                jnp.linspace(-0.02, 0.02, num_x_tr),
                jnp.linspace(-0.02, 0.02, num_y_tr),
                # jnp.linspace(-jnp.pi/15, jnp.pi/15, num_x_rot),
                jnp.linspace(-jnp.pi, jnp.pi, num_z_rot),
            ),
            axis=-1,
        ).reshape(-1, 3)
        cp_delta_poses = jax.vmap(cp_to_pose)(delta_cps)
        print(f"{cp_delta_poses.shape[0]} enums")

        model_args = b3d.model.ModelArgs(
            color_error,
            depth_error,
            inlier_score,
            outlier_prob,
            color_multiplier,
            depth_multiplier,
        )
        arguments = (jnp.arange(1), model_args, self.object_library)

        ## init trace
        gt_trace, _ = model.importance(
            jax.random.PRNGKey(0),
            genjax.choice_map(
                {
                    "camera_pose": camera_pose,
                    "object_pose_0": gt_pose_cam,
                    "object_0": 0,
                    "observed_rgb_depth": (gt_img, gt_depth),
                }
            ),
            arguments,
        )
        b3d.rerun_visualize_trace_t(gt_trace, 0)

        ## get IS scores over the enum grid
        test_poses = gt_trace["object_pose_0"] @ cp_delta_poses
        test_poses_batches = test_poses.split(10)
        scores = jnp.concatenate(
            [
                b3d.enumerate_choices_get_scores_jit(
                    gt_trace, key, genjax.Pytree.const(["object_pose_0"]), poses
                )
                for poses in test_poses_batches
            ]
        )
        samples = jax.random.categorical(
            key, scores, shape=(500,)
        )  # samples from posterior (prior was a uniform)

        if self.rerun:
            ###########
            # rerun visualization
            ###########
            print("generating rerun vizs")

            alternate_camera_pose = Pose.from_position_and_target(
                jnp.array([0.01, 0.000, 1.5]), gt_pose_cam.pos
            )  # alt pose to see occluded side from above

            alternate_view_images, _ = self.renderer.render_attribute_many(
                (alternate_camera_pose.inv() @ test_poses[samples])[:, None, ...],
                self.object_library.vertices,
                self.object_library.faces,
                self.object_library.ranges[jnp.array([0])],
                self.object_library.attributes,
            )

            alternate_view_gt_image, _ = self.renderer.render_attribute(
                (alternate_camera_pose.inv() @ gt_pose_cam)[None, ...],
                self.object_library.vertices,
                self.object_library.faces,
                self.object_library.ranges[jnp.array([0])],
                self.object_library.attributes,
            )

            for t in tqdm(range(len(samples))):
                trace_ = b3d.update_choices_jit(
                    gt_trace,
                    key,
                    genjax.Pytree.const(["object_pose_0"]),
                    test_poses[samples[t]],
                )
                b3d.rerun_visualize_trace_t(trace_, t)
                rr.set_time_sequence("frame", t)
                rr.log("alternate_view_image", rr.Image(alternate_view_images[t, ...]))
                rr.log("alternate_view_image/gt", rr.Image(alternate_view_gt_image))
                rr.log(
                    "text",
                    rr.TextDocument(f"{delta_cps[samples[t]]} \n {scores[samples[t]]}"),
                )

        return samples, scores, delta_cps

    @staticmethod
    def _generate_plot(samples, scores, delta_cps, IMAGE_WIDTH, IMAGE_HEIGHT):
        """
        Optionally, generate a plot depicting the sample likelihoods and posterior sample angles
        """
        print("Generating plot")

        f, axes = plt.subplots(1, 2, figsize=(10, 25))
        circles = []
        circle_radius = 0.4

        angle_to_coord = lambda rad: (
            0.5 + circle_radius * jnp.cos(rad),
            0.5 + circle_radius * jnp.sin(rad),
        )

        for ax in axes:
            ax.set_box_aspect(1)
            ax.axis("off")
            circle = plt.Circle(
                (0.5, 0.5), circle_radius, edgecolor="gray", facecolor="white"
            )
            circles.append(circle)

        axes[0].add_patch(circles[0])
        axes[0].set_title(
            f"Z axes mean particle scores ({len(scores)} hypotheses)\nimg ({IMAGE_WIDTH},{IMAGE_HEIGHT})"
        )
        axes[1].add_patch(circles[1])
        axes[1].set_title(
            f"Z axes samples ({len(samples)} samples)\nimg ({IMAGE_WIDTH},{IMAGE_HEIGHT})"
        )

        ## (1) plot the enumerated scores
        score_viz = []
        for i in tqdm(range(len(scores))):
            score_viz.append(angle_to_coord(delta_cps[i, -1]))
        score_viz = jnp.asarray(score_viz)
        unique_angles, assignments = jnp.unique(score_viz, return_inverse=True, axis=0)
        score_viz_unique = jnp.asarray(
            [
                [*angle, jnp.mean(scores[(assignments == i).reshape(-1)])]
                for (i, angle) in enumerate(unique_angles)
            ]
        )
        normalized_scores = (
            score_viz_unique[:, -1] - score_viz_unique[:, -1].min()
        ) / (score_viz_unique[:, -1].max() - score_viz_unique[:, -1].min())
        sc = axes[0].scatter(
            score_viz_unique[:, 0], score_viz_unique[:, 1], c=normalized_scores
        )
        cbar = plt.colorbar(sc, ax=axes[0], fraction=0.046, pad=0.04)

        ## plot the sampled z angles
        _freqs = dict()
        for sample in tqdm(samples):  # count each unique sample
            if sample.item() not in _freqs:
                _freqs[sample.item()] = 0
            _freqs[sample.item()] += 1
        _freqs_array = jnp.array(
            [unique_sample_occurrence for unique_sample_occurrence in _freqs.values()]
        )
        freqs = _freqs_array / _freqs_array.sum()
        unique_sample_coords = jnp.asarray(
            [
                angle_to_coord(delta_cps[unique_sample, -1])
                for unique_sample in _freqs.keys()
            ]
        )
        sc1 = axes[1].scatter(
            unique_sample_coords[:, 0], unique_sample_coords[:, 1], c=freqs, alpha=0.5
        )

        return f


if __name__ == "__main__":
    ## Setup rerun
    rr.init(f"resolution_invariance")
    rr.connect("127.0.0.1:8812")

    testobj = TestImgResolutionInvariance()
    testobj.test_uncertain(plot=False, rerun=False)
    testobj.test_peaky(plot=False, rerun=False)
