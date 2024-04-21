#### Resolution invariance test for likelihoods
# Test likelihoods for invariance to the resolution of the image space. 
# The inferred posterior distribution of an object's pose 
# should NOT become peakier when the scene/observed images gain resolution.


import jax.numpy as jnp
import jax
import genjax
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
import os
import trimesh
import b3d
from jax.scipy.spatial.transform import Rotation as Rot
from b3d import Pose
import rerun as rr
from tqdm import tqdm
from pathlib import Path
import unittest
       
class TestImgResolutionInvariance(unittest.TestCase):
    def setup(self, rerun, plot):  # TODO pass in mesh path
        self.rerun = rerun
        self.plot = plot
        
        ## load desired mesh and add to library
        MESH_PATH = os.path.join(b3d.get_root_path(),"assets/shared_data_bucket/025_mug/textured.obj")
        mesh = trimesh.load(MESH_PATH) 

        vertices = jnp.array(mesh.vertices) * 5.0  
        vertices = vertices - vertices.mean(0)
        faces = jnp.array(mesh.faces)
        vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
        vertex_colors = (jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0 ) 
        ranges = jnp.array([[0, len(faces)]])
        self.object_library = b3d.MeshLibrary.make_empty_library()
        self.object_library.add_object(vertices, faces, vertex_colors)
        print(f"{self.object_library.get_num_objects()} object(s) in library")
        
    def test_peaky(self,rerun=False, plot=False):
        self.setup(rerun, plot)
        angle_var_bound, mean_var_bound = 1e-4, 1e-4
        self._test_common(-jnp.pi, angle_var_bound, mean_var_bound)
        
    def test_uncertain(self, rerun=False, plot=False):
        self.setup(rerun, plot)
        angle_var_bound, mean_var_bound = 1e-4, 1e-4
        self._test_common(-jnp.pi*0.55, angle_var_bound, mean_var_bound)
        
    def _test_common(self, gt_rotation_angle_z, angle_var_bound=1e-4, mean_var_bound=1e-4):
        #############
        # get posterior per resolution
        ##############
        object_id = 0
        color_error, depth_error = (40.0, 0.02)
        inlier_score, outlier_prob = (5.0, 0.00001)
        color_multiplier, depth_multiplier = (500.0, 500.0)
        num_x_tr, num_y_tr, num_x_rot, num_z_rot = 11, 11, 5, 81
        
            
        test_resolutions = [(50,50), 
                            (75,75), 
                            (100,100), 
                            (150,150), 
                            (200,200)]       
        samples_variances = []
        samples_means = []
        scores_variances = []
        scores_means = []
        for IMAGE_WIDTH, IMAGE_HEIGHT in test_resolutions:
            print(f"========TESTING RESOLUTION ({IMAGE_WIDTH}, {IMAGE_HEIGHT})========")
            
            ## Setup rest of intrinsics and renderer
            fx, fy, cx, cy, near, far = IMAGE_WIDTH*2, IMAGE_HEIGHT*2, IMAGE_WIDTH/2, IMAGE_HEIGHT/2, 0.01, 10.0
            self.renderer = b3d.Renderer(
                IMAGE_WIDTH, IMAGE_HEIGHT, fx, fy, cx, cy, near, far
            )
            
            ###########
            # Setup test image (no self-occlusion)
            ###########
            camera_pose = Pose.from_position_and_target(
                jnp.array([0.0, 4.0, 0.0]),
                jnp.array([0.0, 0.0, 0.0]),
                up=jnp.array([0,0,1])
            )
            _gt_translation = jnp.array([-0.005, 0.01, 0])
            _gt_rotation_angle_z = gt_rotation_angle_z 
            _gt_rotation_z = b3d.Rot.from_euler('z', _gt_rotation_angle_z, degrees=False).quat
            gt_pose_cam = Pose(jnp.array([0,0,0]), _gt_rotation_z) # camera frame pose

            gt_img, gt_depth = self.renderer.render_attribute((camera_pose.inv() @ gt_pose_cam)[None,...], 
                                                        self.object_library.vertices, 
                                                        self.object_library.faces, 
                                                        self.object_library.ranges[jnp.array([object_id])], 
                                                        self.object_library.attributes)
            if self.plot:
                _, axes = plt.subplots(1,1, figsize=(10,10))
                axes.imshow(gt_img)
                plt.title(f"GT image ({IMAGE_WIDTH}, {IMAGE_HEIGHT}):\ntr {_gt_translation}, z rotation (-0.55pi), x rotation (0.03pi)")
                plt.savefig(f"{IMAGE_HEIGHT}_{IMAGE_WIDTH}_gt_{gt_rotation_angle_z}.png")    
                
            samples, scores, pose_enums = self.get_gridding_posterior(camera_pose, gt_pose_cam, gt_img, gt_depth,
                                                                color_error, depth_error,
                                                                inlier_score, outlier_prob,
                                                                color_multiplier, depth_multiplier,
                                                                num_x_tr, num_y_tr, num_x_rot, num_z_rot)

            if self.plot:
                f = self._generate_plot(samples, scores, pose_enums, IMAGE_WIDTH, IMAGE_HEIGHT)
                f.savefig(f"{IMAGE_WIDTH}_{IMAGE_HEIGHT}_viz_{gt_rotation_angle_z}.png",bbox_inches='tight')
            
            samples_variance = jnp.var(pose_enums[samples], axis=0)
            samples_mean = jnp.mean(pose_enums[samples], axis=0)
            samples_variances.append(samples_variance)
            samples_means.append(samples_mean)

            scores_variances.append(jnp.var(scores, axis=0))
            scores_means.append(jnp.mean(scores, axis=0))
            
            del self.renderer

            
        samples_variances, samples_means = jnp.asarray(samples_variances), jnp.asarray(samples_means)
        scores_variances, scores_means = jnp.asarray(scores_variances), jnp.asarray(scores_means)

        #############
        # Asserts on sample statistics
        #############
        # variance peakiness should be similar across resolutions for (1) posterior samples and (2) uniform grid samples
        assert jnp.var(samples_variances[:, 2]) <= angle_var_bound, f"{samples_variances[:,2]} should all be under {angle_var_bound}"
        assert jnp.abs(scores_variances.max() - scores_variances.min()) <= jnp.abs(0.05*jnp.mean(scores_variances)), f"{jnp.var(scores_variances)} should all be under {0.05*jnp.mean(scores_variances)}"
 
    def get_gridding_posterior(self, camera_pose, gt_pose_cam, gt_img, gt_depth, 
                                color_error, depth_error,
                                inlier_score, outlier_prob,
                                color_multiplier, depth_multiplier,
                                num_x_tr=11, num_y_tr=11, num_x_rot=5, num_z_rot=81,
                                ):
        ###########
        # importance sampling
        ###########
        print("scoring hypotheses")

        model = b3d.model_multiobject_gl_factory(self.renderer)
        key = jax.random.PRNGKey(0)

        ## sampling grid
        cp_to_pose = lambda cp: Pose(jnp.array([cp[0], cp[1], 0.0]), b3d.Rot.from_rotvec(jnp.array([cp[2], 0.0, cp[3]])).as_quat())

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

        model_args = b3d.model.ModelArgs(color_error, depth_error,
                                   inlier_score, outlier_prob,
                                   color_multiplier, depth_multiplier)
        arguments = (
                jnp.arange(1),
                model_args,
                self.object_library
            )

        ## init trace 
        gt_trace, _ = model.importance(
            jax.random.PRNGKey(0),
            genjax.choice_map(
                {
                    "camera_pose": camera_pose,
                    "object_pose_0": gt_pose_cam,
                    "object_0": 0,
                    "observed_rgb_depth": (gt_img, gt_depth)
                }
            ), 
            arguments
        )
        b3d.rerun_visualize_trace_t(gt_trace, 0)

        ## get IS scores over the enum grid
        test_poses = gt_trace["object_pose_0"] @ cp_delta_poses
        test_poses_batches = test_poses.split(10)
        scores = jnp.concatenate([b3d.enumerate_choices_get_scores_jit(gt_trace, key, genjax.Pytree.const(["object_pose_0"]), poses) 
                                for poses in test_poses_batches])
        samples = jax.random.categorical(key, scores, shape=(500,))  # samples from posterior (prior was a uniform)


        ###########
        # rerun visualization
        ###########
        if self.rerun:
            print("generating rerun vizs")

            alternate_camera_pose = Pose.from_position_and_target(
                jnp.array([0.01, 0.000, 1.5]),
                gt_pose_cam.pos
            )  # alt pose to see occluded side from above

            alternate_view_images,_  = self.renderer.render_attribute_many(
                (alternate_camera_pose.inv() @  test_poses[samples])[:, None,...],
                self.object_library.vertices, self.object_library.faces,
                self.object_library.ranges[jnp.array([0])],
                self.object_library.attributes
            )

            alternate_view_gt_image,_  = self.renderer.render_attribute(
                (alternate_camera_pose.inv() @  gt_pose_cam)[None,...],
                self.object_library.vertices, self.object_library.faces,
                self.object_library.ranges[jnp.array([0])],
                self.object_library.attributes
            )

            for t in tqdm(range(len(samples))):
                trace_ = b3d.update_choices_jit(gt_trace, key,  genjax.Pytree.const(["object_pose_0"]), test_poses[samples[t]])
                b3d.rerun_visualize_trace_t(trace_, t)
                rr.set_time_sequence("frame", t)
                rr.log("alternate_view_image", rr.Image(alternate_view_images[t,...]))
                rr.log("alternate_view_image/gt", rr.Image(alternate_view_gt_image))
                rr.log("text", rr.TextDocument(f"{delta_cps[samples[t]]} \n {scores[samples[t]]}"))
                
        return samples, scores, delta_cps

    @staticmethod
    def _generate_plot(samples, scores, delta_cps, IMAGE_WIDTH, IMAGE_HEIGHT):
        print("Generating plot")
        
        f, axes = plt.subplots(1,2, figsize=(10,25))
        circles = []
        circle_radius = 0.4

        angle_to_coord = lambda rad: (0.5 + circle_radius*jnp.cos(rad), 0.5 + circle_radius*jnp.sin(rad))
        gt_coord = angle_to_coord(0)

        for ax in axes: 
            ax.set_box_aspect(1)
            ax.axis('off')
            circle = plt.Circle((0.5, 0.5), circle_radius, edgecolor='gray',facecolor='white')
            circles.append(circle)

        axes[0].add_patch(circles[0]); axes[0].set_title(f"Z axes mean particle scores ({len(scores)} hypotheses)\nimg ({IMAGE_WIDTH},{IMAGE_HEIGHT})")
        axes[1].add_patch(circles[1]); axes[1].set_title(f"Z axes samples ({len(samples)} samples)\nimg ({IMAGE_WIDTH},{IMAGE_HEIGHT})")

        ## (1) plot the enumerated scores
        score_viz = []
        for i in tqdm(range(len(scores[::10]))):
            score_viz.append(angle_to_coord(delta_cps[i*10,-1]))
        score_viz = jnp.asarray(score_viz)
        unique_angles, assignments = jnp.unique(score_viz, return_inverse=True, axis=0)
        score_viz_unique = jnp.asarray([[*angle, jnp.mean(scores[::10][(assignments==i).reshape(-1)])] for (i, angle) in enumerate(unique_angles)])
        normalized_scores = (score_viz_unique[:,-1] - score_viz_unique[:,-1].min())/(score_viz_unique[:,-1].max() - score_viz_unique[:,-1].min())
        sc = axes[0].scatter(score_viz_unique[:, 0], score_viz_unique[:, 1], c=normalized_scores)
        cbar = plt.colorbar(sc, ax=axes[0],fraction=0.046, pad=0.04)

        ## plot the sampled z angles
        _freqs = dict()
        for sample in tqdm(samples):  # count each unique sample
            if sample.item() not in _freqs:
                _freqs[sample.item()] = 0
            _freqs[sample.item()] += 1
        _freqs_array = jnp.array([unique_sample_occurrence for unique_sample_occurrence in _freqs.values()])
        freqs = _freqs_array/_freqs_array.sum()
        unique_sample_coords = jnp.asarray([angle_to_coord(delta_cps[unique_sample,-1]) for unique_sample in _freqs.keys()])
        sc1 = axes[1].scatter(unique_sample_coords[:, 0], unique_sample_coords[:, 1], c=freqs, alpha=0.1)   
        # cbar1 = plt.colorbar(sc1, ax=axes[1],fraction=0.046, pad=0.04)
        return f


if __name__ == '__main__':
    ## Setup rerun
    rr.init(f"resolution_invariance")
    rr.connect("127.0.0.1:8812")
        
    testobj = TestImgResolutionInvariance()
    testobj.test_uncertain(plot=False)
    testobj.test_peaky(plot=False)