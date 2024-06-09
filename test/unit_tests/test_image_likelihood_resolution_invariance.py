# #### Resolution invariance test for likelihoods
# # Test likelihoods for invariance to the resolution of the image space.
# # The inferred posterior distribution of an object's pose
# # should NOT become peakier when the scene/observed images gain resolution.


# import jax.numpy as jnp
# import jax
# import os
# import trimesh
# import b3d
# from b3d import Pose
# import unittest

# class UpsamplingRenderer(b3d.Renderer):
#     """
#     Renderer that upsamples rendered image to a desired resolution.
#     Designed for image invariance resolution test, to express images that
#     have more pixels but equal amount of "information"
#     """
#     def __init__(self, *init_args):
#         super().__init__(*init_args)
#         self.IMAGE_WIDTH = self.width
#         self.IMAGE_HEIGHT = self.height

#     def render_attribute(self, *render_inputs):  ## overload
#         rgb, depth = super().render_attribute(*render_inputs)
#         return (jax.image.resize(rgb, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT,3), 'nearest'),
#                 jax.image.resize(depth, (self.IMAGE_WIDTH, self.IMAGE_HEIGHT), 'nearest'))


# class TestImgResolutionInvarianceLikelihood(unittest.TestCase):
#     """
#     Assert that the posterior over poses maintains same landscape (i.e. no significant change to variance)
#     across changes in image resolutions.
#     """
#     def setUp(self):  # TODO pass in mesh path
#         ## load desired mesh and add to library
#         MESH_PATH = os.path.join(b3d.get_root_path(),"assets/shared_data_bucket/025_mug/textured.obj")
#         mesh = trimesh.load(MESH_PATH)

#         vertices = jnp.array(mesh.vertices) * 5.0
#         vertices = vertices - vertices.mean(0)
#         faces = jnp.array(mesh.faces)
#         vertex_colors = vertices * 0.0 + jnp.array([1.0, 0.0, 0.0])
#         vertex_colors = (jnp.array(mesh.visual.to_color().vertex_colors)[...,:3] / 255.0 )
#         ranges = jnp.array([[0, len(faces)]])
#         self.object_library = b3d.MeshLibrary.make_empty_library()
#         self.object_library.add_object(vertices, faces, vertex_colors)
#         print(f"{self.object_library.get_num_objects()} object(s) in library")

#         #####
#         # resolutions to test
#         #####
#         self.TEST_RESOLUTIONS =  [
#                             (320,320),
#                             (160,160),
#                             (80,80),
#                             ]

#         ####
#         # setup renderer
#         ####
#         RENDERER_IMAGE_WIDTH, RENDERER_IMAGE_HEIGHT = self.TEST_RESOLUTIONS[-1]  # lowest resolution (will upsample for image reoslution varying)
#         fx, fy, cx, cy, near, far = RENDERER_IMAGE_WIDTH*2, RENDERER_IMAGE_HEIGHT*2, RENDERER_IMAGE_WIDTH/2, RENDERER_IMAGE_HEIGHT/2, 0.01, 10.0

#         self.renderer = UpsamplingRenderer(
#             RENDERER_IMAGE_WIDTH, RENDERER_IMAGE_HEIGHT, fx, fy, cx, cy, near, far
#         )

#     def test_invariance(self, variance_atol=1e-4):
#         """
#         Across image resolutions, test that image likelihood is same under the same observation model
#         """
#         # model args
#         color_error, depth_error = (40.0, 0.02)
#         inlier_score, outlier_prob = (5.0, 0.00001)
#         color_multiplier, depth_multiplier = (500.0, 500.0)

#         model_args = b3d.model.ModelArgs(color_error, depth_error,
#                             inlier_score, outlier_prob,
#                             color_multiplier, depth_multiplier)

#         # setup scene
#         camera_pose = Pose.from_position_and_target(
#             jnp.array([0.0, 3.0, 0.0]),
#             jnp.array([0.0, 0.0, 0.0]),
#             up=jnp.array([0,0,1])
#         )
#         gt_translation = jnp.array([-0.005, 0.01, 0])
#         gt_rotation_z = b3d.Rot.from_euler('z', -jnp.pi, degrees=False).quat
#         obs_pose = Pose.identity()  # arbitrary pose for observation
#         gt_pose = Pose(gt_translation, gt_rotation_z) # camera frame pose

#         logpdfs = []

#         for IMAGE_WIDTH, IMAGE_HEIGHT in self.TEST_RESOLUTIONS:
#             self.renderer.IMAGE_WIDTH = IMAGE_WIDTH
#             self.renderer.IMAGE_HEIGHT = IMAGE_HEIGHT

#             gt_rgb_img, gt_depth = self.renderer.render_attribute((camera_pose.inv() @ gt_pose)[None,...],
#                                                         self.object_library.vertices,
#                                                         self.object_library.faces,
#                                                         self.object_library.ranges[jnp.array([0])],
#                                                         self.object_library.attributes)
#             obs_rgb_img, obs_depth = self.renderer.render_attribute((camera_pose.inv() @ obs_pose)[None,...],
#                                                         self.object_library.vertices,
#                                                         self.object_library.faces,
#                                                         self.object_library.ranges[jnp.array([0])],
#                                                         self.object_library.attributes)

#             logpdf = self.sensor_model.logpdf(
#                 (obs_rgb_img, obs_depth), gt_rgb_img, gt_depth, model_args
#             )
#             logpdfs.append(logpdf)
#         logpdfs = jnp.asarray(logpdfs)
#         var_logpdfs = jnp.var(logpdfs)

#         message = "logpdf variance across resolutions should be lower"
#         self.assertAlmostEqual(var_logpdfs, 0, delta=variance_atol, msg=message)

# if __name__ == '__main__':
#     testobj = TestImgResolutionInvarianceLikelihood()
#     testobj.setUp()  # automatically run if run in unittest
#     testobj.test_invariance()
